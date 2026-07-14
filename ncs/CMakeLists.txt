cmake_minimum_required(VERSION 3.16)
project(ncs VERSION 1.0.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
    set(PLATFORM_WINDOWS TRUE)
    set(PLATFORM_MACOS FALSE)
elseif(APPLE)
    set(PLATFORM_WINDOWS FALSE)
    set(PLATFORM_MACOS TRUE)
else()
    message(FATAL_ERROR "Unsupported platform")
endif()

if(PLATFORM_MACOS)
    set(ENABLE_CODESIGNING ON CACHE BOOL "Enable automatic codesigning")
    if(ENABLE_CODESIGNING)
        set(CODESIGN_FLAGS --force --deep -s -)
    endif()
endif()

include(cmake/CPM.cmake)
CPMAddPackage("gh:Cycling74/min-api#main")

# libsamplerate -- buffer-mode SNAC/EnCodec modules run at a fixed model
# sample rate (e.g. 24kHz) but read/write buffer~ content assumed to be at
# Max's live samplerate, so encode/decode resample once per call (a plain
# src_simple() call, not the persistent-state streaming resampler used by
# real-time signal modules). Linked in automatically wherever
# create_simple_external/create_onnx_external find the `samplerate` target.
include(FetchContent)
set(LIBSAMPLERATE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(libsamplerate
    GIT_REPOSITORY https://github.com/libsndfile/libsamplerate.git
    GIT_TAG 0.2.2
)
# libsamplerate 0.2.2's own CMakeLists.txt declares
# cmake_minimum_required(VERSION 3.1), which CMake >= 4.0 refuses outright
# (compatibility below 3.5 was removed). Scoped so it only affects this
# vendored dependency's own configure, not the rest of the project.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
FetchContent_MakeAvailable(libsamplerate)
unset(CMAKE_POLICY_VERSION_MINIMUM)

# Locate max-sdk-base (c74_max.h, ext.h etc.)
set(_MSB "${min-api_SOURCE_DIR}/max-sdk-base")
set(_MSB2 "${CMAKE_SOURCE_DIR}/lib/max-sdk/source/max-sdk-base")
if(EXISTS "${min-api_SOURCE_DIR}/.git" AND EXISTS "${min-api_SOURCE_DIR}/.gitmodules")
    execute_process(COMMAND git submodule update --init --depth 1
        WORKING_DIRECTORY "${min-api_SOURCE_DIR}"
        RESULT_VARIABLE _r ERROR_QUIET OUTPUT_QUIET)
endif()
if(NOT EXISTS "${_MSB}/c74support/c74_max.h")
    if(EXISTS "${_MSB2}/c74support/c74_max.h")
        set(_MSB "${_MSB2}")
    else()
        set(_MSB "${CMAKE_BINARY_DIR}/_deps/max-sdk-base-build")
        if(NOT EXISTS "${_MSB}/c74support/c74_max.h")
            execute_process(COMMAND git clone --depth 1
                https://github.com/Cycling74/max-sdk-base.git "${_MSB}"
                RESULT_VARIABLE _r ERROR_QUIET OUTPUT_QUIET)
        endif()
    endif()
endif()
set(MAX_SDK_BASE_DIR "${_MSB}")


if(WIN32)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(ONNX_ARCHIVE_NAME "onnxruntime-win-x64-static_lib-1.20.1.tar.bz2")
        set(ONNX_SUBDIR_NAME  "onnxruntime-win-x64-static_lib-1.20.1")
    else()
        message(FATAL_ERROR "32-bit Windows builds are not supported")
    endif()
elseif(APPLE)
    if(CMAKE_OSX_ARCHITECTURES MATCHES "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        set(ONNX_ARCHIVE_NAME "onnxruntime-osx-x86_64-static_lib-1.22.0.zip")
        set(ONNX_SUBDIR_NAME  "onnxruntime-osx-x86_64-static_lib-1.22.0")
    else()
        set(ONNX_ARCHIVE_NAME "onnxruntime-osx-arm64-static_lib-1.20.0.zip")
        set(ONNX_SUBDIR_NAME  "onnxruntime-osx-arm64-static_lib-1.20.0")
    endif()
endif()
set(ONNX_ARCHIVE_FILE "${CMAKE_SOURCE_DIR}/${ONNX_ARCHIVE_NAME}")
if(NOT EXISTS "${ONNX_ARCHIVE_FILE}")
    message(FATAL_ERROR "ONNX archive not found: ${ONNX_ARCHIVE_FILE}")
endif()
set(ONNX_UNPACK_DIR "${CMAKE_BINARY_DIR}/external/onnxruntime_prebuilt")
set(ONNX_ROOT_DIR "${ONNX_UNPACK_DIR}/${ONNX_SUBDIR_NAME}")
if(NOT EXISTS "${ONNX_ROOT_DIR}/include/onnxruntime_c_api.h")
    file(MAKE_DIRECTORY "${ONNX_UNPACK_DIR}")
    if(WIN32)
        set(EXTRACTION_SUCCESS FALSE)
        find_program(_7Z 7z PATHS "C:/Program Files/7-Zip" "C:/Program Files (x86)/7-Zip")
        if(_7Z AND NOT EXTRACTION_SUCCESS)
            execute_process(COMMAND ${_7Z} x "${ONNX_ARCHIVE_FILE}" -o"${ONNX_UNPACK_DIR}" -y
                RESULT_VARIABLE _r1 OUTPUT_QUIET ERROR_QUIET)
            if(_r1 EQUAL 0)
                file(GLOB _tar "${ONNX_UNPACK_DIR}/*.tar")
                if(_tar)
                    execute_process(COMMAND ${_7Z} x "${_tar}" -o"${ONNX_UNPACK_DIR}" -y
                        RESULT_VARIABLE _r2 OUTPUT_QUIET ERROR_QUIET)
                    if(_r2 EQUAL 0)
                        set(EXTRACTION_SUCCESS TRUE)
                        file(REMOVE ${_tar})
                    endif()
                endif()
            endif()
        endif()
        if(NOT EXTRACTION_SUCCESS)
            message(FATAL_ERROR "Install 7-Zip or manually extract ${ONNX_ARCHIVE_FILE} to ${ONNX_UNPACK_DIR}")
        endif()
    else()
        execute_process(COMMAND unzip -o -q "${ONNX_ARCHIVE_FILE}" -d "${ONNX_UNPACK_DIR}"
            RESULT_VARIABLE _uz)
        if(NOT _uz EQUAL 0)
            message(FATAL_ERROR "Failed to unzip ${ONNX_ARCHIVE_FILE}")
        endif()
    endif()
endif()
if(WIN32)
    if(EXISTS "${ONNX_ROOT_DIR}/lib/onnxruntime.lib")
        set(ONNX_LIB "${ONNX_ROOT_DIR}/lib/onnxruntime.lib")
        set(ONNX_IS_STATIC TRUE)
    else()
        message(FATAL_ERROR "onnxruntime.lib not found")
    endif()
else()
    if(EXISTS "${ONNX_ROOT_DIR}/lib/libonnxruntime.a")
        set(ONNX_LIB "${ONNX_ROOT_DIR}/lib/libonnxruntime.a")
        set(ONNX_IS_STATIC TRUE)
    elseif(EXISTS "${ONNX_ROOT_DIR}/lib/libonnxruntime.dylib")
        set(ONNX_LIB "${ONNX_ROOT_DIR}/lib/libonnxruntime.dylib")
        set(ONNX_IS_STATIC FALSE)
    else()
        message(FATAL_ERROR "libonnxruntime not found")
    endif()
endif()




set(COMMON_INCLUDES ${min-api_SOURCE_DIR}/include
    ${MAX_SDK_BASE_DIR}/c74support
    ${MAX_SDK_BASE_DIR}/c74support/max-includes
    ${CMAKE_SOURCE_DIR}/src/include ${ONNX_ROOT_DIR}/include )

if(PLATFORM_WINDOWS)
    set(PLATFORM_LIBS kernel32 user32 gdi32 winspool shell32 ole32 oleaut32
        uuid comdlg32 advapi32 winmm ws2_32 crypt32 bcrypt)
else()
    find_library(_cf CoreFoundation)
    find_library(_fn Foundation)
    find_library(_cs CoreServices)
    find_library(_se Security)
    find_library(_sc SystemConfiguration)
    find_library(_ob objc)
    set(PLATFORM_LIBS ${_cf} ${_fn} ${_cs} ${_se} ${_sc} ${_ob})
endif()

if(PLATFORM_WINDOWS)
    set(MAX_PACKAGE_NAME "ncs" CACHE STRING "Package folder name")
    set(MAX_USER_PACKAGES_DIR "$ENV{USERPROFILE}/Documents/Max 9/Packages" CACHE PATH "Max Packages directory")
else()
    set(MAX_PACKAGE_NAME "ncs" CACHE STRING "Package folder name")
    set(MAX_USER_PACKAGES_DIR "$ENV{HOME}/Documents/Max 9/Packages" CACHE PATH "Max Packages directory")
endif()
if(MAX_USER_PACKAGES_DIR STREQUAL "" OR MAX_PACKAGE_NAME STREQUAL "")
    message(FATAL_ERROR
        "MAX_USER_PACKAGES_DIR or MAX_PACKAGE_NAME is empty "
        "(MAX_USER_PACKAGES_DIR='${MAX_USER_PACKAGES_DIR}', MAX_PACKAGE_NAME='${MAX_PACKAGE_NAME}'). "
        "This usually means an earlier CMake configure attempt cached an empty value for "
        "one of these — set(... CACHE ...) does not overwrite an existing cache entry. "
        "Delete CMakeCache.txt (or your whole build directory) and reconfigure.")
endif()
set(MAX_PACKAGE_DIR "${MAX_USER_PACKAGES_DIR}/${MAX_PACKAGE_NAME}")
set(MAX_EXTERNALS_DIR "${MAX_PACKAGE_DIR}/externals")
message(STATUS "Deploying externals to: ${MAX_EXTERNALS_DIR}")

# Externals auto-load their ONNX model via BundleResourceLoader, which
# resolves paths relative to the deployed package root (it truncates the
# loaded dylib's own path at "/externals/"). Symlink (not copy) since the
# model exports run into the hundreds of MB across sample rates.
set(MAX_MODELS_DIR "${MAX_PACKAGE_DIR}/models")
if(NOT EXISTS "${MAX_MODELS_DIR}")
    file(MAKE_DIRECTORY "${MAX_PACKAGE_DIR}")
    if(PLATFORM_WINDOWS)
        file(COPY "${CMAKE_SOURCE_DIR}/models/" DESTINATION "${MAX_MODELS_DIR}")
    else()
        file(CREATE_LINK "${CMAKE_SOURCE_DIR}/models" "${MAX_MODELS_DIR}" SYMBOLIC)
    endif()
endif()

function(create_simple_external T S)
    # The CMake target/filename convention is "..._tilde" (CMake- and
    # filesystem-safe); the compiled bundle Max actually discovers on disk
    # must be named with a literal "~" so it matches what min-api registers
    # via class_new AND what Max's filename-driven object-box autocomplete
    # suggests. Only the on-disk OUTPUT_NAME gets the substitution — the
    # CMake target identifier (${T}) stays "_tilde" throughout.
    string(REPLACE "_tilde" "~" MAXNAME "${T}")
    add_library(${T} MODULE ${S})
    target_include_directories(${T} PRIVATE ${COMMON_INCLUDES})
    target_link_options(${T} PRIVATE $<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>)
    if(PLATFORM_WINDOWS)
        find_library(MAXAPI_LIB NAMES MaxAPI
            PATHS "${MAX_SDK_BASE_DIR}/c74support/max-includes/x64" NO_DEFAULT_PATH)
        find_library(MAXAUDIO_LIB NAMES MaxAudio
            PATHS "${MAX_SDK_BASE_DIR}/c74support/msp-includes/x64" NO_DEFAULT_PATH)
        find_library(JITTER_LIB NAMES jitlib
            PATHS "${MAX_SDK_BASE_DIR}/c74support/jit-includes/x64" NO_DEFAULT_PATH)
        target_link_libraries(${T} PRIVATE ${PLATFORM_LIBS} ${MAXAPI_LIB} ${MAXAUDIO_LIB} ${JITTER_LIB})
        set_target_properties(${T} PROPERTIES SUFFIX ".mxe64" PREFIX "" OUTPUT_NAME ${MAXNAME})
    else()
        target_link_libraries(${T} PRIVATE ${PLATFORM_LIBS})
        set_target_properties(${T} PROPERTIES BUNDLE TRUE BUNDLE_EXTENSION mxo OUTPUT_NAME ${MAXNAME}
            INSTALL_NAME_DIR "@rpath" BUILD_WITH_INSTALL_RPATH TRUE
            MACOSX_BUNDLE_INFO_PLIST ${CMAKE_SOURCE_DIR}/cmake/Info.plist.in
            MACOSX_BUNDLE_BUNDLE_NAME ${MAXNAME}
            MACOSX_BUNDLE_GUI_IDENTIFIER "com.behzadhaki.${T}"
            MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0.0"
            MACOSX_BUNDLE_BUNDLE_VERSION "1.0.0"
            MACOSX_BUNDLE_COPYRIGHT "Behzad Haki")
        if(ENABLE_CODESIGNING)
            add_custom_command(TARGET ${T} POST_BUILD
                COMMAND xattr -r -d com.apple.quarantine "$<TARGET_BUNDLE_DIR:${T}>" || true
                COMMAND codesign ${CODESIGN_FLAGS} "$<TARGET_BUNDLE_DIR:${T}>")
        endif()
    endif()
    if(DEFINED FFTW3_FOUND AND FFTW3_FOUND)
        target_include_directories(${T} PRIVATE ${FFTW3_INCLUDE_DIRS})
        target_link_libraries(${T} PRIVATE ${FFTW3_LIBRARIES})
    endif()
    if(TARGET kissfft)
        target_link_libraries(${T} PRIVATE kissfft)
    endif()
    if(TARGET samplerate)
        target_link_libraries(${T} PRIVATE samplerate)
    endif()
    if(PLATFORM_WINDOWS)
        add_custom_command(TARGET ${T} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${MAX_EXTERNALS_DIR}"
            COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:${T}>"
                "${MAX_EXTERNALS_DIR}/$<TARGET_FILE_NAME:${T}>")
    else()
        add_custom_command(TARGET ${T} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${MAX_EXTERNALS_DIR}"
            COMMAND ${CMAKE_COMMAND} -E remove_directory "${MAX_EXTERNALS_DIR}/${MAXNAME}.mxo"
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "$<TARGET_BUNDLE_DIR:${T}>"
                "${MAX_EXTERNALS_DIR}/${MAXNAME}.mxo")
    endif()
endfunction()

function(create_onnx_external T S)
    string(REPLACE "_tilde" "~" MAXNAME "${T}")
    add_library(${T} MODULE ${S})
    target_include_directories(${T} PRIVATE ${COMMON_INCLUDES})
    target_link_options(${T} PRIVATE $<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>)
    target_link_libraries(${T} PRIVATE ${PLATFORM_LIBS} ${ONNX_LIB})
    if(PLATFORM_WINDOWS)
        find_library(MAXAPI_LIB NAMES MaxAPI
            PATHS "${MAX_SDK_BASE_DIR}/c74support/max-includes/x64" NO_DEFAULT_PATH)
        find_library(MAXAUDIO_LIB NAMES MaxAudio
            PATHS "${MAX_SDK_BASE_DIR}/c74support/msp-includes/x64" NO_DEFAULT_PATH)
        find_library(JITTER_LIB NAMES jitlib
            PATHS "${MAX_SDK_BASE_DIR}/c74support/jit-includes/x64" NO_DEFAULT_PATH)
        target_link_libraries(${T} PRIVATE ${PLATFORM_LIBS} ${MAXAPI_LIB} ${MAXAUDIO_LIB} ${JITTER_LIB})
        if(NOT ONNX_IS_STATIC AND EXISTS "${ONNX_ROOT_DIR}/lib/onnxruntime.dll")
            add_custom_command(TARGET ${T} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNX_ROOT_DIR}/lib/onnxruntime.dll" "${CMAKE_BINARY_DIR}/")
        endif()
        set_target_properties(${T} PROPERTIES SUFFIX ".mxe64" PREFIX "" OUTPUT_NAME ${MAXNAME})
    else()
        set_target_properties(${T} PROPERTIES BUNDLE TRUE BUNDLE_EXTENSION mxo OUTPUT_NAME ${MAXNAME}
            INSTALL_NAME_DIR "@rpath" BUILD_WITH_INSTALL_RPATH TRUE
            MACOSX_BUNDLE_INFO_PLIST ${CMAKE_SOURCE_DIR}/cmake/Info.plist.in
            MACOSX_BUNDLE_BUNDLE_NAME ${MAXNAME}
            MACOSX_BUNDLE_GUI_IDENTIFIER "com.behzadhaki.${T}"
            MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0.0"
            MACOSX_BUNDLE_BUNDLE_VERSION "1.0.0"
            MACOSX_BUNDLE_COPYRIGHT "Behzad Haki")
        if(NOT ONNX_IS_STATIC)
            add_custom_command(TARGET ${T} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNX_LIB}" "$<TARGET_FILE_DIR:${T}>/")
        endif()
        if(ENABLE_CODESIGNING)
            add_custom_command(TARGET ${T} POST_BUILD
                COMMAND xattr -r -d com.apple.quarantine "$<TARGET_BUNDLE_DIR:${T}>" || true
                COMMAND codesign ${CODESIGN_FLAGS} "$<TARGET_BUNDLE_DIR:${T}>")
        endif()
    endif()
    if(DEFINED FFTW3_FOUND AND FFTW3_FOUND)
        target_include_directories(${T} PRIVATE ${FFTW3_INCLUDE_DIRS})
        target_link_libraries(${T} PRIVATE ${FFTW3_LIBRARIES})
    endif()
    if(TARGET kissfft)
        target_link_libraries(${T} PRIVATE kissfft)
    endif()
    if(TARGET samplerate)
        target_link_libraries(${T} PRIVATE samplerate)
    endif()
    if(PLATFORM_WINDOWS)
        add_custom_command(TARGET ${T} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${MAX_EXTERNALS_DIR}"
            COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:${T}>"
                "${MAX_EXTERNALS_DIR}/$<TARGET_FILE_NAME:${T}>")
    else()
        add_custom_command(TARGET ${T} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${MAX_EXTERNALS_DIR}"
            COMMAND ${CMAKE_COMMAND} -E remove_directory "${MAX_EXTERNALS_DIR}/${MAXNAME}.mxo"
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "$<TARGET_BUNDLE_DIR:${T}>"
                "${MAX_EXTERNALS_DIR}/${MAXNAME}.mxo")
    endif()
endfunction()


# ===================================================================
# Module discovery
#
# Each module lives in its own src/<name>/ folder with its own
# CMakeLists.txt that calls create_simple_external()/create_onnx_external()
# and registers its target name in the ALL_EXTERNAL_TARGETS global
# property. Adding a module later means adding a new src/<name>/ folder —
# this file never lists module/target names itself, so it never needs to
# be hand-edited to add one.
# ===================================================================
file(GLOB MODULE_DIRS CONFIGURE_DEPENDS RELATIVE "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/src/*")
foreach(MODULE_DIR ${MODULE_DIRS})
    if(EXISTS "${CMAKE_SOURCE_DIR}/src/${MODULE_DIR}/CMakeLists.txt")
        add_subdirectory("src/${MODULE_DIR}")
    endif()
endforeach()

# ===================================================================
# Build-all convenience target
# ===================================================================
get_property(_all_external_targets GLOBAL PROPERTY ALL_EXTERNAL_TARGETS)
set(_all_deps ${_all_external_targets})
if(TARGET CopyExampleModels)
    list(APPEND _all_deps CopyExampleModels)
endif()
add_custom_target(ncsAll DEPENDS ${_all_deps})
