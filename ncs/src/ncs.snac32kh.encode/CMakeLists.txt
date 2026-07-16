create_onnx_external(ncs.snac32kh.encode ${CMAKE_CURRENT_SOURCE_DIR}/ncs.snac32kh.encode.cpp)
set_property(GLOBAL APPEND PROPERTY ALL_EXTERNAL_TARGETS ncs.snac32kh.encode)
