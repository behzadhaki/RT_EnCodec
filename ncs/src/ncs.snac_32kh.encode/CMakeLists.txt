create_onnx_external(ncs.snac_32kh.encode ${CMAKE_CURRENT_SOURCE_DIR}/ncs.snac_32kh.encode.cpp)
set_property(GLOBAL APPEND PROPERTY ALL_EXTERNAL_TARGETS ncs.snac_32kh.encode)
