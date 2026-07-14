create_onnx_external(ncs.snac_32kh.decode ${CMAKE_CURRENT_SOURCE_DIR}/ncs.snac_32kh.decode.cpp)
set_property(GLOBAL APPEND PROPERTY ALL_EXTERNAL_TARGETS ncs.snac_32kh.decode)
