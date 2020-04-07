include_directories(".")

file(GLOB SRC_LIST "*.mlu")
bang_add_library(smcn_kernel STATIC ${SRC_LIST} OPTIONS -std=c++11;-Os;--bang-mlu-arch=MLU270)
