include_directories(".")

file(GLOB SRC_LIST "*.cu")
cuda_add_library(smnv SHARED ${SRC_LIST})
