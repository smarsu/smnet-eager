include_directories(".")

file(GLOB SRC_LIST "*.cc")
add_library(smcn SHARED ${SRC_LIST})
target_link_libraries(smcn libglog.so libcnrt.so smcn_kernel)
