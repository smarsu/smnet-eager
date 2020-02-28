include_directories(".")

file(GLOB SRC_LIST "*.cc" )
add_library(smcb SHARED ${SRC_LIST})
