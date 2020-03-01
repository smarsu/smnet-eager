cmake_minimum_required(VERSION 3.5.1)

message(STATUS "Compile Cblas ... " ${CMAKE_BUILD_TYPE})

set(CMAKE_CXX_STANDARD 11)

add_compile_options(-std=c++11)
add_compile_options(-O3)
add_compile_options(-fPIC)
add_compile_options(-Wall)
add_compile_options(-Werror)

find_package(CUDA REQUIRED)
link_directories(/usr/local/cuda/lib64)
link_directories(/usr/local/lib)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

add_subdirectory(cblas)
add_subdirectory(gpu)
