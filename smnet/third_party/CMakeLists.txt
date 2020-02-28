cmake_minimum_required(VERSION 3.5.1)

message(STATUS "Compile Cblas ... " ${CMAKE_BUILD_TYPE})

add_compile_options(-std=c++11)
add_compile_options(-O2)
add_compile_options(-fPIC)
add_compile_options(-Wall)
add_compile_options(-Werror)

SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11;-use_fast_math;-O3;)

find_package(CUDA REQUIRED)
link_directories(/usr/local/cuda/lib64)

SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

add_subdirectory(cblas)
add_subdirectory(gpu)
