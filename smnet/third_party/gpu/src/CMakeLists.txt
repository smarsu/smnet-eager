include_directories(".")

# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11;-use_fast_math;-O3;)
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11;-O3;)

file(GLOB SRC_LIST "*.cu")
cuda_add_library(smnv SHARED ${SRC_LIST} OPTIONS -std=c++11;-O3;)
target_link_libraries(smnv libglog.so libcudnn.so)
