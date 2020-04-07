cmake_minimum_required(VERSION 3.5.1)

macro(smnet_option TARGET STR VALUE)
  option(${TARGET} ${STR} ${VALUE})
  if (${TARGET})
    add_definitions(-D${TARGET})
  endif()
endmacro()

smnet_option(WITH_CUDA "Compile with cuda" ON)
smnet_option(WITH_MLU "Compile with mlu" OFF)

add_compile_options(-std=c++11)
add_compile_options(-O3)
add_compile_options(-fPIC)
add_compile_options(-Wall)
add_compile_options(-Werror)

if (WITH_CUDA)
  find_package(CUDA REQUIRED)
  link_directories(/usr/local/cuda/lib64)
  link_directories(/usr/local/lib)
endif()

if (WITH_MLU)
  find_package(BANG REQUIRED)
  include_directories(/usr/local/neuware/include)
  link_directories(/usr/local/neuware/lib64)
endif()

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

add_subdirectory(cblas)
if (WITH_CUDA)
  add_subdirectory(gpu)
endif()
if (WITH_MLU)
  add_subdirectory(mlu)
endif()
