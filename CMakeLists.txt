cmake_minimum_required (VERSION 3.9)
project (NeuralNetwork)

# utilities
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindCUDNN.cmake)

# NVCC flags
find_package(CUDA 7 REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++14)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Xcompiler)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-lineinfo)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-G)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-g)
else()
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-DNDEBUG)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3)
endif()

list(APPEND SMS 30 35 50 53 60 61 70 72 75)
foreach(SM ${SMS})
    set(CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};-gencode arch=compute_${SM},code=sm_${SM})
endforeach()
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_75,code=compute_75)

set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# Addresses a bug where code is not compiled as C++11 in non-CUDA code and
# older g++ versions
add_compile_options(-std=c++17)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUDNN_INCLUDE_DIRS})
link_directories(${CUDNN_LIBRARY})

# nn library
include_directories(include)
set(NN_LIB nn)
add_library(nn SHARED
    src/network_impl.cpp
    src/layer.cpp
    src/layers/input.cpp
    src/layers/conv.cpp
    src/layers/sigmoid.cpp
    src/layers/relu.cpp
    src/layers/dropout.cpp
    src/layers/max_pool.cpp
    src/layers/avg_pool.cpp
    src/layers/unpool.cpp
    src/layers/fc.cpp
    src/layers/softmax.cpp
)
target_link_libraries(${NN_LIB} ${CUDA_cublas_LIBRARY})
target_link_libraries(${NN_LIB} ${CUDA_curand_LIBRARY})
target_link_libraries(${NN_LIB} cudnn)
target_link_libraries(${NN_LIB} stdc++fs)
