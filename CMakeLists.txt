# Copyright 2019, Peter Han, All rights reserved.
# This code is released into the public domain.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

cmake_minimum_required (VERSION 3.9)
project (NeuralNetwork)

option(CPPCHECK "use cpplint to check code" ON)

# utilities
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindCUDNN.cmake)

# NVCC flags
find_package(CUDA 8 REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Xcompiler)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-lineinfo)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-G)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-g)
else()
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-DNDEBUG)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3)
endif()

list(APPEND SMS 60 61)
foreach(SM ${SMS})
    set(CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};-gencode arch=compute_${SM},code=sm_${SM})
endforeach()
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_60,code=compute_60)

set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# Addresses a bug where code is not compiled as C++11 in non-CUDA code and
# older g++ versions
add_compile_options(-std=c++17)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUDNN_INCLUDE_DIRS})
link_directories(${CUDNN_LIBRARY})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)

# nn library
include_directories(include)
cuda_add_library(${PROJECT_NAME} SHARED
    src/network_impl.cpp
    src/logger.cpp
    src/layer.cpp
    src/layers/activation.cpp
    src/layers/conv.cpp
    src/layers/dropout.cpp
    src/layers/fc.cpp
    src/layers/input.cpp
    src/layers/pool.cpp
    src/layers/softmax.cpp
    src/layers/unpool.cpp
    src/gpu/compution.cu
)
target_link_libraries(${PROJECT_NAME} ${CUDA_cublas_LIBRARY})
target_link_libraries(${PROJECT_NAME} ${CUDA_curand_LIBRARY})
target_link_libraries(${PROJECT_NAME} cudnn)
target_link_libraries(${PROJECT_NAME} stdc++fs)

#### TEST ####
cuda_add_executable(lenet
    src/lenet.cpp
    src/readubyte.cpp)
link_directories(${CMAKE_BINARY_DIR})
target_link_libraries(lenet ${PROJECT_NAME})


#### STATIC CHECK ####
if(CPPCHECK)
    set(CPPCHECK_TARGET "cpplint")
    find_file(CPPLINT cpplint)
    if(CPPLINT)
        message(STATUS "find cpplint executable")
    else()
        message(FALTAL_ERROR "Cannot find cpplint executable")
    endif()

    file(GLOB_RECURSE
         FILES_TO_CHECK
         src/*.*
         include/*.*)

    add_custom_target(${CPPCHECK_TARGET}
        COMMAND ${CPPLINT} ${FILES_TO_CHECK}
        DEPENDS ${FILES_TO_CHECK}
        COMMENT "cpplint checking")
endif()

list(APPEND URLS http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz)
list(APPEND URLS http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz)
list(APPEND URLS http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz)
list(APPEND URLS http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz)
list(LENGTH URLS COUNT)
math(EXPR COUNT ${COUNT}-1)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
foreach(idx RANGE 0 ${COUNT})
	list(GET URLS ${idx} URL_ITEM)
	get_filename_component(LONGNAME ${URL_ITEM} NAME)
	get_filename_component(SHORTNAME ${URL_ITEM} NAME_WE)

	if(EXISTS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${LONGNAME})
		message("${LONGNAME} already download")
		if(EXISTS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${LONGNAME})
			message("${SHORTNAME} already extracted")
		else()
			execute_process(
				WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
				COMMAND gzip --decompress ${LONGNAME}
				)
		endif()
	else()
		if(EXISTS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${SHORTNAME})
			message("${SHORTNAME} already extracted")
		else()
			execute_process(
				WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
				COMMAND wget ${URL_ITEM}
				)
			execute_process(
				WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
				COMMAND gzip --decompress ${LONGNAME}
				)
		endif()
	endif()
endforeach()

