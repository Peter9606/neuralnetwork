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


# Simple implementation to find CUDNN
# This script defines following variables:
# CUDNN_INCLUDE_DIRS - include directory for cudnn headers
# CUDNN_LIBRARY      - cudnn runtime library

# find cudnn.h and set its folder as include path
find_file(HEADER_PATH
    cudnn.h
    HINTS
    /usr/include
    /usr/local/include)
if("${HEADER_PATH}" STREQUAL HEADER_PATH-NOTFOUND)
    message(FATAL_ERROR "cannot find cudnn header file")
endif()
get_filename_component(CUDNN_INCLUDE_DIRS
    ${HEADER_PATH}
    DIRECTORY)

# find libcudnn.so and set its folder as library path
find_library(LIB_PATH
    libcudnn.so
    HINTS
    /usr/lib
    /usr/local/lib
    /usr/lib/x86_64-linux-gnu)
if("${LIB_PATH}" STREQUAL LIB_PATH-NOTFOUND)
    message(FATAL_ERROR "cannot find cudnn library")
endif()
get_filename_component(CUDNN_LIBRARY
    ${LIB_PATH}
    DIRECTORY)

#message(${CUDNN_INCLUDE_DIRS})
#message(${CUDNN_LIBRARY})
