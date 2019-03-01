/*
 * Copyright 2019, Peter Han, All rights reserved.
 * This code is released into the public domain.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
#pragma once
#include <iostream>
#include <sstream>
#include <string>

// Error handling
// Adapted from the CUDNN classification code
// sample: https://developer.nvidia.com/cuDNN

#define FatalError(s)                                                     \
    do {                                                                  \
        std::stringstream _where, _message;                               \
        _where << __FILE__ << ':' << __LINE__;                            \
        _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__; \
        std::cerr << _message.str() << "\nAborting...\n";                 \
        cudaDeviceReset();                                                \
        system("pause");                                                  \
        exit(1);                                                          \
    } while (0)

#define checkCUDNN(status)                                              \
    do {                                                                \
        std::stringstream _error;                                       \
        if (status != CUDNN_STATUS_SUCCESS) {                           \
            _error << "CUDNN failure: " << cudnnGetErrorString(status); \
            FatalError(_error.str());                                   \
        }                                                               \
    } while (0)

#define checkCudaErrors(status)                                \
    do {                                                       \
        std::stringstream _error;                              \
        if (status != 0) {                                     \
            _error << "Cuda failure: " << status << " "        \
                   << cudaGetErrorString((cudaError_t)status); \
            FatalError(_error.str());                          \
        }                                                      \
    } while (0)
