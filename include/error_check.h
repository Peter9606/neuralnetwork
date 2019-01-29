#pragma once
#include <iostream>
#include <sstream>

// Error handling
// Adapted from the CUDNN classification code
// sample: https://developer.nvidia.com/cuDNN

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    system("pause");                                                   \
    exit(1);                                                           \
} while(0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
        _error << "CUDNN failure: " << cudnnGetErrorString(status);    \
        FatalError(_error.str());                                      \
    }                                                                  \
} while(0)

#define checkCudaErrors(status) do {                                   \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
        _error << "Cuda failure: " << status << " "                    \
        << cudaGetErrorString((cudaError_t)status);                    \
        FatalError(_error.str());                                      \
    }                                                                  \
} while(0)
