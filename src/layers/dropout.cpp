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
#include "neuralnetwork/error_check.h"
#include "neuralnetwork/layers/dropout.h"

namespace nn {
namespace layers {
Dropout::Dropout(const std::string& name,
                 const NetworkConstPtr& network,
                 const LayerConstPtr& up,
                 float dropout_rate)
    : Layer(name, network, up)
    , dropout_rate_(dropout_rate) {
    const Dim dim = up->getDim();
    c_            = dim.c;
    h_            = dim.h;
    w_            = dim.w;

    checkCUDNN(cudnnCreateTensorDescriptor(&y_desc_));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        y_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n_, c_, h_, w_));

    checkCUDNN(cudnnCreateDropoutDescriptor(&dropout_desc_));
    checkCUDNN(cudnnDropoutGetReserveSpaceSize(y_desc_,
                                               &reserve_space_size_in_bytes_));
}

Dropout::~Dropout() {
    checkCUDNN(cudnnDestroyTensorDescriptor(y_desc_));
    checkCUDNN(cudnnDestroyDropoutDescriptor(dropout_desc_));

    checkCudaErrors(cudaFree(d_reserve_space_));
    checkCudaErrors(cudaFree(d_states_));
    checkCudaErrors(cudaFree(d_y_));
    checkCudaErrors(cudaFree(d_dy_));
}

size_t Dropout::prepareFwdPropagation() {
    NetworkConstPtr nn = network_.lock();
    assert(("Network is expired", nn));

    cudnnHandle_t cudnn_handle = nn->getCudnnHandle();
    size_t dropout_state_size;
    checkCUDNN(cudnnDropoutGetStatesSize(cudnn_handle, &dropout_state_size));
    checkCudaErrors(cudaMalloc(&d_states_, dropout_state_size));
    checkCUDNN(cudnnSetDropoutDescriptor(dropout_desc_,
                                         cudnn_handle,
                                         dropout_rate_,
                                         d_states_,
                                         dropout_state_size,
                                         /*Seed*/ time(NULL)));

    checkCudaErrors(
        cudaMalloc(&d_reserve_space_, reserve_space_size_in_bytes_));
    checkCudaErrors(cudaMalloc(&d_y_, getTensorSizeInBytes()));

    return getTensorSizeInBytes() + dropout_state_size +
           reserve_space_size_in_bytes_;
}

size_t Dropout::prepareBwdPropagation() {
    checkCudaErrors(cudaMalloc(&d_dy_, getTensorSizeInBytes()));

    return getTensorSizeInBytes();
}

void Dropout::fwdPropagation() {
    NetworkConstPtr nn = network_.lock();
    assert(("Network is expired", nn));
    LayerConstPtr up = up_.lock();
    assert(("Upstream is expired", up));

    cudnnHandle_t cudnn_handle = nn->getCudnnHandle();
    float* d_x                 = up->getTensor();

    checkCUDNN(cudnnDropoutForward(cudnn_handle,
                                   dropout_desc_,
                                   y_desc_,
                                   d_x,
                                   y_desc_,
                                   d_y_,
                                   d_reserve_space_,
                                   reserve_space_size_in_bytes_));
}

void Dropout::bwdPropagation() {
    NetworkConstPtr nn = network_.lock();
    assert(("Network is expired", nn));
    LayerConstPtr up = up_.lock();
    assert(("Up stream is expired", up));

    cudnnHandle_t cudnn_handle = nn->getCudnnHandle();
    float* d_dx                = up->getGradient();

    checkCUDNN(cudnnDropoutBackward(cudnn_handle,
                                    dropout_desc_,
                                    y_desc_,
                                    d_dy_,
                                    y_desc_,
                                    d_dx,
                                    d_reserve_space_,
                                    reserve_space_size_in_bytes_));
}

}  // namespace layers
}  // namespace nn
