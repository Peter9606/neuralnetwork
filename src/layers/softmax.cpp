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
#include <cassert>
#include <map>

#include "neuralnetwork/error_check.h"
#include "neuralnetwork/layers/softmax.h"

namespace nn {
namespace layers {
Softmax::Softmax(const std::string& name,
                 const NetworkConstPtr& network,
                 const LayerConstPtr& up,
                 bool in_place)
    : Layer(name, network, up)
    , in_place_(in_place) {
    const Dim d = up->getDim();
    c_          = d.c;
    h_          = d.h;
    w_          = d.w;

    if (in_place_) {
        y_desc_ = up->getDescriptor();
    } else {
        checkCUDNN(cudnnCreateTensorDescriptor(&y_desc_));
        checkCUDNN(cudnnSetTensor4dDescriptor(
            y_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n_, c_, h_, w_));
    }
}

Softmax::~Softmax() {
    if (in_place_) {
        return;
    }
    checkCUDNN(cudnnDestroyTensorDescriptor(y_desc_));
    checkCudaErrors(cudaFree(d_y_));
    checkCudaErrors(cudaFree(d_dy_));
}

size_t Softmax::prepareFwdPropagation() {
    if (in_place_) {
        LayerConstPtr up = up_.lock();
        assert(("Upstream is expired", up));
        d_y_ = up->getTensor();
        return 0;
    } else {
        checkCudaErrors(cudaMalloc(&d_y_, getTensorSizeInBytes()));
        return getTensorSizeInBytes();
    }
}

size_t Softmax::prepareBwdPropagation() {
    if (in_place_) {
        LayerConstPtr up = up_.lock();
        assert(("Upstream is expired", up));
        d_dy_ = up->getGradient();
        return 0;
    } else {
        checkCudaErrors(cudaMalloc(&d_dy_, getTensorSizeInBytes()));
        return getTensorSizeInBytes();
    }
}

void Softmax::fwdPropagation() {
    NetworkConstPtr nn = network_.lock();
    assert(("Network is expired", nn));
    LayerConstPtr up = up_.lock();
    assert(("Upstream is expired", up));

    cudnnHandle_t cudnn_handle     = nn->getCudnnHandle();
    const float* alpha             = nn->getAlpha();
    const float* beta              = nn->getBeta();
    cudnnTensorDescriptor_t x_desc = up->getDescriptor();
    float* d_x                     = up->getTensor();

    checkCUDNN(cudnnSoftmaxForward(cudnn_handle,
                                   CUDNN_SOFTMAX_ACCURATE,
                                   CUDNN_SOFTMAX_MODE_CHANNEL,
                                   alpha,
                                   x_desc,
                                   d_x,
                                   beta,
                                   y_desc_,
                                   d_y_));
}

void Softmax::bwdPropagation() {
    NetworkConstPtr nn = network_.lock();
    assert(("Network is expired", nn));
    LayerConstPtr up = up_.lock();
    assert(("Upstream is expired", up));

    cudnnHandle_t cudnn_handle     = nn->getCudnnHandle();
    const float* alpha             = nn->getAlpha();
    const float* beta              = nn->getBeta();
    cudnnTensorDescriptor_t x_desc = up->getDescriptor();
    float* d_dx                    = up->getGradient();

    if (!d_dx) {
        log_->trace("{} bwdPropagation shortcut as no upstream", name_);
        return;
    }

    checkCUDNN(cudnnSoftmaxBackward(cudnn_handle,
                                    CUDNN_SOFTMAX_ACCURATE,
                                    CUDNN_SOFTMAX_MODE_CHANNEL,
                                    alpha,
                                    y_desc_,
                                    d_y_,
                                    y_desc_,
                                    d_dy_,
                                    beta,
                                    x_desc,
                                    d_dx));
}
}  // namespace layers
}  // namespace nn
