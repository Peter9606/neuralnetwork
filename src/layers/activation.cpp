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
#include "neuralnetwork/layers/activation.h"

namespace {
using nn::layers::Activation;

static std::map<Activation::Type, cudnnActivationMode_t> MODE = {
    {Activation::SIGMOID, CUDNN_ACTIVATION_SIGMOID},
    {Activation::RELU, CUDNN_ACTIVATION_RELU},
    {Activation::TANH, CUDNN_ACTIVATION_TANH},
    {Activation::CLIPPED_RELU, CUDNN_ACTIVATION_CLIPPED_RELU},
    {Activation::ELU, CUDNN_ACTIVATION_ELU},
};
}  // namespace

namespace nn {
namespace layers {
Activation::Activation(const std::string& name,
                       const NetworkConstPtr& network,
                       const LayerConstPtr& up,
                       Type type,
                       double coef,
                       bool in_place)
    : Layer(name, network, up)
    , type_(type)
    , coef_(coef)
    , in_place_(in_place) {
    Dim d = up->getDim();
    c_    = d.c;
    h_    = d.h;
    w_    = d.w;

    checkCUDNN(cudnnCreateActivationDescriptor(&activation_desc_));
    checkCUDNN(cudnnSetActivationDescriptor(
        activation_desc_, ::MODE[type_], CUDNN_PROPAGATE_NAN, coef_));

    if (in_place_) {
        y_desc_ = up->getDescriptor();
    } else {
        checkCUDNN(cudnnCreateTensorDescriptor(&y_desc_));
        checkCUDNN(cudnnSetTensor4dDescriptor(
            y_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n_, c_, h_, w_));
    }
}

Activation::~Activation() {
    checkCUDNN(cudnnDestroyActivationDescriptor(activation_desc_));

    if (in_place_) {
        return;
    }

    checkCUDNN(cudnnDestroyTensorDescriptor(y_desc_));
    checkCudaErrors(cudaFree(d_y_));
    checkCudaErrors(cudaFree(d_dy_));
}

size_t Activation::prepareFwdPropagation() {
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

size_t Activation::prepareBwdPropagation() {
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

void Activation::fwdPropagation() {
    NetworkConstPtr nn = network_.lock();
    assert(("Network is expired", nn));
    LayerConstPtr up = up_.lock();
    assert(("Upstream is expired", up));

    cudnnHandle_t cudnn_handle     = nn->getCudnnHandle();
    const float* alpha             = nn->getAlpha();
    const float* beta              = nn->getBeta();
    cudnnTensorDescriptor_t x_desc = up->getDescriptor();
    float* d_x                     = up->getTensor();

    checkCUDNN(cudnnActivationForward(cudnn_handle,
                                      activation_desc_,
                                      alpha,
                                      x_desc,
                                      d_x,
                                      beta,
                                      y_desc_,
                                      d_y_));
}

void Activation::bwdPropagation() {
    NetworkConstPtr nn = network_.lock();
    assert(("Network is expired", nn));
    LayerConstPtr up = up_.lock();
    assert(("Upstream is expired", up));

    cudnnHandle_t cudnn_handle     = nn->getCudnnHandle();
    const float* alpha             = nn->getAlpha();
    const float* beta              = nn->getBeta();
    cudnnTensorDescriptor_t x_desc = up->getDescriptor();
    float* d_x                     = up->getTensor();
    float* d_dx                    = up->getGradient();

    if (!d_dx) {
        log_->trace("{} bwdPropagation shortcut as no upstream", name_);
        return;
    }
    checkCUDNN(cudnnActivationBackward(cudnn_handle,
                                       activation_desc_,
                                       alpha,
                                       y_desc_,
                                       d_y_,
                                       y_desc_,
                                       d_dy_,
                                       x_desc,
                                       d_x,
                                       beta,
                                       x_desc,
                                       d_dx));
}
}  // namespace layers
}  // namespace nn
