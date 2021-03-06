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
#include "neuralnetwork/layers/pool.h"

namespace {
using nn::layers::Pool;

static std::map<Pool::Type, cudnnPoolingMode_t> MODE = {
    {Pool::MAX, CUDNN_POOLING_MAX},
};
}  // namespace

namespace nn {
namespace layers {
Pool::Pool(const std::string& name,
           const NetworkConstPtr& network,
           const LayerConstPtr& up,
           const Window& window,
           const Stride& stride,
           const Pad& pad,
           Type type)
    : Layer(name, network, up)
    , window_(window)
    , pad_(pad)
    , stride_(stride)
    , type_(type) {
    checkCUDNN(cudnnCreatePoolingDescriptor(&pool_desc_));
    checkCUDNN(cudnnSetPooling2dDescriptor(pool_desc_,
                                           ::MODE[type_],
                                           CUDNN_PROPAGATE_NAN,
                                           window_.height,
                                           window_.width,
                                           pad_.vertical,
                                           pad_.horizontal,
                                           stride_.vertical,
                                           stride_.horizontal));

    checkCUDNN(cudnnGetPooling2dForwardOutputDim(
        pool_desc_, up->getDescriptor(), &n_, &c_, &h_, &w_));

    checkCUDNN(cudnnCreateTensorDescriptor(&y_desc_));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        y_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n_, c_, h_, w_));
}

Pool::Pool(const std::string& name,
           const NetworkConstPtr& network,
           const LayerConstPtr& up,
           const Window& window,
           const Stride& stride,
           Type type)
    : Pool(name, network, up, window, stride, {0, 0}, type) {
}

Pool::~Pool() {
    checkCUDNN(cudnnDestroyPoolingDescriptor(pool_desc_));
    checkCUDNN(cudnnDestroyTensorDescriptor(y_desc_));
    checkCudaErrors(cudaFree(d_y_));
    checkCudaErrors(cudaFree(d_dy_));
}

size_t Pool::prepareFwdPropagation() {
    checkCudaErrors(cudaMalloc(&d_y_, getTensorSizeInBytes()));

    return getTensorSizeInBytes();
}

size_t Pool::prepareBwdPropagation() {
    checkCudaErrors(cudaMalloc(&d_dy_, getTensorSizeInBytes()));

    return getTensorSizeInBytes();
}

void Pool::fwdPropagation() {
    NetworkConstPtr nn = network_.lock();
    assert(("Network is expired", nn));
    LayerConstPtr up = up_.lock();
    assert(("Upstream is expired", up));

    cudnnHandle_t cudnn_handle     = nn->getCudnnHandle();
    const float* alpha             = nn->getAlpha();
    const float* beta              = nn->getBeta();
    cudnnTensorDescriptor_t x_desc = up->getDescriptor();
    float* d_x                     = up->getTensor();

    checkCUDNN(cudnnPoolingForward(
        cudnn_handle, pool_desc_, alpha, x_desc, d_x, beta, y_desc_, d_y_));
}

void Pool::bwdPropagation() {
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
    checkCUDNN(cudnnPoolingBackward(cudnn_handle,
                                    pool_desc_,
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
