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
#include <cudnn.h>

#include <cassert>
#include <chrono>  //NOLINT
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "neuralnetwork/error_check.h"
#include "neuralnetwork/gpu/compution.cuh"
#include "neuralnetwork/layers/activation.h"
#include "neuralnetwork/layers/config.h"
#include "neuralnetwork/layers/conv.h"
#include "neuralnetwork/layers/dropout.h"
#include "neuralnetwork/layers/fc.h"
#include "neuralnetwork/layers/input.h"
#include "neuralnetwork/layers/pool.h"
#include "neuralnetwork/layers/softmax.h"
#include "neuralnetwork/layers/unpool.h"
#include "neuralnetwork/network_impl.h"

using nn::layers::Activation;
using nn::layers::Conv;
using nn::layers::Dilation;
using nn::layers::Dropout;
using nn::layers::FC;
using nn::layers::Input;
using nn::layers::Kernel;
using nn::layers::Pad;
using nn::layers::Pool;
using nn::layers::Softmax;
using nn::layers::Stride;
using nn::layers::Unpool;
using nn::layers::Window;
using std::make_shared;
using std::setprecision;
using std::shared_ptr;
using std::stringstream;
using std::vector;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;

namespace nn {
const std::string readableSize(size_t s) {
    constexpr static size_t KB = 1024;
    constexpr static size_t MB = 1024 * KB;
    constexpr static size_t GB = 1024 * MB;
    stringstream ss;
    if (s < KB) {
        ss << s << "B";
    } else if (s < MB) {
        ss << setprecision(3) << static_cast<float>(s) / KB << "KB";
    } else if (s < GB) {
        ss << setprecision(3) << static_cast<float>(s) / MB << "MB";
    } else {
        ss << setprecision(3) << static_cast<float>(s) / GB << "GB";
    }
    return ss.str();
}
}  // namespace nn

namespace nn {
NetworkImpl::NetworkImpl(int batch_size)
    : batch_size_(batch_size)
    , solver_setting_({0.001, SGD}) {
    dim_       = {1, 28, 28};
    loss_type_ = CROSS_ENTROPY;

    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cublasCreate(&cublas_handle_));
    checkCUDNN(cudnnCreate(&cudnn_handle_));
}

NetworkImpl::NetworkImpl(int batch_size,
                         Dim dim,
                         const SolverSetting &solver_setting,
                         LossType loss_type)
    : batch_size_(batch_size)
    , dim_(dim)
    , solver_setting_(solver_setting)
    , loss_type_(loss_type) {
    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cublasCreate(&cublas_handle_));
    checkCUDNN(cudnnCreate(&cudnn_handle_));
}

NetworkImpl::~NetworkImpl() {
    checkCudaErrors(cublasDestroy(cublas_handle_));
    checkCUDNN(cudnnDestroy(cudnn_handle_));
    checkCudaErrors(cudaFree(d_workspace_));
}

void NetworkImpl::prepareTraining() const {
    size_t malloced_size = 0;
    for (auto &&l : layers_) {
        malloced_size += l->prepareFwdPropagation();
        malloced_size += l->prepareBwdPropagation();
        l->loadParameters(nullptr);
    }
    log_->info("Malloced device memory {}", readableSize(malloced_size));
    log_->info("Need workspace memory {}", readableSize(workspace_size_));
    checkCudaErrors(cudaMalloc(&d_workspace_, workspace_size_));
}

void NetworkImpl::prepareInference() const {
    size_t malloced_size = 0;
    for (auto &&l : layers_) {
        malloced_size += l->prepareFwdPropagation();
        l->loadParameters(nullptr);
    }
    log_->info("Malloced device memory {}", readableSize(malloced_size));
    log_->info("Need workspace memory {}", readableSize(workspace_size_));
    checkCudaErrors(cudaMalloc(&d_workspace_, workspace_size_));
}

void NetworkImpl::fwdPropagation(const float *d_data) const {
    checkCudaErrors(
        cudaMemcpyAsync(layers_[0]->getTensor(),
                        d_data,
                        sizeof(float) * batch_size_ * dim_.c * dim_.h * dim_.w,
                        cudaMemcpyDeviceToDevice));
    // TODO(Peter Han): temporary implementation, doesn't work for network of
    // complex topology, for instance, RestNet which has shortcut connection
    // (probably ?)
    for (auto &layer : layers_) {
        layer->fwdPropagation();
    }
}

void NetworkImpl::bwdPropagation(const float *d_label) const {
    computeLoss(d_label);

    for (auto it = layers_.rbegin(); it != layers_.rend(); it++) {
        auto l = *it;
        l->bwdPropagation();
    }
}

void NetworkImpl::updateWeights() const {
    for (auto &layer : layers_) {
        layer->updateWeights();
    }
}

int NetworkImpl::getBatchSize() const {
    return batch_size_;
}

void NetworkImpl::updateMemoryNeeded(size_t inc) const {
    memory_needed_ += inc;
}

SolverSetting NetworkImpl::getSolverSetting() const {
    return solver_setting_;
}

LossType NetworkImpl::getLossType() const {
    return loss_type_;
}

cudnnHandle_t NetworkImpl::getCudnnHandle() const {
    return cudnn_handle_;
}

cublasHandle_t NetworkImpl::getCublasHandle() const {
    return cublas_handle_;
}

const float *NetworkImpl::getAlpha() const {
    return &alpha_;
}

const float *NetworkImpl::getBeta() const {
    return &beta_;
}

size_t NetworkImpl::getWorkspaceSize() const {
    return workspace_size_;
}

float *NetworkImpl::getWorkspace() const {
    return d_workspace_;
}

void NetworkImpl::updateWorkspaceSize(size_t size) const {
    log_->trace("{}[{}]", __FUNCTION__, size);
    if (size < workspace_size_) {
        return;
    }
    workspace_size_ = size;
}

}  // namespace nn
