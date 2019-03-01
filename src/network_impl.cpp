#include "network_impl.h"

// standard
#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include <cudnn.h>

// self
#include "error_check.h"

#include "gpu/compution.cuh"
#include "layers/activation.h"
#include "layers/config.h"
#include "layers/conv.h"
#include "layers/dropout.h"
#include "layers/fc.h"
#include "layers/input.h"
#include "layers/pool.h"
#include "layers/softmax.h"
#include "layers/unpool.h"

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
using std::shared_ptr;
using std::vector;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;

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
    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cublasDestroy(cublas_handle_));
    checkCUDNN(cudnnDestroy(cudnn_handle_));
    checkCudaErrors(cudaFree(d_workspace_));
}

void NetworkImpl::train(shared_ptr<vector<float>> &h_data,
                        shared_ptr<vector<float>> &h_label) const {
    size_t malloced_size = 0;
    for (auto &&l : layers_) {
        malloced_size += l->prepareFwdPropagation();
        malloced_size += l->prepareBwdPropagation();
        l->loadParameters(nullptr);
    }
    log_->info("Malloced device {}KB", malloced_size / 1024);
    log_->info("Need workspace {}KB", workspace_size_ / 1024);
    checkCudaErrors(
        cudaMalloc(&d_workspace_, workspace_size_));  // TODO(Peter Han): move
                                                      // to somewhere good

    checkCudaErrors(cudaDeviceSynchronize());
    auto t1 = high_resolution_clock::now();

    const size_t train_size = h_label->size();

    const int total_iter = 200000;
    for (int iter = 0; iter < total_iter; ++iter) {
        int imageid = iter % (train_size / batch_size_);

        // Prepare current batch on device
        float *d_data       = layers_[0]->getTensor();
        const Dim dim_first = layers_[0]->getDim();
        const size_t data_len =
            batch_size_ * dim_first.c * dim_first.h * dim_first.w;
        checkCudaErrors(cudaMemcpyAsync(d_data,
                                        &h_data->data()[imageid * data_len],
                                        sizeof(float) * data_len,
                                        cudaMemcpyHostToDevice));

        float *d_label;
        checkCudaErrors(cudaMalloc(&d_label, sizeof(float) * batch_size_));
        checkCudaErrors(cudaMemcpyAsync(d_label,
                                        &h_label->data()[imageid * batch_size_],
                                        sizeof(float) * batch_size_,
                                        cudaMemcpyHostToDevice));

        checkCudaErrors(cudaDeviceSynchronize());
        fwdPropagation(d_data);
        bwdPropagation(d_label);
        updateWeights();
        checkCudaErrors(cudaDeviceSynchronize());

        if (iter % 200 == 0) {
            std::vector<float> output(10 * batch_size_);
            std::vector<float> gradient(10 * batch_size_);
            std::vector<float> fcoutput(10 * batch_size_);
            auto last         = *(layers_.rbegin());
            float *d_gradient = last->getGradient();
            float *d_output   = last->getTensor();
            float *d_fcoutput = layers_[1]->getTensor();

            checkCudaErrors(cudaMemcpyAsync(&output[0],
                                            d_output,
                                            sizeof(float) * output.size(),
                                            cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpyAsync(&gradient[0],
                                            d_gradient,
                                            sizeof(float) * gradient.size(),
                                            cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpyAsync(&fcoutput[0],
                                            d_fcoutput,
                                            sizeof(float) * fcoutput.size(),
                                            cudaMemcpyDeviceToHost));

            std::vector<float> grand_truth(batch_size_);
            checkCudaErrors(cudaMemcpyAsync(&grand_truth[0],
                                            d_label,
                                            sizeof(float) * grand_truth.size(),
                                            cudaMemcpyDeviceToHost));
            int num_errors = 0;
            for (int i = 0; i < batch_size_; i++) {
                // Determine classification according to maximal response
                int base   = i * 10;
                int chosen = 0;

                for (int id = 0; id < 10; ++id) {
                    if (output[base + chosen] < output[base + id]) {
                        chosen = id;
                    }
                }
                if (chosen != h_label->at(imageid * batch_size_ + i))
                    ++num_errors;
            }
            float err = (float)num_errors / (float)batch_size_;
            log_->info("Iter: {}, Error rates: {}", iter, err * 100.0f);
        }
    }
    checkCudaErrors(cudaDeviceSynchronize());
    auto t2 = high_resolution_clock::now();

    log_->info("Iteration time: {}",
               duration_cast<microseconds>(t2 - t1).count() / 1000.0f /
                   total_iter);
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

// TODO(Peter Han): let's implement a classifier first
void NetworkImpl::computeLoss(const float *d_label) const {
    const float scale      = 1.0f / static_cast<float>(batch_size_);
    shared_ptr<Layer> last = *(layers_.rbegin());
    const Dim dim          = last->getDim();
    const size_t size     = sizeof(float) * dim.c * dim.h * dim.w * batch_size_;
    const float *d_result = last->getTensor();
    float *d_loss         = last->getGradient();

    checkCudaErrors(
        cudaMemcpyAsync(d_loss, d_result, size, cudaMemcpyDeviceToDevice));

    calculateLossWithGpu(d_label, dim.c * dim.h * dim.w, batch_size_, d_loss);

    // Accounting for batch size in SGD
    checkCudaErrors(cublasSscal(cublas_handle_,
                                dim.c * dim.h * dim.w * batch_size_,
                                &scale,
                                d_loss,
                                1));
}

void NetworkImpl::updateWeights() const {
    for (auto &layer : layers_) {
        layer->updateWeights();
    }
}

// TODO(Peter Han): This should be implemented in sub-class
// but, for temporary solution, LeNet5 is build. Should remove in future
void NetworkImpl::buildNetwork() {
    auto input = make_shared<Input>("Input", shared_from_this(), 1, 28, 28);
    layers_.push_back(input);

    Kernel kernel = {3, 3, 20};
    auto conv1 = make_shared<Conv>("Conv1", shared_from_this(), input, kernel);
    layers_.push_back(conv1);

    Window window = {2, 2};
    Stride stride = {2, 2};
    auto pool1    = make_shared<Pool>(
        "MaxPool1", shared_from_this(), conv1, window, stride, Pool::MAX);
    layers_.push_back(pool1);

    kernel = {3, 3, 50};
    auto conv2_1 =
        make_shared<Conv>("Conv2-1", shared_from_this(), pool1, kernel);
    layers_.push_back(conv2_1);
    auto conv2_2 =
        make_shared<Conv>("Conv2-2", shared_from_this(), conv2_1, kernel);
    layers_.push_back(conv2_2);

    auto pool2 = make_shared<Pool>(
        "MaxPool2", shared_from_this(), conv2_2, window, stride, Pool::MAX);
    layers_.push_back(pool2);

    kernel     = {3, 3, 50};
    auto conv3 = make_shared<Conv>("Conv3", shared_from_this(), pool2, kernel);
    layers_.push_back(conv3);

    auto pool3 = make_shared<Pool>(
        "MaxPool3", shared_from_this(), conv3, window, stride, Pool::MAX);
    layers_.push_back(pool3);

    auto fc1 = make_shared<FC>("FC1", shared_from_this(), pool3, 500);
    layers_.push_back(fc1);

    auto relu1 = make_shared<Activation>("FC1Relu", shared_from_this(), fc1);
    layers_.push_back(relu1);

    auto fc2 = make_shared<FC>("FC2", shared_from_this(), relu1, 10);
    layers_.push_back(fc2);

    auto softmax = make_shared<Softmax>("Softmax", shared_from_this(), fc2);
    layers_.push_back(softmax);
}

int NetworkImpl::getBatchSize() const {
    return batch_size_;
}

void NetworkImpl::updateMemoryNeeded(long inc) const {
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
