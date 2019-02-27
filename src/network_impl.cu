#include "network_impl.h"

// standard
#include <cassert>
#include <chrono>
#include <memory>
#include <vector>

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
using nn::layers::FC;
using nn::layers::Input;
using nn::layers::Pool;
using nn::layers::Softmax;
// using nn::layers::Dropout;
// using nn::layers::Unpool;
using nn::layers::Dilation;
using nn::layers::Kernel;
using nn::layers::Pad;
using nn::layers::Stride;
using nn::layers::Window;
using std::cout;
using std::endl;
using std::make_shared;
using std::shared_ptr;
using std::vector;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;

namespace {
/**
 * Computes ceil(x / y) for integral nonnegative values.
 */
static inline unsigned int RoundUp(unsigned int nominator,
                                   unsigned int denominator) {
    return (nominator + denominator - 1) / denominator;
}
}  // namespace

namespace nn {
NetworkImpl::NetworkImpl(int batch_size,
                         Dim dim,
                         const SolverSetting& solver_setting,
                         const LossType loss_type)
    : batch_size_(batch_size)
    , dim_(dim)
    , solver_setting_(solver_setting)
    , loss_type_(loss_type) {
    checkCudaErrors(cublasCreate(&cublas_handle_));
    checkCUDNN(cudnnCreate(&cudnn_handle_));

    buildNetwork();
}

NetworkImpl::~NetworkImpl() {
    checkCudaErrors(cublasDestroy(cublas_handle_));
    checkCUDNN(cudnnDestroy(cudnn_handle_));
}

void NetworkImpl::train(shared_ptr<vector<float>>& h_data,
                        shared_ptr<vector<float>>& h_label) const {
    // TODO(Peter Han): code in this func copied from culnet.cu, need to be
    // refact!
    checkCudaErrors(cudaDeviceSynchronize());
    auto t1                 = high_resolution_clock::now();
    const size_t train_size = h_label->size();
    for (int iter = 0; iter < 10; ++iter) {
        /*
                int image_id = iter % (train_size / batch_size_);
                checkCudaErrors(cudaMemcpyAsync(
                    d_data,
                    &train_images_float[imageid * context.m_batchSize * width *
           height * channels], sizeof(float) * context.m_batchSize * channels *
           width * height, cudaMemcpyHostToDevice));
        */
    }
    checkCudaErrors(cudaDeviceSynchronize());
    auto t2 = high_resolution_clock::now();

    cout << "Iteration time:"
         << duration_cast<microseconds>(t2 - t1).count() / 1000.0f / 10 << endl;
}

void NetworkImpl::fwdPropagation(const float* d_data) const {
    checkCudaErrors(
        cudaMemcpyAsync(layers_[0]->getTensor(),
                        d_data,
                        sizeof(float) * batch_size_ * dim_.c * dim_.h * dim_.w,
                        cudaMemcpyDeviceToDevice));
    // TODO(Peter Han): temporary implementation, doesn't work for network of
    // complex topology, for instance, RestNet which has shortcut connection
    // (probably ?)
    for (auto&& layer : layers_) {
        layer->fwdPropagation();
    }
}

void NetworkImpl::bwdPropagation(const float* d_label) const {
    auto last = *(layers_.rbegin());
    assert(("last layer is null", last));

    computeLoss(last->getTensor(), d_label, last->getGradient());

    for (auto it = layers_.rbegin(); it != layers_.rend(); it++) {
        auto l = *it;
        l->bwdPropagation();
    }
}

// TODO(Peter Han): let's implement a classifier first
void NetworkImpl::computeLoss(const float* d_label) const {
    const float scale      = 1.0f / static_cast<float>(batch_size_);
    shared_ptr<Layer> last = *(layers_.rbegin());
    const Dim dim          = last->getDim();
    const size_t size      = sizeof(float) * dim.c * dim.h * dim.w;
    const float* d_result  = last->getTensor();
    float* d_loss          = last->getGradient();

    checkCudaErrors(
        cudaMemcpyAsync(d_loss, d_result, size, cudaMemcpyDeviceToDevice));

    constexpr static int BW = 128;
    SoftmaxLossBackprop<<<RoundUp(batch_size_, BW), BW>>>(
        d_label, dim.c * dim.h * dim.w, batch_size_, d_loss);

    // Accounting for batch size in SGD
    checkCudaErrors(cublasSscal(cublas_handle_,
                                dim.c * dim.h * dim.w * batch_size_,
                                &scale,
                                d_loss,
                                1));
}

void NetworkImpl::updateWeights() const {
    for (auto&& layer : layers_) {
        layer->updateWeights();
    }
}

// TODO(Peter Han): This should be implemented in sub-class
// but, for temporary solution, LeNet5 is build. Should remove in future
void NetworkImpl::buildNetwork() {
    auto input = make_shared<Input>("Input", shared_from_this(), 1, 28, 28);
    layers_.push_back(input);

    Kernel kernel = {20, 5, 5};
    auto conv1 = make_shared<Conv>("Conv1", shared_from_this(), input, kernel);
    layers_.push_back(conv1);

    Window window = {2, 2};
    Stride stride = {2, 2};
    auto pool1    = make_shared<Pool>(
        "MaxPool1", shared_from_this(), conv1, window, stride, Pool::MAX);
    layers_.push_back(pool1);

    kernel     = {50, 5, 5};
    auto conv2 = make_shared<Conv>("Conv2", shared_from_this(), pool1, kernel);
    layers_.push_back(conv2);

    auto pool2 = make_shared<Pool>(
        "MaxPool2", shared_from_this(), conv2, window, stride, Pool::MAX);
    layers_.push_back(pool2);

    auto fc1 = make_shared<FC>("FC1", shared_from_this(), pool2, 500);
    layers_.push_back(fc1);

    auto fc2 = make_shared<FC>("FC2", shared_from_this(), fc1, 10);
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

const float* NetworkImpl::getAlpha() const {
    return &alpha_;
}

const float* NetworkImpl::getBeta() const {
    return &beta_;
}

size_t NetworkImpl::getWorkspaceSize() const {
    return workspace_size_;
}

void NetworkImpl::setWorkspaceSize(size_t size) const {
    workspace_size_ = size;
}

float* NetworkImpl::getWorkspace() const {
    return d_workspace_;
}

}  // namespace nn
