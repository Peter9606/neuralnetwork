// standard
#include <chrono>

// self
#include "error_check.h"
#include "network_impl.h"

namespace nn
{
NetworkImpl::NetworkImpl(int batch_size,
                         Dim dim,
                         const SolverSetting& solver_setting,
                         const LossType loss_type)
    : batch_size_(batch_size)
    , dim_(dim)
    , solver_setting_(solver_setting)
    , loss_type_(loss_type)
{
    checkCudaErrors(cublasCreate(&cublas_handle_));
    checkCUDNN(cudnnCreate(&cudnn_handle_));
}

NetworkImpl::~NetworkImpl()
{
    checkCudaErrors(cublasDestroy(cublas_handle_));
    checkCUDNN(cudnnDestroy(cudnn_handle_));
}

void NetworkImpl::train(std::shared_ptr<std::vector<float>>& h_data,
                        std::shared_ptr<std::vector<float>>& h_label) const
{
    const size_t train_size = h_label->size();
    auto t1                 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 10; ++iter)
    {
        /*
                // Train
                int imageid = iter % (train_size / batch_size_);

                // Prepare current batch on device
                checkCudaErrors(cudaMemcpyAsync(
                    d_data,
                    &train_images[imageid * batch_size_ * width * height *
           channels], sizeof(float) * batch_size_ * channels * width * height,
                    cudaMemcpyHostToDevice));
                checkCudaErrors(cudaMemcpyAsync(d_labels,
                                                &train_labels[imageid *
           batch_size_], sizeof(float) * batch_size_, cudaMemcpyHostToDevice));
        */
    }
    checkCudaErrors(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << "Iteration time:"
              << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
                         .count() /
                     1000.0f / 10
              << std::endl;
}

void NetworkImpl::fwdPropagation(const float* d_data) const
{
    checkCudaErrors(
        cudaMemcpyAsync(layers_[0]->getTensor(),
                        d_data,
                        sizeof(float) * batch_size_ * dim_.c * dim_.h * dim_.w,
                        cudaMemcpyDeviceToDevice));
    for (auto&& layer : layers_)
    {
        layer->fwdPropagation();
    }
}

void NetworkImpl::bwdPropagation(const float* d_label) const
{
    // get last layer's gradient pointer

    // compute the lost

    // backprop

    for (auto it = layers_.rbegin(); it != layers_.rend(); it++)
    {
        auto l = *it;
        l->bwdPropagation();
    }
}

void NetworkImpl::updateWeights() const
{
    for (auto&& layer : layers_)
    {
        layer->updateWeights();
    }
}

int NetworkImpl::getBatchSize() const
{
    return batch_size_;
}

void NetworkImpl::updateMemoryNeeded(long inc) const
{
    memory_needed_ += inc;
}

SolverSetting NetworkImpl::getSolverSetting() const
{
    return solver_setting_;
}

LossType NetworkImpl::getLossType() const
{
    return loss_type_;
}

cudnnHandle_t NetworkImpl::getCudnnHandle() const
{
    return cudnn_handle_;
}

cublasHandle_t NetworkImpl::getCublasHandle() const
{
    return cublas_handle_;
}

const float* NetworkImpl::getAlpha() const
{
    return &alpha_;
}

const float* NetworkImpl::getBeta() const
{
    return &beta_;
}

size_t NetworkImpl::getWorkspaceSize() const
{
    return workspace_size_;
}

void NetworkImpl::setWorkspaceSize(size_t size) const
{
    workspace_size_ = size;
}

float* NetworkImpl::getWorkspace() const
{
    return d_workspace_;
}

}  // namespace nn
