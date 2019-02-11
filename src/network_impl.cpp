#include "network_impl.h"

// self
#include "error_check.h"

namespace nn
{
NetworkImpl::NetworkImpl(int batch_size,
                         const SolverSetting& solver_setting,
                         const LossType loss_type)
    : batch_size_(batch_size)
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

int NetworkImpl::getBatchSize() const
{
    return batch_size_;
}

bool NetworkImpl::isInferenceOnly() const
{
    return solver_setting_.type == SolverType::InfOnly;
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
