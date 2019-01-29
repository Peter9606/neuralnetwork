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
}  // namespace nn
