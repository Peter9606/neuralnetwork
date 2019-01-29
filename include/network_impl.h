#pragma once

// cudnn
#include <cudnn.h>

// self
#include "network.h"

namespace nn
{
class NetworkImpl : public Network
{
public:
    /**
     * Constructor build a Network object
     *
     * @param[in] batch_size  batch size
     * @param[in] solver_setting solver setting
     * @param[in] loss_type loss type
     */
    NetworkImpl(int batch_size,
                const SolverSetting& solver_setting,
                LossType loss_type);

    /**
     * get batch size
     *
     * @return batch size
     */
    int getBatchSize() const;

    /**
     * get if inference only
     *
     * @return true if inference only, false otherwise
     */
    bool isInferenceOnly() const final;

    /**
     * update total GPU memory needed by this network
     *
     * @param[in] inc increment GPU memory in byte
     */
    void updateMemoryNeeded(long inc) const final;

    /**
     * get solver setting
     *
     * @return solver setting
     */
    SolverSetting getSolverSetting() const final;

    /**
     * get loss calculation method
     *
     * @return loss type
     */
    LossType getLossType() const final;

private:
    const int batch_size_;
    const SolverSetting solver_setting_;
    const LossType loss_type_;
    mutable long memory_needed_ = 0;
};
}  // namespace nn
