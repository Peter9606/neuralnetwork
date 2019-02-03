#pragma once

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
     * @param[in] inference_only if network is dedicated for inference only
     */
    NetworkImpl(int batch_size,
                const SolverSetting& solver_setting,
                LossType loss_type,
                bool inference_only = false);

    /**
     * destructor
     */
    ~NetworkImpl();

    /**
     * get batch size
     *
     * @return batch size
     */
    int getBatchSize() const final;

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

    /**
     * get cudnn handle
     *
     * @return cudnn handle
     */
    cudnnHandle_t getCudnnHandle() const final;

    /**
     * get cublas handle
     *
     * @return cublas handle
     */
    cublasHandle_t getCublasHandle() const final;

    /**
     * get alpha scaling factor
     *
     * @return alpha scaling factor
     */
    const float* getAlpha() const final;

    /**
     * get beta scaling factor
     *
     * @return beta scaling factor
     */
    const float* getBeta() const final;

    /**
     * get workspace size
     *
     * @return workspace size in bytes
     */
    virtual size_t getWorkspaceSize() const;

    /**
     * set workspace size
     *
     * @param[in] size workspace size in bytes
     */
    virtual void setWorkspaceSize(size_t size) const;

    /**
     * get workspace
     *
     * @return pointer to workspace in device
     */
    virtual float* getWorkspace() const;

    /**
     * get if network only used for inference
     *
     * @return  true if inference only, false otherwise
     */
    virtual bool getInferenceOnly() const;

private:
    float alpha_ = 1.0f;
    float beta_  = 0.0f;
    const int batch_size_;
    const SolverSetting solver_setting_;
    const LossType loss_type_;
    const bool inference_only_;
    mutable long memory_needed_ = 0;

    cudnnHandle_t cudnn_handle_;
    cublasHandle_t cublas_handle_;

    mutable size_t workspace_size_;
    float* d_workspace_ = nullptr;
};  // namespace nn
}  // namespace nn
