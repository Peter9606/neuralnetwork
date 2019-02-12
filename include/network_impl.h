#pragma once
#include <vector>

// self
#include "layer.h"
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
     * @param[in] dim image dimension
     * @param[in] solver_setting solver setting
     * @param[in] loss_type loss type
     */
    NetworkImpl(int batch_size,
                Dim dim,
                const SolverSetting& solver_setting,
                LossType loss_type);

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
     * train nn
     *
     * @param[in] h_data 	data vector located in host
     * @param[in] h_label 	label vector located in host
     */
    void train(std::shared_ptr<std::vector<float>>& h_data,
               std::shared_ptr<std::vector<float>>& h_label) const;

private:
    void fwdPropagation(const float* d_data) const;
    void bwdPropagation(const float* d_label) const;
    void updateWeights() const;

private:
    float alpha_ = 1.0f;
    float beta_  = 0.0f;
    const int batch_size_;
    const Dim dim_;
    const SolverSetting solver_setting_;
    const LossType loss_type_;
    mutable long memory_needed_ = 0;

    cudnnHandle_t cudnn_handle_;
    cublasHandle_t cublas_handle_;

    mutable size_t workspace_size_;
    float* d_workspace_ = nullptr;
    std::vector<std::shared_ptr<Layer>> layers_;
};  // namespace nn
}  // namespace nn
