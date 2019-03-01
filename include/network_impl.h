#pragma once
#include <memory>
#include <vector>

// self
#include "layer.h"
#include "logger.h"
#include "network.h"

using std::shared_ptr;
using std::vector;

namespace nn {
class NetworkImpl : public Network,
                    public std::enable_shared_from_this<NetworkImpl> {
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
     * Constructor build a Network object
     *
     * @param[in] batch_size  batch size
     */
    NetworkImpl(int batch_size);

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
     * @brief update workspace size
     * According to CUDNN guide, convolution forward and backward
     * API need an unidifed workspace size which should be big
     * enough for all conv layers. So each layer calling this API to
     * notify its size to Network.
     *
     * @param[in] size workspace size in bytes
     */
    virtual void updateWorkspaceSize(size_t size) const;

    /**
     * get workspace size
     *
     * @return workspace size in bytes
     */
    virtual size_t getWorkspaceSize() const;

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
    void train(shared_ptr<vector<float>>& h_data,
               shared_ptr<vector<float>>& h_label) const;

    /**
     * @brief compute loss
     * Concrate network should have have implementation
     *
     * @param[in]       d_label     label data in device
     */
    virtual void computeLoss(const float* d_label) const;

    /**
     * @brief construct network topology
     */
    virtual void buildNetwork();

 private:
    void fwdPropagation(const float* d_data) const;
    void bwdPropagation(const float* d_label) const;
    void updateWeights() const;

 private:
    shared_ptr<logger> log_ = Logger::getLogger();
    float alpha_            = 1.0f;
    float beta_             = 0.0f;
    const int batch_size_;
    Dim dim_;
    SolverSetting solver_setting_;
    LossType loss_type_;
    mutable long memory_needed_ = 0;

    cudnnHandle_t cudnn_handle_;
    cublasHandle_t cublas_handle_;

    mutable size_t workspace_size_ = 0;
    mutable float* d_workspace_    = nullptr;
    vector<shared_ptr<Layer>> layers_;
};  // namespace nn
}  // namespace nn
