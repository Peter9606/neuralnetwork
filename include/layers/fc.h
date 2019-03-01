#pragma once
// cudnn
#include <cudnn.h>

// self
#include "layer.h"

namespace nn {
namespace layers {
/** @class FC
 * @brief Fully Connect layer
 */
class FC : public Layer {
 public:
    /**
     * FC constructor
     *
     * @param[in] name              layer name
     * @param[in] network           Network interface handle
     * @param[in] up                upstream layer
     * @param[in] output_size FC    output node number
     */
    FC(const std::string& name,
       const NetworkConstPtr& network,
       const LayerConstPtr& up,
       int output_size);

    /**
     * Desctructor
     */
    virtual ~FC();

    /**
     * @brief load parameters from given data existing in host.
     *
     * parameters consists of two parts: weights and bias
     *
     * @param[in] h_params a float vector contains current layer's
     * parameters
     */
    void loadParameters(const shared_ptr<vector<float>>& h_params) final;

    /**
     * copy parameters from device to host
     *
     * parameters consists of two parts: filters and bias
     *
     * @return a float vector
     */
    shared_ptr<vector<float>> saveParameters() final;

    /**
     * prepare forward propagation
     *
     * @return allocated memory size on GPU in bytes.
     */
    size_t prepareFwdPropagation() final;

    /**
     * prepare backward propagation
     *
     * @return allocated memory size on GPU in bytes.
     */
    size_t prepareBwdPropagation() final;

    /**
     * run forward propagation
     */
    void fwdPropagation() final;

    /**
     * run backward propgation
     */
    void bwdPropagation() final;

    /**
     * update weights
     */
    void updateWeights() final;

    /**
     * get output tensor descriptor
     *
     * @return output tensor descriptor
     */
    cudnnTensorDescriptor_t getDescriptor() const;

    /**
     * get output tensor
     *
     * @return pointer to output tensor on device
     */
    float* getTensor() const;

    /**
     * get gradient w.r.t current layer's output
     *
     * @return graident
     */
    float* getGradient() const;

 private:
    const int input_length_;
    const int output_length_;

    cudnnTensorDescriptor_t y_desc_    = nullptr;
    cudnnTensorDescriptor_t bias_desc_ = nullptr;

    float* d_weight_     = nullptr;
    float* d_bias_       = nullptr;
    float* d_y_          = nullptr;
    float* d_dweight_    = nullptr;
    float* d_dbias_      = nullptr;
    float* d_dy_         = nullptr;
    float* d_one_vector_ = nullptr;
};

}  // namespace layers
}  // namespace nn
