/*
 * Copyright 2019, Peter Han, All rights reserved.
 * This code is released into the public domain.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
#pragma once
#include <cudnn.h>

#include <memory>
#include <string>
#include <vector>

#include "neuralnetwork/layer.h"

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
     * get bias size
     *
     * @return bias size
     */
    size_t getBiasSize() const;

    /**
     * get bias size in bytes
     *
     * @return bias size in bytes
     */
    size_t getBiasSizeInBytes() const;

    /**
     * get weight size
     *
     * @return weight size
     */
    size_t getWeightSize() const;

    /**
     * get weight size in bytes
     *
     * @return weight size in bytes
     */
    size_t getWeightSizeInBytes() const;

 private:
    const size_t input_length_;
    const size_t output_length_;

    cudnnTensorDescriptor_t bias_desc_ = nullptr;

    float* d_weight_     = nullptr;
    float* d_bias_       = nullptr;
    float* d_dweight_    = nullptr;
    float* d_dbias_      = nullptr;
    float* d_one_vector_ = nullptr;
};

}  // namespace layers
}  // namespace nn
