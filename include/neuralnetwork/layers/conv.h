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
#include <memory>
#include <string>
#include <vector>

#include "neuralnetwork/layer.h"
#include "neuralnetwork/layers/config.h"

namespace nn {
namespace layers {
/** @class Conv
 * @brief Convolution layer
 */
class Conv : public Layer {
 public:
    /**
     * Conv constructor
     *
     * @param[in] name      layer name
     * @param[in] network   Network interface handle
     * @param[in] up        upstream layer
     * @param[in] kernel    kernel parameter
     * @param[in] pad       pad parameter
     * @param[in] stride    stride parameter
     * @param[in] dilation  dilation parameter
     */
    Conv(const std::string& name,
         const NetworkConstPtr& network,
         const LayerConstPtr& up,
         const Kernel& kernel,
         const Pad& pad,
         const Stride& stride,
         const Dilation& dilation);

    /**
     * Conv constructor, with dilation set to 1 in both directions
     *
     * @param[in] name      layer name
     * @param[in] network   Network interface handle
     * @param[in] up        upstream layer
     * @param[in] kernel    kernel parameter
     * @param[in] pad       pad parameter
     * @param[in] stride    stride parameter
     */
    Conv(const std::string& name,
         const NetworkConstPtr& network,
         const LayerConstPtr& up,
         const Kernel& kernel,
         const Pad& pad,
         const Stride& stride);

    /**
     * Conv constructor, with stride and dilation set to 1 in both directions
     *
     * @param[in] name      layer name
     * @param[in] network   Network interface handle
     * @param[in] up        upstream layer
     * @param[in] kernel    kernel parameter
     * @param[in] pad       pad parameter
     */
    Conv(const std::string& name,
         const NetworkConstPtr& network,
         const LayerConstPtr& up,
         const Kernel& kernel,
         const Pad& pad);

    /**
     * Conv constructor, with no padding, stride and dilation set to 1 in both
     * directions
     *
     * @param[in] name      layer name
     * @param[in] network   Network interface handle
     * @param[in] up        upstream layer
     * @param[in] kernel    kernel parameter
     */
    Conv(const std::string& name,
         const NetworkConstPtr& network,
         const LayerConstPtr& up,
         const Kernel& kernel);

    /**
     * Desctructor
     */
    virtual ~Conv();

    /**
     * @brief load parameters from given data existing in host.
     *
     * parameters consists of two parts: filters and bias. in either host and
     * device, parameters are continuous, first filters then bias.
     *
     * @param[in] h_params a float vector contains current layer's
     * parameters
     */
    void loadParameters(const shared_ptr<vector<float>>& h_params) final;

    /**
     * copy parameters from device to host
     *
     * parameters consists of two parts: filters and bias. in either host and
     * device, parameters are continuous, first filters then bias.
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
     * get filter size
     *
     * @return filter size
     */
    size_t getFilterSize() const;

    /**
     * get filter size in bytes
     *
     * @return filter size in bytes
     */
    size_t getFilterSizeInBytes() const;

 private:
    const Kernel kernel_;
    const Pad pad_;
    const Stride stride_;
    const Dilation dilation_;
    const int input_channel_;

    cudnnTensorDescriptor_t bias_desc_      = nullptr;
    cudnnFilterDescriptor_t filter_desc_    = nullptr;
    cudnnConvolutionDescriptor_t conv_desc_ = nullptr;

    cudnnConvolutionFwdAlgo_t fwd_algo_;
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_;
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo_;

    float* d_filter_  = nullptr;
    float* d_dfilter_ = nullptr;
    float* d_bias_    = nullptr;
    float* d_dbias_   = nullptr;
};

}  // namespace layers
}  // namespace nn
