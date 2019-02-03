#pragma once
// cudnn
#include <cudnn.h>

// self
#include "layer.h"
#include "layers/config.h"

namespace nn
{
namespace layers
{
/** @class Conv
 * @brief Convolution layer
 */
class Conv : public Layer
{
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
     * get output tensor descriptor
     *
     * @return output tensor descriptor
     */
    cudnnTensorDescriptor_t getYDescriptor() const;

    /**
     * get output tensor
     *
     * @return pointer to output tensor on device
     */
    float* getY() const;

    /**
     * get gradient, for Input layer always return nullptr
     *
     * @return nullptr
     */
    float* getGradient() const;

public:
    // TODO(Peter Han): need pricesly access control over members,
    // currently for debug convinient all set to public
    const Kernel kernel_;
    const Pad pad_;
    const Stride stride_;
    const Dilation dilation_;

    cudnnTensorDescriptor_t y_desc_;
    cudnnTensorDescriptor_t bias_desc_;
    cudnnFilterDescriptor_t filter_desc_;
    cudnnConvolutionDescriptor_t conv_desc_;
    cudnnConvolutionFwdAlgo_t fwd_algo_;

    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_;
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo_;

    float* d_filter_  = nullptr;
    float* d_bias_    = nullptr;
    float* d_y_       = nullptr;
    float* d_dfilter_ = nullptr;
    float* d_dbias_   = nullptr;
    float* d_dy_      = nullptr;
};

}  // namespace layers
}  // namespace nn
