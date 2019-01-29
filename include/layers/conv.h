#pragma once
// cudnn
#include <cudnn.h>

// self
#include "layer.h"

namespace nn
{
namespace layers
{
/** @struct Pad
 * Horizontal and vertical direction padding.
 */
struct Pad
{
    int height;
    int width;
};

/** @struct Stride
 * Horizontal and vertical stride.
 */
struct Stride
{
    int vertical;
    int horizontal;
};

/** @struct Kernel
 * kernel size and kernel number
 */
struct Kernel
{
    int height;
    int width;
    int channel;
};

/** @struct Dilation
 * Filter height and width dilation
 */
struct Dilation
{
    int height;
    int width;
};

/** @class Conv
 * @brief Convolution layer
 *
 * Current only support one upstream layer.
 */
class Conv : public Layer
{
public:
    /**
     * Conv constructor
     *
     * @param[in] name      layer name
     * @param[in] network   Network interface handle
     * @param[in] upstreams upstream layer vector
     * @param[in] pad       pad parameter
     * @param[in] stride    stride parameter
     * @param[in] kernel    kernel parameter
     * @param[in] dilation  dilation parameter
     */
    Conv(const std::string& name,
         const NetworkConstPtr& network,
         const LayerConstPtrVec& upstreams,
         const Pad& pad,
         const Stride& stride,
         const Kernel& kernel,
         const Dilation& dilation);

    /**
     * Desctructor
     */
    virtual ~Conv();

    /**
     * forward propagation
     */
    virtual void fwdPropagation();

    /**
     * backward propagation
     */
    virtual void bwdPropagation();

    /**
     * prepare forward propagation
     */
    virtual void setFwdPropagation();

private:
    /**
     * create convolution tensor and alloc memory on device, gradient as well
     * if applicable
     *
     * @return total size of device memory in byte
     */
    size_t Conv::constructConvlutionTensor();

    /**
     * destroy convolution tensor and its allocated memory from device, gradient
     * as well if applicable
     */
    void Conv::destroyConvolutionTensor();

    /**
     * construct filter tensor and alloc memory on device, gradient as
     * well if applicable
     */
    size_t Conv::constructFilterTensor();

    /**
     * destroy filter tensor and its allocated memory from deice, gradient as
     * well if applicable
     */
    void Conv::destroyFilterTensor();

    /**
     * create bias tensor and alloc memory on device, gradient as well if
     * applicable
     */
    size_t Conv::constructBiasTensor();

    /**
     * destroy bias tensor and its allocated memory from device, gradient as
     * well if applicable
     */
    void Conv::destroyBiasTensor();

private:
    struct  // Adam parameters
    {
        // size_t t = 0;
        float* d_m_f = nullptr;  // Adam moment parameter for filter
        float* d_v_f = nullptr;
        float* d_m_b = nullptr;  // Adam moment parameter for bias
        float* d_v_b = nullptr;
        float* tmp_f = nullptr;  // for g*g usage and other uses
        float* tmp_b = nullptr;
    } adam;

    struct  // mSGD parameters
    {
        float* d_v_f = nullptr;
        float* d_v_b = nullptr;
    } msgd;

private:
    const Pad pad_;
    const Stride stride_;
    const Kernel kernel_;
    const Dilation dilation_;
    const LayerConstPtr upstream_;

    float* filter_          = nullptr;
    float* filter_gradient_ = nullptr;
    float* bias_            = nullptr;
    float* bias_gradient_   = nullptr;

    cudnnFilterDescriptor_t filter_desc_;
    cudnnTensorDescriptor_t bias_desc_;
    cudnnConvolutionDescriptor_t conv_desc_;

    cudnnConvolutionFwdAlgo_t fwd_algo_;
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_;
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo_;
};

}  // namespace layers
}  // namespace nn
