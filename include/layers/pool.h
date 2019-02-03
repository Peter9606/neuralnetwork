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
/** @class Pool
 * @brief Pool support max pooling
 */
class Pool : public Layer
{
public:
    enum Type : int
    {
        MAX,
    };

public:
    /**
     * Pool constructor
     *
     * @param[in] name              layer name
     * @param[in] network           Network interface handle
     * @param[in] up                upstream layer
     * @param[in] window            pooling window
     * @param[in] pad               padding
     * @param[in] stride            stride
     * @param[in] type              pooling type
     */
    Pool(const std::string& name,
         const NetworkConstPtr& network,
         const LayerConstPtr& up,
         const Window& window,
         const Pad& pad,
         const Stride& stride,
         Type type);

    /**
     * Desctructor
     */
    virtual ~Pool();

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
     * get output dimension
     */
    Dim getDim() const;

    /**
     * get gradient, for Input layer always return nullptr
     *
     * @return nullptr
     */
    float* getGradient() const;

private:
    const Window window_;
    const Pad pad_;
    const Stride stride_;
    const Type type_;

    cudnnPoolingDescriptor_t pool_desc_;
    cudnnTensorDescriptor_t y_desc_;

    float* d_y_  = nullptr;
    float* d_dy_ = nullptr;
};

}  // namespace layers
}  // namespace nn
