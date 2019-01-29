#pragma once
// standard
#include <cassert>
#include <memory>
#include <string>
#include <vector>

// cudnn
#include <cudnn.h>

// self
#include "network.h"

namespace nn
{
/** @class Layer
 * @brief Layer reprensent a common base layer of a typical neural network.
 *
 * A layer could have multiple upstream and downstream layers, but can only
 * produce only type of output tensor named tensor_.
 * Layer has  the functions to forward and backward propagation.
 */
class Layer
{
public:
    enum Type : int
    {
        INPUT = 0,
        CONV,
        RELU,
        FC,
        SOFTMAX,
        DROPOUT,
        UNPOOL,
    };

public:
    /**
     * Layer constructor
     *
     * @param[in] name          layer name
     * @param[in] type          layer type
     * @param[in] network       Network interface handle
     * @param[in] upstreams     upstream layer vector
     */
    Layer(const std::string& name,
          Type type,
          const NetworkConstPtr network,
          const std::vector<std::shared_ptr<Layer const>>& upstreams);

    /**
     * Layer destructor
     */
    virtual ~Layer();

    /**
     * Get layer name
     *
     * @return layer name
     */
    const std::string getName() const;

    /**
     * Get layer type
     *
     * @return layer type
     */
    Layer::Type getType() const;

    /**
     * Get output tensor's dimention in x, y and z
     *
     * @return output tensor's dimention
     */
    dim3 getOutputTensorDim() const;

    /**
     * Get output tensor pointer which is pointing to device memory
     *
     * @return output tensor pointer
     */
    float* getOutputTensor() const;

    /**
     * Get grident pointer of output tensor which points to device memeory
     *
     * @return grident pointer
     */
    float* getOutputTensorGradient() const;

    /**
     * Get output tensor descriptor
     *
     * @return get output tensor's descriptor
     */
    cudnnTensorDescriptor_t getOutputTensorDesc() const;

    /**
     * Link upstream
     */
    virtual void linkTensor(std::shared_ptr<Layer const>& src);

    /**
     * Prepare forward propagation
     * each layer should have its own implmentation
     */
    virtual void setFwdPropagation(){};

    /**
     * Forward propagation, each layer should have its own implementation
     */
    virtual void fwdPropagation(){};

    /**
     * Backward propagation, each layer should have its own implementation
     */
    virtual void bwdPropagation(){};

    /**
     * << operator to stringizing a Layer object
     */
    friend std::ostream& operator<<(std::ostream& os, const Layer& layer);

private:
    /**
     * create output tensor and malloc memory on device, gradient as well if
     * applicable
     *
     * @return total size of device memory in byte
     */
    size_t constructOutputTensor();

    /**
     * destroy output tensor and its allocated memory from device, gradient as
     * well if applicable. It's safe to call this even no constructOutputTensor
     * is called before.
     */
    void destroyOutputTensor();

protected:
    const std::string name_;
    const Layer::Type type_;
    const NetworkConstPtr network_;
    const std::vector<std::shared_ptr<Layer const>> upstreams_;

    cudnnTensorDescriptor_t output_tensor_desc_;
    dim3 output_tensor_dim_;
    float* output_tensor_          = nullptr;
    float* output_tensor_gradient_ = nullptr;
};

using LayerPtr         = std::shared_ptr<Layer>;
using LayerConstPtr    = std::shared_ptr<Layer const>;
using LayerPtrVec      = std::vector<LayerPtr>;
using LayerConstPtrVec = std::vector<LayerConstPtr>;

}  // namespace nn
