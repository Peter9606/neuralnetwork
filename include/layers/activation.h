#pragma once
// cudnn
#include <cudnn.h>

// self
#include "layer.h"

namespace nn
{
namespace layers
{
/** @class Activation
 * @brief Activation layer, support ReLU, Sigmoid, Tanh, Clipped ReLU and ELU
 */
class Activation : public Layer
{
public:
    enum Type : int
    {
        SIGMOID,
        RELU,
        TANH,
        CLIPPED_RELU,
        ELU,
    };

public:
    /**
     * Activation constructor
     *
     * @param[in] name              layer name
     * @param[in] network           Network interface handle
     * @param[in] up                upstream layer
     * @param[in] type              activation type
     * @param[in] coef              clipping threashold for clipped relu or
     * alpha for elu
     * @param[in] in_place          in place
     */
    Activation(const std::string& name,
               const NetworkConstPtr& network,
               const LayerConstPtr& up,
               Type type     = RELU,
               double coef   = 0.0,
               bool in_place = true);

    /**
     * Desctructor
     */
    virtual ~Activation();

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
     * @return nullptr
     */
    float* getGradient() const;

private:
    const Type type_;
    const double coef_;
    const bool in_place_;

    cudnnActivationDescriptor_t activation_desc_;
    cudnnTensorDescriptor_t y_desc_;

    float* d_y_  = nullptr;
    float* d_dy_ = nullptr;
};

}  // namespace layers
}  // namespace nn
