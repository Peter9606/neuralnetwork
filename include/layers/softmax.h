#pragma once
// cudnn
#include <cudnn.h>

// self
#include "layer.h"
#include "layers/config.h"

namespace nn {
namespace layers {
/** @class Softmax
 * @brief Softmax
 */
class Softmax : public Layer {
 public:
    /**
     * Softmax constructor
     *
     * @param[in] name              layer name
     * @param[in] network           Network interface handle
     * @param[in] up                upstream layer
     * @param[in] in_place          in place or not, default is false
     */
    Softmax(const std::string& name,
            const NetworkConstPtr& network,
            const LayerConstPtr& up,
            bool in_place = false);

    /**
     * Desctructor
     */
    virtual ~Softmax();

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
     * get output dimension
     */
    Dim getDim() const;

    /**
     * get gradient w.r.t current layer's output
     *
     * @return gradient
     */
    float* getGradient() const;

 private:
    const bool in_place_;

    cudnnTensorDescriptor_t y_desc_;
    float* d_y_  = nullptr;
    float* d_dy_ = nullptr;
};

}  // namespace layers
}  // namespace nn
