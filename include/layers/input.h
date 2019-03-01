#pragma once
// self
#include "layer.h"

namespace nn {
namespace layers {
/** @class Input
 * @brief A layer take the input data, sepcificly image in CNN.
 */
class Input : public Layer {
 public:
    /**
     * constructor of Input layer
     *
     * @param[in] name      layer name
     * @param[in] network   Network handle
     * @param[in] c         image channel
     * @param[in] h         image height
     * @param[in] w         image width
     */
    Input(const std::string& name,
          const NetworkConstPtr& network,
          int c,
          int h,
          int w);

    /**
     * destroctor
     */
    virtual ~Input();

    /**
     * @brief prepare forward propagation
     * Set tensor descriptor and alloc memory on GPU
     *
     * @return total allocated GPU memory in bytes
     */
    size_t prepareFwdPropagation() final;

    /**
     * run forward propagation
     */
    void fwdPropagation() final;

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

 private:
    cudnnTensorDescriptor_t y_desc_ = nullptr;

    float* d_y_ = nullptr;
};

}  // namespace layers
}  // namespace nn
