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
#include <string>
#include "neuralnetwork/layer.h"

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
