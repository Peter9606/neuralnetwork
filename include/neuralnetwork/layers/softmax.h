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

#include <string>

#include "neuralnetwork/layer.h"
#include "neuralnetwork/layers/config.h"

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
     * get gradient w.r.t current layer's output
     *
     * @return gradient
     */
    float* getGradient() const;

 private:
    const bool in_place_;

    cudnnTensorDescriptor_t y_desc_ = nullptr;
    float* d_y_                     = nullptr;
    float* d_dy_                    = nullptr;
};

}  // namespace layers
}  // namespace nn
