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
/** @class Activation
 * @brief Activation layer, support ReLU, Sigmoid, Tanh, Clipped ReLU and ELU
 */
class Activation : public Layer {
 public:
    enum Type : int {
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

 private:
    const Type type_;
    const double coef_;
    const bool in_place_;

    cudnnActivationDescriptor_t activation_desc_ = nullptr;

    size_t y_size_in_bytes_ = 0;
};

}  // namespace layers
}  // namespace nn
