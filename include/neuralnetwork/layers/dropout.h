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
/** @class Dropout
 * @brief Dropout layer
 */
class Dropout : public Layer {
 public:
    /**
     * Dropout constructor
     *
     * @param[in] name              layer name
     * @param[in] network           Network interface handle
     * @param[in] up                upstream layer
     * @param[in] dropout_rate      dropout rate
     */
    Dropout(const std::string& name,
            const NetworkConstPtr& network,
            const LayerConstPtr& up,
            float dropout_rate);

    /**
     * Desctructor
     */
    virtual ~Dropout();

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
    const float dropout_rate_;

    float* d_reserve_space_                = nullptr;
    float* d_states_                       = nullptr;
    cudnnDropoutDescriptor_t dropout_desc_ = nullptr;
    size_t reserve_space_size_in_bytes_;
};

}  // namespace layers
}  // namespace nn
