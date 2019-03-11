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
#include "neuralnetwork/layers/config.h"

namespace nn {
namespace layers {
/** @class Pool
 * @brief Pool support max pooling
 */
class Pool : public Layer {
 public:
    enum Type : int {
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
     * @param[in] stride            stride
     * @param[in] pad               padding
     * @param[in] type              pooling type
     */
    Pool(const std::string& name,
         const NetworkConstPtr& network,
         const LayerConstPtr& up,
         const Window& window,
         const Stride& stride,
         const Pad& pad,
         Type type);

    /**
     * Pool constructor with no padding
     *
     * @param[in] name              layer name
     * @param[in] network           Network interface handle
     * @param[in] up                upstream layer
     * @param[in] window            pooling window
     * @param[in] stride            stride
     * @param[in] type              pooling type
     */
    Pool(const std::string& name,
         const NetworkConstPtr& network,
         const LayerConstPtr& up,
         const Window& window,
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

 private:
    const Window window_;
    const Pad pad_;
    const Stride stride_;
    const Type type_;

    cudnnPoolingDescriptor_t pool_desc_ = nullptr;
};

}  // namespace layers
}  // namespace nn
