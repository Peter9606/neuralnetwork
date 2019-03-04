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

#include <memory>
#include <string>
#include <vector>

#include "neuralnetwork/logger.h"
#include "neuralnetwork/network.h"

namespace nn {
using std::shared_ptr;
using std::vector;
using std::weak_ptr;

/** @struct Dim
 * A dimension for channel, height and width
 */
struct Dim {
    int c;
    int h;
    int w;
};

/** @class Layer
 * @brief Interface class reprensent a common base layer of a typical neural
 * network.
 * Currently a layer could have at most one up layer and multiple down
 * layers.
 * Below figure illustrates what is upstream & downstream image:
 * image...--> upstream --> current layer --> downstream -->...result
 */
class Layer : public std::enable_shared_from_this<Layer> {
 public:
    enum Type : int {
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
     * @param[in] name      layer name
     * @param[in] network   Network interface handle
     * @param[in] up        upstream Layer handle
     */
    explicit Layer(const std::string& name,
                   const NetworkConstPtr& network,
                   const shared_ptr<Layer const>& up);

    /**
     * Layer destructor
     */
    virtual ~Layer() = default;

    /**
     * load parameters from given data existing in host.
     * do nothing in default implementation.
     *
     * @param[in] h_params a float vector contains current layer's parameters
     */
    virtual void loadParameters(const shared_ptr<vector<float>>& h_params);

    /**
     * copy parameters from device to host
     * do nothing in default implementation.
     *
     * @return float vector which contains current layer's parameters
     */
    virtual shared_ptr<vector<float>> saveParameters();

    /**
     * prepare forward propagation
     * do nothing in default implementation.
     * concrete class should set various type descriptor, alloc memory on gpu
     * Network implementation will collect all layers' returned size.
     *
     * @return allocated memory size on GPU in bytes.
     */
    virtual size_t prepareFwdPropagation();

    /**
     * prepare backward propagation
     * do nothing in default implementation.
     * concrete class should set various type descriptor, alloc memory on gpu
     * Network implementation will collect all layers' returned size.
     *
     * @return allocated memory size on GPU in bytes.
     */
    virtual size_t prepareBwdPropagation();

    /**
     * run forward propagation
     */
    virtual void fwdPropagation();

    /**
     * run backward propgation
     *
     * @param[in] d_downstream_gradient downstream layer's output tensor
     * gradient
     * @return pointer to gradient on device w.r.t. current layer's output
     */
    virtual void bwdPropagation();

    /**
     * update weights
     * do nothing in default implmentation.
     */
    virtual void updateWeights();

    /**
     * get output tensor
     *
     * @return pointer to output tensor on device
     */
    virtual float* getTensor() const;

    /**
     * get output tensor descriptor
     *
     * @return output tensor descriptor
     */
    virtual cudnnTensorDescriptor_t getDescriptor() const;

    /**
     * @brief get graident w.r.t current layer's output
     *
     * The flow of backward propagation:
     *   1. downstream layer acquires upstream's gradient pointer which has
     *      already been allocated with proper device memory
     *   2. downstream caculates the gradient w.r.t input
     *
     * NOTE: if a layer doesn't need its gradient w.r.t to output, then just
     * return nullptr, which is also the default implementation's behavior
     *
     * @return gradient of current layer
     */
    virtual float* getGradient() const;

    /**
     * get output dimension
     *
     * @return dim of output tensor, in which x, y and z represents height,
     * width and channel respectively
     */
    virtual Dim getDim() const;

    /**
     * @brief get output channels
     * simple return c_
     *
     * @return output channel number
     */
    int getChannel() const;

    /**
     * @brief get output size
     * return the result of c_ * n_ * w_ * h_
     *
     * @return the length of output tensor
     */
    size_t getTensorSize() const;

    /**
     * @brief get output size in bytes
     *
     * @return the length in bytes of output tensor
     */
    size_t getTensorSizeInBytes() const;

 protected:
    const std::shared_ptr<logger> log_ = Logger::getLogger();
    const std::string name_;
    const weak_ptr<Network const> network_;
    const weak_ptr<Layer const> up_;

    cudnnTensorDescriptor_t y_desc_ = nullptr;
    float* d_y_                     = nullptr;
    float* d_dy_                    = nullptr;

    int n_ = 0; /** output image number  */
    int c_ = 0; /** output image channel */
    int h_ = 0; /** output imgae height  */
    int w_ = 0; /** output image width   */
};

using LayerPtr          = shared_ptr<Layer>;
using LayerConstPtr     = shared_ptr<Layer const>;
using LayerWeakPtr      = weak_ptr<Layer>;
using LayerWeakConstPtr = weak_ptr<Layer const>;
}  // namespace nn
