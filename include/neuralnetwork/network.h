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
#include <cublas_v2.h>
#include <cudnn.h>

#include <memory>

namespace nn {
using std::shared_ptr;
using std::weak_ptr;

/** @struct ADAM solver
 */
struct Adam {
    float alpha = 0.001;  // value from ADAM paper
    float alpha_t;        // this will be calcualted at each step
    float beta1   = 0.9;
    float beta2   = 0.999;
    float epsilon = 1.0e-8;
    float epsilon_t;
    size_t t = 0;
};

/** @struct SGD with Momentum
 */
struct Msgd {
    float momentum = 0.9;     // momentum
    float L2       = 0.0005;  // L2 or weight decay
    float lr       = 0.01;    // learning rate
    size_t t       = 0;
};

/** @enum SolverType
 */
enum SolverType {
    SGD = 0,  // Stochastic Gradient Descent
    mSGD,     // mini-batch Gradient Descent with momentum
    ADAM,
    InfOnly,
};

/** @struct SolverSetting
 */
struct SolverSetting {
    SolverType type;
    /*
        union {
            Adam adam;
            Msgd msgd;
        } setting;
    */
};

/** @enum LossType
 */
enum LossType {
    CROSS_ENTROPY = 0,  // for classification mainly
    COMPOSITIONAL
};

class Network {
 public:
    virtual ~Network() = default;

    /**
     * get batch size
     *
     * @return batch size
     */
    virtual int getBatchSize() const = 0;

    /**
     * update total GPU memory needed by this network
     *
     * @param[in] inc increment GPU memory in byte
     */
    virtual void updateMemoryNeeded(size_t inc) const = 0;

    /**
     * get learning rate
     *
     * @return get learning rate
     */
    virtual float getLearningRate() const = 0;

    /**
     * get loss calculation method
     *
     * @return loss type
     */
    virtual LossType getLossType() const = 0;

    /**
     * get cudnn handle
     *
     * @return cudnn handle
     */
    virtual cudnnHandle_t getCudnnHandle() const = 0;

    /**
     * get cublas handle
     *
     * @return cublas handle
     */
    virtual cublasHandle_t getCublasHandle() const = 0;

    /**
     * get alpha scaling factor
     *
     * @return alpha scaling factor
     */
    virtual const float* getAlpha() const = 0;

    /**
     * get beta scaling factor
     *
     * @return beta scaling factor
     */
    virtual const float* getBeta() const = 0;

    /**
     * @brief get workspace size
     * This API should be called after all Conv layers called
     * updateWorkspaceSize
     *
     * @return workspace size in bytes
     */
    virtual size_t getWorkspaceSize() const = 0;

    /**
     * @brief update workspace size
     * According to CUDNN guide, convolution forward and backward API need an
     * unidifed workspace size which should be big enough for all conv layers.
     * So each layer calling this API to notify its size to Network.
     *
     * @param[in] size workspace size in bytes
     */
    virtual void updateWorkspaceSize(size_t size) const = 0;

    /**
     * get workspace
     *
     * @return pointer to workspace in device
     */
    virtual float* getWorkspace() const = 0;

    /**
     * @brief if network is inference only
     *
     * @return true if inference only, otherwise false
     */
    virtual bool isInferenceOnly() const = 0;
};

using NetworkPtr          = shared_ptr<Network>;
using NetworkConstPtr     = shared_ptr<Network const>;
using NetworkWeakPtr      = weak_ptr<Network>;
using NetworkWeakConstPtr = weak_ptr<Network const>;
}  // namespace nn
