#pragma once
#include <memory>

// CUDA
#include <vector_types.h>

namespace nn
{
/** @struct ADAM solver
 */
struct Adam
{
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
struct Msgd
{
    float momentum = 0.9;     // momentum
    float L2       = 0.0005;  // L2 or weight decay
    float lr       = 0.01;    // learning rate
    size_t t       = 0;
};

/** @enum SolverType
 */
enum SolverType
{
    SGD = 0,  // Stochastic Gradient Descent
    mSGD,     // mini-batch Gradient Descent with momentum
    ADAM,
    InfOnly,
};

/** @struct SolverSetting
 */
struct SolverSetting
{
    SolverType type;
    union
    {
        Adam adam;
        Msgd msgd;
    } setting;
};

/** @enum LossType
 */
enum LossType
{
    CORSS_ENTROPY = 0,  // for classification mainly
    COMPOSITIONAL
};

class Network
{
public:
    /**
     * get batch size
     *
     * @return batch size
     */
    virtual int getBatchSize() const = 0;

    /**
     * get if inference only
     *
     * @return true if inference only, false otherwise
     */
    virtual bool isInferenceOnly() const = 0;

    /**
     * update total GPU memory needed by this network
     *
     * @param[in] inc increment GPU memory in byte
     */
    virtual void updateMemoryNeeded(long inc) const = 0;

    /**
     * get solver setting
     *
     * @return get solver setting
     */
    virtual SolverSetting getSolverSetting() const = 0;

    /**
     * get loss calculation method
     *
     * @return loss type
     */
    virtual LossType getLossType() const = 0;
};

using NetworkPtr      = std::shared_ptr<Network>;
using NetworkConstPtr = std::shared_ptr<Network const>;
}  // namespace nn
