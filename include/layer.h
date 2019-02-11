#pragma once
// standard
#include <cassert>
#include <memory>
#include <string>
#include <vector>

// cudnn
#include <cudnn.h>

// self
#include "network.h"

namespace nn
{
using std::shared_ptr;
using std::vector;
using std::weak_ptr;

/** @struct Dim
 * A dimension for channel, height and width
 */
struct Dim
{
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
class Layer : public std::enable_shared_from_this<Layer>
{
public:
    enum Type : int
    {
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
    virtual ~Layer() = 0;

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
    virtual void fwdPropagation() = 0;

    /**
     * run backward propgation
     *
     * @param[in] d_downstream_gradient downstream layer's output tensor
     * gradient
     * @return pointer to gradient on device w.r.t. current layer's output
     */
    virtual void bwdPropagation() = 0;

    /**
     * update weights
     * do nothing in default implmentation.
     */
    virtual void updateWeights();

    /**
     * get output tensor descriptor
     *
     * @return output tensor descriptor
     */
    virtual cudnnTensorDescriptor_t getDescriptor() const = 0;

    /**
     * get output tensor
     *
     * @return pointer to output tensor on device
     */
    virtual float* getTensor() const = 0;

    /**
     * get output dimension
     *
     * @return dim of output tensor, in which x, y and z represents height,
     * width and channel respectively
     */
    virtual Dim getDim() const;

    /**
     * get graident w.r.t current layer's output
     *
     * @return gradient of current layer
     */
    virtual float* getGradient() const = 0;

    /**
     * append downstream layer
     *
     * @param[in] layer downstream layer
     */
    virtual void appendDownstream(const shared_ptr<Layer const>& layer) const;

protected:
    const std::string name_;
    const weak_ptr<Network const> network_;
    const weak_ptr<Layer const> up_;
    mutable vector<weak_ptr<Layer const>> down_vector_;

    int n_; /** output image number  */
    int c_; /** output image channel */
    int h_; /** output imgae height  */
    int w_; /** output image width   */
};

using LayerPtr          = shared_ptr<Layer>;
using LayerConstPtr     = shared_ptr<Layer const>;
using LayerWeakPtr      = weak_ptr<Layer>;
using LayerWeakConstPtr = weak_ptr<Layer const>;
}  // namespace nn
