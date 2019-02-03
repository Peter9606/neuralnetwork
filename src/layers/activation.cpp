#include <cassert>
#include <map>

#include "error_check.h"
#include "layers/activation.h"

namespace
{
using nn::layers::Activation;

static std::map<Activation::Type, cudnnActivationMode_t> MODE = {
    {Activation::SIGMOID, CUDNN_ACTIVATION_SIGMOID},
    {Activation::RELU, CUDNN_ACTIVATION_RELU},
    {Activation::TANH, CUDNN_ACTIVATION_TANH},
    {Activation::CLIPPED_RELU, CUDNN_ACTIVATION_CLIPPED_RELU},
    {Activation::ELU, CUDNN_ACTIVATION_ELU},
};
}  // namespace

namespace nn
{
namespace layers
{
Activation::Activation(const std::string& name,
                       const NetworkConstPtr& network,
                       const LayerConstPtr& up,
                       Type type,
                       double coef,
                       bool in_place)
    : Layer(name, network, up)
    , type_(type)
    , coef_(coef)
    , in_place_(in_place)

{
    Dim d = up->getDim();
    c_    = d.c;
    h_    = d.h;
    w_    = d.w;
}

Activation::~Activation()
{
    checkCUDNN(cudnnDestroyActivationDescriptor(activation_desc_));

    if (!in_place_)
    {
        checkCUDNN(cudnnDestroyTensorDescriptor(y_desc_));
        checkCudaErrors(cudaFree(d_y_));
        checkCudaErrors(cudaFree(d_dy_));
    }
}

size_t Activation::prepareFwdPropagation()
{
    NetworkConstPtr nn = network_.lock();
    assert(("Network is expired", nn));
    LayerConstPtr up = up_.lock();
    assert(("Upstream is expired", up));

    checkCUDNN(cudnnCreateActivationDescriptor(&activation_desc_));
    checkCUDNN(cudnnSetActivationDescriptor(
        activation_desc_, ::MODE[type_], CUDNN_PROPAGATE_NAN, coef_));

    size_t total = 0;
    if (!in_place_)
    {
        total = n_ * c_ * h_ * w_;
        checkCudaErrors(cudaMalloc(&d_y_, total));

        checkCUDNN(cudnnCreateTensorDescriptor(&y_desc_));
        checkCUDNN(cudnnSetTensor4dDescriptor(
            y_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n_, c_, h_, w_));
    }
    else
    {
        y_desc_ = up->getYDescriptor();
        d_y_    = up->getY();
    }

    return total;
}

size_t Activation::prepareBwdPropagation()
{
    LayerConstPtr up = up_.lock();
    assert(("Upstream is expired", up));

    size_t total = 0;
    if (!in_place_)
    {
        total = n_ * c_ * h_ * w_;
        checkCudaErrors(cudaMalloc(&d_dy_, total));
    }
    else
    {
        d_dy_ = up->getGradient();
    }
    return total;
}

void Activation::fwdPropagation()
{
    NetworkConstPtr nn = network_.lock();
    assert(("Network is expired", nn));
    LayerConstPtr up = up_.lock();
    assert(("Upstream is expired", up));

    cudnnHandle_t cudnn_handle     = nn->getCudnnHandle();
    const float* alpha             = nn->getAlpha();
    const float* beta              = nn->getBeta();
    cudnnTensorDescriptor_t x_desc = up->getYDescriptor();
    float* d_x                     = up->getY();

    checkCUDNN(cudnnActivationForward(cudnn_handle,
                                      activation_desc_,
                                      alpha,
                                      x_desc,
                                      d_x,
                                      beta,
                                      y_desc_,
                                      d_y_));
}

void Activation::bwdPropagation()
{
    NetworkConstPtr nn = network_.lock();
    assert(("Network is expired", nn));
    LayerConstPtr up = up_.lock();
    assert(("Upstream is expired", up));

    /***
        cudnnHandle_t cudnn_handle     = nn->getCudnnHandle();
        const float* alpha             = nn->getAlpha();
        const float* beta              = nn->getBeta();
        cudnnTensorDescriptor_t x_desc = down->getYDescriptor();
        float* d_x                     = down->getY();
        float* d_dx                    = down->getGradient();

        checkCUDNN(cudnnActivationBackward(cudnn_handle,
                                           activation_desc_,
                                           alpha,
                                           y_desc_,
                                           d_y_,
                                           y_desc_,
                                           d_dy_,
                                           x_desc,
                                           d_x,
                                           beta,
                                           x_desc,
                                           d_dx));
    ***/
}

void Activation::updateWeights()
{
}

cudnnTensorDescriptor_t Activation::getYDescriptor() const
{
    return y_desc_;
}

float* Activation::getY() const
{
    return d_y_;
}

}  // namespace layers
}  // namespace nn
