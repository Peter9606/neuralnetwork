#include <cassert>
#include <map>

#include "error_check.h"
#include "layers/softmax.h"

namespace nn
{
namespace layers
{
Softmax::Softmax(const std::string& name,
                 const NetworkConstPtr& network,
                 const LayerConstPtr& up,
                 bool in_place)
    : Layer(name, network, up)
    , in_place_(in_place)

{
    const Dim d = up->getDim();
    c_          = d.c;
    h_          = d.h;
    w_          = d.w;
}

Softmax::~Softmax()
{
    if (!in_place_)
    {
        checkCUDNN(cudnnDestroyTensorDescriptor(y_desc_));
        checkCudaErrors(cudaFree(d_y_));
        checkCudaErrors(cudaFree(d_dy_));
    }
}

size_t Softmax::prepareFwdPropagation()
{
    size_t total = 0;
    if (!in_place_)
    {
        const size_t size = n_ * c_ * h_ * w_;
        checkCudaErrors(cudaMalloc(&d_y_, size));

        checkCUDNN(cudnnCreateTensorDescriptor(&y_desc_));
        checkCUDNN(cudnnSetTensor4dDescriptor(
            y_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n_, c_, h_, w_));

        NetworkConstPtr network = network_.lock();
        assert(("Network is expired", network));
        if (!network->isInferenceOnly())
        {
            checkCudaErrors(cudaMalloc(&d_dy_, size));
        }
    }
    else
    {
        LayerConstPtr up = up_.lock();
        assert(("Upstream is expired", up));
        y_desc_ = up->getDescriptor();
        d_y_    = up->getTensor();
        d_dy_   = up->getGradient();
    }
    return total;
}

size_t Softmax::prepareBwdPropagation()
{
    size_t total = 0;
    if (!in_place_)
    {
        const size_t size = n_ * c_ * h_ * w_;
        checkCudaErrors(cudaMalloc(&d_dy_, size));
        total = size;
    }
    return total;
}

void Softmax::fwdPropagation()
{
    NetworkConstPtr nn = network_.lock();
    assert(("Network is expired", nn));
    LayerConstPtr up = up_.lock();
    assert(("Upstream is expired", up));

    cudnnHandle_t cudnn_handle     = nn->getCudnnHandle();
    const float* alpha             = nn->getAlpha();
    const float* beta              = nn->getBeta();
    cudnnTensorDescriptor_t x_desc = up->getDescriptor();
    float* d_x                     = up->getTensor();

    checkCUDNN(cudnnSoftmaxForward(cudnn_handle,
                                   CUDNN_SOFTMAX_ACCURATE,
                                   CUDNN_SOFTMAX_MODE_CHANNEL,
                                   alpha,
                                   x_desc,
                                   d_x,
                                   beta,
                                   y_desc_,
                                   d_y_));
}

void Softmax::bwdPropagation()
{
    NetworkConstPtr nn = network_.lock();
    assert(("Network is expired", nn));
    cudnnHandle_t cudnn_handle = nn->getCudnnHandle();
    const float* alpha         = nn->getAlpha();
    const float* beta          = nn->getBeta();

    /* TODO(Peter Han): not complete
        checkCUDNN(cudnnSoftmaxBackward(cudnn_handle,
                                        CUDNN_SOFTMAX_ACCURATE,
                                        CUDNN_SOFTMAX_MODE_CHANNEL,
                                        alpha,
                                        tensor,
                                        data,
                                        tensor,
                                        diff_data,
                                        beta,
                                        dst->tensor,
                                        dst->diff_data));
    */
}

void Softmax::updateWeights()
{
}

cudnnTensorDescriptor_t Softmax::getDescriptor() const
{
    return y_desc_;
}

float* Softmax::getTensor() const
{
    return d_y_;
}

}  // namespace layers
}  // namespace nn
