#include <cassert>
#include <map>

#include "error_check.h"
#include "layers/pool.h"

namespace
{
using nn::layers::Pool;

static std::map<Pool::Type, cudnnPoolingMode_t> MODE = {
    {Pool::MAX, CUDNN_POOLING_MAX},
};
}  // namespace

namespace nn
{
namespace layers
{
Pool::Pool(const std::string& name,
           const NetworkConstPtr& network,
           const LayerConstPtr& up,
           const Window& window,
           const Pad& pad,
           const Stride& stride,
           Type type)
    : Layer(name, network, up)
    , window_(window_)
    , pad_(pad)
    , stride_(stride)
    , type_(type)

{
}

Pool::~Pool()
{
    checkCUDNN(cudnnDestroyPoolingDescriptor(pool_desc_));
    checkCUDNN(cudnnDestroyTensorDescriptor(y_desc_));
    checkCudaErrors(cudaFree(d_y_));
    checkCudaErrors(cudaFree(d_dy_));
}

size_t Pool::prepareFwdPropagation()
{
    checkCUDNN(cudnnCreatePoolingDescriptor(&pool_desc_));

    checkCUDNN(cudnnSetPooling2dDescriptor(pool_desc_,
                                           ::MODE[type_],
                                           CUDNN_PROPAGATE_NAN,
                                           window_.height,
                                           window_.width,
                                           pad_.vertical,
                                           pad_.horizontal,
                                           stride_.vertical,
                                           stride_.horizontal));
    /*
        int n;
        int c;
        int h;
        int w;
        checkCUDNN(cudnnGetPooling2dForwardOutputDim(
            pool_desc_, up->getYDescriptor(), &n, &c, &h, &w));

        checkCUDNN(cudnnCreateTensorDescriptor(&y_desc_));
        checkCUDNN(cudnnSetTensor4dDescriptor(
            y_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));

        const size_t len = n * c * h * w;
        checkCudaErrors(cudaMalloc(&d_y_, len));
        device_usage_ = len;

        if (!network->isInferenceOnly())
        {
            const Dim input_dim = up->getDim();
            const size_t flen   = n * input_dim.h * input_dim.w * input_dim.c;
            checkCudaErrors(cudaMalloc(&d_dy_, flen));
            device_usage_ += flen;
        }
    */
}

size_t Pool::prepareBwdPropagation()
{
}

void Pool::fwdPropagation()
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

    checkCUDNN(cudnnPoolingForward(
        cudnn_handle, pool_desc_, alpha, x_desc, d_x, beta, y_desc_, d_y_));
}

void Pool::bwdPropagation()
{
    /** No need down_

        NetworkConstPtr nn = network_.lock();
        assert(("Network is expired", nn));
        LayerConstPtr up = up_.lock();
        assert(("Upstream is expired", up));
        LayerConstPtr down = down_.lock();
        assert(("Downstream is expired", down));

        cudnnHandle_t cudnn_handle        = nn->getCudnnHandle();
        const float* alpha                = nn->getAlpha();
        const float* beta                 = nn->getBeta();
        cudnnTensorDescriptor_t down_desc = down->getYDescriptor();
        float* d_down_tensor              = down->getY();
        float* d_down_gradient            = down->getGradient();
        cudnnTensorDescriptor_t up_desc   = up->getYDescriptor();
        float* d_up_tensor                = up->getY();

        checkCUDNN(cudnnPoolingBackward(cudnn_handle,
                                        pool_desc_,
                                        alpha,
                                        y_desc_,
                                        d_y_,
                                        y_desc_,
                                        d_down_gradient,
                                        up_desc,
                                        d_up_tensor,
                                        beta,
                                        up_desc,
                                        d_dy_));
    ***/
}

cudnnTensorDescriptor_t Pool::getYDescriptor() const
{
    return y_desc_;
}

float* Pool::getY() const
{
    return d_y_;
}
}  // namespace layers
}  // namespace nn
