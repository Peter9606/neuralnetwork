// self
#include "error_check.h"
#include "layers/input.h"

namespace nn
{
namespace layers
{
Input::Input(const std::string& name,
             const NetworkConstPtr& network,
             int c,
             int h,
             int w)
    : Layer(name, network, nullptr)
{
    c_ = c;
    h_ = h;
    w_ = w;
}

Input::~Input()
{
    checkCUDNN(cudnnDestroyTensorDescriptor(y_desc_));
    checkCudaErrors(cudaFree(d_y_));
}

size_t Input::prepareFwdPropagation()
{
    const size_t size       = sizeof(float) * n_ * c_ * h_ * w_;
    NetworkConstPtr network = network_.lock();
    assert(("Network is expired", network));

    checkCUDNN(cudnnCreateTensorDescriptor(&y_desc_));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        y_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n_, c_, h_, w_));

    checkCudaErrors(cudaMalloc(&d_y_, size));

    return size;
}

void Input::fwdPropagation()
{
    // do nothing here.
}

cudnnTensorDescriptor_t Input::getDescriptor() const
{
    return y_desc_;
}

float* Input::getTensor() const
{
    return d_y_;
}

}  // namespace layers
}  // namespace nn
