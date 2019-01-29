#include "error_check.h"
#include "layer.h"

namespace nn
{
Layer::Layer(const std::string& name,
             Layer::Type type,
             const NetworkConstPtr network,
             const LayerConstPtrVec& upstreams)
    : name_(name)
    , type_(type)
    , network_(network)
    , upstreams_(upstreams)
{
}

Layer::~Layer()
{
    destroyOutputTensor();
}

size_t Layer::constructOutputTensor()
{
    checkCUDNN(cudnnCreateTensorDescriptor(&output_tensor_desc_));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_tensor_desc_,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          network_->getBatchSize(),
                                          output_tensor_dim_.x,
                                          output_tensor_dim_.y,
                                          output_tensor_dim_.z));

    // size of current layer's tensor
    size_t memory_size = sizeof(float) * network_->getBatchSize() *
                         output_tensor_dim_.x * output_tensor_dim_.y *
                         output_tensor_dim_.z;

    // allocate memory for tensor on GPU
    checkCudaErrors(cudaMalloc(&output_tensor_, memory_size));

    // allocate memory for gradient on GPU if applicable
    if (type_ != INPUT && !network_->isInferenceOnly())
    {
        checkCudaErrors(cudaMalloc(&output_tensor_gradient_, memory_size));
        memory_size *= 2;
    }

    std::cout << this << " memory allocated " << memory_size << "MB"
              << std::endl;
    return memory_size;
}

void Layer::destroyOutputTensor()
{
    // cudaFree 0 at will
    checkCudaErrors(cudaFree(output_tensor_));
    output_tensor_ = nullptr;

    checkCudaErrors(cudaFree(output_tensor_gradient_));
    output_tensor_gradient_ = nullptr;

    // it's safe to destory a null pointer
    checkCUDNN(cudnnDestroyTensorDescriptor(output_tensor_desc_));
}

void Layer::linkTensor(LayerConstPtr& src)
{
    assert(type_ == RELU);
    output_tensor_          = src->getOutputTensor();
    output_tensor_dim_      = src->getOutputTensorDim();
    output_tensor_gradient_ = src->getOutputTensorGradient();
}

const std::string Layer::getName() const
{
    return name_;
}

Layer::Type Layer::getType() const
{
    return type_;
}

dim3 Layer::getOutputTensorDim() const
{
    return output_tensor_dim_;
}

float* Layer::getOutputTensor() const
{
    return output_tensor_;
}

float* Layer::getOutputTensorGradient() const
{
    return output_tensor_gradient_;
}

cudnnTensorDescriptor_t Layer::getOutputTensorDesc() const
{
    return output_tensor_desc_;
}

std::ostream& operator<<(std::ostream& os, const Layer& layer)
{
    os << "Layer[" << layer.name_ << "]";
    return os;
}

}  // namespace nn
