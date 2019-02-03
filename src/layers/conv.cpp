#include <cassert>

#include "error_check.h"
#include "layers/conv.h"

namespace nn
{
namespace layers
{
Conv::Conv(const std::string& name,
           const NetworkConstPtr& network,
           const LayerConstPtr& up,
           const Kernel& kernel,
           const Pad& pad,
           const Stride& stride,
           const Dilation& dilation)
    : Layer(name, network, up)
    , kernel_(kernel)
    , pad_(pad)
    , stride_(stride)
    , dilation_(dilation)
{
}

Conv::Conv(const std::string& name,
           const NetworkConstPtr& network,
           const LayerConstPtr& up,
           const Kernel& kernel,
           const Pad& pad,
           const Stride& stride)
    : Conv(name, network, up, kernel, pad, stride, {1, 1})
{
}

Conv::Conv(const std::string& name,
           const NetworkConstPtr& network,
           const LayerConstPtr& up,
           const Kernel& kernel,
           const Pad& pad)
    : Conv(name, network, up, kernel, pad, {1, 1})
{
}

Conv::Conv(const std::string& name,
           const NetworkConstPtr& network,
           const LayerConstPtr& up,
           const Kernel& kernel)
    : Conv(name, network, up, kernel, {0, 0}, {1, 1})
{
}

Conv::~Conv()
{
    checkCUDNN(cudnnDestroyTensorDescriptor(y_desc_));
    checkCUDNN(cudnnDestroyTensorDescriptor(bias_desc_));

    checkCUDNN(cudnnDestroyFilterDescriptor(filter_desc_));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(conv_desc_));

    checkCudaErrors(cudaFree(d_bias_));
    checkCudaErrors(cudaFree(d_filter_));
    checkCudaErrors(cudaFree(d_y_));
    checkCudaErrors(cudaFree(d_dbias_));
    checkCudaErrors(cudaFree(d_dfilter_));
    checkCudaErrors(cudaFree(d_dy_));
}

void Conv::loadParameters(const shared_ptr<vector<float>>& h_params)
{
    // before loading parameters, all pointers on device should be null,
    // otherwise memory leak on device, which also is NOT the scenario
    assert(!d_filter_);
    assert(!d_bias_);

    // only one bias corresponding to one filter
    const size_t filter_num = kernel_.channel;
    size_t one_filter_count = h_params->size() / filter_num - 1;  // 1 is bias
    size_t filter_bytes     = one_filter_count * sizeof(float) * filter_num;
    size_t bias_bytes       = 1 * sizeof(float) * filter_num;

    // move filters and bias to device
    checkCudaErrors(cudaMemcpyAsync(
        d_filter_, h_params->data(), filter_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(
        cudaMemcpyAsync(d_bias_,
                        &h_params->data()[one_filter_count * filter_num],
                        bias_bytes,
                        cudaMemcpyHostToDevice));
}

shared_ptr<vector<float>> Conv::saveParameters()
{
    assert(d_filter_);
    assert(d_bias_);

    // TODO(Peter Han): NUST have already known the input feature map size
    // and channels
}

size_t Conv::prepareFwdPropagation()
{
    NetworkConstPtr nn = network_.lock();
    assert(("Network is expired", nn));
    LayerConstPtr up = up_.lock();
    assert(("Upstream is expired", up));

    // create descriptors
    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc_));
    checkCUDNN(cudnnCreateFilterDescriptor(&filter_desc_));
    checkCUDNN(cudnnCreateTensorDescriptor(&bias_desc_));
    checkCUDNN(cudnnCreateTensorDescriptor(&y_desc_));

    // compute output dimension and set descriptors
    cudnnTensorDescriptor_t x_desc = up->getYDescriptor();
    int n;
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
        conv_desc_, x_desc, filter_desc_, &n, &c_, &h_, &w_));
    assert(("Batch size doesn't match", n == n_));
    assert(("Channel doesn't match", kernel_.channel == c_));

    checkCUDNN(cudnnSetConvolution2dDescriptor(conv_desc_,
                                               pad_.vertical,
                                               pad_.horizontal,
                                               stride_.vertical,
                                               stride_.horizontal,
                                               dilation_.height,
                                               dilation_.width,
                                               CUDNN_CROSS_CORRELATION,
                                               CUDNN_DATA_FLOAT));
    checkCUDNN(cudnnSetFilter4dDescriptor(filter_desc_,
                                          CUDNN_DATA_FLOAT,
                                          CUDNN_TENSOR_NCHW,
                                          kernel_.channel,
                                          c_,
                                          h_,
                                          w_));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        y_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n_, c_, h_, w_));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        bias_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, c_, 1, 1));

    // alloc memory
    const size_t tensor_size = sizeof(float) * n_ * c_ * h_ * w_;
    const size_t bias_size   = sizeof(float) * c_;
    const size_t filter_size =
        sizeof(float) * kernel_.height * kernel_.width * kernel_.channel;
    const size_t total = tensor_size + bias_size + filter_size;

    checkCudaErrors(cudaMalloc(&d_y_, tensor_size));
    checkCudaErrors(cudaMalloc(&d_bias_, bias_size));
    checkCudaErrors(cudaMalloc(&d_filter_, filter_size));

    // find best suited algorithm and update Network's workspace size
    size_t workspace_size = 0;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(
        nn->getCudnnHandle(),
        up->getYDescriptor(),
        filter_desc_,
        conv_desc_,
        y_desc_,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        0,
        &fwd_algo_));
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(nn->getCudnnHandle(),
                                                       up->getYDescriptor(),
                                                       filter_desc_,
                                                       conv_desc_,
                                                       y_desc_,
                                                       fwd_algo_,
                                                       &workspace_size));
    nn->updateWorkspaceSize(workspace_size);

    return total;
}

size_t Conv::prepareBwdPropagation()
{
    NetworkConstPtr nn = network_.lock();
    assert(("Network is expired", nn));
    LayerConstPtr up = up_.lock();
    assert(("Upstream is expired", up));

    size_t workspace_size          = 0;
    cudnnHandle_t cudnn_handle     = nn->getCudnnHandle();
    cudnnTensorDescriptor_t x_desc = up->getYDescriptor();
    const size_t tensor_size       = sizeof(float) * n_ * c_ * h_ * w_;
    const size_t bias_size         = sizeof(float) * c_;
    const size_t filter_size =
        sizeof(float) * kernel_.height * kernel_.width * kernel_.channel;
    const size_t total = tensor_size + bias_size + filter_size;

    checkCudaErrors(cudaMalloc(&d_dy_, tensor_size));
    checkCudaErrors(cudaMalloc(&d_dbias_, bias_size));
    checkCudaErrors(cudaMalloc(&d_dfilter_, filter_size));

    checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
        cudnn_handle,
        x_desc,
        y_desc_, /* dyDesc == yDesc */
        conv_desc_,
        filter_desc_,
        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
        0,
        &bwd_filter_algo_));
    checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle,
                                                              x_desc,
                                                              y_desc_,
                                                              conv_desc_,
                                                              filter_desc_,
                                                              bwd_filter_algo_,
                                                              &workspace_size));
    nn->updateWorkspaceSize(workspace_size);

    checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
        cudnn_handle,
        filter_desc_,
        y_desc_,
        conv_desc_,
        x_desc,
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
        0,
        &bwd_data_algo_));
    checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle,
                                                            filter_desc_,
                                                            y_desc_,
                                                            conv_desc_,
                                                            x_desc,
                                                            bwd_data_algo_,
                                                            &workspace_size));
    nn->updateWorkspaceSize(workspace_size);

    return total;
}

void Conv::fwdPropagation()
{
    NetworkConstPtr nn = network_.lock();
    assert(("Network is expired", nn));
    LayerConstPtr up = up_.lock();
    assert(("Upstream is expired", up));

    cudnnHandle_t cudnn_handle     = nn->getCudnnHandle();
    float* d_workspace             = nn->getWorkspace();
    size_t workspace_size          = nn->getWorkspaceSize();
    const float* alpha             = nn->getAlpha();
    const float* beta              = nn->getBeta();
    cudnnTensorDescriptor_t x_desc = up->getYDescriptor();
    float* d_x                     = up->getY();

    checkCUDNN(cudnnConvolutionForward(cudnn_handle,
                                       alpha,
                                       x_desc,
                                       d_x,
                                       filter_desc_,
                                       d_filter_,
                                       conv_desc_,
                                       fwd_algo_,
                                       d_workspace,
                                       workspace_size,
                                       beta,
                                       y_desc_,
                                       d_y_));
    checkCUDNN(cudnnAddTensor(
        cudnn_handle, &alpha, bias_desc_, d_bias_, alpha, y_desc_, d_y_));
}

void Conv::bwdPropagation()
{
    // TODO(Peter Han): Should review all bwdPropagation methods, figure out x,
    // y, dx, dy .  Dont have to hold down_ in layer ???

    /*
        NetworkConstPtr nn = network_.lock();
        assert(("Network is expired", nn));
        LayerConstPtr up = up_.lock();
        assert(("Upstream is expired", up));
        LayerConstPtr down = down_.lock();
        assert(("Downstream is expired", down));

        cudnnHandle_t cudnnHandle      = nn->getCudnnHandle();
        float* d_workspace             = nn->getWorkspace();
        size_t workspace_size          = nn->getWorkspaceSize();
        const float* alpha             = nn->getAlpha();
        const float* beta              = nn->getBeta();
        cudnnTensorDescriptor_t x_desc = up->getYDescriptor();
        float* d_x                     = up->getY();
        float* d_downstream_gradient   = down->getGradient();

        checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle,
                                                alpha,
                                                y_desc_,
                                                d_downstream_gradient,
                                                beta,
                                                bias_desc_,
                                                d_bias_gradient_));

        checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle,
                                                  alpha,
                                                  x_desc,
                                                  d_x,
                                                  y_desc_,
                                                  d_dy_,
                                                  conv_desc_,
                                                  bwd_filter_algo_,
                                                  d_workspace,
                                                  workspace_size,
                                                  beta,
                                                  filter_desc_,
                                                  d_filter_gradient_));

        checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle,
                                                alpha,
                                                filter_desc_,
                                                d_filter_,
                                                y_desc_,
                                                d_downstream_gradient,
                                                conv_desc_,
                                                bwd_data_algo_,
                                                d_workspace,
                                                workspace_size,
                                                beta,
                                                x_desc,
                                                d_dy_));
    */
}

void Conv::updateWeights()
{
}

cudnnTensorDescriptor_t Conv::getYDescriptor() const
{
    return y_desc_;
}

float* Conv::getY() const
{
    return d_y_;
}

}  // namespace layers
}  // namespace nn
