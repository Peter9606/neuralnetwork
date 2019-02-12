#include <cassert>
#include <random>

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
    checkCudaErrors(cudaFree(d_dbias_));
    checkCudaErrors(cudaFree(d_filter_));
    checkCudaErrors(cudaFree(d_dfilter_));
    checkCudaErrors(cudaFree(d_y_));
    checkCudaErrors(cudaFree(d_dy_));
}

void Conv::loadParameters(const shared_ptr<vector<float>>& h_params)
{
    /*
        // before loading parameters, all pointers on device should be null,
        // otherwise memory leak on device, which also is NOT the scenario
        assert(!d_filter_);
        assert(!d_bias_);

        // only one bias corresponding to one filter
        const size_t filter_num = kernel_.channel;
        size_t one_filter_count = h_params->size() / filter_num - 1;  // 1 is
       bias size_t filter_bytes     = one_filter_count * sizeof(float) *
       filter_num; size_t bias_bytes       = 1 * sizeof(float) * filter_num;

        // move filters and bias to device
        checkCudaErrors(cudaMemcpyAsync(
            d_filter_, h_params->data(), filter_bytes, cudaMemcpyHostToDevice));
        checkCudaErrors(
            cudaMemcpyAsync(d_bias_,
                            &h_params->data()[one_filter_count * filter_num],
                            bias_bytes,
                            cudaMemcpyHostToDevice));
    */
    LayerConstPtr up = up_.lock();
    assert(("Upstream is expired", up));

    const int input_channel = up->getDim().c;
    const size_t filter_length =
        input_channel * kernel_.width * kernel_.height * kernel_.channel;
    std::vector<float> h_vect(filter_length);

    std::random_device rd;
    std::mt19937 gen(rd());
    float w = sqrt(3.0f / (input_channel * kernel_.width * kernel_.height));
    std::uniform_real_distribution<> dist(-w, w);
    for (auto&& ite : h_vect)
    {
        ite = static_cast<float>(dist(gen));
    }
    cudaMemcpyAsync(d_filter_,
                    &h_vect[0],
                    sizeof(float) * h_vect.size(),
                    cudaMemcpyHostToDevice);
}

shared_ptr<vector<float>> Conv::saveParameters()
{
    assert(d_filter_);
    assert(d_bias_);

    // TODO(Peter Han)
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

    // compute output dimension based on input and filter
    cudnnTensorDescriptor_t x_desc = up->getDescriptor();
    checkCUDNN(cudnnSetFilter4dDescriptor(filter_desc_,
                                          CUDNN_DATA_FLOAT,
                                          CUDNN_TENSOR_NCHW,
                                          kernel_.channel,
                                          c_,
                                          h_,
                                          w_));
    checkCUDNN(cudnnSetConvolution2dDescriptor(conv_desc_,
                                               pad_.vertical,
                                               pad_.horizontal,
                                               stride_.vertical,
                                               stride_.horizontal,
                                               dilation_.height,
                                               dilation_.width,
                                               CUDNN_CROSS_CORRELATION,
                                               CUDNN_DATA_FLOAT));

    int n;
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
        conv_desc_, x_desc, filter_desc_, &n, &c_, &h_, &w_));
    // sanity check
    assert(("Batch size doesn't match", n == n_));
    assert(("Channel doesn't match", kernel_.channel == c_));

    checkCUDNN(cudnnSetTensor4dDescriptor(
        y_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n_, c_, h_, w_));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        bias_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, c_, 1, 1));

    // alloc memory
    const size_t tensor_size = sizeof(float) * n_ * c_ * h_ * w_;
    const size_t bias_size   = sizeof(float) * c_;
    const int input_channel  = up->getDim().c;
    const size_t filter_size = sizeof(float) * kernel_.height * kernel_.width *
                               input_channel * kernel_.channel;

    checkCudaErrors(cudaMalloc(&d_y_, tensor_size));
    checkCudaErrors(cudaMalloc(&d_bias_, bias_size));
    checkCudaErrors(cudaMalloc(&d_filter_, filter_size));

    // find best suited algorithm and update Network's workspace size
    size_t workspace_size = 0;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(
        nn->getCudnnHandle(),
        up->getDescriptor(),
        filter_desc_,
        conv_desc_,
        y_desc_,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        0,
        &fwd_algo_));
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(nn->getCudnnHandle(),
                                                       up->getDescriptor(),
                                                       filter_desc_,
                                                       conv_desc_,
                                                       y_desc_,
                                                       fwd_algo_,
                                                       &workspace_size));
    // notify network to update workspace size
    nn->updateWorkspaceSize(workspace_size);

    return tensor_size + bias_size + filter_size;
}

size_t Conv::prepareBwdPropagation()
{
    NetworkConstPtr nn = network_.lock();
    assert(("Network is expired", nn));
    LayerConstPtr up = up_.lock();
    assert(("Upstream is expired", up));

    cudnnHandle_t cudnn_handle     = nn->getCudnnHandle();
    cudnnTensorDescriptor_t x_desc = up->getDescriptor();
    const size_t tensor_size       = sizeof(float) * n_ * c_ * h_ * w_;
    const size_t bias_size         = sizeof(float) * c_;
    const int input_channel        = up->getDim().c;
    const size_t filter_size = sizeof(float) * kernel_.height * kernel_.width *
                               input_channel * kernel_.channel;
    const size_t total = tensor_size + bias_size + filter_size;

    checkCudaErrors(cudaMalloc(&d_dy_, tensor_size));
    checkCudaErrors(cudaMalloc(&d_dbias_, bias_size));
    checkCudaErrors(cudaMalloc(&d_dfilter_, filter_size));

    size_t workspace_size = 0;
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

    return tensor_size + bias_size + filter_size;
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
    cudnnTensorDescriptor_t x_desc = up->getDescriptor();
    float* d_x                     = up->getTensor();

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
    NetworkConstPtr nn = network_.lock();
    assert(("Network is expired", nn));
    LayerConstPtr up = up_.lock();
    assert(("Upstream is expired", up));

    cudnnHandle_t cudnn_handle     = nn->getCudnnHandle();
    float* d_workspace             = nn->getWorkspace();
    size_t workspace_size          = nn->getWorkspaceSize();
    const float* alpha             = nn->getAlpha();
    const float* beta              = nn->getBeta();
    cudnnTensorDescriptor_t x_desc = up->getDescriptor();
    float* d_x                     = up->getTensor();
    float* d_dx                    = up->getGradient();

    checkCUDNN(cudnnConvolutionBackwardBias(
        cudnn_handle, alpha, y_desc_, d_dy_, beta, bias_desc_, d_dbias_));

    checkCUDNN(cudnnConvolutionBackwardFilter(cudnn_handle,
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
                                              d_dfilter_));

    if (!d_dx)
    {
        return;
    }
    checkCUDNN(cudnnConvolutionBackwardData(cudnn_handle,
                                            alpha,
                                            filter_desc_,
                                            d_filter_,
                                            y_desc_,
                                            d_dy_,
                                            conv_desc_,
                                            bwd_data_algo_,
                                            d_workspace,
                                            workspace_size,
                                            beta,
                                            x_desc,
                                            d_dx));
}

void Conv::updateWeights()
{
    NetworkConstPtr nn = network_.lock();
    assert(("Network is expired", nn));
    LayerConstPtr up = up_.lock();
    assert(("Upstream is expired", up));

    const SolverSetting setting = nn->getSolverSetting();
    const float learning_rate   = setting.learning_rate;
    const size_t bias_size      = sizeof(float) * c_;
    const int input_channel     = up->getDim().c;
    const size_t filter_size = sizeof(float) * kernel_.height * kernel_.width *
                               input_channel * kernel_.channel;

    cublasHandle_t cublas_handle = nn->getCublasHandle();

    checkCudaErrors(cublasSaxpy(cublas_handle,
                                static_cast<int>(filter_size),
                                &learning_rate,
                                d_dfilter_,
                                1,
                                d_filter_,
                                1));
    checkCudaErrors(cublasSaxpy(cublas_handle,
                                static_cast<int>(bias_size),
                                &learning_rate,
                                d_dbias_,
                                1,
                                d_bias_,
                                1));
}

cudnnTensorDescriptor_t Conv::getDescriptor() const
{
    return y_desc_;
}

float* Conv::getTensor() const
{
    return d_y_;
}

float* Conv::getGradient() const
{
    return d_dy_;
}

}  // namespace layers
}  // namespace nn
