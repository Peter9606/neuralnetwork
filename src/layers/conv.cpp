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
#include <cassert>
#include <random>

#include "neuralnetwork/error_check.h"
#include "neuralnetwork/layers/conv.h"

namespace nn {
namespace layers {
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
    , input_channel_(up->getDim().c) {
    checkCUDNN(cudnnCreateFilterDescriptor(&filter_desc_));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc_));
    checkCUDNN(cudnnCreateTensorDescriptor(&bias_desc_));
    checkCUDNN(cudnnCreateTensorDescriptor(&y_desc_));

    checkCUDNN(cudnnSetFilter4dDescriptor(filter_desc_,
                                          CUDNN_DATA_FLOAT,
                                          CUDNN_TENSOR_NCHW,
                                          kernel_.channel,
                                          input_channel_,
                                          kernel_.height,
                                          kernel_.width));
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
    cudnnTensorDescriptor_t x_desc = up->getDescriptor();
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
        conv_desc_, x_desc, filter_desc_, &n, &c_, &h_, &w_));

    // sanity check
    assert(("Batch size doesn't match", n == n_));
    assert(("Channel doesn't match", kernel_.channel == c_));

    checkCUDNN(cudnnSetTensor4dDescriptor(
        y_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n_, c_, h_, w_));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        bias_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, c_, 1, 1));

    cudnnHandle_t cudnn_handle = network->getCudnnHandle();
    {
        // find best suited algorithm and update Network's workspace size
        size_t workspace_size = 0;
        checkCUDNN(cudnnGetConvolutionForwardAlgorithm(
            cudnn_handle,
            x_desc,
            filter_desc_,
            conv_desc_,
            y_desc_,
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            0,
            &fwd_algo_));
        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle,
                                                           x_desc,
                                                           filter_desc_,
                                                           conv_desc_,
                                                           y_desc_,
                                                           fwd_algo_,
                                                           &workspace_size));
        // notify network to update workspace size
        network->updateWorkspaceSize(workspace_size);
    }

    {
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
        checkCUDNN(
            cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle,
                                                           x_desc,
                                                           y_desc_,
                                                           conv_desc_,
                                                           filter_desc_,
                                                           bwd_filter_algo_,
                                                           &workspace_size));
        network->updateWorkspaceSize(workspace_size);

        checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
            cudnn_handle,
            filter_desc_,
            y_desc_,
            conv_desc_,
            x_desc,
            CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
            0,
            &bwd_data_algo_));
        checkCUDNN(
            cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle,
                                                         filter_desc_,
                                                         y_desc_,
                                                         conv_desc_,
                                                         x_desc,
                                                         bwd_data_algo_,
                                                         &workspace_size));
        network->updateWorkspaceSize(workspace_size);
    }
}

Conv::Conv(const std::string& name,
           const NetworkConstPtr& network,
           const LayerConstPtr& up,
           const Kernel& kernel,
           const Pad& pad,
           const Stride& stride)
    : Conv(name, network, up, kernel, pad, stride, {1, 1}) {
}

Conv::Conv(const std::string& name,
           const NetworkConstPtr& network,
           const LayerConstPtr& up,
           const Kernel& kernel,
           const Pad& pad)
    : Conv(name, network, up, kernel, pad, {1, 1}) {
}

Conv::Conv(const std::string& name,
           const NetworkConstPtr& network,
           const LayerConstPtr& up,
           const Kernel& kernel)
    : Conv(name, network, up, kernel, {0, 0}, {1, 1}) {
}

Conv::~Conv() {
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

void Conv::loadParameters(const shared_ptr<vector<float>>& h_params) {
    LayerConstPtr up = up_.lock();
    assert(("Upstream is expired", up));
    const int input_channel = up->getChannel();
    std::random_device rd;
    std::mt19937 gen(rd());
    float w = sqrt(3.0f / (input_channel * kernel_.width * kernel_.height));
    std::uniform_real_distribution<> dist(-w, w);

    {
        std::vector<float> filter(getFilterSize());
        for (auto&& ite : filter) {
            ite = static_cast<float>(dist(gen));
        }
        cudaMemcpyAsync(d_filter_,
                        &filter[0],
                        getFilterSizeInBytes(),
                        cudaMemcpyHostToDevice);
    }
    {
        std::vector<float> bias(getBiasSize());
        for (auto&& ite : bias) {
            ite = static_cast<float>(dist(gen));
        }
        cudaMemcpyAsync(
            d_bias_, &bias[0], getBiasSizeInBytes(), cudaMemcpyHostToDevice);
    }
}

shared_ptr<vector<float>> Conv::saveParameters() {
    assert(d_filter_);
    assert(d_bias_);

    // TODO(Peter Han)
}

size_t Conv::prepareFwdPropagation() {
    checkCudaErrors(cudaMalloc(&d_y_, getTensorSizeInBytes()));
    checkCudaErrors(cudaMalloc(&d_bias_, getBiasSizeInBytes()));
    checkCudaErrors(cudaMalloc(&d_filter_, getFilterSizeInBytes()));

    return getTensorSizeInBytes() + getBiasSizeInBytes() +
           getFilterSizeInBytes();
}

size_t Conv::prepareBwdPropagation() {
    checkCudaErrors(cudaMalloc(&d_dy_, getTensorSizeInBytes()));
    checkCudaErrors(cudaMalloc(&d_dbias_, getBiasSizeInBytes()));
    checkCudaErrors(cudaMalloc(&d_dfilter_, getFilterSizeInBytes()));

    return getTensorSizeInBytes() + getBiasSizeInBytes() +
           getFilterSizeInBytes();
}

void Conv::fwdPropagation() {
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

void Conv::bwdPropagation() {
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

    if (!d_dx) {
        log_->trace("{} bwdPropagation shortcut as no upstream", name_);
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

void Conv::updateWeights() {
    NetworkConstPtr nn = network_.lock();
    assert(("Network is expired", nn));

    const SolverSetting setting = nn->getSolverSetting();
    const float learning_rate   = -setting.learning_rate;

    cublasHandle_t cublas_handle = nn->getCublasHandle();

    checkCudaErrors(cublasSaxpy(cublas_handle,
                                static_cast<int>(getFilterSize()),
                                &learning_rate,
                                d_dfilter_,
                                1,
                                d_filter_,
                                1));
    checkCudaErrors(cublasSaxpy(cublas_handle,
                                static_cast<int>(getBiasSize()),
                                &learning_rate,
                                d_dbias_,
                                1,
                                d_bias_,
                                1));
}

size_t Conv::getBiasSize() const {
    assert(c_ != 0);
    return c_;
}

size_t Conv::getBiasSizeInBytes() const {
    return sizeof(float) * getBiasSize();
}

size_t Conv::getFilterSize() const {
    assert(input_channel_ != 0);
    return kernel_.height * kernel_.width * kernel_.channel * input_channel_;
}

size_t Conv::getFilterSizeInBytes() const {
    return sizeof(float) * getFilterSize();
}
}  // namespace layers
}  // namespace nn
