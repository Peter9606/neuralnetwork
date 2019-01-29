#include <cassert>

#include "error_check.h"
#include "layers/conv.h"

namespace nn
{
namespace layers
{
Conv::Conv(const std::string& name,
           const NetworkConstPtr& network,
           const LayerConstPtrVec& upstreams,
           const Pad& pad,
           const Stride& stride,
           const Kernel& kernel,
           const Dilation& dilation)
    : Layer(name, Layer::CONV, network, upstreams)
    , pad_(pad)
    , stride_(stride)
    , kernel_(kernel)
    , dilation_(dilation)
    , upstream_(upstreams[0])
{
}

Conv::~Conv()
{
    destroyConvolutionTensor();
    destroyFilterTensor();
    destroyBiasTensor();
}

size_t Conv::constructConvlutionTensor()
{
    // convolution descriptor
    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc_));
    checkCUDNN(cudnnSetConvolution2dDescriptor(conv_desc_,
                                               pad_.height,
                                               pad_.width,
                                               stride_.horizontal,
                                               stride_.vertical,
                                               dilation_.height,
                                               dilation_.width,
                                               CUDNN_CROSS_CORRELATION,
                                               CUDNN_DATA_FLOAT));
}

void Conv::destroyConvolutionTensor()
{
    checkCUDNN(cudnnDestroyConvolutionDescriptor(conv_desc_));
}

size_t Conv::constructFilterTensor()
{
    size_t total_space  = 0;
    int input_channel   = upstream_->getOutputTensorDim().z;
    size_t filter_space = sizeof(float) * input_channel * kernel_.height *
                          kernel_.width * kernel_.channel;

    checkCUDNN(cudnnCreateFilterDescriptor(&filter_desc_));
    checkCUDNN(cudnnSetFilter4dDescriptor(filter_desc_,
                                          CUDNN_DATA_FLOAT,
                                          CUDNN_TENSOR_NCHW,
                                          kernel_.channel,
                                          input_channel,
                                          kernel_.height,
                                          kernel_.width));
    total_space += filter_sapce;
    checkCudaErrors(cudaMalloc(&filter_, filter_space));

    if (network_->isInferenceOnly())
    {
        return total;
    }

    checkCudaErrors(cudaMalloc(&filter_gradient_, filter_space));
    total_space += filter_sapce;
    return total_space;
}

void Conv::destroyFilterTensor()
{
    checkCUDNN(cudnnDestroyFilterDescriptor(filter_desc_));
    checkCudaErrors(cudaFree(filter_));
    filter_ = nullptr;
    checkCudaErrors(cudaFree(filter_gradient_));
    filter_gradient = nullptr;
}

size_t Conv::constructBiasTensor()
{
    size_t total_space = 0;
    checkCUDNN(cudnnCreateTensorDescriptor(&bias_desc_));
    checkCUDNN(cudnnSetTensor4dDescriptor(bias_desc_,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          1,
                                          kernel_.channel,
                                          1,
                                          1));
    size_t bias_space = sizeof(float) * kernel_.channel;
    checkCudaErrors(cudaMalloc(&bias_, bias_space));
    total_space += bias_sapce;
    if (network_->isInferenceOnly())
    {
        checkCudaErrors(cudaMalloc(&bias_gradient_, bias_space));
        total_space += bias_sapce;
    }
    return total_space;
}

void Conv::destroyBiasTensor()
{
    checkCUDNN(cudnnDestroyTensorDescriptor(bias_desc_));
    checkCudaErrors(cudaFree(bias_));
    bias_ = nullptr;
    checkCudaErrors(cudaFree(bias_gradient_));
    bias_gradient_ = nullptr;
}

size_t Conv::constructSolver()
{
    size_t total_space = 0;

    // solver part
    const SolverSetting solver_setting = network_->getSolverSetting();
    const SolverType solver_type       = solver_setting.type;
    if (solver_type == ADAM)
    {
        size_t space = sizeof(float) * kernel_.channels;
        checkCudaErrors(cudaMalloc(&adam.d_m_b, space));
        checkCudaErrors(cudaMemset(adam.d_m_b, 0, space));
        checkCudaErrors(cudaMalloc(&adam.d_v_b, space));
        checkCudaErrors(cudaMemset(adam.d_v_b, 0, space));
        checkCudaErrors(cudaMalloc(&adam.tmp_b, space));

        total_space += space * 5;
    }
    else if (solver_type == mSGD)
    {
        size_t space = sizeof(float) * kernel_.channels;
        checkCudaErrors(cudaMalloc(&msgd.d_v_b, space));

        total_space += space;
    }

    // initialize parameters
    // TODO(Peter Han): move to other dedicate method
    if (solver_type != SolverType::InfOnly)
    {
        size_t input_channel = upstream_.getOutputDim().z;
        size_t len =
            kernel_.height * kernel_.width * kernel_.output * input_channel;
        std::vector<float> h_filter(len);
        std::vector<float> h_bias(len);

        std::random_device rd;
        std::mt19937 gen(rd());
        float w = 1.0f / (kernel_.height * kernel_.width * input_channel);
        std::uniform_real_distribution<> d(-w, w);
        for (auto&& iter : h_filter)
            iter = static_cast<float>(d(gen));
        for (auto&& iter : h_bias)
            iter = 0.0f;
        checkCudaErrors(cudaMemcpy(filter_,
                                   h_filter.data(),
                                   sizeof(float) * h_filter.size(),
                                   cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(bias_,
                                   h_bias.data(),
                                   sizeof(float) * h_bias.size(),
                                   cudaMemcpyHostToDevice));
    }

    return total_space;
}

void Conv::destroySolver()
{
    checkCudaErrors(cudaFree(adam.d_m_b));
    checkCudaErrors(cudaFree(adam.d_v_b));
    checkCudaErrors(cudaFree(adam.tmp_b));
    checkCudaErrors(cudaFree(msgd.d_v_b));
}

// TODO(Peter Han): terrible function name, should refact
size_t Conv::dealWithWorkspace()
{
    int output_batch_size;
    int output_channel;
    int output_height;
    int output_width;
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
        desc, src->tensor, filterDesc, &output_height, &output_width));

    size_t image_bytes = output_batch_size * output_channels * output_height *
                         output_width * sizeof(float);

    assert(network_->getBatchSize() == output_batch_size);
    assert(output_channel == kernel_.channel);

    out_height = oheight;
    out_width  = owidth;

    // WTF!
    define_tensor(nn);

    bool with_limit = true;
    cudnnConvolutionFwdPreference_t fwdPref;
    cudnnConvolutionBwdDataPreference_t bwdDPref;
    cudnnConvolutionBwdFilterPreference_t bwdFPref;
    size_t memory_limit;
    if (with_limit)
    {
        memory_limit = 1073741824 * 1.5;
        fwdPref      = CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT;
        bwdDPref     = CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT;
        bwdFPref     = CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT;
    }
    else
    {
        memory_limit = 0;
        fwdPref      = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
        bwdDPref     = CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST;
        bwdFPref     = CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST;
    }
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(nn->cudnnHandle,
                                                   src->tensor,
                                                   filterDesc,
                                                   desc,
                                                   tensor,
                                                   fwdPref,
                                                   memory_limit,
                                                   &fw_algo));
    size_t workspace_bytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(nn->cudnnHandle,
                                                       src->tensor,
                                                       filterDesc,
                                                       desc,
                                                       tensor,
                                                       fw_algo,
                                                       &workspace_bytes));
    if (workspace_bytes > nn->m_workspaceSize)
        nn->m_workspaceSize = workspace_bytes;
    if (nn->update_type != ONLY_INFERENCE)
    {
        checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(nn->cudnnHandle,
                                                              src->tensor,
                                                              tensor,
                                                              desc,
                                                              filterDesc,
                                                              bwdFPref,
                                                              memory_limit,
                                                              &bwF_algo));
        checkCUDNN(
            cudnnGetConvolutionBackwardFilterWorkspaceSize(nn->cudnnHandle,
                                                           src->tensor,
                                                           tensor,
                                                           desc,
                                                           filterDesc,
                                                           bwF_algo,
                                                           &workspace_bytes));
        nn->m_workspaceSize = std::max(nn->m_workspaceSize, workspace_bytes);
        checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(nn->cudnnHandle,
                                                            filterDesc,
                                                            tensor,
                                                            desc,
                                                            src->tensor,
                                                            bwdDPref,
                                                            memory_limit,
                                                            &bwD_algo));
        checkCUDNN(
            cudnnGetConvolutionBackwardDataWorkspaceSize(nn->cudnnHandle,
                                                         filterDesc,
                                                         tensor,
                                                         desc,
                                                         src->tensor,
                                                         bwD_algo,
                                                         &workspace_bytes));
        nn->m_workspaceSize = std::max(nn->m_workspaceSize, workspace_bytes);
    }
}

void Conv::setFwdPropagation()
{
}

}  // namespace layers
}  // namespace nn
