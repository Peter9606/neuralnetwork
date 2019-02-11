#include <cassert>

#include "error_check.h"
#include "layers/fc.h"

namespace nn
{
namespace layers
{
FC::FC(const std::string& name,
       const NetworkConstPtr& network,
       const LayerConstPtr& up,
       int output_length)
    : Layer(name, network, up)
    , input_length_(up->getDim().h * up->getDim().w * up->getDim().c)
    , output_length_(output_length)
{
    c_ = output_length_;
    h_ = 1;
    w_ = 1;
}

FC::~FC()
{
    checkCUDNN(cudnnDestroyTensorDescriptor(y_desc_));

    checkCudaErrors(cudaFree(d_weight_));
    checkCudaErrors(cudaFree(d_bias_));
    checkCudaErrors(cudaFree(d_y_));
    checkCudaErrors(cudaFree(d_one_vector_));
    checkCudaErrors(cudaFree(d_dbias_));
    checkCudaErrors(cudaFree(d_dweight_));
    checkCudaErrors(cudaFree(d_dy_));
}

size_t FC::prepareFwdPropagation()
{
    checkCUDNN(cudnnCreateTensorDescriptor(&y_desc_));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        y_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n_, c_, h_, w_));
    const size_t tensor_size = sizeof(float) * n_ * c_ * h_ * w_;
    const size_t weight_size = sizeof(float) * input_length_ * output_length_;
    const size_t bias_size   = sizeof(float) * output_length_;
    const size_t vector_size = sizeof(float) * n_;
    const size_t total = tensor_size + weight_size + bias_size + vector_size;

    checkCudaErrors(cudaMalloc(&d_y_, tensor_size));
    checkCudaErrors(cudaMalloc(&d_weight_, weight_size));
    checkCudaErrors(cudaMalloc(&d_bias_, bias_size));
    checkCudaErrors(cudaMalloc(&d_one_vector_, vector_size));

    // create a one-vector on device
    // TODO(Peter Han): No need do it multiple times for one Network instance,
    // as it only varies for different batch size. So move it to Network class
    // TODO(Peter Han): do it in GPU
    std::vector<float> h_one_vector(n_);
    for (auto&& i : h_one_vector)
    {
        i = 1;
    }
    // FIXME(Peter Han): sync or async version, which one?
    checkCudaErrors(cudaMemcpy(d_one_vector_,
                               h_one_vector.data(),
                               sizeof(float) * n_,
                               cudaMemcpyHostToDevice));

    return total;
}

size_t FC::prepareBwdPropagation()
{
    const size_t tensor_size = sizeof(float) * n_ * c_ * h_ * w_;
    const size_t weight_size = sizeof(float) * input_length_ * output_length_;
    const size_t bias_size   = sizeof(float) * output_length_;
    const size_t total       = tensor_size + weight_size + bias_size;

    checkCudaErrors(cudaMalloc(&d_dy_, tensor_size));
    checkCudaErrors(cudaMalloc(&d_dweight_, weight_size));
    checkCudaErrors(cudaMalloc(&d_dbias_, bias_size));

    return total;
}

void FC::loadParameters(const shared_ptr<vector<float>>& h_params)
{
    // before loading parameters, all pointers on device should be null,
    // otherwise memory leak on device, which also is NOT the scenario
    assert(!d_bias_);
    assert(!d_weight_);

    // only one bias corresponding to one filter
    /*
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
    */
}

shared_ptr<vector<float>> FC::saveParameters()
{
    assert(d_weight_);
    assert(d_bias_);
}

void FC::fwdPropagation()
{
    NetworkConstPtr nn = network_.lock();
    assert(("Network is expired", nn));
    LayerConstPtr up = up_.lock();
    assert(("Upstream is expired", up));

    cublasHandle_t cublas_handle = nn->getCublasHandle();
    const float* alpha           = nn->getAlpha();
    const float* beta            = nn->getBeta();
    float* d_x                   = up->getTensor();

    // d_y_(output_length_ * batch_size) = d_weight_.T (output_length_ *
    // input_length_) * d_src_tensor (input_length_ * batch_size)
    checkCudaErrors(cublasSgemm(cublas_handle,
                                CUBLAS_OP_T,
                                CUBLAS_OP_N,
                                output_length_,
                                n_,
                                input_length_,
                                alpha,
                                d_weight_,
                                input_length_,
                                d_x,
                                input_length_,
                                beta,
                                d_y_,
                                output_length_));

    // d_y_ += d_bias_ * d_one_vector_.T
    checkCudaErrors(cublasSgemm(cublas_handle,
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                output_length_,
                                n_,
                                1,
                                alpha,
                                d_bias_,
                                output_length_,
                                d_one_vector_,
                                1,
                                alpha,
                                d_y_,
                                output_length_));
}

void FC::bwdPropagation()
{
    /***
        NetworkConstPtr nn = network_.lock();
        assert(("Network is expired", nn));
        LayerConstPtr up = up_.lock();
        assert(("Up stream is expired", up));
        LayerConstPtr down = down_.lock();
        assert(("Down stream is expired", down));

        cublasHandle_t cublas_handle = nn->getCublasHandle();
        const float* alpha           = nn->getAlpha();
        const float* beta            = nn->getBeta();
        float* d_x                   = up->getTensor();
        float* d_downstream_gradient = down->getGradient();

        // compute derivative w.r.t. weights
        checkCudaErrors(cublasSgemm(cublas_handle,
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_T,
                                    input_length_,
                                    output_length_,
                                    n_,
                                    alpha,
                                    d_x,
                                    input_length_,
                                    d_downstream_gradient,
                                    output_length_,
                                    beta,
                                    d_dweight_,
                                    input_length_));

        // compute derivative w.r.t. bias
        checkCudaErrors(cublasSgemv(cublas_handle,
                                    CUBLAS_OP_N,
                                    output_length_,
                                    n_,
                                    alpha,
                                    d_downstream_gradient,
                                    output_length_,
                                    d_one_vector_,
                                    1,
                                    beta,
                                    d_dbias_,
                                    1));

        // compute derivative w.r.t. data (for upstream layer)
        checkCudaErrors(cublasSgemm(cublas_handle,
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    input_length_,
                                    n_,
                                    output_length_,
                                    alpha,
                                    d_weight_,
                                    input_length_,
                                    d_downstream_gradient,
                                    output_length_,
                                    beta,
                                    d_dy_,
                                    input_length_));
    */
}

void FC::updateWeights()
{
}

cudnnTensorDescriptor_t FC::getDescriptor() const
{
    return y_desc_;
}

float* FC::getTensor() const
{
    return d_y_;
}
}  // namespace layers
}  // namespace nn
