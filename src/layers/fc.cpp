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
#include "neuralnetwork/layers/fc.h"

namespace nn {
namespace layers {
FC::FC(const std::string& name,
       const NetworkConstPtr& network,
       const LayerConstPtr& up,
       int output_length)
    : Layer(name, network, up)
    , input_length_(up->getDim().h * up->getDim().c * up->getDim().w)
    , output_length_(output_length) {
    c_ = output_length_;
    h_ = 1;
    w_ = 1;

    checkCUDNN(cudnnCreateTensorDescriptor(&y_desc_));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        y_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n_, c_, h_, w_));
}

FC::~FC() {
    checkCUDNN(cudnnDestroyTensorDescriptor(y_desc_));

    checkCudaErrors(cudaFree(d_weight_));
    checkCudaErrors(cudaFree(d_bias_));
    checkCudaErrors(cudaFree(d_y_));
    checkCudaErrors(cudaFree(d_dbias_));
    checkCudaErrors(cudaFree(d_dweight_));
    checkCudaErrors(cudaFree(d_dy_));
    checkCudaErrors(cudaFree(d_one_vector_));
}

size_t FC::prepareFwdPropagation() {
    const size_t vector_size = sizeof(float) * n_;

    checkCudaErrors(cudaMalloc(&d_y_, getTensorSizeInBytes()));
    checkCudaErrors(cudaMalloc(&d_weight_, getWeightSizeInBytes()));
    checkCudaErrors(cudaMalloc(&d_bias_, getBiasSizeInBytes()));
    checkCudaErrors(cudaMalloc(&d_one_vector_, vector_size));

    // create a one-vector on device
    // TODO(Peter Han): No need do it multiple times for one Network
    // instance, as it only varies for different batch size. So move it to
    // Network class
    // TODO(Peter Han): do it in GPU
    std::vector<float> h_one_vector(n_);
    for (auto&& i : h_one_vector) {
        i = 1;
    }
    // FIXME(Peter Han): sync or async version, which one?
    checkCudaErrors(cudaMemcpyAsync(d_one_vector_,
                                    h_one_vector.data(),
                                    sizeof(float) * n_,
                                    cudaMemcpyHostToDevice));

    return getTensorSizeInBytes() + getWeightSizeInBytes() +
           getBiasSizeInBytes() + vector_size;
}

size_t FC::prepareBwdPropagation() {
    checkCudaErrors(cudaMalloc(&d_dy_, getTensorSizeInBytes()));
    checkCudaErrors(cudaMalloc(&d_dweight_, getWeightSizeInBytes()));
    checkCudaErrors(cudaMalloc(&d_dbias_, getBiasSizeInBytes()));

    return getTensorSizeInBytes() + getWeightSizeInBytes() +
           getBiasSizeInBytes();
}

void FC::loadParameters(const shared_ptr<vector<float>>& h_params) {
    // before loading parameters, all pointers on device should be null,
    // otherwise memory leak on device, which also is NOT the scenario

    assert(d_bias_);
    assert(d_weight_);

    // only one bias corresponding to one filter
    /*
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

    std::random_device rd;
    std::mt19937 gen(rd());
    float w = sqrt(3.0f / (input_length_ * output_length_));
    std::uniform_real_distribution<> dist(-w, w);
    {
        std::vector<float> weight(getWeightSize());
        for (auto&& ite : weight) {
            ite = static_cast<float>(dist(gen));
        }
        cudaMemcpyAsync(d_weight_,
                        &weight[0],
                        getWeightSizeInBytes(),
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

shared_ptr<vector<float>> FC::saveParameters() {
    assert(d_weight_);
    assert(d_bias_);
}

void FC::fwdPropagation() {
    NetworkConstPtr nn = network_.lock();
    assert(("Network is expired", nn));
    LayerConstPtr up = up_.lock();
    assert(("Upstream is expired", up));

    cublasHandle_t cublas_handle = nn->getCublasHandle();
    const float* alpha           = nn->getAlpha();
    const float* beta            = nn->getBeta();
    float* d_x                   = up->getTensor();

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

void FC::bwdPropagation() {
    NetworkConstPtr nn = network_.lock();
    assert(("Network is expired", nn));
    LayerConstPtr up = up_.lock();
    assert(("Up stream is expired", up));

    cublasHandle_t cublas_handle = nn->getCublasHandle();
    const float* alpha           = nn->getAlpha();
    const float* beta            = nn->getBeta();
    float* d_x                   = up->getTensor();
    float* d_dx                  = up->getGradient();

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
                                d_dy_,
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
                                d_dy_,
                                output_length_,
                                d_one_vector_,
                                1,
                                beta,
                                d_dbias_,
                                1));

    if (!d_dx) {
        log_->trace("{} bwdPropagation shortcut as no upstream", name_);
        return;
    }
    // compute derivative w.r.t. data
    checkCudaErrors(cublasSgemm(cublas_handle,
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                input_length_,
                                n_,
                                output_length_,
                                alpha,
                                d_weight_,
                                input_length_,
                                d_dy_,
                                output_length_,
                                beta,
                                d_dx,
                                input_length_));
}

void FC::updateWeights() {
    NetworkConstPtr nn = network_.lock();
    assert(("Network is expired", nn));
    LayerConstPtr up = up_.lock();
    assert(("Up stream is expired", up));

    cublasHandle_t cublas_handle = nn->getCublasHandle();
    const float learning_rate    = -nn->getSolverSetting().learning_rate;

    checkCudaErrors(cublasSaxpy(cublas_handle,
                                static_cast<int>(getWeightSize()),
                                &learning_rate,
                                d_dweight_,
                                1,
                                d_weight_,
                                1));
    checkCudaErrors(cublasSaxpy(cublas_handle,
                                static_cast<int>(getBiasSize()),
                                &learning_rate,
                                d_dbias_,
                                1,
                                d_bias_,
                                1));
}

size_t FC::getBiasSize() const {
    assert(output_length_);
    return output_length_;
}

size_t FC::getBiasSizeInBytes() const {
    return sizeof(float) * getBiasSize();
}

size_t FC::getWeightSize() const {
    return input_length_ * output_length_;
}

size_t FC::getWeightSizeInBytes() const {
    return sizeof(float) * getWeightSize();
}
}  // namespace layers
}  // namespace nn
