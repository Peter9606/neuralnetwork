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
#include <iostream>
#include <memory>
#include <vector>

#include "neuralnetwork/error_check.h"
#include "neuralnetwork/gpu/compution.cuh"
#include "neuralnetwork/layers/activation.h"
#include "neuralnetwork/layers/config.h"
#include "neuralnetwork/layers/conv.h"
#include "neuralnetwork/layers/dropout.h"
#include "neuralnetwork/layers/fc.h"
#include "neuralnetwork/layers/input.h"
#include "neuralnetwork/layers/pool.h"
#include "neuralnetwork/layers/softmax.h"
#include "neuralnetwork/layers/unpool.h"
#include "neuralnetwork/lenet.h"
#include "neuralnetwork/network_impl.h"
#include "neuralnetwork/readubyte.h"

using nn::layers::Activation;
using nn::layers::Conv;
using nn::layers::Dilation;
using nn::layers::Dropout;
using nn::layers::FC;
using nn::layers::Input;
using nn::layers::Kernel;
using nn::layers::Pad;
using nn::layers::Pool;
using nn::layers::Softmax;
using nn::layers::Stride;
using nn::layers::Unpool;
using nn::layers::Window;
using spdlog::logger;
using std::make_shared;
using std::shared_ptr;
using std::vector;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;

namespace nn {
LeNet::LeNet(int batch_size)
    : NetworkImpl(batch_size, {1, 28, 28}, {SGD}, CROSS_ENTROPY) {
}

LeNet::~LeNet() {
}

float LeNet::test(const shared_ptr<vector<float>> &h_data,
                  const shared_ptr<vector<float>> &h_label) const {
    const size_t size     = h_label->size();
    const size_t kTotal   = size / batch_size_;
    const size_t data_len = batch_size_ * dim_.c * dim_.h * dim_.w;
    int num_errors        = 0;
    this->inference_only_ = true;

    for (int iter = 0; iter < kTotal - 1; ++iter) {
        // Prepare current batch on device
        float *d_data = layers_[0]->getTensor();
        checkCudaErrors(cudaMemcpyAsync(d_data,
                                        &h_data->data()[iter * data_len],
                                        sizeof(float) * data_len,
                                        cudaMemcpyHostToDevice));
        checkCudaErrors(cudaDeviceSynchronize());

        fwdPropagation(d_data);

        std::vector<float> output(10 * batch_size_);
        auto last       = *(layers_.rbegin());
        float *d_output = last->getTensor();

        checkCudaErrors(cudaMemcpyAsync(&output[0],
                                        d_output,
                                        sizeof(float) * output.size(),
                                        cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaDeviceSynchronize());

        for (int i = 0; i < batch_size_; i++) {
            // Determine classification according to maximal response
            int base   = i * 10;
            int chosen = 0;

            for (int id = 0; id < 10; ++id) {
                if (output[base + chosen] < output[base + id]) {
                    chosen = id;
                }
            }
            if (chosen != h_label->at(iter * 64 + i)) {
                ++num_errors;
            }
        }
    }
    return static_cast<float>(num_errors) / (batch_size_ * (kTotal - 1));
}

void LeNet::train(const shared_ptr<vector<float>> &h_data_train,
                  const shared_ptr<vector<float>> &h_label_train,
                  const shared_ptr<vector<float>> &h_data_test,
                  const shared_ptr<vector<float>> &h_label_test) const {
    prepareTraining();
    checkCudaErrors(cudaDeviceSynchronize());
    auto t1 = high_resolution_clock::now();

    const size_t train_size = h_label_train->size();

    const int total_iter = 10000;
    float *d_label;
    checkCudaErrors(cudaMalloc(&d_label, sizeof(float) * batch_size_));
    for (int iter = 0; iter < total_iter; ++iter) {
        int imageid = iter % (train_size / batch_size_);

        // Prepare current batch on device
        float *d_data         = layers_[0]->getTensor();
        const size_t data_len = batch_size_ * dim_.c * dim_.h * dim_.w;
        checkCudaErrors(
            cudaMemcpyAsync(d_data,
                            &h_data_train->data()[imageid * data_len],
                            sizeof(float) * data_len,
                            cudaMemcpyHostToDevice));

        checkCudaErrors(
            cudaMemcpyAsync(d_label,
                            &h_label_train->data()[imageid * batch_size_],
                            sizeof(float) * batch_size_,
                            cudaMemcpyHostToDevice));

        fwdPropagation(d_data);
        bwdPropagation(d_label);
        updateWeights();
        updateLearningRate(iter);

        if (iter % 100 == 0) {
            auto err = test(h_data_test, h_label_test);
            log_->info(
                "Test error rates @ iteration {}: {}", iter, err * 100.0f);
        }
    }
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(d_label));
    auto t2 = high_resolution_clock::now();

    log_->info("Time per iteration: {}",
               duration_cast<microseconds>(t2 - t1).count() / 1000.0f /
                   total_iter);

    auto err = test(h_data_test, h_label_test);
    log_->info("Error rates@test: {}", err * 100.0f);
}

void LeNet::computeLoss(const float *d_label) const {
    const float scale      = 1.0f / static_cast<float>(batch_size_);
    shared_ptr<Layer> last = *(layers_.rbegin());
    const Dim dim          = last->getDim();
    const size_t size     = sizeof(float) * dim.c * dim.h * dim.w * batch_size_;
    const float *d_result = last->getTensor();
    float *d_loss         = last->getGradient();

    checkCudaErrors(
        cudaMemcpyAsync(d_loss, d_result, size, cudaMemcpyDeviceToDevice));

    calculateLossWithGpu(d_label, dim.c * dim.h * dim.w, batch_size_, d_loss);

    // Accounting for batch size in SGD
    checkCudaErrors(cublasSscal(cublas_handle_,
                                dim.c * dim.h * dim.w * batch_size_,
                                &scale,
                                d_loss,
                                1));
}

void LeNet::buildNetwork() {
    auto input = make_shared<Input>("Input", shared_from_this(), 1, 28, 28);
    layers_.push_back(input);

    Kernel kernel = {5, 5, 20};
    auto conv1 = make_shared<Conv>("Conv1", shared_from_this(), input, kernel);
    layers_.push_back(conv1);

    Window window = {2, 2};
    Stride stride = {2, 2};
    auto pool1    = make_shared<Pool>(
        "Pool1", shared_from_this(), conv1, window, stride, Pool::MAX);
    layers_.push_back(pool1);

    kernel     = {5, 5, 50};
    auto conv2 = make_shared<Conv>("Conv2", shared_from_this(), pool1, kernel);
    layers_.push_back(conv2);

    auto pool2 = make_shared<Pool>(
        "Pool2", shared_from_this(), conv2, window, stride, Pool::MAX);
    layers_.push_back(pool2);

    auto fc1 = make_shared<FC>("FC1", shared_from_this(), pool2, 500);
    layers_.push_back(fc1);

    /*
    auto dropout =
        make_shared<Dropout>("Dropout", shared_from_this(), fc1, 0.5);
    layers_.push_back(dropout);
    */

    auto relu1 = make_shared<Activation>("FC1Relu", shared_from_this(), fc1);
    layers_.push_back(relu1);

    auto fc2 = make_shared<FC>("FC2", shared_from_this(), relu1, 10);
    layers_.push_back(fc2);

    auto softmax = make_shared<Softmax>("Softmax", shared_from_this(), fc2);
    layers_.push_back(softmax);
}

void LeNet::updateLearningRate(int iter) const {
    static const float kBASE  = 0.01f;
    static const float kGAMMA = 0.0001f;
    static const float kPOWER = 0.75f;

    learning_rate_ =
        static_cast<float>(kBASE * pow((1.0 + kGAMMA * iter), (-kPOWER)));
}
}  // namespace nn

int main() try {
    size_t width                        = 1;
    size_t height                       = 1;
    size_t channels                     = 1;
    int batch_size                      = 64;
    const char *const train_images_path = "train-images-idx3-ubyte";
    const char *const train_labels_path = "train-labels-idx1-ubyte";
    const char *const test_images_path  = "t10k-images-idx3-ubyte";
    const char *const test_labels_path  = "t10k-labels-idx1-ubyte";

    shared_ptr<spdlog::logger> log = spdlog::default_logger();
    log->info("Reading input data");

    // Read dataset sizes
    size_t train_size = ReadUByteDataset(train_images_path,
                                         train_labels_path,
                                         nullptr,
                                         nullptr,
                                         &width,
                                         &height);
    size_t test_size  = ReadUByteDataset(
        test_images_path, test_labels_path, nullptr, nullptr, &width, &height);

    if (train_size == 0 || test_size == 0) {
        return 1;
    }

    vector<uint8_t> train_images(train_size * width * height * channels);
    vector<uint8_t> train_labels(train_size);
    vector<uint8_t> test_images(test_size * width * height * channels);
    vector<uint8_t> test_labels(test_size);

    // Read data from datasets
    if (ReadUByteDataset(train_images_path,
                         train_labels_path,
                         &train_images[0],
                         &train_labels[0],
                         &width,
                         &height) != train_size) {
        return 2;
    }
    if (ReadUByteDataset(test_images_path,
                         test_labels_path,
                         &test_images[0],
                         &test_labels[0],
                         &width,
                         &height) != test_size) {
        return 2;
    }

    log->info("Done. Training dataset size: {}, test dataset size:{}",
              train_size,
              test_size);
    log->info("Batch size {}", batch_size);

    // Normalize training set to be in [0,1]
    shared_ptr<vector<float>> train_images_float =
        make_shared<vector<float>>(train_images.size());
    shared_ptr<vector<float>> train_labels_float =
        make_shared<vector<float>>(train_size);
    shared_ptr<vector<float>> test_images_float =
        make_shared<vector<float>>(test_images.size());
    shared_ptr<vector<float>> test_labels_float =
        make_shared<vector<float>>(test_size);

    for (size_t i = 0; i < train_size * channels * width * height; ++i)
        (*train_images_float)[i] = static_cast<float>(train_images[i] / 255.0f);

    for (size_t i = 0; i < train_size; ++i)
        (*train_labels_float)[i] = static_cast<float>(train_labels[i]);

    for (size_t i = 0; i < test_size * channels * width * height; ++i)
        (*test_images_float)[i] = static_cast<float>(test_images[i] / 255.0f);

    for (size_t i = 0; i < test_size; ++i)
        (*test_labels_float)[i] = static_cast<float>(test_labels[i]);

    shared_ptr<nn::LeNet> lenet = make_shared<nn::LeNet>(batch_size);
    lenet->buildNetwork();
    lenet->train(train_images_float,
                 train_labels_float,
                 test_images_float,
                 test_labels_float);
    return 0;
} catch (const std::runtime_error &err) {
    std::cout << "runtime error happened" << std::endl;
    return 1;
}
