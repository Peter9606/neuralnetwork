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

#include "neuralnetwork/network_impl.h"
#include "neuralnetwork/readubyte.h"

using nn::Network;
using nn::NetworkImpl;
using std::cout;
using std::endl;
using std::make_shared;
using std::shared_ptr;
using std::vector;

int main() try {
    int iterations                      = 100;
    size_t width                        = 1;
    size_t height                       = 1;
    size_t channels                     = 1;
    int batch_size                      = 128;
    const char* const train_images_path = "train-images-idx3-ubyte";
    const char* const train_labels_path = "train-labels-idx1-ubyte";

    std::cout << "Reading input data" << std::endl;

    // Read dataset sizes
    size_t train_size = ReadUByteDataset(train_images_path,
                                         train_labels_path,
                                         nullptr,
                                         nullptr,
                                         &width,
                                         &height);

    if (train_size == 0) {
        return 1;
    }

    vector<uint8_t> train_images(train_size * width * height * channels);
    vector<uint8_t> train_labels(train_size);

    // Read data from datasets
    if (ReadUByteDataset(train_images_path,
                         train_labels_path,
                         &train_images[0],
                         &train_labels[0],
                         &width,
                         &height) != train_size) {
        return 2;
    }

    cout << "Done. Training dataset size: " << train_size << endl;

    cout << "Batch size :" << batch_size << " iterations: " << iterations
         << endl;

    // Normalize training set to be in [0,1]
    shared_ptr<vector<float>> train_images_float =
        make_shared<vector<float>>(train_images.size());
    shared_ptr<vector<float>> train_labels_float =
        make_shared<vector<float>>(train_size);

    for (size_t i = 0; i < train_size * channels * width * height; ++i)
        (*train_images_float)[i] = static_cast<float>(train_images[i] / 255.0f);

    for (size_t i = 0; i < train_size; ++i)
        (*train_labels_float)[i] = static_cast<float>(train_labels[i]);

    shared_ptr<NetworkImpl> lenet = make_shared<NetworkImpl>(batch_size);
    lenet->buildNetwork();
    lenet->train(train_images_float, train_labels_float);
    return 0;
} catch (const std::runtime_error& err) {
    std::cout << "runtime error happened" << std::endl;
}
