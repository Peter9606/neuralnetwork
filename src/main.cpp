#include <iostream>
#include <memory>
#include <vector>

#include "network_impl.h"
#include "readubyte.h"

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
    int batch_size                      = 48;
    const char* const train_images_path = "train-images-idx3-ubyte";
    const char* const train_labels_path = "train-labels-idx1-ubyte";

    std::cout << "Reading input data" << std::endl;

    // Read dataset sizes
    size_t train_size = ReadUByteDataset(
        train_images_path, train_labels_path, nullptr, nullptr, width, height);

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
                         width,
                         height) != train_size) {
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
        (*train_images_float)[i] = (float)train_images[i] / 255.0f;

    for (size_t i = 0; i < train_size; ++i)
        (*train_labels_float)[i] = (float)train_labels[i];

    shared_ptr<NetworkImpl> lenet = make_shared<NetworkImpl>(batch_size);
    lenet->buildNetwork();
    lenet->train(train_images_float, train_labels_float);
    return 0;
} catch (const std::runtime_error& err) {
    std::cout << "runtime error happened" << std::endl;
}
