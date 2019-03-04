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
#include "neuralnetwork/error_check.h"
#include "neuralnetwork/layers/input.h"

namespace nn {
namespace layers {
Input::Input(const std::string& name,
             const NetworkConstPtr& network,
             int c,
             int h,
             int w)
    : Layer(name, network, nullptr) {
    c_ = c;
    h_ = h;
    w_ = w;

    checkCUDNN(cudnnCreateTensorDescriptor(&y_desc_));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        y_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n_, c_, h_, w_));
}

Input::~Input() {
    checkCUDNN(cudnnDestroyTensorDescriptor(y_desc_));
    checkCudaErrors(cudaFree(d_y_));
}

size_t Input::prepareFwdPropagation() {
    checkCudaErrors(cudaMalloc(&d_y_, getTensorSizeInBytes()));

    return getTensorSizeInBytes();
}

void Input::fwdPropagation() {
    // do nothing here.
}
}  // namespace layers
}  // namespace nn
