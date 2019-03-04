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
#include "neuralnetwork/layer.h"

namespace nn {
Layer::Layer(const std::string& name,
             const NetworkConstPtr& network,
             const LayerConstPtr& up)
    : name_(name)
    , network_(network)
    , up_(up)
    , n_(network->getBatchSize()) {
    log_->info("Layer {} is being created", name_);
    /**
     * A principle for create and set various types of descriptor and malloc
     * memory on GPU:
     * 1. descriptor: MUST complete setting descriptor at constructor
     * 2. gpu memory: alloc them in prepareFwdPropagation and
     * prepareBwdPropagation
     *
     * NOTE: descriptor handle and memory can be safely destroied even they are
     * not created or allocated already.
     */
}

void Layer::loadParameters(const shared_ptr<vector<float>>& h_params) {
}

shared_ptr<vector<float>> Layer::saveParameters() {
    return nullptr;
}

size_t Layer::prepareFwdPropagation() {
    return 0;
}

size_t Layer::prepareBwdPropagation() {
    return 0;
}

void Layer::fwdPropagation() {
}

void Layer::bwdPropagation() {
}

void Layer::updateWeights() {
}

Dim Layer::getDim() const {
    assert(c_ != 0);
    assert(h_ != 0);
    assert(w_ != 0);
    Dim d = {c_, h_, w_};
    return d;
}

int Layer::getChannel() const {
    assert(c_ != 0);
    return c_;
}

size_t Layer::getTensorSize() const {
    assert(n_ != 0);
    assert(c_ != 0);
    assert(h_ != 0);
    assert(w_ != 0);
    return c_ * h_ * w_ * n_;
}

size_t Layer::getTensorSizeInBytes() const {
    return sizeof(float) * getTensorSize();
}

cudnnTensorDescriptor_t Layer::getDescriptor() const {
    return y_desc_;
}

float* Layer::getTensor() const {
    return d_y_;
}

float* Layer::getGradient() const {
    return d_dy_;
}

}  // namespace nn
