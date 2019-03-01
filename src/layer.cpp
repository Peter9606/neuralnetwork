// self
#include "error_check.h"
#include "layer.h"

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
    Dim d = {c_, h_, w_};
    return d;
}

float* Layer::getGradient() const {
    return nullptr;
}

}  // namespace nn
