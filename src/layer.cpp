// self
#include "error_check.h"
#include "layer.h"

namespace nn
{
Layer::Layer(const std::string& name,
             const NetworkConstPtr& network,
             const LayerConstPtr& up)
    : name_(name)
    , network_(network)
    , up_(up)
    , n_(network->getBatchSize())
{
    /**
     * A principle for create and set various types of descriptor and malloc
     * memory on GPU:
     * do it in prepareFwdPropagation and prepareBwdPropagation as long
     * as possible.
     *
     * NOTE: descriptor handle and memory can be safely destroied even they are
     * not created or allocated already.
     */
}

void Layer::loadParameters(const shared_ptr<vector<float>>& h_params)
{
}

shared_ptr<vector<float>> Layer::saveParameters()
{
    return nullptr;
}

size_t Layer::prepareFwdPropagation()
{
    return 0;
}

size_t Layer::prepareBwdPropagation()
{
    return 0;
}

void Layer::updateWeights()
{
}

Dim Layer::getDim() const
{
    Dim d = {c_, h_, w_};
    return d;
}

float* Layer::getGradient() const
{
    return nullptr;
}

}  // namespace nn
