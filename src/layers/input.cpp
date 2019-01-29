// self
#include "error_check.h"
#include "layers/input.h"

namespace nn
{
namespace layers
{
Input::Input(const std::string& name,
             const NetworkConstPtr& network,
             int width,
             int height,
             int channel)
    : Layer(name, Layer::INPUT, network, {})
    , height_(height)
    , width_(width)
    , channel_(channel)
{
}

void Input::inputData(float* data)
{
    const size_t len =
        network_->getBatchSize() * channel_ * height_ * width_ * sizeof(float);
    checkCudaErrors(cudaMemcpy(data, tensor_, len, cudaMemcpyDeviceToDevice));
}

}  // namespace layers
}  // namespace nn
