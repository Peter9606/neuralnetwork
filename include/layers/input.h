#pragma once
// self
#include "layer.h"

namespace nn
{
namespace layers
{
/** @class Input
 * @brief A layer take the input data, sepcificly image in CNN.
 *
 * As all other types of layer, Input layer only handle data which already
 * exists in device. So it's the client's responsibility to move image data from
 * host to device, and then call pass to Input layer.
 */
class Input : public Layer
{
public:
    /**
     * Constructor of Input layer
     *
     * @param[in] name      layer name
     * @param[in] network   Network interface handle
     * @param[in] height    height of input image
     * @param[in] width     width of input image
     * @param[in] channel   channel of input image
     */
    Input(const std::string& name,
          const NetworkConstPtr& network,
          int height,
          int width,
          int channel);

    virtual ~Input() = default;

    /**
     * Pass image data into network via Input layer
     *
     * @param[in] data      data of batch images
     */
    void inputData(float* data);

private:
    const int height_;
    const int width_;
    const int channel_;
    const LayerConstPtr upstream_;
};

}  // namespace layers
}  // namespace nn
