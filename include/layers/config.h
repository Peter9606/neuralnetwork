#pragma once

namespace nn
{
namespace layers
{
/** @struct Window
 * Pooling window
 */
struct Window
{
    int height;
    int width;
};

/** @struct Pad
 * Horizontal and vertical direction padding.
 */
struct Pad
{
    int horizontal;
    int vertical;
};

/** @struct Stride
 * Horizontal and vertical stride.
 */
struct Stride
{
    int vertical;
    int horizontal;
};

/** @struct Kernel
 * kernel size and kernel number
 */
struct Kernel
{
    int height;
    int width;
    int channel;
};

/** @struct Dilation
 * Filter height and width dilation
 */
struct Dilation
{
    int height;
    int width;
};

}  // namespace layers
}  // namespace nn
