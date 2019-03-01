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
#pragma once
namespace nn {
namespace layers {
/** @struct Window
 * Pooling window
 */
struct Window {
    int height;
    int width;
};

/** @struct Pad
 * Horizontal and vertical direction padding.
 */
struct Pad {
    int horizontal;
    int vertical;
};

/** @struct Stride
 * Horizontal and vertical stride.
 */
struct Stride {
    int vertical;
    int horizontal;
};

/** @struct Kernel
 * kernel size and kernel number
 */
struct Kernel {
    int height;
    int width;
    int channel;
};

/** @struct Dilation
 * Filter height and width dilation
 */
struct Dilation {
    int height;
    int width;
};

}  // namespace layers
}  // namespace nn
