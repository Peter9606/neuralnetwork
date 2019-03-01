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

/**
 * Computes the backpropagation results of the Softmax loss for each result in a
 * batch. Uses the softmax values obtained from forward propagation to compute
 * the difference.
 *
 * @param[in]   label       The training batch label values.
 * @param[in]   num_labels  The number of possible labels.
 * @param[in]   batch_size  The size of the trained batch.
 * @param[out]  diff        The resulting gradient.
 */
void calculateLossWithGpu(const float* d_label,
                          int label_size,
                          int batch_size,
                          float* d_loss);
