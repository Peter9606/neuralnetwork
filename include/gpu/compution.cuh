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
