#include "gpu/compution.cuh"

#define BW 512

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
__global__ void SoftmaxLoss(const float *label,
                            int num_labels,
                            int batch_size,
                            float *diff)  // or also Cross-Entropy
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) {
        return;
    }
    const int label_value = static_cast<int>(label[idx]);
    // For each item in the batch, decrease the result of the label's value by 1
    diff[idx * num_labels + label_value] -= 1.0f;
}

/**
 * Computes ceil(x / y) for integral nonnegative values.
 */
static unsigned int RoundUp(unsigned int nominator, unsigned int denominator) {
    return (nominator + denominator - 1) / denominator;
}

void calculateLossWithGpu(const float *d_label,
                          int label_size,
                          int batch_size,
                          float *d_loss) {
    SoftmaxLoss<<<RoundUp(batch_size, BW), BW>>>(
        d_label, label_size, batch_size, d_loss);
}
