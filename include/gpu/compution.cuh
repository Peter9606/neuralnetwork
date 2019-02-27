void SoftmaxLossBackprop(const float* label,
                         int num_labels,
                         int batch_size,
                         float* diff);

void calculate_loss_with_gpu(float* d_label,
                             int label_size,
                             int batch_size,
                             float* d_loss);
