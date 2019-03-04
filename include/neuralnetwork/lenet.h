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
#include <memory>
#include <vector>

// self
#include "neuralnetwork/network_impl.h"

using std::shared_ptr;
using std::vector;

namespace nn {

class LeNet : public NetworkImpl {
 public:
    /**
     * Constructor build a Network object
     *
     * @param[in] batch_size  batch size
     */
    explicit LeNet(int batch_size);

    /**
     * destructor
     */
    ~LeNet();

    /**
     * train nn
     *
     * @param[in] h_data 	data vector located in host
     * @param[in] h_label 	label vector located in host
     */
    void train(const shared_ptr<vector<float>>& h_data,
               const shared_ptr<vector<float>>& h_label) const;

    /**
     * @brief construct network topology
     */
    virtual void buildNetwork();

 protected:
    /**
     * @brief compute loss
     * Concrate network should have have implementation
     *
     * @param[in]       d_label     label data in device
     */
    virtual void computeLoss(const float* d_label) const;
};
}  // namespace nn
