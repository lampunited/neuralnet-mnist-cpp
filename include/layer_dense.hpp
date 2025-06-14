#ifndef LAYER_DENSE_HPP
#define LAYER_DENSE_HPP

#include "layer.hpp"
#include <vector>

class DenseLayer : public Layer {
public:
    DenseLayer(int input_size, int output_size);
    ~DenseLayer();

    // Forward pass: z = W * x + b
    std::vector<double> forward(const std::vector<double>& input) override;

    // Backward pass: compute gradients, return grad w.r.t. input
    std::vector<double> backward(const std::vector<double>& grad_output) override;

    // Update weights & biases using stored gradients
    void update_weights(double learning_rate) override;

    void get_weights(std::vector<std::vector<double>>& W,
                     std::vector<double>& b) const {
        W = weights_;
        b = biases_;
    }

    void set_weights(const std::vector<std::vector<double>>& W,
                     const std::vector<double>& b) {
        weights_ = W;
        biases_  = b;
    }

private:
    int input_size_;
    int output_size_;

    // Parameters
    std::vector<std::vector<double>> weights_;  // [output_size_][input_size_]
    std::vector<double> biases_;                // [output_size_]

    // Caches for backprop
    std::vector<double> input_cache_;                 // [input_size_]
    std::vector<std::vector<double>> grad_weights_;   // same shape as weights_
    std::vector<double> grad_biases_;                 // [output_size_]
    std::vector<double> grad_input_;                  // [input_size_]
};

#endif // LAYER_DENSE_HPP
