#include "layer_dense.hpp"
#include <random>

// Constructor: initialize weights (Gaussian ~N(0, 0.01)) and zero biases
DenseLayer::DenseLayer(int input_size, int output_size)
    : input_size_(input_size), output_size_(output_size)
{
    std::mt19937 gen(42);
    std::normal_distribution<> dist(0.0, 1.0);

    weights_.resize(output_size_, std::vector<double>(input_size_));
    biases_.resize(output_size_);

    for (int o = 0; o < output_size_; ++o) {
        biases_[o] = 0.0;
        for (int i = 0; i < input_size_; ++i) {
            weights_[o][i] = 0.01 * dist(gen);
        }
    }
}

// Destructor: nothing to do (vectors are self-managed)
DenseLayer::~DenseLayer() {}

// Forward: z = W * x + b
std::vector<double> DenseLayer::forward(const std::vector<double>& input) {
    input_cache_ = input;  // store for backprop
    std::vector<double> output(output_size_, 0.0);

    for (int o = 0; o < output_size_; ++o) {
        double sum = biases_[o];
        for (int i = 0; i < input_size_; ++i) {
            sum += weights_[o][i] * input[i];
        }
        output[o] = sum;
    }
    return output;
}

// Backward: given grad_output (dL/dz of this layer), compute:
//   grad_input[j] = sum_o (weights_[o][j] * grad_output[o])
//   grad_weights_[o][j] = grad_output[o] * input_cache_[j]
//   grad_biases_[o] = grad_output[o]
std::vector<double> DenseLayer::backward(const std::vector<double>& grad_output) {
    // Resize gradient storages
    grad_input_.assign(input_size_, 0.0);
    grad_weights_.assign(output_size_, std::vector<double>(input_size_, 0.0));
    grad_biases_.assign(output_size_, 0.0);

    // Compute gradients
    for (int o = 0; o < output_size_; ++o) {
        grad_biases_[o] = grad_output[o];
        for (int i = 0; i < input_size_; ++i) {
            grad_weights_[o][i] = grad_output[o] * input_cache_[i];
            grad_input_[i] += weights_[o][i] * grad_output[o];
        }
    }
    return grad_input_;
}

// Update W_ij ← W_ij − lr * grad_weights_[o][i], b_o ← b_o − lr * grad_biases_[o]
void DenseLayer::update_weights(double learning_rate) {
    for (int o = 0; o < output_size_; ++o) {
        for (int i = 0; i < input_size_; ++i) {
            weights_[o][i] -= learning_rate * grad_weights_[o][i];
        }
        biases_[o] -= learning_rate * grad_biases_[o];
    }
}
