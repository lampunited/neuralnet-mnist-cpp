#include "layer_dense.hpp"
#include <random>
#include <algorithm>

DenseLayer::DenseLayer(int input_size, int output_size)
    : input_size_(input_size),
      output_size_(output_size),
      weights_(output_size, std::vector<double>(input_size)),
      biases_(output_size, 0.0),
      dW_accum_(output_size, std::vector<double>(input_size, 0.0)),
      db_accum_(output_size, 0.0)
{
    std::mt19937 gen(std::random_device{}());
    double he_std = std::sqrt(2.0 / double(input_size_));
    std::normal_distribution<> dist(0.0, he_std);

    for (int o = 0; o < output_size_; ++o) {
        for (int i = 0; i < input_size_; ++i) {
            weights_[o][i] = dist(gen);
        }
    }
}

DenseLayer::~DenseLayer() {}

std::vector<double> DenseLayer::forward(const std::vector<double>& input) {
    input_cache_ = input;
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

std::vector<double> DenseLayer::backward(const std::vector<double>& grad_output) {
    std::vector<double> grad_input(input_size_, 0.0);
    for (int o = 0; o < output_size_; ++o) {
        db_accum_[o] += grad_output[o];
        for (int i = 0; i < input_size_; ++i) {
            dW_accum_[o][i] += grad_output[o] * input_cache_[i];
            grad_input[i] += weights_[o][i] * grad_output[o];
        }
    }
    return grad_input;
}

void DenseLayer::zero_grad() {
    for (auto &row : dW_accum_) {
        std::fill(row.begin(), row.end(), 0.0);
    }
    std::fill(db_accum_.begin(), db_accum_.end(), 0.0);
}

void DenseLayer::update_weights(double learning_rate) {
    for (int o = 0; o < output_size_; ++o) {
        biases_[o] -= learning_rate * db_accum_[o];
        for (int i = 0; i < input_size_; ++i) {
            weights_[o][i] -= learning_rate * dW_accum_[o][i];
        }
    }
}