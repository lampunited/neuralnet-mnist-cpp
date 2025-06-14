#include "activations.hpp"
#include <algorithm>
#include <cmath>

// ----- ReLU -----
// Forward: out_i = max(0, input_i).  Store mask 1 if input>0, else 0.
std::vector<double> ReLU::forward(const std::vector<double>& input) {
    size_t N = input.size();
    mask_.resize(N);
    std::vector<double> output(N);

    for (size_t i = 0; i < N; ++i) {
        if (input[i] > 0.0) {
            mask_[i] = 1.0;
            output[i] = input[i];
        } else {
            mask_[i] = 0.0;
            output[i] = 0.0;
        }
    }
    return output;
}

// Backward: grad_input[i] = grad_output[i] * mask_[i]
std::vector<double> ReLU::backward(const std::vector<double>& grad_output) {
    size_t N = grad_output.size();
    std::vector<double> grad_input(N);
    for (size_t i = 0; i < N; ++i) {
        grad_input[i] = grad_output[i] * mask_[i];
    }
    return grad_input;
}

// ----- Softmax -----
// Forward: out_i = exp(z_i − max_z) / sum_j exp(z_j − max_z)
std::vector<double> Softmax::forward(const std::vector<double>& input) {
    double max_val = *std::max_element(input.begin(), input.end());
    size_t N = input.size();
    std::vector<double> exp_vals(N);
    double sum_exp = 0.0;

    for (size_t i = 0; i < N; ++i) {
        exp_vals[i] = std::exp(input[i] - max_val);
        sum_exp += exp_vals[i];
    }

    output_cache_.resize(N);
    for (size_t i = 0; i < N; ++i) {
        output_cache_[i] = exp_vals[i] / sum_exp;
    }
    return output_cache_;
}

// Backward: In cross-entropy + softmax, grad is (output − target), so
// we assume the upstream grad_output already equals (output - target). Just pass it through.
std::vector<double> Softmax::backward(const std::vector<double>& grad_output) {
    return grad_output; 
}

