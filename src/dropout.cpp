#include "dropout.hpp"

Dropout::Dropout(double p)
  : p_(p),
    training_(true),
    rng_(std::random_device{}()),
    dist_(1.0 - p)   
{}

std::vector<double> Dropout::forward(const std::vector<double>& input) {
    size_t N = input.size();
    mask_.resize(N);
    std::vector<double> out(N);

    if (training_) {
        for (size_t i = 0; i < N; ++i) {
            mask_[i] = dist_(rng_) ? 1.0 : 0.0;
            out[i] = input[i] * mask_[i];
        }
    } else {
        double keep = 1.0 - p_;
        for (size_t i = 0; i < N; ++i) {
            out[i] = input[i] * keep;
        }
    }
    return out;
}

std::vector<double> Dropout::backward(const std::vector<double>& grad_output) {
    size_t N = grad_output.size();
    std::vector<double> grad_input(N);
    if (training_) {
        for (size_t i = 0; i < N; ++i) {
            grad_input[i] = grad_output[i] * mask_[i];
        }
    } else {
        double keep = 1.0 - p_;
        for (size_t i = 0; i < N; ++i) {
            grad_input[i] = grad_output[i] * keep;
        }
    }
    return grad_input;
}

void Dropout::set_training(bool train) {
    training_ = train;
}
