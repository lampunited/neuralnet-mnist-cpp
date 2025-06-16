#pragma once
#include "layer.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

class ReLU : public Layer {
public:
    std::vector<double> forward(const std::vector<double>& in) override {
        input_cache_ = in;
        std::vector<double> out(in.size());
        for (size_t i = 0; i < in.size(); ++i) {
            out[i] = std::max(0.0, in[i]);
        }
        return out;
    }

    std::vector<double> backward(const std::vector<double>& grad_output) override {
        std::vector<double> grad(grad_output.size());
        for (size_t i = 0; i < grad.size(); ++i) {
            grad[i] = (input_cache_[i] > 0.0 ? grad_output[i] : 0.0);
        }
        return grad;
    }

    void update_weights(double) override {}  
    void zero_grad() override {}            

private:
    std::vector<double> input_cache_;
};

class Softmax : public Layer {
public:
    std::vector<double> forward(const std::vector<double>& in) override;
    std::vector<double> backward(const std::vector<double>& grad_output) override;

    void update_weights(double) override {}
    void zero_grad() override {}
};
