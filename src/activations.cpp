#include "activations.hpp"
#include <algorithm>
#include <cmath>

std::vector<double> Softmax::forward(const std::vector<double>& in) {
    double mx = *std::max_element(in.begin(), in.end());
    std::vector<double> exps(in.size());
    double sum = 0.0;
    for (size_t i = 0; i < in.size(); ++i) {
        exps[i] = std::exp(in[i] - mx);
        sum += exps[i];
    }
    for (double &e : exps) e /= sum;
    return exps;
}

std::vector<double> Softmax::backward(const std::vector<double>& grad_output) {
    return grad_output;
}
