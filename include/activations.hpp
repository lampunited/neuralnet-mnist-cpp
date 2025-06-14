#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include "layer.hpp"
#include <vector>

class ReLU : public Layer {
public:
    ReLU() = default;
    ~ReLU() override = default;

    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& grad_output) override;
    void update_weights(double) override {}  // no parameters in ReLU

private:
    std::vector<double> mask_;  // tracks which inputs > 0
};

class Softmax : public Layer {
public:
    Softmax() = default;
    ~Softmax() override = default;

    // Forward: compute probabilities
    std::vector<double> forward(const std::vector<double>& input) override;

    // Backward: for cross-entropy with Softmax, gradient is (output - target),
    // but here we simply pass through grad_output as-is because Net.train does softmax+CE in one step.
    std::vector<double> backward(const std::vector<double>& grad_output) override;
    void update_weights(double) override {}  // no parameters in Softmax

private:
    std::vector<double> output_cache_;  // stores softmax output
};

#endif // ACTIVATIONS_HPP
