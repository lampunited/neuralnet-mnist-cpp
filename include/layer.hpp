#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>

// Abstract base class for any layer
class Layer {
public:
    virtual ~Layer() = default;

    // Forward: given input vector, produce output vector
    virtual std::vector<double> forward(const std::vector<double>& input) = 0;

    // Backward: given gradient w.r.t. this layer's output,
    // return gradient w.r.t. this layer's input
    virtual std::vector<double> backward(const std::vector<double>& grad_output) = 0;

    // Update weights using the stored gradients and the learning rate
    virtual void update_weights(double learning_rate) = 0;
};

#endif // LAYER_HPP
