#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>

class Layer {
public:
    virtual ~Layer() = default;
    virtual std::vector<double> forward(const std::vector<double>& input) = 0;
    virtual std::vector<double> backward(const std::vector<double>& grad_output) = 0;
    virtual void update_weights(double learning_rate) = 0;
    virtual void zero_grad() = 0;
};

#endif 
