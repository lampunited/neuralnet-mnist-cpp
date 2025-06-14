#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>
#include <string>
#include "layer.hpp"

class NeuralNetwork {
public:
    NeuralNetwork() = default;
    ~NeuralNetwork();

    // Add a layer (caller is responsible for `new`-ing it)
    void add_layer(Layer* layer);

    // Forward pass (returns probabilities from Softmax)
    std::vector<double> predict(const std::vector<double>& input);

    // Train on (inputs, one-hot targets) using SGD + cross-entropy
    void train(const std::vector<std::vector<double>>& inputs,
               const std::vector<std::vector<double>>& targets,
               int epochs, double learning_rate);

    void save_weights(const std::string& filepath) const;
    void load_weights(const std::string& filepath);

private:
    std::vector<Layer*> layers_;
};

#endif // NEURAL_NETWORK_HPP
