// include/neural_network.hpp
#pragma once
#include "layer.hpp"
#include <vector>
#include <string>

class NeuralNetwork {
public:
    ~NeuralNetwork();
    const std::vector<Layer*>& get_layers() const { return layers_; }
    void add_layer(Layer* layer);

    std::vector<double> predict(const std::vector<double>& input);

    void train(const std::vector<std::vector<double>>& inputs,
               const std::vector<std::vector<double>>& targets,
               int epochs,
               double learning_rate,
               int batch_size = 1);

    void save_weights(const std::string& filepath) const;
    void load_weights(const std::string& filepath);

private:
    std::vector<Layer*> layers_;
};
