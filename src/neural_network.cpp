// src/neural_network.cpp

#include "neural_network.hpp"
#include "layer_dense.hpp"

#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <iomanip> 

// Destructor: delete all layers
NeuralNetwork::~NeuralNetwork() {
    for (auto layer : layers_) {
        delete layer;
    }
}

// Add a layer to the network
void NeuralNetwork::add_layer(Layer* layer) {
    layers_.push_back(layer);
}

// Forward pass
std::vector<double> NeuralNetwork::predict(const std::vector<double>& input) {
    std::vector<double> activation = input;
    for (auto layer : layers_) {
        activation = layer->forward(activation);
    }
    return activation;
}

// Train with SGD + cross-entropy loss
void NeuralNetwork::train(const std::vector<std::vector<double>>& inputs,
                          const std::vector<std::vector<double>>& targets,
                          int epochs, double learning_rate)
{
    const size_t N = inputs.size();
    for (int e = 0; e < epochs; ++e) {
        double epoch_loss = 0.0;
        for (size_t i = 0; i < N; ++i) {
            // Forward
            auto activation = inputs[i];
            for (auto layer : layers_) {
                activation = layer->forward(activation);
            }
            // Compute loss + gradient (softmax + cross-entropy)
            std::vector<double> grad(activation.size());
            for (size_t j = 0; j < activation.size(); ++j) {
                epoch_loss += -targets[i][j] * std::log(std::max(activation[j], 1e-12));
                grad[j] = activation[j] - targets[i][j];
            }
            // Backward
            for (int li = (int)layers_.size() - 1; li >= 0; --li) {
                grad = layers_[li]->backward(grad);
            }
            // Update
            for (auto layer : layers_) {
                layer->update_weights(learning_rate);
            }
        }
        std::cout << "Epoch " << (e + 1) << "/" << epochs
                  << "   Avg. Loss = " << (epoch_loss / (double)N) << "\n";
    }
}

// Save all DenseLayer weights & biases to a binary file
void NeuralNetwork::save_weights(const std::string& filepath) const {
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs) throw std::runtime_error("Cannot open file for writing: " + filepath);

    uint32_t L = static_cast<uint32_t>(layers_.size());
    ofs.write(reinterpret_cast<char*>(&L), sizeof(L));
    std::cout << "[DEBUG] Saving weights for " << L << " layers to " << filepath << std::endl;

    for (size_t idx = 0; idx < layers_.size(); ++idx) {
        auto layer = layers_[idx];
        auto dl = dynamic_cast<DenseLayer*>(layer);
        if (dl) {
            uint8_t type = 1;
            ofs.write(reinterpret_cast<char*>(&type), sizeof(type));

            std::vector<std::vector<double>> W;
            std::vector<double> b;
            dl->get_weights(W, b);

            uint32_t out_sz = static_cast<uint32_t>(W.size());
            uint32_t in_sz  = out_sz > 0 ? static_cast<uint32_t>(W[0].size()) : 0;
            ofs.write(reinterpret_cast<char*>(&in_sz),  sizeof(in_sz));
            ofs.write(reinterpret_cast<char*>(&out_sz), sizeof(out_sz));

            std::cout << "[DEBUG] Layer " << idx << " is DenseLayer with shape [" 
                      << out_sz << " x " << in_sz << "]\n";

            for (uint32_t i = 0; i < out_sz; ++i) {
                ofs.write(reinterpret_cast<char*>(W[i].data()), in_sz * sizeof(double));
            }
            ofs.write(reinterpret_cast<char*>(b.data()), out_sz * sizeof(double));
        } else {
            uint8_t type = 0;
            ofs.write(reinterpret_cast<char*>(&type), sizeof(type));
        }
    }
    std::cout << "[DEBUG] Done saving weights.\n";
}

// Load weights from a binary file created by save_weights()
void NeuralNetwork::load_weights(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs) throw std::runtime_error("Cannot open file for reading: " + filepath);

    uint32_t L = 0;
    ifs.read(reinterpret_cast<char*>(&L), sizeof(L));
    std::cout << "[DEBUG] Loading weights from " << filepath << "\n";
    std::cout << "[DEBUG] File specifies " << L << " layers\n";

    if (L != layers_.size()) {
        throw std::runtime_error("Layer count mismatch: file has " +
                                 std::to_string(L) + " layers, network has " +
                                 std::to_string(layers_.size()));
    }

    for (uint32_t idx = 0; idx < L; ++idx) {
        uint8_t type = 0;
        ifs.read(reinterpret_cast<char*>(&type), sizeof(type));

        if (type == 1) {
            auto dl = dynamic_cast<DenseLayer*>(layers_[idx]);
            if (!dl) throw std::runtime_error("Expected DenseLayer at index " + std::to_string(idx));

            uint32_t in_sz = 0, out_sz = 0;
            ifs.read(reinterpret_cast<char*>(&in_sz),  sizeof(in_sz));
            ifs.read(reinterpret_cast<char*>(&out_sz), sizeof(out_sz));

            std::vector<std::vector<double>> W(out_sz, std::vector<double>(in_sz));
            for (uint32_t i = 0; i < out_sz; ++i) {
                ifs.read(reinterpret_cast<char*>(W[i].data()), in_sz * sizeof(double));
            }
            std::vector<double> b(out_sz);
            ifs.read(reinterpret_cast<char*>(b.data()), out_sz * sizeof(double));

            dl->set_weights(W, b);

            std::cout << "[DEBUG] Loaded DenseLayer " << idx << " with shape [" 
                      << out_sz << " x " << in_sz << "]\n";
            std::cout << std::fixed << std::setprecision(5)
                      << "         Example W[0][0] = " << W[0][0]
                      << ", b[0] = " << b[0] << "\n";
        }
    }
    std::cout << "[DEBUG] Done loading weights.\n";
}
