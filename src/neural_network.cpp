#include "neural_network.hpp"
#include "layer.hpp"
#include "layer_dense.hpp"
#include "dropout.hpp"

#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <random>
#include <numeric>

NeuralNetwork::~NeuralNetwork() {
    for (auto* L : layers_) delete L;
}

void NeuralNetwork::add_layer(Layer* layer) {
    layers_.push_back(layer);
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& input) {
    auto activation = input;
    for (auto* L : layers_) activation = L->forward(activation);
    return activation;
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& inputs,
                          const std::vector<std::vector<double>>& targets,
                          int epochs,
                          double learning_rate,
                          int batch_size)
{
    size_t N = inputs.size();
    std::mt19937 rng(std::random_device{}());

    for (int e = 0; e < epochs; ++e) {
        double epoch_loss = 0.0;

        std::vector<size_t> idx(N);
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), rng);

        for (size_t start = 0; start < N; start += batch_size) {
            size_t end = std::min(start + batch_size, N);

            for (auto* L : layers_) L->zero_grad();

            for (size_t b = start; b < end; ++b) {
                auto activation = inputs[idx[b]];
                for (auto* L : layers_) activation = L->forward(activation);

                std::vector<double> grad(activation.size());
                for (size_t j = 0; j < activation.size(); ++j) {
                    double y = targets[idx[b]][j];
                    double p = std::max(activation[j], 1e-12);
                    epoch_loss += -y * std::log(p);
                    grad[j] = activation[j] - y;
                }

                for (int li = (int)layers_.size() - 1; li >= 0; --li)
                    grad = layers_[li]->backward(grad);
            }

            double lr = learning_rate / double(end - start);
            for (auto* L : layers_) L->update_weights(lr);
        }

        std::cout << "Epoch " << (e + 1) << "/" << epochs
                  << "   Avg. Loss = " << (epoch_loss / double(N)) << "\n";
    }
}

void NeuralNetwork::save_weights(const std::string& filepath) const {
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs) throw std::runtime_error("failed to open " + filepath);
    uint32_t L = (uint32_t)layers_.size();
    ofs.write(reinterpret_cast<char*>(&L), sizeof(L));

    for (uint32_t i = 0; i < L; ++i) {
        auto* dl = dynamic_cast<DenseLayer*>(layers_[i]);
        uint8_t type = dl ? 1 : 0;
        ofs.write(reinterpret_cast<char*>(&type), sizeof(type));
        if (dl) {
            std::vector<std::vector<double>> W;
            std::vector<double> b;
            dl->get_weights(W, b);
            uint32_t out_sz = (uint32_t)W.size();
            uint32_t in_sz  = out_sz ? (uint32_t)W[0].size() : 0;
            ofs.write(reinterpret_cast<char*>(&in_sz), sizeof(in_sz));
            ofs.write(reinterpret_cast<char*>(&out_sz), sizeof(out_sz));
            for (auto& row : W)
                ofs.write(reinterpret_cast<char*>(row.data()),
                          row.size() * sizeof(double));
            ofs.write(reinterpret_cast<char*>(b.data()),
                      b.size() * sizeof(double));
        }
    }
}

void NeuralNetwork::load_weights(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs) throw std::runtime_error("failed to open " + filepath);
    uint32_t Lfile;
    ifs.read(reinterpret_cast<char*>(&Lfile), sizeof(Lfile));
    if (Lfile != layers_.size())
        throw std::runtime_error("layer count mismatch");
    for (uint32_t i = 0; i < Lfile; ++i) {
        uint8_t type;
        ifs.read(reinterpret_cast<char*>(&type), sizeof(type));
        if (type == 1) {
            auto* dl = dynamic_cast<DenseLayer*>(layers_[i]);
            uint32_t in_sz, out_sz;
            ifs.read(reinterpret_cast<char*>(&in_sz),  sizeof(in_sz));
            ifs.read(reinterpret_cast<char*>(&out_sz), sizeof(out_sz));
            std::vector<std::vector<double>> W(out_sz, std::vector<double>(in_sz));
            for (uint32_t o = 0; o < out_sz; ++o)
                ifs.read(reinterpret_cast<char*>(W[o].data()),
                         in_sz * sizeof(double));
            std::vector<double> b(out_sz);
            ifs.read(reinterpret_cast<char*>(b.data()), out_sz * sizeof(double));
            dl->set_weights(W, b);
        }
    }
}

void DenseLayer::get_weights(std::vector<std::vector<double>>& W,
                             std::vector<double>& b) const
{
    W = weights_;
    b = biases_;
}

void DenseLayer::set_weights(const std::vector<std::vector<double>>& W,
                             const std::vector<double>& b)
{
    if (W.size() != weights_.size() ||
        (W.size() > 0 && W[0].size() != weights_[0].size()) ||
        b.size() != biases_.size())
    {
        throw std::runtime_error("DenseLayer::set_weights dimension mismatch");
    }
    weights_ = W;
    biases_  = b;
}