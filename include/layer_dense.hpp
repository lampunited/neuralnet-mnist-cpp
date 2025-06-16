#pragma once
#include "layer.hpp"
#include <vector>
#include <algorithm>

class DenseLayer : public Layer {
public:
    DenseLayer(int input_size, int output_size);
    ~DenseLayer();

    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& grad_output) override;

    void update_weights(double learning_rate) override;
    void zero_grad() override;

    void get_weights(std::vector<std::vector<double>>& W,
                     std::vector<double>& b) const;
    void set_weights(const std::vector<std::vector<double>>& W,
                     const std::vector<double>& b);

private:
    int input_size_;
    int output_size_;

    std::vector<std::vector<double>> weights_;
    std::vector<double> biases_;

    std::vector<double> input_cache_;

    std::vector<std::vector<double>> dW_accum_;
    std::vector<double>              db_accum_;
};
