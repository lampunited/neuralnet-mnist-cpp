#pragma once
#include "layer.hpp"
#include <vector>
#include <random>

class Dropout : public Layer {
public:
    explicit Dropout(double p);

    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& grad_output) override;

    void update_weights(double) override {}
    void zero_grad() override {}
    void set_training(bool train);
    
private:
    double p_;                       
    bool   training_;                
    std::vector<double> mask_;      
    std::mt19937        rng_;        
    std::bernoulli_distribution dist_; 
};
