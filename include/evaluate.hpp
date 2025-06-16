// include/evaluate.hpp
#pragma once
#include "neural_network.hpp"
double evaluate_accuracy(NeuralNetwork&, const std::vector<std::vector<double>>&,
                        const std::vector<std::vector<double>>&);
