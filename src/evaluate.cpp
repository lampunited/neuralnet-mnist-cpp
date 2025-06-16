#include "neural_network.hpp"
#include <vector>
#include <algorithm>

double evaluate_accuracy(NeuralNetwork& net,
                         const std::vector<std::vector<double>>& images,
                         const std::vector<std::vector<double>>& labels)
{
    size_t correct = 0;
    for (size_t i = 0; i < images.size(); ++i) {
        auto probs = net.predict(images[i]);
        int pred = std::max_element(probs.begin(), probs.end()) - probs.begin();
        int actual = std::max_element(labels[i].begin(), labels[i].end()) - labels[i].begin();
        if (pred == actual) ++correct;
    }
    return double(correct) / images.size();
}
