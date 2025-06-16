#include <iostream>
#include "neural_network.hpp"
#include "mnist_loader.hpp"
#include "layer_dense.hpp"
#include "activations.hpp"
#include "dropout.hpp"
#include "evaluate.hpp"

int main() {

    std::vector<std::vector<double>> test_images, test_labels;
    load_mnist_data(
        "../../data/t10k-images.idx3-ubyte",
        "../../data/t10k-labels.idx1-ubyte",
        test_images, test_labels
    );

    NeuralNetwork net;
    net.add_layer(new DenseLayer(784, 512));
    net.add_layer(new ReLU());
    net.add_layer(new Dropout(0.2));
    net.add_layer(new DenseLayer(512, 256));
    net.add_layer(new ReLU());
    net.add_layer(new Dropout(0.2));
    net.add_layer(new DenseLayer(256, 128));
    net.add_layer(new ReLU());
    net.add_layer(new Dropout(0.2));
    net.add_layer(new DenseLayer(128, 10));
    net.add_layer(new Softmax());

    net.load_weights("trained_weights.bin");
    for (Layer* L : net.get_layers())
        if (auto* d = dynamic_cast<Dropout*>(L))
            d->set_training(false);

    double acc = evaluate_accuracy(net, test_images, test_labels);
    std::cout << "Accuracy = " << (acc * 100.0) << "%\n";
    return 0;
}
