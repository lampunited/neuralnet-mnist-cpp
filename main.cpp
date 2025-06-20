#include <iostream>
#include "neural_network.hpp"
#include "mnist_loader.hpp"
#include "layer_dense.hpp"
#include "activations.hpp"
#include "evaluate.hpp"            
#include "dropout.hpp"

int main() {
    std::vector<std::vector<double>> train_images;
    std::vector<std::vector<double>> train_labels;

    load_mnist_data(
        "../../data/train-images.idx3-ubyte",
        "../../data/train-labels.idx1-ubyte",
        train_images, train_labels
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

    // epochs = 50, lr = 0.001, batchsiz = 64
    net.train(train_images, train_labels, 50, 0.1, 64);

    net.save_weights("trained_weights.bin");

    return 0;
}
