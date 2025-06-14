#include <iostream>
#include "neural_network.hpp"
#include "mnist_loader.hpp"
#include "layer_dense.hpp"
#include "activations.hpp"

int main() {
    std::cout << "[INFO] Starting main()\n";

    std::vector<std::vector<double>> train_images;
    std::vector<std::vector<double>> train_labels;

    std::cout << "[INFO] Loading MNIST data...\n";

    load_mnist_data(
        "C:/Users/steve/OneDrive/Desktop/neuralnet_mnist_cpp/data/train-images.idx3-ubyte",
        "C:/Users/steve/OneDrive/Desktop/neuralnet_mnist_cpp/data/train-labels.idx1-ubyte",
        train_images, train_labels
    );

    std::cout << "[INFO] Data loaded. Creating network...\n";

    NeuralNetwork net;
    net.add_layer(new DenseLayer(784, 256));
    net.add_layer(new ReLU());
    net.add_layer(new DenseLayer(256, 128));
    net.add_layer(new ReLU());
    net.add_layer(new DenseLayer(128, 10));
    net.add_layer(new Softmax());

    std::cout << "[INFO] Starting training...\n";
    net.train(train_images, train_labels, 30, 0.005); 

    std::cout << "[INFO] Training finished. Saving weights...\n";
    net.save_weights("trained_weights.bin");
    std::cout << "[OK] Weights saved to trained_weights.bin\n";

    std::cout << "[INFO] Done.\n";
    return 0;
}
