#ifndef MNIST_LOADER_HPP
#define MNIST_LOADER_HPP

#include <string>
#include <vector>

// Load MNIST images and labels (labels → one-hot vectors of size 10).
// images will be size [num_images][784], values in [0,1].
// labels will be size [num_images][10], one-hot encoded.
void load_mnist_data(const std::string& image_path,
                     const std::string& label_path,
                     std::vector<std::vector<double>>& images,
                     std::vector<std::vector<double>>& labels);

#endif // MNIST_LOADER_HPP
