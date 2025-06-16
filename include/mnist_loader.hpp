#ifndef MNIST_LOADER_HPP
#define MNIST_LOADER_HPP

#include <string>
#include <vector>

void load_mnist_data(const std::string& image_path,const std::string& label_path,
                std::vector<std::vector<double>>& images,
                std::vector<std::vector<double>>& labels);

#endif 
