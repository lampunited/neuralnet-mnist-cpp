#include "mnist_loader.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

static int read_int(std::ifstream& ifs) {
    int value = 0;
    for (int i = 0; i < 4; ++i) {
        if (ifs.eof()) throw std::runtime_error("error while reading 4-byte int");
        value = (value << 8) | static_cast<unsigned char>(ifs.get());
    }
    return value;
}

void load_mnist_data(const std::string& image_path,
                     const std::string& label_path,
                     std::vector<std::vector<double>>& images,
                     std::vector<std::vector<double>>& labels)
{

    std::ifstream ifs_images(image_path, std::ios::binary);
    std::ifstream ifs_labels(label_path, std::ios::binary);

    if (!ifs_images) {
        std::cerr << "failed to open image file: " << image_path << std::endl;
        throw std::runtime_error("image file open failed.");
    }
    if (!ifs_labels) {
        std::cerr << "failed to open label file: " << label_path << std::endl;
        throw std::runtime_error("label file open failed.");
    }

    int magic_images = read_int(ifs_images);
    int num_images = read_int(ifs_images);
    int rows = read_int(ifs_images);
    int cols = read_int(ifs_images);

    int magic_labels = read_int(ifs_labels);
    int num_labels = read_int(ifs_labels);

    if (magic_images != 2051) throw std::runtime_error("invalid magic number in image file.");
    if (magic_labels != 2049) throw std::runtime_error("invalid magic number in label file.");
    if (num_images != num_labels) throw std::runtime_error("image and label count mismatch.");

    images.resize(num_images, std::vector<double>(rows * cols));
    labels.resize(num_images, std::vector<double>(10, 0.0));

    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < rows * cols; ++j) {
            if (ifs_images.eof()) throw std::runtime_error("error in image data.");
            unsigned char pixel = ifs_images.get();
            images[i][j] = static_cast<double>(pixel) / 255.0;
        }

        if (ifs_labels.eof()) throw std::runtime_error("error in label data.");
        unsigned char label = ifs_labels.get();
        if (label > 9) throw std::runtime_error("invalid label value.");
        labels[i][label] = 1.0;

        if (i % 10000 == 0) {
            std::cout << "loaded " << i << " samples...\n";
        }
    }
}
