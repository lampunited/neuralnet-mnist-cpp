#include "mnist_loader.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

static int read_int(std::ifstream& ifs) {
    int value = 0;
    for (int i = 0; i < 4; ++i) {
        if (ifs.eof()) throw std::runtime_error("❌ Unexpected EOF while reading 4-byte int");
        value = (value << 8) | static_cast<unsigned char>(ifs.get());
    }
    return value;
}

void load_mnist_data(const std::string& image_path,
                     const std::string& label_path,
                     std::vector<std::vector<double>>& images,
                     std::vector<std::vector<double>>& labels)
{
    std::cout << "[DEBUG] Entering load_mnist_data()\n";
    std::cout << "[DEBUG] Attempting to open files...\n";

    std::ifstream ifs_images(image_path, std::ios::binary);
    std::ifstream ifs_labels(label_path, std::ios::binary);

    if (!ifs_images) {
        std::cerr << "❌ ERROR: Failed to open image file: " << image_path << std::endl;
        throw std::runtime_error("Image file open failed.");
    }
    if (!ifs_labels) {
        std::cerr << "❌ ERROR: Failed to open label file: " << label_path << std::endl;
        throw std::runtime_error("Label file open failed.");
    }

    std::cout << "[DEBUG] Files opened. Reading headers...\n";

    int magic_images = read_int(ifs_images);
    int num_images = read_int(ifs_images);
    int rows = read_int(ifs_images);
    int cols = read_int(ifs_images);

    int magic_labels = read_int(ifs_labels);
    int num_labels = read_int(ifs_labels);

    std::cout << "[DEBUG] magic_images: " << magic_images
              << ", num_images: " << num_images
              << ", rows: " << rows << ", cols: " << cols << "\n";
    std::cout << "[DEBUG] magic_labels: " << magic_labels
              << ", num_labels: " << num_labels << "\n";

    if (magic_images != 2051) throw std::runtime_error("❌ Invalid magic number in image file.");
    if (magic_labels != 2049) throw std::runtime_error("❌ Invalid magic number in label file.");
    if (num_images != num_labels) throw std::runtime_error("❌ Image and label count mismatch.");

    images.resize(num_images, std::vector<double>(rows * cols));
    labels.resize(num_images, std::vector<double>(10, 0.0));

    std::cout << "[DEBUG] Reading image and label data...\n";

    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < rows * cols; ++j) {
            if (ifs_images.eof()) throw std::runtime_error("❌ Unexpected EOF in image data.");
            unsigned char pixel = ifs_images.get();
            images[i][j] = static_cast<double>(pixel) / 255.0;
        }

        if (ifs_labels.eof()) throw std::runtime_error("❌ Unexpected EOF in label data.");
        unsigned char label = ifs_labels.get();
        if (label > 9) throw std::runtime_error("❌ Invalid label value.");
        labels[i][label] = 1.0;

        if (i % 10000 == 0) {
            std::cout << "[DEBUG] Loaded " << i << " samples...\n";
        }
    }

    std::cout << "[INFO] Successfully loaded " << num_images << " images and labels.\n";
}
