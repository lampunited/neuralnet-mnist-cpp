#include "httplib/httplib.h"
#include "nlohmann/json.hpp"

#include "neural_network.hpp"
#include "layer_dense.hpp"
#include "activations.hpp"
#include "mnist_loader.hpp"

using json = nlohmann::json;

int main() {
    NeuralNetwork net;
    net.add_layer(new DenseLayer(784, 256));
    net.add_layer(new ReLU());
    net.add_layer(new DenseLayer(256, 128));
    net.add_layer(new ReLU());
    net.add_layer(new DenseLayer(128, 10));
    net.add_layer(new Softmax());

    net.load_weights("trained_weights.bin");
    std::cout << "[OK] Loaded weights from trained_weights.bin\n";

    httplib::Server svr;

    svr.set_mount_point("/", "../../ui");

    svr.Post("/predict", [&](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");

        try {
            auto j = json::parse(req.body);
            auto pixels = j.at("pixels").get<std::vector<double>>();
            if (pixels.size() != 784) throw std::runtime_error("Expected 784 pixels");

            auto probs = net.predict(pixels);
            int digit = static_cast<int>(std::max_element(probs.begin(), probs.end()) - probs.begin());
            double confidence = probs[digit];

            json reply = {
                {"digit", digit},
                {"confidence", confidence}
            };

            res.set_content(reply.dump(), "application/json");
        } catch (const std::exception& e) {
            json err = {{"error", e.what()}};
            res.status = 400;
            res.set_content(err.dump(), "application/json");
        }
    });

    const int port = 8080;
    std::cout << "Server listening on http://localhost:" << port << "/" << std::endl;
    svr.listen("0.0.0.0", port);

    return 0;
}