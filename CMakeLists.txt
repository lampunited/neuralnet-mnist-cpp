# At top, after project(...)
find_package(Threads REQUIRED)

# Existing core library
add_library(mnist_core
    src/neural_network.cpp
    src/layer_dense.cpp
    src/activations.cpp
    src/mnist_loader.cpp
)
target_include_directories(mnist_core PUBLIC include)

# Server
add_executable(classifier_server
    src/server/classifier_server.cpp
)
target_include_directories(classifier_server PRIVATE
    include
    include/httplib
    include/nlohmann
)
target_link_libraries(classifier_server PRIVATE mnist_core Threads::Threads)
