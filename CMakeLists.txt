cmake_minimum_required(VERSION 3.10)
project(neuralnet_mnist_cpp)

# For std::thread (used by httplib)
find_package(Threads REQUIRED)

# Core library with all model & utility code
add_library(mnist_core
    src/neural_network.cpp
    src/layer_dense.cpp
    src/activations.cpp
    src/dropout.cpp
    src/mnist_loader.cpp
    src/evaluate.cpp
)
# Make headers in include/ public to anything linking mnist_core
target_include_directories(mnist_core PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Training executable (the one with main.cpp)
add_executable(neuralnet
    main.cpp
)
target_include_directories(neuralnet PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(neuralnet PRIVATE 
    mnist_core 
    Threads::Threads
)

# Evaluation-only executable
add_executable(neuralnet_eval
    eval.cpp
)
target_include_directories(neuralnet_eval PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(neuralnet_eval PRIVATE
    mnist_core
    Threads::Threads
)

# Server executable
add_executable(classifier_server
    src/server/classifier_server.cpp
)
target_include_directories(classifier_server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/httplib
    ${CMAKE_CURRENT_SOURCE_DIR}/include/nlohmann
)
target_link_libraries(classifier_server PRIVATE 
    mnist_core 
    Threads::Threads
)
