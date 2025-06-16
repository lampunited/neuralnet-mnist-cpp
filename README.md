Download raw MNIST files from https://www.kaggle.com/datasets/hojjatk/mnist-dataset?select=t10k-labels.idx1-ubyte and place into /data folder

git clone                                                  
cd mnist-cpp-net                          
mkdir build                              
cd build                                      
cmake ..                                              
cmake --build . --config Release                                      
cd Release                                              
./neuralnet                                          
./classifier_server                                      
./evaluate_net                                  


