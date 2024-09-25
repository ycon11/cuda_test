#!/bin/bash

export PATH=/usr/local/cuda-11.4/bin/:${PATH}
cd build
cmake -D CMAKE_BUILD_TYPE=Debug -D CMAKE_INSTALL_PREFIX=/home/yinkang/workspace/cuda_test/build ..
make -j${nproc} 