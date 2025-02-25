#!/bin/bash

# Usage: ./install_cuda.sh <cuda_version> <distro_version>
#   e.g. ./install_cuda.sh 11.8 ubuntu-20.04

# Replace '.' with '-' ex: 11.8 -> 11-8
cuda_version=$(echo "$1" | tr "." "-")

# Removes '-' and '.' ex: ubuntu-20.04 -> ubuntu2004
OS=$(echo "$2" | tr -d ".\-")

# Detect system architecture for the CUDA repo
arch=$(uname -m)
if [ "$arch" = "x86_64" ]; then
    repo_arch="x86_64"
elif [ "$arch" = "aarch64" ]; then
    repo_arch="sbsa"
else
    echo "Unsupported architecture: $arch"
    exit 1
fi

# Fetch and install the CUDA public key package
wget -nv "https://developer.download.nvidia.com/compute/cuda/repos/${OS}/${repo_arch}/cuda-keyring_1.1-1_all.deb"
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb

# Update and install the requested CUDA version
sudo apt -qq update
sudo apt -y install \
    "cuda-${cuda_version}" \
    "cuda-nvcc-${cuda_version}" \
    "cuda-libraries-dev-${cuda_version}"

# Cleanup
sudo apt clean

# Put nvcc on PATH
export PATH="/usr/local/cuda-$1/bin:${PATH}"

# Test nvcc
nvcc --version

# Log gcc, g++, c++ versions
gcc --version
g++ --version
c++ --version
