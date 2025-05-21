#!/bin/bash
set -o errexit -o pipefail -o noclobber

while [[ $# -gt 0 ]]; do
    case $1 in
    -v | --version)
        version="$2"
        shift 2 # Removes both the option and its value
        ;;
    *)
        echo "Unknown option: $1"
        shift # Removes the unknown option
        ;;
    esac
done

pushd /tmp

# Download the LLVM installer script
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh

# Add the LLVM repository directly to avoid using add-apt-repository
echo "deb http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs)-${version} main" | sudo tee /etc/apt/sources.list.d/llvm-${version}.list

# Update and install LLVM
sudo apt-get update
sudo apt-get install -y llvm-${version} llvm-${version}-dev llvm-${version}-runtime clang-${version} clang-tools-${version} lld-${version}

popd

LLVM_BIN_DIR=/usr/bin
for binary in ${LLVM_BIN_DIR}/*-${version}; do
    base_name=$(basename "$binary" -${version})
    ln -sfv "$binary" "${LLVM_BIN_DIR}/${base_name}"
    echo "Created/updated symlink: ${base_name} -> $(basename $binary)"
done
