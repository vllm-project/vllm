#!/bin/bash
set -eux

python_executable=python3

# Update paths
# Install requirements
$python_executable -m pip install -r requirements/rocm.txt

# Limit the number of parallel jobs to avoid OOM
export MAX_JOBS=1
# Make sure release wheels are built for the following architectures
export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX"

rm -f "$(which sccache)"

export MAX_JOBS=32

# Build
$python_executable setup.py bdist_wheel --dist-dir=dist
