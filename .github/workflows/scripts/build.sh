#!/bin/bash
set -eux

python_executable=python3

# Update paths
# Install requirements
$python_executable -m pip install -r requirements-rocm.txt

# Limit the number of parallel jobs to avoid OOM
export MAX_JOBS=1
# Make sure release wheels are built for the following architectures
export PYTORCH_ROCM_ARCH="gfx90a;gfx942"

rm -f "$(which sccache)"

export MAX_JOBS=32

# Build
$python_executable setup.py bdist_wheel --dist-dir=dist
cd gradlib
$python_executable setup.py bdist_wheel --dist-dir=dist
cd ..