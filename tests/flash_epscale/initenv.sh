#!/bin/bash

set -e
export LD_LIBRARY_PATH=/usr/local/cuda-13/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-13/bin:$PATH
echo "Initializing environment..."
cd ./vllm
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install --editable . -v
pip3 install meson ninja pybind11 tomlkit
pip install nixl-cu13==1.0.1
pip install ray[default]