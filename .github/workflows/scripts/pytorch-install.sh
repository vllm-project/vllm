#!/bin/bash

python_executable=python$1
cuda_version=$2

# Install torch
$python_executable -m pip install numpy pyyaml scipy ipython mkl mkl-include ninja cython typing pandas typing-extensions dataclasses setuptools && conda clean -ya
# Temporarily fix the PyTorch version to v2.0.1
$python_executable -m pip install torch==2.0.1+cu${cuda_version//./} --index-url https://download.pytorch.org/whl/cu${cuda_version//./}

# Print version information
$python_executable --version
$python_executable -c "import torch; print('PyTorch:', torch.__version__)"
$python_executable -c "import torch; print('CUDA:', torch.version.cuda)"
$python_executable -c "from torch.utils import cpp_extension; print (cpp_extension.CUDA_HOME)"
