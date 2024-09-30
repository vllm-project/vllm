export VERBOSE=1
export CUDA_HOME=/usr/local/cuda-12
export TORCH_CUDA_ARCH_LIST="6.0"
export CUDACXX=/usr/local/cuda/bin/nvcc 
export PATH=/usr/local/cuda/bin:$PATH

pip install -e . -vvv
