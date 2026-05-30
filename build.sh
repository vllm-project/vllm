

cd `dirname $0`

# export VLLM_CUTLASS_SRC_DIR=`pwd`/cutlass-src
export CMAKE_BUILD_TYPE=Release

# export TORCH_CUDA_ARCH_LIST=8.0
export VLLM_INSTALL_PUNICA_KERNELS=1
source ../init.sh

export TORCH_CUDA_ARCH_LIST="8.0;8.9"
export NVCC_THREADS=1
export MAX_JOBS=12
export BUILD_VERSION=$(cat version.txt)

#export C_INCLUDE_PATH=`pwd`/cutlass-src/include
#export CPP_INCLUDE_PATH=`pwd`/cutlass-src/include
#export CUTLASS_INCLUDE_DIR=`pwd`/cutlass-src/include
rm -rf dist
echo $PATH
# rm -rf dist build
python setup.py bdist_wheel
mkdir -p ../wheel/vllm/$torch_cuda_combine_name/
cp dist/* ../wheel/vllm/$torch_cuda_combine_name/


