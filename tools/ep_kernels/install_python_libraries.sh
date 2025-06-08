set -ex

# prepare workspace directory
WORKSPACE=$1
if [ -z "$WORKSPACE" ]; then
    export WORKSPACE=$(pwd)/ep_kernels_workspace
fi

if [ ! -d "$WORKSPACE" ]; then
    mkdir -p $WORKSPACE
fi

# install dependencies if not installed
pip3 install cmake torch ninja

# build gdrcopy, required by nvshmem
pushd $WORKSPACE
wget https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v2.4.4.tar.gz
mkdir -p gdrcopy_src
tar -xvf v2.4.4.tar.gz -C gdrcopy_src --strip-components=1
pushd gdrcopy_src
make -j$(nproc)
make prefix=$WORKSPACE/gdrcopy_install install
popd

# build nvshmem
pushd $WORKSPACE
mkdir -p nvshmem_src
wget https://developer.download.nvidia.com/compute/redist/nvshmem/3.2.5/source/nvshmem_src_3.2.5-1.txz
tar -xvf nvshmem_src_3.2.5-1.txz -C nvshmem_src --strip-components=1
pushd nvshmem_src
wget https://github.com/deepseek-ai/DeepEP/raw/main/third-party/nvshmem.patch
git init
git apply -vvv nvshmem.patch

# assume CUDA_HOME is set correctly
export GDRCOPY_HOME=$WORKSPACE/gdrcopy_install
export NVSHMEM_SHMEM_SUPPORT=0
export NVSHMEM_UCX_SUPPORT=0
export NVSHMEM_USE_NCCL=0
export NVSHMEM_IBGDA_SUPPORT=1
export NVSHMEM_PMIX_SUPPORT=0
export NVSHMEM_TIMEOUT_DEVICE_POLLING=0
export NVSHMEM_USE_GDRCOPY=1
export NVSHMEM_IBRC_SUPPORT=1

# remove MPI dependency
export NVSHMEM_BUILD_TESTS=0
export NVSHMEM_BUILD_EXAMPLES=0
export NVSHMEM_MPI_SUPPORT=1
export NVSHMEM_BUILD_HYDRA_LAUNCHER=1
export NVSHMEM_BUILD_TXZ_PACKAGE=0
export CMAKE_CUDA_ARCHITECTURES=90a

cmake -S . -B $WORKSPACE/nvshmem_build/ \
-DCMAKE_INSTALL_PREFIX=$WORKSPACE/nvshmem_install \
-DCMAKE_CUDA_ARCHITECTURES="90a" \
-DNVSHMEM_MPI_SUPPORT=0            \
-DNVSHMEM_PMIX_SUPPORT=0           \
-DNVSHMEM_IBRC_SUPPORT=1           \
-DNVSHMEM_IBGDA_SUPPORT=1          \
-DNVSHMEM_IBDEVX_SUPPORT=1         \
-DNVSHMEM_USE_GDRCOPY=1            \
-DNVSHMEM_BUILD_TESTS=0            \
-DNVSHMEM_BUILD_EXAMPLES=0         \
-DNVSHMEM_LIBFABRIC_SUPPORT=0      \
-DGDRCOPY_HOME=${GDRCOPY_HOME}    


cd $WORKSPACE/nvshmem_build/
make -j$(nproc)
make install

popd

export CMAKE_PREFIX_PATH=$WORKSPACE/nvshmem_install:$CMAKE_PREFIX_PATH

# build and install pplx, require pytorch installed
pushd $WORKSPACE
git clone https://github.com/ppl-ai/pplx-kernels
cd pplx-kernels
# see https://github.com/pypa/pip/issues/9955#issuecomment-838065925
# PIP_NO_BUILD_ISOLATION=0 disables build isolation
PIP_NO_BUILD_ISOLATION=0 TORCH_CUDA_ARCH_LIST=9.0a+PTX uv pip install --no-build-isolation -vvv -e  .
popd

# build and install deepep, require pytorch installed
pushd $WORKSPACE
git clone https://github.com/deepseek-ai/DeepEP
cd DeepEP
export NVSHMEM_DIR=$WORKSPACE/nvshmem_install
PIP_NO_BUILD_ISOLATION=0 uv pip install --no-build-isolation -vvv -e  .
popd
