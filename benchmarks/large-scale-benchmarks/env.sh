export WORKSPACE=~/vllm/tools/ep_kernels/ep_kernels_workspace
export CMAKE_PREFIX_PATH=$WORKSPACE/nvshmem_install:$CMAKE_PREFIX_PATH
export NVSHMEM_DIR=$WORKSPACE/nvshmem_install
export NVSHMEM_HOME=$WORKSPACE/nvshmem_install
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH"
export PATH="${NVSHMEM_DIR}/bin:$PATH"
