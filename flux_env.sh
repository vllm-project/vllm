#Point to the directory containing the flux .so files:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/nm-vllm/flux_experiment/lib 

export NVSHMEM_BOOTSTRAP_MPI_PLUGIN=nvshmem_bootstrap_torch.so

# Env variables for symmetric heap allocation.
# These are needed for supporting CUDA_VISIBLE DEVICES
# This is big enough for llama3 8b, but should be set correctly
export NVSHMEM_SYMMETRIC_SIZE=$((8*1024**3))
export NVSHMEM_DISABLE_CUDA_VMM=1 # moving from cpp to shell

# Not sure if these are needed
export CUDA_DEVICE_MAX_CONNECTIONS=1
export BYTED_TORCH_BYTECCL=O0
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:=23}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:=3}
export NVSHMEM_IB_GID_INDEX=3
