# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# can only run on machines with p2p access across GPUs
# can only run with torchrun:
# torchrun --nproc_per_node=2 tests/distributed/test_ca_buffer_sharing.py

import ctypes

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from vllm.distributed.device_communicators.custom_all_reduce import (  # noqa
    CustomAllreduce,
)

# create a cpu process group for communicating metadata (ipc handle)
dist.init_process_group(backend="gloo")
rank = local_rank = dist.get_rank()
world_size = dist.get_world_size()

# every process sets its own device (differently)
lib = CudaRTLibrary()
lib.cudaSetDevice(rank)

buffer_size_in_bytes = 1024
byte_value = 2  # the value we write to the buffer for verification

pointers = CustomAllreduce.create_shared_buffer(buffer_size_in_bytes)

print(f"Rank {rank} has pointers {pointers}")

dist.barrier()
torch.cuda.synchronize()

if rank == 0:
    # the first rank tries to write to all buffers
    for p in pointers:
        pointer = ctypes.c_void_p(p)
        lib.cudaMemset(pointer, byte_value, buffer_size_in_bytes)

dist.barrier()
torch.cuda.synchronize()

host_data = (ctypes.c_char * buffer_size_in_bytes)()

# all ranks read from all buffers, and check if the data is correct
for p in pointers:
    pointer = ctypes.c_void_p(p)
    lib.cudaMemcpy(host_data, pointer, buffer_size_in_bytes)
    for i in range(buffer_size_in_bytes):
        assert ord(host_data[i]) == byte_value, (
            f"Rank {rank} failed"
            f" to verify buffer {p}. Expected {byte_value}, "
            f"got {ord(host_data[i])}"
        )

print(f"Rank {rank} verified all buffers")

dist.barrier()
torch.cuda.synchronize()

CustomAllreduce.free_shared_buffer(pointers)
