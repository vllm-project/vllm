# can only run on machines with p2p access across GPUs
# can only run with torchrun --nproc_per_node=n 

import ctypes
import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

dist.init_process_group(backend="gloo")
rank = local_rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(local_rank)

pointers = CustomAllreduce.create_shared_buffer(1024)

dist.barrier()
torch.cuda.synchronize()

lib = CudaRTLibrary()

if rank == 0:
    for p in pointers:
        pointer = ctypes.c_void_p(p)
        lib.cudaMemset(pointer, 2, 1024)

dist.barrier()
torch.cuda.synchronize()

host_data = (ctypes.c_char * 1024)()

for p in pointers:
    pointer = ctypes.c_void_p(p)
    lib.cudaMemcpy(host_data, pointer, 1024)
    for i in range(1024):
        assert ord(host_data[i]) == 2

dist.barrier()
torch.cuda.synchronize()

CustomAllreduce.free_shared_buffer(pointers)
