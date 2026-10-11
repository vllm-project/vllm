"""Separate commInitRank (structural) from channel buffers (P2P FIFO, lazy).
Bypass pynccl's built-in warmup by calling NCCLLibrary directly.
"""
import os
import ctypes
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank: int, world_size: int, port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    from vllm.distributed.device_communicators.pynccl_wrapper import (
        NCCLLibrary, ncclUniqueId, ncclComm_t, buffer_type, cudaStream_t,
        ncclDataTypeEnum, ncclRedOpTypeEnum)

    nccl = NCCLLibrary()

    # ---- broadcast uniqueId via gloo ----
    if rank == 0:
        uid = nccl.ncclGetUniqueId()
        buf = torch.ByteTensor(list(uid.internal))
    else:
        uid = ncclUniqueId()
        buf = torch.zeros(128, dtype=torch.uint8)
    dist.broadcast(buf, src=0)
    if rank != 0:
        for i, b in enumerate(buf.tolist()):
            uid.internal[i] = b

    def free():
        return torch.cuda.mem_get_info(rank)[0]

    torch.cuda.synchronize()
    base = free()
    if rank == 0:
        print(f"\n=== baseline free = {base/2**20:.0f} MiB ===\n"
              "--- comm #1 (first, one-time load included) ---", flush=True)

    def report(label, ref):
        f = free()
        used = (ref - f) / 2**20
        if rank == 0:
            print(f"  [{label:45s}] Δ since prev = {used:7.1f} MiB", flush=True)
        return f

    # --- comm #1 ---
    prev = base
    comm1 = nccl.ncclCommInitRank(world_size, uid, rank)
    prev = report("ncclCommInitRank #1 alone (no all_reduce yet)", prev)

    # First all_reduce -> triggers lazy channel buffer allocation
    stream = torch.cuda.current_stream()
    data = torch.zeros(1, device=f"cuda:{rank}")
    nccl.ncclAllReduce(buffer_type(data.data_ptr()),
                       buffer_type(data.data_ptr()), 1,
                       ncclDataTypeEnum.from_torch(data.dtype),
                       ncclRedOpTypeEnum.from_torch(torch.distributed.ReduceOp.SUM),
                       comm1, cudaStream_t(stream.cuda_stream))
    torch.cuda.synchronize()
    prev = report("first all_reduce on comm#1 (lazy channel buffers)", prev)

    # --- comm #2 (second uniqueId) ---
    if rank == 0:
        print("\n--- comm #2 (steady) ---", flush=True)
    if rank == 0:
        uid2 = nccl.ncclGetUniqueId()
        buf = torch.ByteTensor(list(uid2.internal))
    else:
        uid2 = ncclUniqueId()
        buf = torch.zeros(128, dtype=torch.uint8)
    dist.broadcast(buf, src=0)
    if rank != 0:
        for i, b in enumerate(buf.tolist()):
            uid2.internal[i] = b

    comm2 = nccl.ncclCommInitRank(world_size, uid2, rank)
    prev = report("ncclCommInitRank #2 alone (no all_reduce yet)", prev)

    data = torch.zeros(1, device=f"cuda:{rank}")
    nccl.ncclAllReduce(buffer_type(data.data_ptr()),
                       buffer_type(data.data_ptr()), 1,
                       ncclDataTypeEnum.from_torch(data.dtype),
                       ncclRedOpTypeEnum.from_torch(torch.distributed.ReduceOp.SUM),
                       comm2, cudaStream_t(stream.cuda_stream))
    torch.cuda.synchronize()
    prev = report("first all_reduce on comm#2 (lazy channel buffers)", prev)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    mp.spawn(worker, args=(2, 29583), nprocs=2)
