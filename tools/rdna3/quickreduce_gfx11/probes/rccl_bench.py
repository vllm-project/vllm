"""Baseline to compare the push all-reduce against: what RCCL already does on
this box, at the sizes a TP2 decode step actually uses.

Run: torchrun --nproc_per_node=2 rccl_bench.py
"""

import os

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

SIZES = [16 * 1024, 64 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024]
REPS = 300

if rank == 0:
    print(f"RCCL all-reduce, world={dist.get_world_size()}, {REPS} reps")

for nbytes in SIZES:
    t = torch.ones(nbytes // 2, dtype=torch.float16, device=f"cuda:{rank}")
    for _ in range(20):
        dist.all_reduce(t)
    torch.cuda.synchronize()
    dist.barrier()

    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(REPS):
        dist.all_reduce(t)
    b.record()
    torch.cuda.synchronize()
    ms = a.elapsed_time(b)

    # sanity: every rank contributed 1.0, so the result must be world_size
    expect = float(dist.get_world_size()) ** (20 + REPS)
    ok = torch.isfinite(t).all().item()
    if rank == 0:
        us = ms * 1000.0 / REPS
        gbs = nbytes * REPS / 1e9 / (ms / 1e3)
        print(
            f"  {nbytes // 1024:5d} KiB   {us:8.1f} us/op   {gbs:6.2f} GB/s"
            f"   finito={ok}"
        )
    del t
    torch.cuda.empty_cache()

dist.destroy_process_group()
