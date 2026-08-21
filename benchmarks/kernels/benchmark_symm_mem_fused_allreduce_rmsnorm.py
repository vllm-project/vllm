# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark: fused (NVLink P2P + RMSNorm) vs unfused (NCCL+RMSNorm).

For each (n_tokens, hidden_size, dtype) combination this script reports:
    * unfused latency:  NCCL all_reduce  ->  standalone RMSNorm
    * fused latency:    SymmMem fused all_reduce + RMSNorm Triton kernel
    * speedup:          unfused / fused
    * effective HBM bw: counts the activation traffic only

Run on a node with TP-many GPUs (default reads ``torch.cuda.device_count()``):

    python benchmarks/kernels/benchmark_symm_mem_fused_allreduce_rmsnorm.py
"""

import argparse
import os
from functools import partial as partial_fn

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.distributed.device_communicators.symm_mem_fused_norm import (
    fused_allreduce_rmsnorm,
    is_supported,
)

try:
    import torch.distributed._symmetric_memory as torch_symm_mem
except ImportError:  # pragma: no cover
    torch_symm_mem = None


SHAPES = [
    (1, 4096),
    (4, 4096),
    (32, 4096),
    (128, 4096),
    (512, 4096),
    (1024, 4096),
    (32, 8192),
    (256, 8192),
    (1024, 8192),
    (32, 16384),
    (256, 16384),
    (1024, 16384),
]


def _time_iters(fn, iters: int) -> float:
    """Median latency of ``fn`` in milliseconds using cuda events."""
    s = torch.cuda.Stream()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(s):
            start.record()
            fn()
            end.record()
        s.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def _unfused_step(
    ar_buf: torch.Tensor,
    partial: torch.Tensor,
    rms_norm_fn: torch.nn.RMSNorm,
) -> None:
    """One unfused iteration: NCCL all_reduce + torch.nn.RMSNorm."""
    ar_buf.copy_(partial)
    dist.all_reduce(ar_buf, op=dist.ReduceOp.SUM)
    _ = rms_norm_fn(ar_buf)


def _fused_step(
    x_symm: torch.Tensor,
    partial: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    world_size: int,
) -> None:
    """One fused iteration: symm-mem fused all_reduce + RMSNorm."""
    x_symm.copy_(partial)
    fused_allreduce_rmsnorm(x_symm, weight, 1e-5, world_size=world_size, out=out)


def _worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29555"
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{rank}"),
    )

    # Minimal TP group init (avoid pulling in full vllm.config)
    from torch.distributed._symmetric_memory import enable_symm_mem_for_group
    enable_symm_mem_for_group(dist.group.WORLD.group_name)

    import vllm.distributed.parallel_state as ps
    group_ranks = [list(range(world_size))]
    ps._TP = ps.GroupCoordinator(
        group_ranks=group_ranks,
        local_rank=rank,
        torch_distributed_backend="nccl",
        use_device_communicator=False,
    )

    group = dist.group.WORLD

    if not is_supported():
        if rank == 0:
            print(
                "symm_mem fused norm backend is not supported on this box; "
                "skipping benchmark."
            )
        dist.destroy_process_group()
        return

    if rank == 0:
        print(
            f"\n{'shape':<14}{'dtype':<7}"
            f"{'unfused (ms)':>14}{'fused (ms)':>12}"
            f"{'speedup':>10}{'unfused BW':>13}{'fused BW':>13}"
        )
        print("-" * 90)

    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.dtype]

    for shape in SHAPES:
        n_rows, n_cols = shape
        elem = torch.tensor([], dtype=dtype).element_size()
        bytes_per_iter = n_rows * n_cols * elem

        torch.manual_seed(42 + rank)
        partial = torch.randn(shape, dtype=dtype, device=f"cuda:{rank}") * 0.5
        torch.manual_seed(0)
        weight = torch.randn(n_cols, dtype=dtype, device=f"cuda:{rank}") * 0.1

        x_symm = torch_symm_mem.empty(shape, device=f"cuda:{rank}", dtype=dtype)
        torch_symm_mem.rendezvous(x_symm, group.group_name)

        ar_buf = torch.empty_like(x_symm)
        out = torch.empty_like(x_symm)

        # Use torch.nn.RMSNorm as the real fused baseline
        rms_norm_fn = torch.nn.RMSNorm(n_cols, eps=1e-5, device=f"cuda:{rank}",
                                        dtype=dtype)
        rms_norm_fn.weight.data.copy_(weight)

        unfused = partial_fn(_unfused_step, ar_buf, partial, rms_norm_fn)
        fused = partial_fn(_fused_step, x_symm, partial, weight, out, world_size)

        for _ in range(args.warmup):
            unfused()
        for _ in range(args.warmup):
            fused()
        torch.cuda.synchronize()
        dist.barrier()

        t_un = _time_iters(unfused, args.iters)
        t_fu = _time_iters(fused, args.iters)
        speedup = t_un / t_fu
        # Unfused: copy(1) + allreduce(~2) + rms r/w (2) ~ 5x activation bytes
        unfused_gbps = 5 * bytes_per_iter / (t_un / 1000) / 1e9
        # Fused: copy(1) + local read(1) + peer read(1) + write(1) ~ 4x bytes
        fused_gbps = 4 * bytes_per_iter / (t_fu / 1000) / 1e9

        if rank == 0:
            print(
                f"{n_rows}x{n_cols:<10}{args.dtype:<7}"
                f"{t_un:>13.3f} {t_fu:>11.3f} "
                f"{speedup:>8.2f}x {unfused_gbps:>9.1f} GB/s "
                f"{fused_gbps:>9.1f} GB/s"
            )

    dist.barrier()
    # Note: skip destroy_process_group() to avoid SIGSEGV during cleanup
    # with symmetric memory + PyTorch nightly. Process exit handles cleanup.


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=torch.cuda.device_count())
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    args = parser.parse_args()

    mp.spawn(_worker, args=(args.world_size, args), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
