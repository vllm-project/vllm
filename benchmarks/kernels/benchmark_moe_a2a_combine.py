#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark the FlashInfer one-sided MoE all-to-all combine leg in BF16 vs FP8.

Measures the effect of VLLM_FLASHINFER_MOE_A2A_LOW_PRECISION_COMBINE, which
transmits expert outputs as fp8_e4m3 instead of bf16, halving the bytes that
cross NVLink on the combine leg.

The kernel enforces a dispatch -> combine state machine, so combine cannot be
timed on its own. This times the dispatch+combine pair under both transports.
Dispatch is byte-identical in each, so the difference between the two pair
times is attributable entirely to the combine leg.

Spawns its own workers (one per GPU) rather than relying on torchrun, mirroring
tests/distributed/test_mnnvl_alltoall.py so the distributed setup matches the
path that is known to work with the one-sided manager.

Usage:
    python3 benchmarks/kernels/benchmark_moe_a2a_combine.py

    # DeepSeek-R1 shapes, decode- through prefill-sized token counts
    python3 benchmarks/kernels/benchmark_moe_a2a_combine.py \
        --hidden-size 7168 --top-k 8 --num-experts 256 \
        --tokens-per-rank 8 32 128 512 2048
"""

import argparse
import os
import statistics
import sys
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument(
        "--tokens-per-rank",
        type=int,
        nargs="+",
        default=[8, 32, 128, 512, 2048],
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Ranks to launch. Defaults to the visible GPU count.",
    )
    parser.add_argument(
        "--fp8-first",
        action="store_true",
        help="Run the fp8 configuration before bf16. Run both ways: a saving "
        "that follows position rather than dtype is a warmup artifact.",
    )
    return parser.parse_args()


def _init_dp(
    world_size: int,
    rank: int,
    local_rank: int,
    master_addr: str,
    port: str,
    dp_port: str,
) -> None:
    """Initialize tp=pp=1 with data_parallel_size=world_size.

    Mirrors _init_dp_environment in tests/distributed/test_mnnvl_alltoall.py.
    """
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.parallel import ParallelConfig
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )

    vllm_config = VllmConfig()
    vllm_config.parallel_config = ParallelConfig(
        data_parallel_size=world_size,
        data_parallel_rank=rank,
        # Defaults to 127.0.0.1, which strands ranks on other nodes.
        data_parallel_master_ip=master_addr,
        _data_parallel_master_port_list=[int(dp_port)],
    )
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,  # tp * pp = 1; each process is one DP rank
            rank=0,
            distributed_init_method=f"tcp://{master_addr}:{port}",
            local_rank=local_rank,
        )
        ensure_model_parallel_initialized(1, 1)


def make_manager(low_precision: bool, args):
    """Build a one-sided manager with the fp8-combine flag set or cleared."""
    from vllm.distributed.device_communicators.all2all import (
        FlashInferNVLinkOneSidedManager,
    )
    from vllm.distributed.parallel_state import get_dp_group

    # envs.__getattr__ is uncached by default, so initialize() picks this up.
    os.environ["VLLM_FLASHINFER_MOE_A2A_LOW_PRECISION_COMBINE"] = (
        "1" if low_precision else "0"
    )

    manager = FlashInferNVLinkOneSidedManager(get_dp_group().cpu_group)
    manager.initialize(
        max_num_tokens=max(args.tokens_per_rank),
        top_k=args.top_k,
        num_experts=args.num_experts,
        hidden_size=args.hidden_size,
        # nvfp4 activations plus their fp8 block scales, matching the payload
        # layout the mnnvl tests exercise.
        x_bytes_per_token=args.hidden_size // 2,
        x_sf_bytes_per_token=args.hidden_size // 16,
    )
    assert manager.low_precision_combine == low_precision, (
        f"requested low_precision={low_precision} but the manager resolved to "
        f"{manager.low_precision_combine}; the installed FlashInfer build "
        "likely has no `use_low_precision` parameter on MoeAlltoAll.combine()"
    )
    return manager


def make_inputs(args, tokens: int, world_size: int, rank: int, device):
    torch.manual_seed(rank + 42)
    hidden = args.hidden_size
    x = torch.randint(0, 256, (tokens, hidden // 2), device=device, dtype=torch.uint8)
    x_sf = torch.randint(
        0, 256, (tokens, hidden // 16), device=device, dtype=torch.uint8
    )
    topk_ids = torch.randint(
        0, args.num_experts, (tokens, args.top_k), device=device, dtype=torch.int32
    )
    topk_weights = torch.rand(tokens, args.top_k, device=device, dtype=torch.float32)
    expert_output = torch.ones(
        world_size, tokens, hidden, device=device, dtype=torch.bfloat16
    )
    return [x, x_sf, topk_ids, topk_weights], topk_ids, expert_output


def check_correctness(
    manager, payloads, topk_ids, expert_output, args, tokens, world_size, device
) -> None:
    """Combine of all-ones expert output must equal the distinct-rank count.

    Integers 1..top_k are exactly representable in fp8_e4m3, so this holds for
    both transports. A workspace/dtype mismatch corrupts the buffer silently,
    so this must pass before any timing is trusted.
    """
    manager.moe_alltoall.dispatch(
        token_selected_experts=topk_ids,
        input_payloads=payloads,
        runtime_max_tokens_per_rank=tokens,
    )
    output = torch.empty(tokens, args.hidden_size, device=device, dtype=torch.bfloat16)
    manager.combine_into(
        payload=expert_output,
        runtime_max_tokens_per_rank=tokens,
        output=output,
    )
    experts_per_rank = args.num_experts // world_size
    expert_ranks = topk_ids // experts_per_rank
    num_distinct = torch.tensor(
        [len(set(row.tolist())) for row in expert_ranks],
        device=device,
        dtype=torch.bfloat16,
    ).unsqueeze(1)
    torch.testing.assert_close(output, num_distinct.expand_as(output))


def time_op(fn, warmup: int, iters: int, group) -> float:
    """Median over iterations of the per-iteration max across ranks.

    A rank-local time is not a distributed latency: a collective is only done
    when its slowest participant is, so each sample is MAX-reduced across ranks
    before the median. Collectives are scoped to the DP group because the
    default process group here is per-rank and would make them no-ops.
    """
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()

    samples = []
    for _ in range(iters):
        dist.barrier(group=group)
        torch.accelerator.synchronize()
        start = time.perf_counter()
        fn()
        torch.accelerator.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)  # microseconds

    local = torch.tensor(samples, device="cuda", dtype=torch.float64)
    dist.all_reduce(local, op=dist.ReduceOp.MAX, group=group)
    return statistics.median(local.tolist())


def benchmark_tokens(manager, args, tokens, world_size, rank, device, group):
    """Return the dispatch+combine time for one token count, in microseconds.

    The kernel enforces a dispatch -> combine state machine, so dispatch cannot
    be timed on its own. Dispatch is byte-identical in both configurations, so
    the difference between the bf16 and fp8 pair times is attributable entirely
    to the combine leg.
    """
    payloads, topk_ids, expert_output = make_inputs(
        args, tokens, world_size, rank, device
    )
    check_correctness(
        manager, payloads, topk_ids, expert_output, args, tokens, world_size, device
    )
    output = torch.empty(tokens, args.hidden_size, device=device, dtype=torch.bfloat16)

    def dispatch_and_combine():
        manager.moe_alltoall.dispatch(
            token_selected_experts=topk_ids,
            input_payloads=payloads,
            runtime_max_tokens_per_rank=tokens,
        )
        manager.combine_into(
            payload=expert_output,
            runtime_max_tokens_per_rank=tokens,
            output=output,
        )

    return time_op(dispatch_and_combine, args.warmup, args.iters, group)


def _run_bench(
    rank, world_size, local_rank, master_addr, port, dp_port, args, err_queue=None
):
    """Run the full sweep on one rank. Rank 0 prints the table."""
    try:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        torch.accelerator.set_device_index(local_rank)
        _init_dp(world_size, rank, local_rank, master_addr, port, dp_port)

        from vllm.distributed.parallel_state import get_dp_group

        device = torch.device(f"cuda:{local_rank}")
        group = get_dp_group().device_group

        results = {}
        order = (True, False) if args.fp8_first else (False, True)
        for low_precision in order:
            manager = make_manager(low_precision, args)
            try:
                for tokens in args.tokens_per_rank:
                    results[low_precision, tokens] = benchmark_tokens(
                        manager, args, tokens, world_size, rank, device, group
                    )
                    dist.barrier(group=group)
            finally:
                manager.cleanup()
            dist.barrier(group=group)

        if rank == 0:
            report(results, args, world_size)
    except Exception:
        tb = f"[Rank {rank}]\n{traceback.format_exc()}"
        print(tb, file=sys.stderr, flush=True)
        if err_queue is not None:
            err_queue.put(tb)
        sys.exit(1)


def report(results, args, world_size) -> None:
    import flashinfer

    print(
        f"\nworld_size={world_size} gpu={torch.get_device_module().get_device_name(0)}"
    )
    print(f"flashinfer={flashinfer.__version__} torch={torch.__version__}")
    print(
        f"hidden={args.hidden_size} top_k={args.top_k} "
        f"experts={args.num_experts} warmup={args.warmup} iters={args.iters}\n"
    )
    header = (
        f"{'tokens':>7} {'bf16 pair':>11} {'fp8 pair':>11} "
        f"{'saved':>10} {'speedup':>8} {'saved GB/s':>11}"
    )
    print(header)
    print("-" * len(header))
    for tokens in args.tokens_per_rank:
        bf16 = results[False, tokens]
        fp8 = results[True, tokens]
        saved = bf16 - fp8
        # fp8 halves the combine payload, so this many bytes never cross NVLink.
        bytes_saved = world_size * tokens * args.hidden_size
        saved_gbs = bytes_saved / saved / 1e3 if saved > 0 else float("nan")
        speedup = bf16 / fp8 if fp8 > 0 else float("nan")
        print(
            f"{tokens:>7} {bf16:>10.1f}us {fp8:>10.1f}us "
            f"{saved:>9.1f}us {speedup:>7.2f}x {saved_gbs:>11.1f}"
        )
    print("\ntimes are dispatch+combine; the kernel forbids timing dispatch alone")
    print("dispatch is identical in both, so 'saved' is the combine-leg gain")
    print("'saved GB/s' = bytes fp8 avoided sending / time saved")


def main() -> None:
    args = parse_args()

    # Multi-node: srun launches one task per rank, so just be that rank.
    # SLURM_PROCID/LOCALID/NTASKS are set by srun; the ports come from the
    # submitting script so every rank agrees on them.
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        _run_bench(
            rank=int(os.environ["SLURM_PROCID"]),
            world_size=int(os.environ["SLURM_NTASKS"]),
            local_rank=int(os.environ["SLURM_LOCALID"]),
            master_addr=os.environ["MASTER_ADDR"],
            port=os.environ["BENCH_PORT"],
            dp_port=os.environ["BENCH_DP_PORT"],
            args=args,
        )
        return

    # Single node: spawn one worker per GPU ourselves.
    from vllm.utils.network_utils import get_open_port

    world_size = args.world_size or torch.accelerator.device_count()
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    port = str(get_open_port())
    dp_port = str(get_open_port())
    err_queue: mp.Queue = mp.Queue()

    procs = []
    for rank in range(world_size):
        p = mp.Process(
            target=_run_bench,
            args=(rank, world_size, rank, "localhost", port, dp_port, args, err_queue),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    errors = []
    while not err_queue.empty():
        errors.append(err_queue.get_nowait())
    if errors:
        raise SystemExit("Worker(s) failed:\n" + "\n---\n".join(errors))


if __name__ == "__main__":
    main()
