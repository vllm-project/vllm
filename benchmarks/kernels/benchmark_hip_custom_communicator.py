# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark HipCustomCommunicator's ops against vLLM's own path.

Each op is timed against what vLLM runs without it: the TP all-reduce, then
vLLM's RMSNorm or Kimi-K3's AttnRes, timed as one call. Both are captured in a
CUDA graph. Every replay is one sample.

Usage:
    torchrun --nproc_per_node=8 benchmark_hip_custom_communicator.py
    torchrun --nproc_per_node=8 benchmark_hip_custom_communicator.py \
        --quantization int4 --num-tokens 1 64 1024 --output-jsonl int4.jsonl
"""

import argparse
import dataclasses
import json
import os
import statistics
from collections.abc import Callable
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.distributed.device_communicators.hip_custom_communicator import (
        HipCustomCommunicator,
    )

OPS = ("all_reduce", "all_reduce_add_rms_norm", "all_reduce_add_attn_res_rms_norm")
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
EPS = 1e-5
NUM_BLOCKS = 4
TOTAL_BLOCKS = 10
WRITE_INDEX = -1


def time_graph(
    fn: Callable[[], object],
    device: torch.device,
    comm: "HipCustomCommunicator | None",
    warmup: int,
    replays: int,
    cycles: int,
) -> list[float]:
    """Capture `fn` `cycles` times in a CUDA graph. Returns us per call, a
    sample per replay. `comm` is the HipCustomCommunicator whose capture wraps
    the graph, or None for vLLM's path alone."""
    from vllm.distributed.parallel_state import graph_capture

    fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with (
        graph_capture(device=device) as context,
        comm.capture() if comm is not None else nullcontext(),
        torch.cuda.graph(graph, stream=context.stream),
    ):
        for _ in range(cycles):
            fn()
    for _ in range(warmup):
        graph.replay()
    torch.accelerator.synchronize()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(replays)
    ]
    for start, end in events:
        start.record()
        graph.replay()
        end.record()
    torch.accelerator.synchronize()
    return [start.elapsed_time(end) * 1000 / cycles for start, end in events]


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--num-tokens",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
    )
    parser.add_argument("--hidden-dim", type=int, default=7168)
    parser.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    parser.add_argument("--ops", nargs="+", choices=OPS, default=list(OPS))
    parser.add_argument("--quantization", choices=["exact", "int4"], default="exact")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--output-jsonl", type=str, help="every sample, a shape a line")
    return parser.parse_args()


def main() -> None:
    args = parse_cli()

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "Run with torchrun: "
            "torchrun --nproc_per_node=8 benchmark_hip_custom_communicator.py"
        )
    # vLLM's path is the baseline: AITER's all-reduce, with QuickReduce INT4
    # when ours moves INT4.
    os.environ["VLLM_ROCM_USE_HIP_CUSTOM_COMMUNICATOR"] = "0"
    os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
    os.environ.setdefault(
        "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION",
        "INT4" if args.quantization == "int4" else "NONE",
    )

    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import tensor_model_parallel_all_reduce
    from vllm.distributed.device_communicators.hip_custom_communicator import (
        HipCustomCommunicator,
    )
    from vllm.distributed.parallel_state import (
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.ir.ops import fused_add_rms_norm, rms_norm
    from vllm.models.kimi_k3.amd.ops.attn_res import attn_res

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', rank))}")
    torch.accelerator.set_device_index(device)
    dtype = DTYPES[args.dtype]
    cols = args.hidden_dim
    # As served: CUDA graphs on decode, so vLLM's RMSNorm is the kernel it serves.
    served = VllmConfig(
        compilation_config=CompilationConfig(cudagraph_mode="FULL_DECODE_ONLY")
    )
    priority = served.kernel_config.ir_op_priority
    shapes = [
        (op, residual, tokens)
        for op in args.ops
        for residual in ([False, True] if op == "all_reduce_add_rms_norm" else [None])
        for tokens in args.num_tokens
    ]

    with set_current_vllm_config(served), priority.set_priority():
        init_distributed_environment()
        initialize_model_parallel(tensor_model_parallel_size=world)
        group = get_tp_group()
        comm = HipCustomCommunicator(
            group.cpu_group,
            group.device_group,
            device,
            max_rows=max(args.num_tokens),
            max_cols=cols,
            quantization=args.quantization,
        )
        assert not comm.disabled, comm.lifecycle
        if rank == 0:
            print(
                f"world {world}, hidden {cols}, {args.dtype}, {args.quantization}, "
                f"baseline rms_norm {priority.rms_norm}"
            )
            print(
                f"{'op':<34} {'residual':>8} {'tokens':>6} {'ours p50':>9} "
                f"{'base p50':>9} {'ratio':>6}  config"
            )

        write = rank == 0 and args.output_jsonl is not None
        if write:
            with open(args.output_jsonl, "w"):
                pass
        try:
            for op, residual, tokens in shapes:
                g = torch.Generator(device=device).manual_seed(rank)
                x = torch.randn(tokens, cols, generator=g, device=device)
                x = x.mul(1e-3).to(dtype)
                weight = torch.ones(cols, dtype=dtype, device=device)
                match op:
                    case "all_reduce":
                        admitted = comm.should_allreduce(x)

                        def ours(x=x):
                            return comm.all_reduce(x)

                        def baseline(x=x):
                            return tensor_model_parallel_all_reduce(x)

                    case "all_reduce_add_rms_norm":
                        r = (
                            torch.randn(tokens, cols, generator=g, device=device).to(
                                dtype
                            )
                            if residual
                            else None
                        )
                        admitted = comm.should_allreduce_add_rms_norm(x, weight, r)

                        def ours(x=x, weight=weight, r=r):
                            return comm.all_reduce_add_rms_norm(
                                x, weight, EPS, residual=r
                            )

                        def baseline(x=x, weight=weight, r=r):
                            summed = tensor_model_parallel_all_reduce(x)
                            if r is None:
                                return rms_norm(summed, weight, EPS)
                            return fused_add_rms_norm(summed, r, weight, EPS)

                    case "all_reduce_add_attn_res_rms_norm":
                        prefix = torch.randn(tokens, cols, generator=g, device=device)
                        prefix = prefix.to(dtype)
                        blocks = torch.randn(
                            tokens, TOTAL_BLOCKS, cols, generator=g, device=device
                        ).to(dtype)
                        qk = torch.randn(cols, generator=g, device=device)
                        qk = qk.mul(0.1).to(dtype)
                        admitted = comm.should_allreduce_add_attn_res_rms_norm(
                            x, prefix, blocks, weight, qk, weight, NUM_BLOCKS,
                            WRITE_INDEX,
                        )  # fmt: skip

                        def ours(
                            x=x, prefix=prefix, blocks=blocks, weight=weight, qk=qk
                        ):
                            return comm.all_reduce_add_attn_res_rms_norm(
                                x, prefix, blocks, weight, qk, weight, NUM_BLOCKS,
                                WRITE_INDEX, EPS, EPS,
                            )  # fmt: skip

                        def baseline(
                            x=x, prefix=prefix, blocks=blocks, weight=weight, qk=qk
                        ):
                            summed = tensor_model_parallel_all_reduce(x)
                            return attn_res(
                                prefix, summed, blocks, weight, qk, weight, NUM_BLOCKS,
                                WRITE_INDEX, EPS, EPS,
                            )  # fmt: skip

                if not admitted:
                    if rank == 0:
                        print(f"{op:<34} {residual!s:>8} {tokens:>6}  refused")
                    continue
                config = comm.tuned(op, tokens, cols, residual)
                ours_us = time_graph(
                    ours, device, comm, args.warmup, args.replays, args.cycles
                )
                baseline_us = time_graph(
                    baseline, device, None, args.warmup, args.replays, args.cycles
                )
                row = {
                    "op": op,
                    "residual": residual,
                    "tokens": tokens,
                    "hidden": cols,
                    "dtype": args.dtype,
                    "world": world,
                    "quantization": args.quantization,
                    "config": dataclasses.asdict(config),
                    "baseline_rms_norm": priority.rms_norm,
                    "ours_us": ours_us,
                    "baseline_us": baseline_us,
                }
                if write:
                    with open(args.output_jsonl, "a") as f:
                        f.write(json.dumps(row) + "\n")
                if rank == 0:
                    ours_p50 = statistics.median(ours_us)
                    base_p50 = statistics.median(baseline_us)
                    print(
                        f"{op:<34} {residual!s:>8} {tokens:>6} {ours_p50:>9.2f} "
                        f"{base_p50:>9.2f} {ours_p50 / base_p50:>6.3f}  "
                        f"{config.algorithm} {config.quantization}"
                    )
        finally:
            comm.close()


if __name__ == "__main__":
    main()
