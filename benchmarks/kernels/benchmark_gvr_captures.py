# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark GVR kernel variants on captured model logits."""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import torch

from vllm.models.deepseek_v32.nvidia.ops.gvr_topk import (
    prepare_gvr_hints,
    store_decode_gvr_state,
)
from vllm.models.deepseek_v32.nvidia.ops.gvr_topk_cutedsl import GvrTopKKernel

_TOPK = 2048
_RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024
_CONFIGS: dict[str, dict] = {
    "default": {},
    "separate_state_store": {"_state_mode": "separate"},
    "current_pipeline": {"_state_mode": "current"},
    "fused_state_store": {"_state_mode": "fused"},
    "fused_hint_and_state": {"_state_mode": "full"},
    "glm_rungs3": {
        "r0_qfracs": (0.35, 0.05, 0.01),
        "r0_vseed": False,
    },
    "glm_rungs3_vseed": {
        "r0_qfracs": (0.35, 0.05, 0.01),
        "r0_vseed": True,
    },
    "glm_rungs2": {
        "r0_qfracs": (0.35, 0.05),
        "r0_vseed": False,
    },
    "secant": {"enable_r0": False},
    "cluster1": {
        "cluster_size": 1,
        "num_threads": 1024,
        "enable_warp_parallel_reduce": True,
    },
    "cluster2": {"cluster_size": 2},
    "cluster8": {"cluster_size": 8},
    "cluster16": {"cluster_size": 16},
    "baseline_shape_cs4": {
        "cluster_size": 4,
        "num_threads": 1024,
        "min_blocks_per_mp": 1,
        "enable_warp_parallel_reduce": True,
    },
    "baseline_shape_cs8": {
        "cluster_size": 8,
        "num_threads": 1024,
        "min_blocks_per_mp": 1,
        "enable_warp_parallel_reduce": True,
    },
    "baseline_shape_cs16": {
        "cluster_size": 16,
        "num_threads": 1024,
        "min_blocks_per_mp": 1,
        "enable_warp_parallel_reduce": True,
    },
    "smem_cache": {
        "enable_smem_cache": True,
        "smem_cache_elems": 32768,
    },
    "constant_load": {"use_constant_hint": True},
    "load128": {"use_256bit_load": False},
    "threads1024": {
        "num_threads": 1024,
        "enable_warp_parallel_reduce": True,
    },
    "no_count_unroll": {"enable_unroll_4": False},
    "no_collect_unroll": {"enable_phase3_unroll": False},
    "p2_leader": {"p2_warp_redundant": False},
    "p4_leader": {"p4_warp_redundant": False},
    "p4_snap": {"enable_p4_rank_scatter": False},
    "p4_snap_load128": {
        "enable_p4_rank_scatter": False,
        "use_256bit_load": False,
    },
    "p4_snap_smem": {
        "enable_p4_rank_scatter": False,
        "enable_smem_cache": True,
        "smem_cache_elems": 32768,
    },
    "p4_snap_threads1024": {
        "enable_p4_rank_scatter": False,
        "num_threads": 1024,
        "enable_warp_parallel_reduce": True,
    },
}


def _time_variant(
    logits: torch.Tensor,
    hints: torch.Tensor,
    seq_lens: torch.Tensor,
    reference: torch.Tensor,
    overrides: dict,
    nodes: int,
    repeats: int,
) -> float:
    overrides = dict(overrides)
    state_mode = overrides.pop("_state_mode", None)
    output = torch.empty_like(reference)
    previous = None
    state_valid = None
    request_indices = None
    prepared_hints = None
    if state_mode is not None:
        previous = torch.empty_like(reference)
        state_valid = torch.zeros(
            logits.shape[0], dtype=torch.bool, device=logits.device
        )
        request_indices = torch.arange(
            logits.shape[0], dtype=torch.int64, device=logits.device
        )
        if state_mode == "full":
            previous.copy_(hints)
            state_valid.fill_(True)
        elif state_mode == "current":
            previous.copy_(hints)
            state_valid.fill_(True)
            prepared_hints = torch.empty_like(hints)

    def launch_once() -> None:
        if state_mode in ("fused", "full"):
            GvrTopKKernel.launch(
                logits,
                hints,
                seq_lens,
                output,
                _TOPK,
                previous_topk=previous,
                state_valid=state_valid,
                request_indices=request_indices,
                fuse_hint_prepare=state_mode == "full",
                **overrides,
            )
        else:
            kernel_hints = hints
            if state_mode == "current":
                kernel_hints = prepare_gvr_hints(
                    previous,
                    state_valid,
                    request_indices,
                    seq_lens,
                    prepared_hints,
                )
            GvrTopKKernel.launch(
                logits, kernel_hints, seq_lens, output, _TOPK, **overrides
            )
            if state_mode in ("separate", "current"):
                store_decode_gvr_state(output, request_indices, previous, state_valid)

    launch_once()
    torch.cuda.synchronize()

    actual_values = logits.gather(1, output.long()).sort(1).values
    reference_values = logits.gather(1, reference.long()).sort(1).values
    torch.testing.assert_close(actual_values, reference_values, rtol=0, atol=0)
    if state_mode is not None:
        torch.testing.assert_close(previous, output, rtol=0, atol=0)
        assert state_valid.all()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(nodes):
            launch_once()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (nodes * repeats)


def _time_baseline(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    reference: torch.Tensor,
    nodes: int,
    repeats: int,
) -> tuple[float, int]:
    output = torch.empty_like(reference)
    workspace = torch.empty(
        _RADIX_TOPK_WORKSPACE_SIZE, dtype=torch.uint8, device=logits.device
    )
    max_seq_len = int(seq_lens.max().item())

    def launch_once() -> None:
        if logits.shape[0] <= 32:
            torch.ops._C.cooperative_topk(
                logits,
                seq_lens,
                output,
                workspace,
                _TOPK,
                max_seq_len,
            )
        else:
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                output,
                workspace,
                _TOPK,
                max_seq_len,
            )

    launch_once()
    torch.cuda.synchronize()
    actual_values = logits.gather(1, output.long()).sort(1).values
    reference_values = logits.gather(1, reference.long()).sort(1).values
    mismatched_rows = int((actual_values != reference_values).any(dim=1).sum().item())
    if logits.shape[0] > 32 and mismatched_rows:
        raise AssertionError(
            f"persistent_topk mismatched the reference on {mismatched_rows} rows"
        )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(nodes):
            launch_once()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    latency = start.elapsed_time(end) * 1000 / (nodes * repeats)
    return latency, mismatched_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("--nodes", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--pattern", default="gvr_b*_call*.pt")
    parser.add_argument(
        "--dtypes",
        default="float32",
        help="Comma-separated input dtypes: float32,float16,bfloat16.",
    )
    parser.add_argument(
        "--target-batches",
        help="Comma-separated batches made by repeating each captured batch.",
    )
    parser.add_argument(
        "--fp32-baseline",
        action="store_true",
        help="Also benchmark FP32 baseline (alias for --baseline-dtypes=float32).",
    )
    parser.add_argument(
        "--baseline-dtypes",
        default="",
        help=("Comma-separated production baseline dtypes: float32,float16,bfloat16."),
    )
    parser.add_argument(
        "--configs",
        default=",".join(_CONFIGS),
        help="Comma-separated variant names; default is always included.",
    )
    args = parser.parse_args()
    config_names = list(dict.fromkeys(("default", *args.configs.split(","))))
    unknown = set(config_names) - _CONFIGS.keys()
    if unknown:
        raise ValueError(f"unknown configs: {sorted(unknown)}")

    dtype_by_name = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype_names = [name for name in dict.fromkeys(args.dtypes.split(",")) if name]
    unknown_dtypes = set(dtype_names) - dtype_by_name.keys()
    if unknown_dtypes:
        raise ValueError(f"unknown dtypes: {sorted(unknown_dtypes)}")
    baseline_dtype_names = [name for name in args.baseline_dtypes.split(",") if name]
    if args.fp32_baseline:
        baseline_dtype_names.append("float32")
    baseline_dtype_names = list(dict.fromkeys(baseline_dtype_names))
    unknown_baseline_dtypes = set(baseline_dtype_names) - dtype_by_name.keys()
    if unknown_baseline_dtypes:
        raise ValueError(f"unknown baseline dtypes: {sorted(unknown_baseline_dtypes)}")
    target_batches = (
        None
        if args.target_batches is None
        else [int(value) for value in args.target_batches.split(",")]
    )

    results: dict[tuple[int, int, str, str], list[float]] = defaultdict(list)
    baseline_results: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    baseline_mismatches: dict[tuple[int, int, str], int] = defaultdict(int)
    paths = sorted(args.capture_dir.glob(args.pattern))
    if not paths:
        raise ValueError(f"no captures found in {args.capture_dir}")

    for path in paths:
        tensors = torch.load(path, map_location="cuda", weights_only=True)
        captured_batch = tensors["logits"].shape[0]
        if "target_kv_len" in tensors:
            kv_length = int(tensors["target_kv_len"])
        else:
            actual_length = int(tensors["seq_lens"].max())
            nominal_lengths = (10000, 50000, 100000, 200000)
            kv_length = min(
                nominal_lengths, key=lambda value: abs(value - actual_length)
            )
        batches = target_batches or [captured_batch]
        for batch in batches:
            if batch % captured_batch != 0:
                raise ValueError(
                    f"target batch {batch} is not a multiple of {captured_batch}"
                )
            copies = batch // captured_batch
            hints = tensors["hints"].repeat(copies, 1)
            seq_lens = tensors["seq_lens"].repeat(copies)
            for dtype_name in baseline_dtype_names:
                dtype = dtype_by_name[dtype_name]
                captured_logits = tensors["logits"].to(dtype)
                logits = captured_logits.repeat(copies, 1)
                column = torch.arange(
                    captured_logits.shape[1], device=captured_logits.device
                )
                valid_logits = captured_logits.masked_fill(
                    column >= tensors["seq_lens"][:, None], float("-inf")
                )
                captured_reference = torch.topk(valid_logits, _TOPK, dim=1).indices.to(
                    torch.int32
                )
                reference = captured_reference.repeat(copies, 1)
                latency, mismatched_rows = _time_baseline(
                    logits,
                    seq_lens,
                    reference,
                    args.nodes,
                    args.repeats,
                )
                baseline_results[kv_length, batch, dtype_name].append(latency)
                baseline_mismatches[kv_length, batch, dtype_name] += mismatched_rows
                backend = "cooperative" if batch <= 32 else "persistent"
                print(
                    f"{path.name} kv={kv_length} batch={batch} dtype={dtype_name} "
                    f"baseline_{backend}: {latency:.3f} us "
                    f"mismatched_rows={mismatched_rows}",
                    flush=True,
                )
            for dtype_name in dtype_names:
                dtype = dtype_by_name[dtype_name]
                captured_logits = tensors["logits"].to(dtype)
                logits = captured_logits.repeat(copies, 1)
                column = torch.arange(
                    captured_logits.shape[1], device=captured_logits.device
                )
                valid_logits = captured_logits.masked_fill(
                    column >= tensors["seq_lens"][:, None], float("-inf")
                )
                captured_reference = torch.topk(valid_logits, _TOPK, dim=1).indices.to(
                    torch.int32
                )
                reference = captured_reference.repeat(copies, 1)
                for name in config_names:
                    overrides = _CONFIGS[name]
                    latency = _time_variant(
                        logits,
                        hints,
                        seq_lens,
                        reference,
                        overrides,
                        args.nodes,
                        args.repeats,
                    )
                    results[kv_length, batch, dtype_name, name].append(latency)
                    print(
                        f"{path.name} kv={kv_length} batch={batch} "
                        f"dtype={dtype_name} "
                        f"{name}: {latency:.3f} us",
                        flush=True,
                    )

    print("aggregate")
    result_groups = sorted(
        {(kv_length, batch, dtype) for kv_length, batch, dtype, _ in results}
    )
    for kv_length, batch, dtype_name in result_groups:
        defaults = results[kv_length, batch, dtype_name, "default"]
        baseline = sum(defaults) / len(defaults)
        for name in config_names:
            latencies = results[kv_length, batch, dtype_name, name]
            mean = sum(latencies) / len(latencies)
            speedups = [
                default / current for default, current in zip(defaults, latencies)
            ]
            geomean = math.exp(sum(map(math.log, speedups)) / len(speedups))
            print(
                f"kv={kv_length} batch={batch} dtype={dtype_name} {name}: "
                f"mean={mean:.3f} us "
                f"speedup={baseline / mean:.3f}x paired-gm={geomean:.3f}x"
            )
        if {
            "separate_state_store",
            "fused_state_store",
        }.issubset(config_names):
            separate = results[kv_length, batch, dtype_name, "separate_state_store"]
            fused = results[kv_length, batch, dtype_name, "fused_state_store"]
            paired = [old / new for old, new in zip(separate, fused)]
            print(
                f"kv={kv_length} batch={batch} dtype={dtype_name} "
                "state-store fusion: "
                f"speedup={sum(separate) / sum(fused):.3f}x "
                f"paired-gm={math.exp(sum(map(math.log, paired)) / len(paired)):.3f}x"
            )
        if {"current_pipeline", "fused_hint_and_state"}.issubset(config_names):
            current = results[kv_length, batch, dtype_name, "current_pipeline"]
            fused = results[kv_length, batch, dtype_name, "fused_hint_and_state"]
            paired = [old / new for old, new in zip(current, fused)]
            print(
                f"kv={kv_length} batch={batch} dtype={dtype_name} "
                "full pipeline fusion: "
                f"speedup={sum(current) / sum(fused):.3f}x "
                f"paired-gm={math.exp(sum(map(math.log, paired)) / len(paired)):.3f}x"
            )

    for kv_length, batch, baseline_dtype in sorted(baseline_results):
        baseline = baseline_results[kv_length, batch, baseline_dtype]
        baseline_mean = sum(baseline) / len(baseline)
        print(
            f"kv={kv_length} batch={batch} {baseline_dtype}_baseline: "
            f"mean={baseline_mean:.3f} us "
            "mismatched_rows="
            f"{baseline_mismatches[kv_length, batch, baseline_dtype]}"
        )
        key = kv_length, batch, baseline_dtype, "default"
        if key not in results:
            continue
        gvr = results[key]
        gvr_mean = sum(gvr) / len(gvr)
        paired = [old / new for old, new in zip(baseline, gvr)]
        print(
            f"kv={kv_length} batch={batch} {baseline_dtype}_baseline/gvr: "
            f"speedup={baseline_mean / gvr_mean:.3f}x "
            f"paired-gm={math.exp(sum(map(math.log, paired)) / len(paired)):.3f}x"
        )


if __name__ == "__main__":
    main()
