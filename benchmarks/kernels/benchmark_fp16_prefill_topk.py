# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark FP32 against reduced-precision prefill top-k on model logits."""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import torch

from vllm import _custom_ops as ops

_TOPK = 2048
_DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _time_prefill(
    logits: torch.Tensor,
    row_ends: torch.Tensor,
    reference: torch.Tensor,
    nodes: int,
    repeats: int,
) -> float:
    num_rows = logits.shape[0]
    row_starts = torch.zeros_like(row_ends)
    output = torch.empty_like(reference)

    def launch_once() -> None:
        ops.top_k_per_row_prefill(
            logits,
            row_starts,
            row_ends,
            output,
            num_rows,
            logits.stride(0),
            logits.stride(1),
            _TOPK,
        )

    launch_once()
    torch.cuda.synchronize()
    actual_values = logits.gather(1, output.long()).sort(1).values
    reference_values = logits.gather(1, reference.long()).sort(1).values
    torch.testing.assert_close(actual_values, reference_values, rtol=0, atol=0)

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("--pattern", default="gvr_b*_call*.pt")
    parser.add_argument("--target-batches")
    parser.add_argument(
        "--reduced-dtypes",
        default="fp16,bf16",
        help="Comma-separated reduced dtypes to compare with FP32.",
    )
    parser.add_argument("--nodes", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()

    target_batches = (
        None
        if args.target_batches is None
        else [int(value) for value in args.target_batches.split(",")]
    )
    reduced_dtypes = [_DTYPES[name] for name in args.reduced_dtypes.split(",")]
    paths = sorted(args.capture_dir.glob(args.pattern))
    if not paths:
        raise ValueError(f"no captures found in {args.capture_dir}")

    results: dict[tuple[int, int, torch.dtype], list[float]] = defaultdict(list)
    overlaps: dict[tuple[int, int, torch.dtype], list[float]] = defaultdict(list)
    for path in paths:
        tensors = torch.load(path, map_location="cuda", weights_only=True)
        captured_logits = tensors["logits"]
        captured_batch = captured_logits.shape[0]
        actual_length = int(tensors["seq_lens"].max())
        kv_length = int(
            tensors.get(
                "target_kv_len",
                min(
                    (10000, 50000, 100000, 200000),
                    key=lambda value: abs(value - actual_length),
                ),
            )
        )
        for batch in target_batches or [captured_batch]:
            if batch % captured_batch:
                raise ValueError(
                    f"target batch {batch} is not a multiple of {captured_batch}"
                )
            copies = batch // captured_batch
            row_ends = tensors["seq_lens"].repeat(copies)
            references: dict[torch.dtype, torch.Tensor] = {}
            for dtype in (torch.float32, *reduced_dtypes):
                logits = captured_logits.to(dtype).repeat(copies, 1)
                column = torch.arange(logits.shape[1], device=logits.device)
                valid_logits = logits.masked_fill(
                    column >= row_ends[:, None], float("-inf")
                )
                reference = torch.topk(valid_logits, _TOPK, dim=1).indices.to(
                    torch.int32
                )
                references[dtype] = reference
                latency = _time_prefill(
                    logits, row_ends, reference, args.nodes, args.repeats
                )
                results[kv_length, batch, dtype].append(latency)
                print(
                    f"{path.name} kv={kv_length} batch={batch} "
                    f"dtype={dtype} prefill_topk={latency:.3f} us",
                    flush=True,
                )

            fp32_sets = [set(row.tolist()) for row in references[torch.float32].cpu()]
            for dtype in reduced_dtypes:
                reduced_sets = [set(row.tolist()) for row in references[dtype].cpu()]
                overlaps[kv_length, batch, dtype].extend(
                    len(fp32 & reduced) / _TOPK
                    for fp32, reduced in zip(fp32_sets, reduced_sets, strict=True)
                )

    print("aggregate")
    pairs = sorted({(kv_length, batch) for kv_length, batch, _ in results})
    for kv_length, batch in pairs:
        fp32 = results[kv_length, batch, torch.float32]
        fp32_mean = sum(fp32) / len(fp32)
        for dtype in reduced_dtypes:
            reduced = results[kv_length, batch, dtype]
            reduced_mean = sum(reduced) / len(reduced)
            paired = [old / new for old, new in zip(fp32, reduced, strict=True)]
            overlap = sum(overlaps[kv_length, batch, dtype]) / len(
                overlaps[kv_length, batch, dtype]
            )
            print(
                f"kv={kv_length} batch={batch} fp32={fp32_mean:.3f} us "
                f"{dtype}={reduced_mean:.3f} us "
                f"speedup={fp32_mean / reduced_mean:.3f}x "
                f"paired-gm="
                f"{math.exp(sum(map(math.log, paired)) / len(paired)):.3f}x "
                f"fp32-index-overlap={overlap:.6%}"
            )


if __name__ == "__main__":
    main()
