# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep QSA sparse-attention launch configs for one production-shaped batch.

``qsa._select_config`` picks ``(BLOCK_N, num_warps, num_tiles, num_splits)`` for
the Qwen4Exp QSA split-K kernel from an architecture-specific table keyed on
``base_programs = num_rows * num_kv_heads``. Re-measure the table with this
script before changing it for a new GPU or cache dtype.

The batch is production-shaped: 8192-token prefix, page 240, token top-k 2048
expanded by the real ``expand_qsa_block_indices`` kernel into 2051 selection
columns plus the trailing count column, random physical pages, every row real.
For each candidate ``(BLOCK_N, target_splits, num_warps)`` the selector is
replaced in this process only; the kernel body, ``num_stages`` and the merge
launch stay untouched. A candidate must match the in-tree config's output (which
the test suite checks against an FP32 reference) within the test tolerance, both
eagerly and after a CUDA-graph replay on changed inputs. Graph replays are then
timed over three candidate orders, hot and with an evicted L2. The in-tree config
is timed twice, so its second row bounds run-to-run noise.

    python benchmarks/kernels/benchmark_qsa_select_config.py \\
        --requests 128 --query-len 4 --group 3 --kv-heads 1 \\
        --candidates 64,4,2 32,8,1 32,4,1
"""

import argparse
import random
import statistics

import torch

from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops
from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import expand_qsa_block_indices
from vllm.triton_utils import triton

PREFIX_TOKENS = 8192
HEAD_DIM = 256
PAGE_SIZE = 240
COMPRESS_RATIO = 4
TOKEN_TOPK = 2048
SELECTION_WIDTH = TOKEN_TOPK + COMPRESS_RATIO - 1
TOLERANCE = dict(rtol=2e-2, atol=2e-2)  # as in the kernel's correctness test
DEFAULT_CANDIDATES = ["64,64,2", "32,64,4", "32,16,1", "32,8,1", "32,4,1", "64,4,2"]


def parse_candidate(text: str) -> tuple[int, int, int, int]:
    """``BLOCK_N,target_splits,num_warps`` -> the selector's 4-tuple."""
    block_n, target_splits, num_warps = (int(field) for field in text.split(","))
    num_tiles = triton.cdiv(SELECTION_WIDTH, block_n)
    return block_n, num_warps, num_tiles, min(target_splits, num_tiles)


def build_batch(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    """Production-shaped synthetic decode/verify batch on the current device."""
    rows = args.requests * args.query_len
    pages_per_request = triton.cdiv(PREFIX_TOKENS + args.query_len, PAGE_SIZE)
    num_pages = args.requests * pages_per_request
    block_topk = TOKEN_TOPK // COMPRESS_RATIO
    compressed_blocks = PREFIX_TOKENS // COMPRESS_RATIO + 1
    generator = torch.Generator().manual_seed(args.seed)

    positions = (PREFIX_TOKENS + torch.arange(args.query_len)).repeat(args.requests)
    visible = ((positions + 1) // COMPRESS_RATIO).to(torch.int32)
    # Per-request random priority over the compressed blocks; each row keeps the
    # blocks it can see, so the causal tail of a verify step differs per row.
    block_indices = torch.full((rows, block_topk), -1, dtype=torch.int32)
    for request in range(args.requests):
        order = torch.randperm(compressed_blocks, generator=generator)
        for step in range(args.query_len):
            row = request * args.query_len + step
            selected = order[order < visible[row]][:block_topk]
            block_indices[row, : selected.numel()] = selected.to(torch.int32)

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    q = torch.randn(
        rows, args.group * args.kv_heads, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    kv_cache = torch.randn(
        num_pages, PAGE_SIZE, args.kv_heads, 2 * HEAD_DIM, device=device
    ).to(torch.bfloat16)
    k_cache, v_cache = kv_cache.split(HEAD_DIM, dim=-1)
    k_scale, v_scale = 1.0, 1.0
    if args.kv_dtype == "fp8":
        # Stored values are the scaled ones, as reshape_and_cache writes them.
        k_scale, v_scale = 0.5, 2.0
        k_cache = (k_cache / k_scale).to(torch.float8_e4m3fn)
        v_cache = (v_cache / v_scale).to(torch.float8_e4m3fn)
    packed = torch.empty(rows, SELECTION_WIDTH + 1, device=device, dtype=torch.int32)
    expand_qsa_block_indices(
        block_indices.to(device),
        positions.to(device),
        visible.to(device),
        COMPRESS_RATIO,
        TOKEN_TOPK,
        packed,
    )
    return {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "packed": packed,
        "block_table": torch.randperm(num_pages, generator=generator)
        .reshape(args.requests, pages_per_request)
        .to(torch.int32)
        .to(device),
        "token_to_req": torch.arange(args.requests, dtype=torch.int32)
        .repeat_interleave(args.query_len)
        .to(device),
        "output_gate": torch.randn_like(q),
        "out": torch.empty_like(q),
        "k_scale": torch.tensor(k_scale),
        "v_scale": torch.tensor(v_scale),
    }


def run_attention(batch: dict[str, torch.Tensor], prefill: bool) -> torch.Tensor:
    return qsa_ops.qsa_sparse_paged_attention(
        batch["q"],
        batch["k_cache"],
        batch["v_cache"],
        batch["packed"],
        batch["block_table"],
        batch["token_to_req"],
        prefill,
        out=batch["out"],
        k_scale=float(batch["k_scale"]),
        v_scale=float(batch["v_scale"]),
        output_gate=batch["output_gate"],
    )


def capture(batch: dict[str, torch.Tensor], prefill: bool) -> torch.cuda.CUDAGraph:
    """Warm the kernel on a side stream, then capture one call into a graph."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run_attention(batch, prefill)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run_attention(batch, prefill)
    torch.accelerator.synchronize()
    return graph


def replay_us(graph: torch.cuda.CUDAGraph, evict: torch.Tensor | None) -> float:
    """Microseconds for one replay; ``evict`` zeroes an L2-sized buffer first."""
    if evict is not None:
        evict.zero_()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument(
        "--query-len", type=int, default=1, help="rows per request (1 + MTP tokens)"
    )
    parser.add_argument("--group", type=int, default=3, help="query heads per kv head")
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--kv-dtype", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--prefill", action="store_true", help="use_prefill_config")
    parser.add_argument("--candidates", nargs="+", default=DEFAULT_CANDIDATES)
    parser.add_argument("--replays", type=int, default=9)
    parser.add_argument("--seed", type=int, default=913256)
    args = parser.parse_args()

    rows = args.requests * args.query_len
    is_fp8 = args.kv_dtype == "fp8"
    current = qsa_ops._select_config(
        rows, args.kv_heads, args.prefill, SELECTION_WIDTH, is_fp8
    )
    arms = [("current", current), ("current (repeat)", current)]
    arms += [(text, parse_candidate(text)) for text in args.candidates]
    properties = torch.cuda.get_device_properties(0)
    print(f"{properties.name} sm{properties.major}{properties.minor}")
    print(
        f"rows={rows} group={args.group} kv_heads={args.kv_heads} "
        f"base_programs={rows * args.kv_heads} prefill={args.prefill} "
        f"kv_dtype={args.kv_dtype}; in-tree config (block_n, warps, tiles, "
        f"splits) = {current}"
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    original_select = qsa_ops._select_config
    with torch.inference_mode():
        batch = build_batch(args)
        graphs: dict[str, torch.cuda.CUDAGraph] = {}
        outputs: dict[str, torch.Tensor] = {}
        for label, config in arms:
            qsa_ops._select_config = lambda *_, config=config: config
            try:
                batch["out"] = torch.full_like(batch["q"], float("nan"))
                run_attention(batch, args.prefill)
                torch.accelerator.synchronize()
                torch.testing.assert_close(
                    batch["out"], outputs.get("current", batch["out"]), **TOLERANCE
                )
                graphs[label] = capture(batch, args.prefill)
                outputs[label] = batch["out"]  # the buffer this graph writes
            finally:
                qsa_ops._select_config = original_select

        # Change the inputs and replay every graph over a NaN-poisoned output: a
        # config that read stale data or skipped rows would no longer agree.
        batch["q"].mul_(0.875)
        batch["output_gate"].add_(0.125)
        v_cache = batch["v_cache"]
        v_cache.copy_((v_cache.float() * 0.9375).to(v_cache.dtype))
        for label in graphs:
            outputs[label].fill_(float("nan"))
            graphs[label].replay()
        torch.accelerator.synchronize()
        for label in graphs:
            torch.testing.assert_close(outputs[label], outputs["current"], **TOLERANCE)
        print(f"all {len(arms)} arms agree with the in-tree config after replay")

        evict = torch.empty(
            2 * properties.L2_cache_size, device="cuda", dtype=torch.uint8
        )
        labels = [label for label, _ in arms]
        orders = [
            labels,
            labels[::-1],
            random.Random(args.seed).sample(labels, len(labels)),
        ]
        medians: dict[bool, dict[str, float]] = {}
        for cold in (False, True):
            samples: dict[str, list[float]] = {label: [] for label in labels}
            for order in orders:
                for _ in range(args.replays):
                    for label in order:
                        samples[label].append(
                            replay_us(graphs[label], evict if cold else None)
                        )
            medians[cold] = {k: statistics.median(v) for k, v in samples.items()}

    header = f"{'candidate':<18}{'n/splits/warps':>16}{'hot us':>10}{'cold us':>10}"
    print(f"\n{header}{'d hot':>8}{'d cold':>8}")
    for label, (block_n, num_warps, _, num_splits) in arms:
        hot, cold = medians[False][label], medians[True][label]
        print(
            f"{label:<18}{f'{block_n}/{num_splits}/{num_warps}':>16}{hot:>10.2f}"
            f"{cold:>10.2f}{hot / medians[False]['current'] - 1:>8.1%}"
            f"{cold / medians[True]['current'] - 1:>8.1%}"
        )
    print(
        f"\nmedian of 3 orders x {args.replays} replays; cold zeroes 2x L2 before each "
        "replay; negative deltas are faster than the in-tree config"
    )


if __name__ == "__main__":
    main()
