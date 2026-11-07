# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the NVFP4 MoE-experts quantization kernels: vLLM vs FlashInfer.

Unlike benchmark_nvfp4_quant.py (dense `scaled_fp4_quant`) or
benchmark_cutlass_moe_nvfp4.py (whole cutlass_moe_fp4 block, including
GEMMs), this isolates just the per-expert quantization kernel for each
backend:

  vLLM (ragged/packed layout, CUTLASS grouped-GEMM convention):
    - scaled_fp4_experts_quant
    - silu_and_mul_scaled_fp4_experts_quant

  FlashInfer (batched/masked layout, one padded [B, M_max, K] tensor):
    - scaled_fp4_grouped_quantize
    - silu_and_mul_scaled_nvfp4_experts_quantize

Both backends are fed the *same* per-expert token counts (derived once per
num_tokens from a random routing via get_cutlass_moe_mm_data) so the
comparison reflects real backend/kernel differences rather than differing
workloads. Note the
two backends fundamentally differ in layout: vLLM operates on exactly
sum(counts) ragged rows, while FlashInfer pads every expert up to
max(counts) rows (plus internal 128-row tiling) -- so some of FlashInfer's
extra time on skewed routing is inherent padding overhead, not raw kernel
speed. That skew sensitivity is itself part of what this benchmark is
meant to surface.

Reported latency is the median over the timed calls (with p25-p75) of the
whole op as called from Python, so at small num_tokens it includes host-side
overhead (wrapper, output allocation, dispatch) comparable to the kernel itself.

By default, each op runs at Qwen3.8-Flash-Next's MoE shape (topk 10, hidden
size 2560, intermediate size 640) for every combination of DEFAULT_NUM_EXPERTS
and DEFAULT_NUM_TOKENS.
"""

import argparse
import functools
import os

import pandas as pd
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    scaled_fp4_grouped_quantize as flashinfer_scaled_fp4_grouped_quantize,
)
from vllm.utils.flashinfer import (
    silu_and_mul_scaled_nvfp4_experts_quantize as flashinfer_silu_and_mul_quantize,
)

if not current_platform.has_device_capability(100):
    raise RuntimeError("NVFP4 requires compute capability of 10.0 (Blackwell)")

PROVIDERS = ["vllm", "flashinfer"]
GLOBAL_SF = 448.0 * 6.0  # FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX

# Default sweep: single-token decode up to a 16384-token prefill chunk.
DEFAULT_NUM_TOKENS = [1, 16, 64, 256, 1024, 4096, 16384]
# Expert counts for simulating MoE models.
DEFAULT_NUM_EXPERTS = [512, 256, 128]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark NVFP4 MoE-experts quantization: vLLM vs FlashInfer"
    )
    parser.add_argument(
        "--num-experts",
        nargs="+",
        type=int,
        default=DEFAULT_NUM_EXPERTS,
        help="Expert counts to benchmark, e.g. --num-experts 512 64 16.",
    )
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=2560)
    parser.add_argument("--intermediate-size", type=int, default=640)
    parser.add_argument(
        "--num-tokens",
        nargs="+",
        type=int,
        default=DEFAULT_NUM_TOKENS,
        help="Token counts to benchmark, e.g. --num-tokens 256 16384.",
    )
    parser.add_argument(
        "--providers", nargs="+", type=str, default=PROVIDERS, choices=PROVIDERS
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Directory to write one CSV per op and expert count to.",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=100,
        help="Timed calls per data point, after 25 untimed warmup calls. "
        "1 skips the warmup, so an ncu capture sees one launch per data point.",
    )
    parser.add_argument(
        "--read-flush",
        action="store_true",
        help="Flush L2 before each timed call by reading a buffer instead of "
        "zeroing it, which leaves L2 full of dirty lines whose write-back "
        "competes with the timed kernel.",
    )
    parser.add_argument(
        "--routing-skew",
        type=float,
        default=0.4,
        help="Zipf exponent s of expert popularity (rank^-s, ranks shuffled "
        "over expert ids); each token draws topk distinct experts from it. "
        "0 routes uniformly; larger values concentrate tokens on fewer experts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Routing seed, so runs against different builds see the same "
        "per-expert token counts.",
    )
    return parser.parse_args()


args = _parse_args()

# vLLM's scaled_fp4_experts_quant/silu_and_mul_scaled_fp4_experts_quant size
# their output_scales buffer to VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE * topk
# rows -- a fixed worst-case bound (default 163840) chosen so the op never
# needs a device->host sync to learn the real per-expert row count. Bound it
# to this run's max num_tokens instead, so the vLLM side's buffer is not
# several times larger than anything FlashInfer's side (sized to the real
# per-expert max) allocates. num_tokens * topk rows alone is not enough: the
# kernel pads each expert's scale rows up to a multiple of 128, so the buffer
# needs up to num_experts * 127 more rows -- without them the largest point
# writes scale factors past the end of the allocation. vllm.envs reads the
# variable lazily, so setting it after the imports is fine.
_max_tokens_per_expert = max(args.num_tokens) + -(
    -max(args.num_experts) * 127 // args.topk
)
os.environ.setdefault("VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE", str(_max_tokens_per_expert))

_flush_buf: torch.Tensor | None = None
# (op_name, num_experts, num_tokens, provider) -> (median, p25, p75 in us, n_calls)
_stats: dict[tuple[str, int, int, str], tuple[float, float, float, int]] = {}


def _flush_l2(read: bool) -> None:
    """Evict L2 with a buffer several times its size.

    Reading the buffer leaves L2 clean. Zeroing it leaves L2 full of dirty
    lines whose write-back competes with the timed kernel.
    """
    global _flush_buf
    if _flush_buf is None:
        props = torch.cuda.get_device_properties(
            torch.accelerator.current_device_index()
        )
        flush_bytes = max(4 * props.L2_cache_size, 256 << 20)
        _flush_buf = torch.zeros(flush_bytes // 4, dtype=torch.int32, device="cuda")
    if read:
        _flush_buf.sum()
    else:
        _flush_buf.zero_()


def _timed_runs(fn, n: int, read_flush: bool) -> torch.Tensor:
    """Per-call latency in ms of n calls, each preceded by an untimed flush."""
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    for i in range(n):
        _flush_l2(read_flush)
        starts[i].record()
        fn()
        ends[i].record()
    torch.accelerator.synchronize()
    return torch.tensor([s.elapsed_time(e) for s, e in zip(starts, ends)])


def _do_bench(fn, read_flush, n_repeat, n_warmup=25):
    """Median, p25 and p75 of fn's latency in ms, with a selectable L2 flush."""
    _timed_runs(fn, n_warmup, read_flush)
    times = _timed_runs(fn, n_repeat, read_flush)
    p25, median, p75 = torch.quantile(times, torch.tensor([0.25, 0.5, 0.75])).tolist()
    return median, p25, p75, n_repeat


@functools.cache
def _route(num_tokens: int, e: int, n: int, k: int, topk: int, device: str):
    """Build a random routing once, return vLLM offsets + per-expert counts.

    Cached so every provider sees the same routing for a given num_tokens.
    """
    gen = torch.Generator(device=device).manual_seed(args.seed + num_tokens)
    if args.routing_skew > 0:
        # Each token draws topk distinct experts with probability proportional
        # to a Zipf popularity, so a few experts get most rows. Shuffling the
        # ranks keeps the hot experts from always being the lowest ids.
        ranks = torch.randperm(e, generator=gen, device=device) + 1
        popularity = ranks.float().pow(-args.routing_skew)
        topk_ids = torch.multinomial(
            popularity.expand(num_tokens, e).contiguous(),
            topk,
            replacement=False,
            generator=gen,
        ).to(torch.int32)
    else:
        gating = torch.randn(num_tokens, e, device=device, generator=gen)
        topk_ids = torch.topk(gating, topk, dim=-1).indices.to(torch.int32)

    expert_offsets = torch.empty((e + 1), dtype=torch.int32, device=device)
    blockscale_offsets = torch.empty((e + 1), dtype=torch.int32, device=device)
    problem_sizes1 = torch.empty((e, 3), dtype=torch.int32, device=device)
    problem_sizes2 = torch.empty((e, 3), dtype=torch.int32, device=device)
    a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)

    ops.get_cutlass_moe_mm_data(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        e,
        n,
        k,
        blockscale_offsets,
        is_gated=True,
    )
    counts = (expert_offsets[1:] - expert_offsets[:-1]).clamp_min(1)
    return a_map, expert_offsets, blockscale_offsets, counts


def _benchmark(
    op_name: str,
    gated: bool,
    num_tokens: int,
    provider: str,
    e: int,
    topk: int,
    hidden_size: int,
    intermediate_size: int,
) -> None:
    """Time one (op, num_experts, num_tokens, provider) point into _stats."""
    device = "cuda"
    dtype = torch.bfloat16
    k = hidden_size
    n = intermediate_size
    width = 2 * n if gated else k

    a_map, expert_offsets, blockscale_offsets, counts = _route(
        num_tokens, e, n, k, topk, device
    )

    if provider == "vllm":
        m_topk = a_map.numel()
        gscale = torch.full((e,), GLOBAL_SF, dtype=torch.float32, device=device)
        if gated:
            a = torch.randn(m_topk, width, dtype=dtype, device=device)
            fn = lambda: ops.silu_and_mul_scaled_fp4_experts_quant(
                a, gscale, expert_offsets, blockscale_offsets, topk
            )
        else:
            hidden_states = torch.randn(num_tokens, k, dtype=dtype, device=device)
            a = ops.shuffle_rows(hidden_states, a_map)
            fn = lambda: ops.scaled_fp4_experts_quant(
                a, gscale, expert_offsets, blockscale_offsets, topk
            )
    else:
        m_max = int(counts.max().item())
        a_batched = torch.randn(e, m_max, width, dtype=dtype, device=device)
        mask = counts.to(torch.int32)
        gscale = torch.full((e,), GLOBAL_SF, dtype=torch.float32, device=device)
        if gated:
            fn = lambda: flashinfer_silu_and_mul_quantize(a_batched, mask, gscale)
        else:
            fn = lambda: flashinfer_scaled_fp4_grouped_quantize(a_batched, mask, gscale)

    n_warmup = 25 if args.num_iters > 1 else 0
    median_ms, p25_ms, p75_ms, n_calls = _do_bench(
        fn, args.read_flush, args.num_iters, n_warmup
    )
    _stats[op_name, e, num_tokens, provider] = (
        median_ms * 1e3,
        p25_ms * 1e3,
        p75_ms * 1e3,
        n_calls,
    )


def _save_csv(op_name: str, num_experts: int, save_path: str) -> None:
    """Write median/p25/p75 (us) and call count per provider and num_tokens."""
    rows = []
    for num_tokens in args.num_tokens:
        row: dict[str, float | int] = {"num_tokens": num_tokens}
        for provider in args.providers:
            median_us, p25_us, p75_us, n_calls = _stats[
                op_name, num_experts, num_tokens, provider
            ]
            row[f"{provider}_median_us"] = median_us
            row[f"{provider}_p25_us"] = p25_us
            row[f"{provider}_p75_us"] = p75_us
            row[f"{provider}_n"] = n_calls
        rows.append(row)
    os.makedirs(save_path, exist_ok=True)
    path = os.path.join(save_path, f"{op_name}_E{num_experts}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)


# Fixed column widths, so tables for different ops and expert counts line up.
_TOKENS_WIDTH = 10
_CELL_WIDTH = 36


def _print_table(op_name: str, num_experts: int) -> None:
    """Print median, p25-p75 and call count per provider."""
    print(f"\nNVFP4 MoE-Experts {op_name} Latency (E={num_experts}):")
    print(
        "num_tokens".rjust(_TOKENS_WIDTH)
        + "".join(p.rjust(_CELL_WIDTH) for p in args.providers)
    )
    for num_tokens in args.num_tokens:
        line = str(num_tokens).rjust(_TOKENS_WIDTH)
        for provider in args.providers:
            median_us, p25_us, p75_us, n_calls = _stats[
                op_name, num_experts, num_tokens, provider
            ]
            cell = f"{median_us:.2f}us ({p25_us:.2f}-{p75_us:.2f}, n={n_calls})"
            line += cell.rjust(_CELL_WIDTH)
        print(line)


OP_QUANT = "scaled_fp4_experts_quant"
OP_SILU_MUL = "silu_and_mul_scaled_fp4_experts_quant"


if __name__ == "__main__":
    shape = dict(
        topk=args.topk,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
    )
    print(f"config: {shape}, num_experts={args.num_experts}")
    for num_experts in args.num_experts:
        for op_name, gated in ((OP_QUANT, False), (OP_SILU_MUL, True)):
            for num_tokens in args.num_tokens:
                for provider in args.providers:
                    _benchmark(
                        op_name, gated, num_tokens, provider, e=num_experts, **shape
                    )
            _print_table(op_name, num_experts)
            if args.save_path:
                _save_csv(op_name, num_experts, args.save_path)

    print("\nBenchmark finished!")
