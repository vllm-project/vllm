# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark: fused QK-Norm + mRoPE vs. the unfused 3-launch path.

Unfused path (what Qwen3-VL-class models run today):
    rms_norm(q) -> rms_norm(k) -> mrope(q, k)          # 3 kernel launches
Fused path (this PR):
    fused_qk_norm_mrope(qkv, ...)                       # 1 kernel launch

Run:
    python benchmarks/fused_kernels/fused_qk_norm_mrope_benchmark.py
"""

import statistics
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import product

import torch
import torch.utils.benchmark as TBenchmark
from torch.utils.benchmark import Measurement as TMeasurement
from tqdm import tqdm

# Imported for side effects: loading vLLM registers the C ops (torch.ops._C.rms_norm,
# torch.ops._C.fused_qk_norm_mrope) and the mRoPE custom op (torch.ops.vllm.mrope).
# The module import is safe; only *instantiating* MRotaryEmbedding needs a vLLM config.
import vllm.model_executor.layers.rotary_embedding.mrope  # noqa: F401,E402

# mRoPE rotation is always neox-style; the fused kernel is invoked with is_neox=True.
IS_NEOX = True
EPS = 1e-6


@dataclass
class bench_params_t:
    num_tokens: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    mrope_section: tuple[int, int, int]
    dtype: torch.dtype

    def description(self) -> str:
        return (
            f"N {self.num_tokens} "
            f"x Hq {self.num_heads} "
            f"x Hkv {self.num_kv_heads} "
            f"x D {self.head_dim} "
            f"x DT {self.dtype}"
        )


def get_bench_params() -> list[bench_params_t]:
    """Representative Qwen-VL-class attention geometries.

    mrope_section sums to head_dim // 2 (full rotary). All are non-interleaved.
    """
    NUM_TOKENS = [16, 128, 512, 2048, 8192]
    # (num_heads, num_kv_heads, head_dim, mrope_section)
    HEAD_CONFIGS = [
        (16, 2, 128, (16, 24, 24)),  # small VL
        (28, 4, 128, (16, 24, 24)),  # ~Qwen2.5-VL-7B
        (40, 8, 128, (16, 24, 24)),  # larger GQA
    ]
    DTYPES = [torch.bfloat16, torch.float16]

    params: list[bench_params_t] = []
    for num_tokens, (nh, nkv, hd, ms), dtype in product(
        NUM_TOKENS, HEAD_CONFIGS, DTYPES
    ):
        params.append(bench_params_t(num_tokens, nh, nkv, hd, ms, dtype))
    return params


def _rmsnorm_cuda(
    x_flat: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    out = torch.empty_like(x_flat)
    torch.ops._C.rms_norm(out, x_flat, weight, eps)
    return out


def make_unfused(
    qkv, positions, q_weight, k_weight, cos_sin_cache, p: bench_params_t
) -> Callable:
    q_size = p.num_heads * p.head_dim
    kv_size = p.num_kv_heads * p.head_dim
    ms = p.mrope_section

    def unfused():
        q, k, _ = qkv.split([q_size, kv_size, kv_size], dim=-1)
        qn = _rmsnorm_cuda(q.reshape(-1, p.head_dim).contiguous(), q_weight, EPS)
        kn = _rmsnorm_cuda(k.reshape(-1, p.head_dim).contiguous(), k_weight, EPS)
        torch.ops.vllm.mrope(
            positions,
            qn.view(p.num_tokens, q_size),
            kn.view(p.num_tokens, kv_size),
            cos_sin_cache,
            p.head_dim,
            p.head_dim,
            ms[0],
            ms[1],
            ms[2],
            False,
        )

    return unfused


def make_fused(
    qkv, positions, q_weight, k_weight, cos_sin_cache, p: bench_params_t
) -> Callable:
    ms = p.mrope_section

    def fused():
        torch.ops._C.fused_qk_norm_mrope(
            qkv,
            p.num_heads,
            p.num_kv_heads,
            p.num_kv_heads,
            p.head_dim,
            EPS,
            q_weight,
            k_weight,
            cos_sin_cache,
            IS_NEOX,
            positions,
            ms[0],
            ms[1],
        )

    return fused


def bench_fn(fn: Callable, label: str, sub_label: str, description: str) -> TMeasurement:
    return TBenchmark.Timer(
        stmt="fn()",
        globals={"fn": fn},
        label=label,
        sub_label=sub_label,
        description=description,
    ).blocked_autorange(min_run_time=1.0)


def bench(p: bench_params_t, label: str, sub_label: str) -> Iterable[TMeasurement]:
    device = "cuda"
    q_size = p.num_heads * p.head_dim
    kv_size = p.num_kv_heads * p.head_dim
    total = q_size + 2 * kv_size

    qkv = torch.randn(p.num_tokens, total, dtype=p.dtype, device=device)
    positions = torch.randint(
        0, 4096, (3, p.num_tokens), dtype=torch.long, device=device
    )
    q_weight = torch.randn(p.head_dim, dtype=p.dtype, device=device)
    k_weight = torch.randn(p.head_dim, dtype=p.dtype, device=device)

    # A random cos/sin cache is sufficient for timing: the kernel's memory
    # traffic (gather cos_sin_cache[positions]) and FLOPs are identical
    # regardless of the values, and the benchmark makes no correctness assertion.
    cos_sin_cache = torch.randn(4096, p.head_dim, dtype=p.dtype, device=device)

    return [
        bench_fn(
            make_unfused(qkv.clone(), positions, q_weight, k_weight, cos_sin_cache, p),
            label,
            sub_label,
            "unfused_rmsnorm+mrope",
        ),
        bench_fn(
            make_fused(qkv.clone(), positions, q_weight, k_weight, cos_sin_cache, p),
            label,
            sub_label,
            "fused_qk_norm_mrope",
        ),
    ]


def print_speedups(
    params: list[bench_params_t], timers: list[TMeasurement]
) -> None:
    """Per-config unfused/fused speedup plus summary. timers are laid out as
    [unfused_0, fused_0, unfused_1, fused_1, ...] matching ``params`` order."""
    print("\n" + "=" * 80)
    print("SPEEDUP (unfused / fused, from median times)")
    print("=" * 80)
    print(f"{'config':<50}{'unfused(us)':>13}{'fused(us)':>12}{'speedup':>10}")
    print("-" * 85)
    speedups = []
    for i, p in enumerate(params):
        unf = timers[2 * i].median * 1e6
        fus = timers[2 * i + 1].median * 1e6
        sp = unf / fus
        speedups.append(sp)
        print(f"{p.description():<50}{unf:>13.1f}{fus:>12.1f}{sp:>9.2f}x")
    print("-" * 85)
    print(
        f"min {min(speedups):.2f}x | "
        f"median {statistics.median(speedups):.2f}x | "
        f"mean {statistics.mean(speedups):.2f}x | "
        f"max {max(speedups):.2f}x"
    )


def main():
    torch.set_default_device("cuda")
    params = get_bench_params()
    print(f"Running {len(params)} configurations (~2s each)...\n")

    timers: list[TMeasurement] = []
    for p in tqdm(params):
        timers.extend(bench(p, "qk-norm-mrope", p.description()))

    print("\n" + "=" * 80)
    print("FINAL COMPARISON — unfused (3 launches) vs fused (1 launch)")
    print("=" * 80)
    TBenchmark.Compare(timers).print()

    print_speedups(params, timers)


if __name__ == "__main__":
    main()
