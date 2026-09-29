# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark: fused QK-Norm + mRoPE vs. the unfused paths.

Baselines, all timed under a CUDA graph (launch overhead excluded):
    compiled_native   torch.compile (Inductor) of native RMSNorm + native mRoPE,
                      the stock path with enable_qk_norm_rope_fusion off.
    unfused_customop  rms_norm CUDA kernel x2 + Triton mRoPE (custom ops on).
    fused             fused_qk_norm_mrope, one launch.

Run:
    python benchmarks/fused_kernels/fused_qk_norm_mrope_benchmark.py
"""

import statistics
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product

import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
from vllm.triton_utils import triton

EPS = 1e-6
IS_NEOX = True


@dataclass
class bench_params_t:
    num_tokens: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    mrope_section: tuple[int, int, int]
    mrope_interleaved: bool
    dtype: torch.dtype

    def description(self) -> str:
        return (
            f"N {self.num_tokens} x Hq {self.num_heads} x Hkv {self.num_kv_heads} "
            f"x D {self.head_dim} x {str(self.dtype)[6:]}"
        )


def get_bench_params() -> list[bench_params_t]:
    NUM_TOKENS = [1, 16, 64, 256, 1024, 4096, 16384]
    # (num_heads, num_kv_heads, head_dim, mrope_section, interleaved)
    HEAD_CONFIGS = [
        (32, 8, 128, (24, 20, 20), True),  # Qwen3-VL-8B
        (16, 4, 128, (16, 24, 24), False),
        (40, 8, 128, (16, 24, 24), False),
    ]
    DTYPES = [torch.bfloat16, torch.float16]
    return [
        bench_params_t(n, nh, nkv, hd, ms, il, dt)
        for n, (nh, nkv, hd, ms, il), dt in product(NUM_TOKENS, HEAD_CONFIGS, DTYPES)
    ]


def make_layers(p: bench_params_t) -> tuple[RMSNorm, RMSNorm, MRotaryEmbedding]:
    q_norm = RMSNorm(p.head_dim, eps=EPS).to(dtype=p.dtype)
    k_norm = RMSNorm(p.head_dim, eps=EPS).to(dtype=p.dtype)
    q_norm.weight.data.normal_(mean=1.0, std=0.1)
    k_norm.weight.data.normal_(mean=1.0, std=0.1)
    rope = MRotaryEmbedding(
        p.head_dim,
        rotary_dim=p.head_dim,
        max_position_embeddings=4096,
        base=1e6,
        is_neox_style=IS_NEOX,
        dtype=p.dtype,
        mrope_section=list(p.mrope_section),
        mrope_interleaved=p.mrope_interleaved,
    )
    return q_norm, k_norm, rope


def make_compiled_native(
    qkv, positions, q_norm, k_norm, rope, p: bench_params_t
) -> Callable:
    q_size = p.num_heads * p.head_dim
    kv_size = p.num_kv_heads * p.head_dim

    def native(qkv, positions):
        q, k, _ = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = q_norm.forward_native(q.view(-1, p.num_heads, p.head_dim)).view(q.shape)
        k = k_norm.forward_native(k.view(-1, p.num_kv_heads, p.head_dim)).view(k.shape)
        return rope.forward_native(positions, q, k)

    compiled = torch.compile(native, dynamic=False)
    compiled(qkv, positions)
    return lambda: compiled(qkv, positions)


def make_unfused_customop(
    qkv, positions, q_norm, k_norm, rope, p: bench_params_t
) -> Callable:
    q_size = p.num_heads * p.head_dim
    kv_size = p.num_kv_heads * p.head_dim
    ms = p.mrope_section

    def unfused():
        q, k, _ = qkv.split([q_size, kv_size, kv_size], dim=-1)
        qn = torch.empty_like(q)
        kn = torch.empty_like(k)
        torch.ops._C.rms_norm(
            qn.view(-1, p.head_dim), q.reshape(-1, p.head_dim), q_norm.weight, EPS
        )
        torch.ops._C.rms_norm(
            kn.view(-1, p.head_dim), k.reshape(-1, p.head_dim), k_norm.weight, EPS
        )
        torch.ops.vllm.mrope(
            positions,
            qn,
            kn,
            rope.cos_sin_cache,
            p.head_dim,
            p.head_dim,
            ms[0],
            ms[1],
            ms[2],
            p.mrope_interleaved,
            IS_NEOX,
        )

    return unfused


def make_fused(qkv, positions, q_norm, k_norm, rope, p: bench_params_t) -> Callable:
    ms = p.mrope_section

    def fused():
        torch.ops._C.fused_qk_norm_mrope(
            qkv,
            p.num_heads,
            p.num_kv_heads,
            p.num_kv_heads,
            p.head_dim,
            EPS,
            q_norm.weight,
            k_norm.weight,
            rope.cos_sin_cache,
            IS_NEOX,
            positions,
            ms[0],
            ms[1],
            p.mrope_interleaved,
        )

    return fused


def bench(p: bench_params_t) -> dict[str, float]:
    total = (p.num_heads + 2 * p.num_kv_heads) * p.head_dim
    qkv = torch.randn(p.num_tokens, total, dtype=p.dtype)
    positions = torch.randint(0, 4096, (3, p.num_tokens), dtype=torch.long)
    q_norm, k_norm, rope = make_layers(p)
    variants = {
        "compiled_native": make_compiled_native,
        "unfused_customop": make_unfused_customop,
        "fused": make_fused,
    }
    us = {}
    for name, make in variants.items():
        fn = make(qkv.clone(), positions, q_norm, k_norm, rope, p)
        us[name] = triton.testing.do_bench_cudagraph(fn) * 1e3
    torch._dynamo.reset()
    return us


def main():
    torch.set_default_device("cuda")
    params = get_bench_params()
    print(
        f"{'config':<44}{'compiled_native':>16}{'unfused_customop':>17}"
        f"{'fused':>9}{'x native':>10}{'x customop':>12}"
    )
    speedups: dict[str, list[float]] = {"native": [], "customop": []}
    with set_current_vllm_config(VllmConfig()):
        for p in params:
            us = bench(p)
            s_native = us["compiled_native"] / us["fused"]
            s_customop = us["unfused_customop"] / us["fused"]
            speedups["native"].append(s_native)
            speedups["customop"].append(s_customop)
            print(
                f"{p.description():<44}{us['compiled_native']:>14.1f}us"
                f"{us['unfused_customop']:>15.1f}us{us['fused']:>7.1f}us"
                f"{s_native:>9.2f}x{s_customop:>11.2f}x",
                flush=True,
            )
    for key, sp in speedups.items():
        print(
            f"fused vs {key}: geomean {statistics.geometric_mean(sp):.2f}x, "
            f"range {min(sp):.2f}x-{max(sp):.2f}x"
        )


if __name__ == "__main__":
    main()
