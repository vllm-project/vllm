# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark: fused QK-Norm + mRoPE vs. the unfused compiled paths.

The attention-input block of a Qwen3-VL-class layer is compiled through the
vLLM pass pipeline (IR lowering, clone elimination, functionalization fix-up)
in three configurations, all timed under a CUDA graph:
    compiled_native    enable_qk_norm_rope_fusion off, the stock path: native
                       RMSNorm and native mRoPE compiled by Inductor.
    compiled_customop  fusion off with +rotary_embedding: Triton mRoPE.
    compiled_fused     fusion on: one fused_qk_norm_mrope launch.

Run from the repository root:
    python benchmarks/fused_kernels/fused_qk_norm_mrope_benchmark.py
"""

import statistics
from dataclasses import dataclass
from itertools import product

import torch

import vllm.compilation.passes.fusion.qk_norm_rope_fusion as fusion_module
from tests.compile.backend import TestBackend
from vllm.compilation.passes.ir.clone_elimination import UnsafeCloneEliminationPass
from vllm.compilation.passes.ir.lowering_pass import VllmIRLoweringPass
from vllm.compilation.passes.utility.fix_functionalization import (
    FixFunctionalizationPass,
)
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.compilation.passes.utility.split_coalescing import SplitCoalescingPass
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    ModelConfig,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
from vllm.triton_utils import triton
from vllm.v1.attention.backend import AttentionType

EPS = 1e-6
IS_NEOX = True

# name -> (custom_ops, run the fusion pass)
VARIANTS = {
    "compiled_native": (["-rms_norm", "-rotary_embedding"], False),
    "compiled_customop": (["-rms_norm", "+rotary_embedding"], False),
    "compiled_fused": (["-rms_norm", "+rotary_embedding"], True),
}


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
        (16, 2, 64, (8, 12, 12), True),
        (8, 8, 256, (32, 48, 48), True),
    ]
    DTYPES = [torch.bfloat16, torch.float16]
    return [
        bench_params_t(n, nh, nkv, hd, ms, il, dt)
        for (nh, nkv, hd, ms, il), dt, n in product(HEAD_CONFIGS, DTYPES, NUM_TOKENS)
    ]


class QKNormMRoPE(torch.nn.Module):
    """Attention-input block, written as in Qwen3Attention.forward."""

    def __init__(self, p: bench_params_t, vllm_config: VllmConfig) -> None:
        super().__init__()
        self.head_dim = p.head_dim
        self.q_size = p.num_heads * p.head_dim
        self.kv_size = p.num_kv_heads * p.head_dim
        # Registers the layer geometry the fusion pass reads.
        self.attn = Attention(
            num_heads=p.num_heads,
            head_size=p.head_dim,
            scale=p.head_dim**-0.5,
            num_kv_heads=p.num_kv_heads,
            cache_config=vllm_config.cache_config,
            prefix="model.layers.0.self_attn.attn",
            attn_type=AttentionType.DECODER,
        )
        self.q_norm = RMSNorm(p.head_dim, eps=EPS)
        self.k_norm = RMSNorm(p.head_dim, eps=EPS)
        self.rotary_emb = MRotaryEmbedding(
            p.head_dim,
            rotary_dim=p.head_dim,
            max_position_embeddings=4096,
            base=1e6,
            is_neox_style=IS_NEOX,
            dtype=p.dtype,
            mrope_section=list(p.mrope_section),
            mrope_interleaved=p.mrope_interleaved,
        )

    def forward(self, qkv: torch.Tensor, positions: torch.Tensor):
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q = self.q_norm(q_by_head).view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k = self.k_norm(k_by_head).view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v


def bench_variant(
    p: bench_params_t,
    custom_ops: list[str],
    fuse: bool,
    qkv: torch.Tensor,
    positions: torch.Tensor,
) -> float:
    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=p.dtype),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=custom_ops,
            pass_config=PassConfig(
                enable_qk_norm_rope_fusion=fuse, eliminate_noops=True
            ),
        ),
    )
    with (
        set_current_vllm_config(vllm_config),
        vllm_config.kernel_config.ir_op_priority.set_priority(),
    ):
        model = QKNormMRoPE(p, vllm_config)
        passes = [NoOpEliminationPass(vllm_config), SplitCoalescingPass(vllm_config)]
        fusion_pass = None
        if fuse:
            fusion_pass = fusion_module.QKNormMRoPEFusionPass(vllm_config)
            passes.append(fusion_pass)
        # Same closing sequence as PostGradPassManager.__call__.
        passes += [
            PostCleanupPass(vllm_config),
            VllmIRLoweringPass(vllm_config),
            UnsafeCloneEliminationPass(vllm_config),
            PostCleanupPass(vllm_config),
            FixFunctionalizationPass(vllm_config),
        ]
        compiled = torch.compile(model, backend=TestBackend(*passes), dynamic=False)
        compiled(qkv, positions)
        if fusion_pass is not None:
            assert fusion_pass.matched_count == 1, "fusion pass did not match"
        us = triton.testing.do_bench_cudagraph(lambda: compiled(qkv, positions)) * 1e3
    torch._dynamo.reset()
    return us


def bench(p: bench_params_t) -> dict[str, float]:
    torch.set_default_dtype(p.dtype)
    total = (p.num_heads + 2 * p.num_kv_heads) * p.head_dim
    qkv = torch.randn(p.num_tokens, total, dtype=p.dtype)
    positions = torch.randint(0, 4096, (3, p.num_tokens), dtype=torch.long)
    # The pass reads the mRoPE layout from the HF config; a synthetic
    # ModelConfig has none.
    fusion_module._discover_mrope_configs = lambda config: (
        (p.mrope_section, p.mrope_interleaved),
    )
    return {
        name: bench_variant(p, custom_ops, fuse, qkv, positions)
        for name, (custom_ops, fuse) in VARIANTS.items()
    }


def main():
    torch.set_default_device("cuda")
    print(
        f"{'config':<42}{'compiled_native':>17}{'compiled_customop':>19}"
        f"{'compiled_fused':>16}{'x native':>10}{'x customop':>12}"
    )
    speedups: dict[str, list[float]] = {"native": [], "customop": []}
    for p in get_bench_params():
        us = bench(p)
        s_native = us["compiled_native"] / us["compiled_fused"]
        s_customop = us["compiled_customop"] / us["compiled_fused"]
        speedups["native"].append(s_native)
        speedups["customop"].append(s_customop)
        print(
            f"{p.description():<42}{us['compiled_native']:>15.1f}us"
            f"{us['compiled_customop']:>17.1f}us{us['compiled_fused']:>14.1f}us"
            f"{s_native:>9.2f}x{s_customop:>11.2f}x",
            flush=True,
        )
    for key, sp in speedups.items():
        print(
            f"fused vs compiled {key}: geomean {statistics.geometric_mean(sp):.2f}x, "
            f"range {min(sp):.2f}x-{max(sp):.2f}x"
        )


if __name__ == "__main__":
    main()
