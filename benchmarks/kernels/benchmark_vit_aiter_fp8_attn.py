# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark complete AITER BF16 and FP8 ViT attention calls on ROCm.

The FP8 timing includes Q/K/V quantization and attention. Dynamic scales are
calibrated once per input before timing so the measured path matches serving
with a static scale file.

Example:
    python benchmarks/kernels/benchmark_vit_aiter_fp8_attn.py \
        --seq-lens 2304 4096 8192 16384 --head-dim 72

"""

from types import SimpleNamespace

import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.multimodal import MultiModalConfig
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def make_attention(num_heads: int, head_dim: int, fp8: bool) -> MMEncoderAttention:
    mm_config = MultiModalConfig(
        mm_encoder_attn_backend=AttentionBackendEnum.ROCM_AITER_FA,
        mm_encoder_attn_dtype="fp8" if fp8 else None,
    )
    vllm_config = VllmConfig()
    vllm_config.model_config = SimpleNamespace(multimodal_config=mm_config)
    with set_current_vllm_config(vllm_config):
        return MMEncoderAttention(num_heads, head_dim).to("cuda")


def bench(
    seq_lens: list[int],
    num_heads: int,
    head_dim: int,
    warmup_ms: int,
    repeat_ms: int,
) -> None:
    if not current_platform.is_rocm():
        raise RuntimeError("This benchmark requires ROCm and AITER.")

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        fp8_attention = make_attention(num_heads, head_dim, fp8=True)
        bf16_attention = make_attention(num_heads, head_dim, fp8=False)
    finally:
        torch.set_default_dtype(old_dtype)

    print(
        f"{'seq_len':>8} {'BF16 ms':>12} {'FP8 ms':>12} "
        f"{'speedup':>10} {'FP8/BF16':>12}"
    )
    print("-" * 60)
    for seq_len in seq_lens:
        torch.manual_seed(0)
        qkv = torch.randn(
            1,
            seq_len,
            3,
            num_heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        q, k, v = qkv.unbind(dim=2)
        cu_seqlens = torch.tensor([0, seq_len], device="cuda", dtype=torch.int32)
        max_seqlen = torch.tensor(seq_len, device="cuda", dtype=torch.int32)

        # Calibrate per-tensor scales once, then benchmark the static-scale path.
        fp8_attention._fp8_dynamic_scale = True
        fp8_attention._forward_aiter_fp8(q, k, v, cu_seqlens, max_seqlen)
        fp8_attention._fp8_dynamic_scale = False

        bf16_ms = triton.testing.do_bench(
            lambda q=q, k=k, v=v, cu=cu_seqlens, ms=max_seqlen: (
                bf16_attention._forward_fa(q, k, v, cu, ms)
            ),
            warmup=warmup_ms,
            rep=repeat_ms,
        )
        fp8_ms = triton.testing.do_bench(
            lambda q=q, k=k, v=v, cu=cu_seqlens, ms=max_seqlen: (
                fp8_attention._forward_aiter_fp8(q, k, v, cu, ms)
            ),
            warmup=warmup_ms,
            rep=repeat_ms,
        )
        speedup = bf16_ms / fp8_ms
        ratio = fp8_ms / bf16_ms
        print(
            f"{seq_len:>8} {bf16_ms:>12.3f} {fp8_ms:>12.3f} "
            f"{speedup:>9.2f}x {ratio:>12.3f}"
        )


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark AITER BF16 vs FP8 ViT attention."
    )
    parser.add_argument(
        "--seq-lens", type=int, nargs="+", default=[2304, 4096, 8192, 16384]
    )
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=72)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--repeat-ms", type=int, default=500)
    args = parser.parse_args()
    bench(
        args.seq_lens,
        args.num_heads,
        args.head_dim,
        args.warmup_ms,
        args.repeat_ms,
    )
