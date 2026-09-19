# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark MoonViT's packed-QKV complex RoPE implementations."""

import torch

from vllm.benchmarks.lib.utils import default_vllm_config
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.layers.rotary_embedding.vision import (
    apply_fused_qk_complex_rope,
)
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser


def get_benchmark(
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str,
):
    token_counts = [256, 1024, 4096, 16384]

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["num_tokens"],
            x_vals=token_counts,
            line_arg="provider",
            line_vals=["existing", "fused"],
            line_names=["Existing ApplyRotaryEmb x2", "Fused Q/K complex RoPE"],
            styles=[("blue", "-"), ("red", "-")],
            ylabel="us",
            plot_name=f"moonvit-vision-rope-h{num_heads}-d{head_dim}-{dtype}",
            args={},
        )
    )
    @default_vllm_config()
    def benchmark(num_tokens: int, provider: str):
        qkv = torch.randn(
            num_tokens,
            3,
            num_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )
        query, key, _ = torch.unbind(qkv, dim=1)
        angles = torch.randn(num_tokens, head_dim // 2, device=device)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)
        apply_rotary_emb = ApplyRotaryEmb(
            enforce_enable=True,
            is_neox_style=False,
            enable_fp32_compute=True,
        )

        if provider == "existing":

            def run():
                rope_cos = freqs_cis.real.contiguous()
                rope_sin = freqs_cis.imag.contiguous()
                return (
                    apply_rotary_emb.forward_cuda(query, rope_cos, rope_sin),
                    apply_rotary_emb.forward_cuda(key, rope_cos, rope_sin),
                )

        else:

            def run():
                return apply_fused_qk_complex_rope(query, key, freqs_cis)

        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(run, quantiles=quantiles)
        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark MoonViT packed-QKV complex RoPE."
    )
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16"],
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-path", default="./configs/vision_rope/")
    args = parser.parse_args()

    benchmark = get_benchmark(
        args.num_heads,
        args.head_dim,
        getattr(torch, args.dtype),
        args.device,
    )
    benchmark.run(print_data=True, save_path=args.save_path)
