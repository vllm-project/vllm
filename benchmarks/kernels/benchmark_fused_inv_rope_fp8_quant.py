# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm.benchmarks.lib.utils import default_vllm_config
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.attention.ops.deepseek_v4_ops import fused_inv_rope_fp8_quant

num_tokens_range = [2**i for i in range(0, 14)]


def make_cos_sin_cache(
    max_pos: int,
    rope_dim: int,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    """Create a synthetic cos_sin_cache matching the layout used by
    DeepseekV4ScalingRotaryEmbedding._compute_cos_sin_cache.

    Shape: [max_pos, rope_dim] where first half is cos, second half is sin.
    The fused kernel requires fp32; callers can override dtype if passing
    the cache into the bf16-only paths.
    """
    half = rope_dim // 2
    # Use random but bounded frequencies so cos/sin are well-behaved
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
    )
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [max_pos, half]
    cos = freqs.cos()
    sin = freqs.sin()
    cache = torch.cat((cos, sin), dim=-1)  # [max_pos, rope_dim]
    return cache.to(dtype)


def get_benchmark(num_heads, n_groups, head_dim, rope_dim, device):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["num_tokens"],
            x_vals=num_tokens_range,
            line_arg="provider",
            line_vals=["vllm"],
            line_names=["vLLM"],
            styles=[("blue", "-")],
            ylabel="us",
            plot_name="Fused FP8 RoPE kernel perf",
            args={},
        )
    )
    @default_vllm_config()
    def benchmark(num_tokens, provider):
        max_pos = 4096
        heads_per_group = num_heads // n_groups

        # Create inputs
        o = torch.randn(
            num_tokens, num_heads, head_dim, device=device, dtype=torch.bfloat16
        )
        positions = torch.randint(
            0, max_pos, (num_tokens,), device=device, dtype=torch.long
        )
        cos_sin_cache = make_cos_sin_cache(
            max_pos, rope_dim, dtype=torch.float32, device=device
        )

        quantiles = [0.5, 0.2, 0.8]

        if provider == "vllm":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: fused_inv_rope_fp8_quant(
                    o.clone(),
                    positions,
                    cos_sin_cache,
                    n_groups,
                    heads_per_group,
                ),
                quantiles=quantiles,
            )
        else:
            raise ValueError(f"Invalid provider: {provider}")

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the fused_inv_rope_fp8_quant kernel."
    )
    parser.add_argument("--num-heads", type=int, default=128)
    parser.add_argument(
        "--n-groups",
        type=int,
        default=8,
    )
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--save-path", type=str, default="./configs/fused_inv_rope_fp8_quant/"
    )
    args = parser.parse_args()

    # Get the benchmark function
    benchmark = get_benchmark(
        args.num_heads, args.n_groups, args.head_dim, args.rope_dim, args.device
    )
    # Run performance benchmark
    benchmark.run(print_data=True, save_path=args.save_path)
