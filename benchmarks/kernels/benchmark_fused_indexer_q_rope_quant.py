# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark fused_indexer_q_rope_quant: FP8 vs MXFP4 vs NVFP4.

All providers are forced onto the Triton path (has_cutedsl patched to False)
because NVFP4 has no CuteDSL kernel yet — this keeps the comparison
apples-to-apples. Shapes match the DeepSeek-V4 sparse indexer.

Usage:
    python benchmarks/kernels/benchmark_fused_indexer_q_rope_quant.py
"""

from unittest import mock

import torch

from vllm.models.deepseek_v4.common.ops import fused_indexer_q_rope_quant
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser

# DeepSeek-V4 indexer shapes (same as the unit test).
HEAD_DIM = 128
ROPE_DIM = 64
N_HEAD = 64
MAX_POS = 8192

num_tokens_range = [2**i for i in range(0, 15, 2)]  # 1 .. 16384

PROVIDERS = ["fp8", "mxfp4", "nvfp4"]


def _bytes_moved(num_tokens: int, provider: str) -> int:
    """Approximate DRAM traffic per call (read + write), for GB/s."""
    q_in = num_tokens * N_HEAD * HEAD_DIM * 2  # bf16
    weights_in = num_tokens * N_HEAD * 2  # bf16
    weights_out = num_tokens * N_HEAD * 4  # fp32
    pos = num_tokens * 8
    if provider == "fp8":
        q_out = num_tokens * N_HEAD * HEAD_DIM  # fp8, no scale tensor
        scale_out = 0
    else:
        block = 32 if provider == "mxfp4" else 16
        q_out = num_tokens * N_HEAD * HEAD_DIM // 2  # packed e2m1
        scale_out = num_tokens * N_HEAD * (HEAD_DIM // block)  # 1B/block
    return q_in + weights_in + weights_out + pos + q_out + scale_out


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=num_tokens_range,
        line_arg="provider",
        line_vals=PROVIDERS,
        line_names=["FP8", "MXFP4", "NVFP4"],
        styles=[("blue", "-"), ("green", "-"), ("red", "-")],
        ylabel="us",
        plot_name="fused-indexer-q-rope-quant",
        args={},
    )
)
def benchmark(num_tokens, provider):
    device = "cuda"
    torch.manual_seed(0)

    q = torch.randn(num_tokens, N_HEAD, HEAD_DIM, dtype=torch.bfloat16, device=device)
    positions = torch.randint(
        0, MAX_POS, (num_tokens,), dtype=torch.int64, device=device
    )
    cos_sin_cache = torch.randn(MAX_POS, ROPE_DIM, dtype=torch.float32, device=device)
    weights = torch.randn(num_tokens, N_HEAD, dtype=torch.bfloat16, device=device)
    softmax_scale = HEAD_DIM**-0.5
    head_scale = N_HEAD**-0.5

    def run():
        return fused_indexer_q_rope_quant(
            positions,
            q,
            cos_sin_cache,
            weights,
            softmax_scale,
            head_scale,
            use_fp4=(provider == "mxfp4"),
            use_nvfp4=(provider == "nvfp4"),
        )

    with mock.patch(
        "vllm.models.deepseek_v4.common.ops.fused_indexer_q.has_cutedsl",
        return_value=False,
    ):
        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(run, quantiles=quantiles)

    gbps = _bytes_moved(num_tokens, provider) / (ms * 1e-3) / 1e9
    print(
        f"  num_tokens={num_tokens:>6} {provider:>6}:"
        f" {ms * 1000:8.2f} us  {gbps:7.1f} GB/s"
    )
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark fused indexer-Q RoPE+quant kernels (FP8/MXFP4/NVFP4)."
    )
    parser.add_argument("--save-path", type=str, default=None)
    args = parser.parse_args()

    benchmark.run(print_data=True, save_path=args.save_path)
