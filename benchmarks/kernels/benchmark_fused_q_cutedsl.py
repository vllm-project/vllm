# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the CuTeDSL and Triton implementations of the DSA fused query op.

``fused_q`` dispatches to the CuTeDSL kernel on SM100 when the dtypes and
shapes are supported and falls back to Triton otherwise. This benchmark forces
each path so the two can be compared directly.
"""

import argparse
from unittest.mock import patch

import torch

import vllm.models.deepseek_v32.common.kernels as K
from vllm.triton_utils import triton

# GLM-5.2 / DeepSeek-V3.2 DSA dimensions.
NOPE_DIM = 512
ROPE_DIM = 64
INDEX_HEADS = 32
INDEX_HEAD_DIM = 128
MAX_POS = 16384

# num_heads is per-rank: 128 attention heads over TP 8/4/2.
NUM_HEADS = [16, 32, 64]
# MTP=5 makes a concurrency-N decode batch 6*N tokens.
NUM_TOKENS = [1, 6, 24, 96, 384, 1536, 4096]


def make_inputs(num_tokens: int, num_heads: int):
    dev = "cuda"
    bf16 = torch.bfloat16
    return dict(
        positions=torch.randint(0, MAX_POS, (num_tokens,), device=dev),
        q_pe=torch.randn(num_tokens, num_heads, ROPE_DIM, device=dev, dtype=bf16),
        q_pe_cos_sin_cache=torch.randn(MAX_POS, ROPE_DIM, device=dev, dtype=bf16),
        index_q=torch.randn(
            num_tokens, INDEX_HEADS, INDEX_HEAD_DIM, device=dev, dtype=bf16
        ),
        index_q_cos_sin_cache=torch.randn(MAX_POS, ROPE_DIM, device=dev, dtype=bf16),
        ql_nope=torch.randn(num_tokens, num_heads, NOPE_DIM, device=dev, dtype=bf16),
        q_scale=torch.tensor([0.5], device=dev, dtype=torch.float32),
        index_weights=torch.randn(
            num_tokens, INDEX_HEADS, device=dev, dtype=torch.float32
        ),
        index_weights_softmax_scale=INDEX_HEAD_DIM**-0.5,
        index_weights_head_scale=INDEX_HEADS**-0.5,
    )


def run(kwargs, *, cutedsl: bool):
    # is_fused_q_cutedsl_supported gates the dispatch inside fused_q.
    from vllm.models.deepseek_v32.nvidia.ops import fused_q_cutedsl as C

    with patch.object(C, "is_fused_q_cutedsl_supported", lambda *_, **__: cutedsl):
        K.fused_q(**kwargs)


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["num_tokens"],
            x_vals=NUM_TOKENS,
            line_arg="provider",
            line_vals=["triton", "cutedsl"],
            line_names=["Triton", "CuTeDSL"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="us",
            plot_name=f"fused_q-num_heads-{nh}",
            args={"num_heads": nh},
        )
        for nh in NUM_HEADS
    ]
)
def benchmark(num_tokens, num_heads, provider):
    kwargs = make_inputs(num_tokens, num_heads)
    fn = lambda: run(kwargs, cutedsl=provider == "cutedsl")  # noqa: E731
    fn()  # warm up JIT / kernel compilation before timing
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-path", type=str, default=None)
    args = parser.parse_args()

    from vllm.platforms import current_platform
    from vllm.utils.import_utils import has_cutedsl

    if not (current_platform.has_device_capability(100) and has_cutedsl()):
        raise SystemExit("fused_q CuTeDSL requires an SM100 device and cutlass-dsl")

    benchmark.run(print_data=True, show_plots=False, save_path=args.save_path)
