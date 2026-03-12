# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse

import torch

from vllm import _custom_ops as ops
from vllm.triton_utils import triton

# DeepSeek V3 dimensions
NOPE_DIM = 512
ROPE_DIM = 64
NUM_HEADS = 128

NUM_TOKENS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


def get_configs():
    return NUM_TOKENS


def make_inputs(num_tokens, dtype):
    """Create inputs matching the real code path.

    Args:
        contiguous_nope: If False, simulate the transposed BMM output
                         (non-contiguous nope with stride pattern from
                         [N,B,L].transpose(0,1)).
    """
    # Simulate: bmm output [N, B, L].transpose(0, 1) -> [B, N, L]
    raw = torch.randn(NUM_HEADS, num_tokens, NOPE_DIM, dtype=dtype, device="cuda")
    ql_nope = raw.transpose(0, 1)

    q_pe = torch.randn(num_tokens, NUM_HEADS, ROPE_DIM, dtype=dtype, device="cuda")
    return ql_nope, q_pe


# ---- Non-contiguous nope benchmark (real code path) ----
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=get_configs(),
        line_arg="provider",
        line_vals=["torch_cat", "concat_mla_q"],
        line_names=["torch.cat", "concat_mla_q (v8)"],
        styles=[("blue", "--"), ("green", "-")],
        ylabel="Latency (us)",
        plot_name="concat_mla_q-transposed",
        args={},
    )
)
def bench_transposed(num_tokens, provider):
    dtype = torch.bfloat16
    ql_nope, q_pe = make_inputs(num_tokens, dtype)

    q_out = torch.empty(
        num_tokens, NUM_HEADS, NOPE_DIM + ROPE_DIM, dtype=dtype, device="cuda"
    )

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch_cat":
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: torch.cat((ql_nope, q_pe), dim=-1), quantiles=quantiles, rep=500
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: ops.concat_mla_q(ql_nope, q_pe, q_out), quantiles=quantiles, rep=500
        )

    return ms * 1000, max_ms * 1000, min_ms * 1000  # us


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark concat_mla_q vs torch.cat")
    parser.add_argument(
        "--save-path", type=str, default=None, help="Path to save benchmark results"
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("CONCAT MLA Q KERNEL BENCHMARKS")
    print("=" * 70)
    print(f"Dimensions: nope={NOPE_DIM}, rope={ROPE_DIM}, heads={NUM_HEADS}")
    print(
        f"Per-head output: {NOPE_DIM + ROPE_DIM} bf16 = "
        f"{(NOPE_DIM + ROPE_DIM) * 2} bytes"
    )
    print(f"num_tokens (decode=batch_size, prefill=chunk_size): {NUM_TOKENS}")
    print("=" * 70)

    print("\n--- Non-contiguous nope inputs (transposed BMM output) ---")
    bench_transposed.run(print_data=True, save_path=args.save_path)

    print("\n" + "=" * 70)
    print("Benchmarking complete!")
    print("=" * 70)
