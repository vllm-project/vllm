# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark and correctness check for ``ops.dsv4_norm_router_gemm``.

Two implementations are compared:

  1. ``unfused``   — ``vllm_ops.rms_norm`` then ``ops.dsv3_router_gemm``,
                     i.e. the current vLLM hot path (two kernel launches).
  2. ``fused``     — ``ops.dsv4_norm_router_gemm``, the new single-kernel
                     fused path.

Both produce ``(normed_x: bf16, router_logits: fp32)``.  The correctness
check verifies that ``fused`` and ``unfused`` agree to within ~1 bf16
ULP — that is the precision floor for this op.
"""

import argparse

import torch

from vllm import _custom_ops as vllm_ops
from vllm.triton_utils import triton

# The fused dsv4_norm_router_gemm kernel is templated only for DSV4-Pro
# (hidden_size=7168, num_experts=384).  Other shapes fall back to the
# unfused path on the Python side (NormGatedLinear), so benchmark only
# the configuration that the fused kernel actually targets.
HIDDEN_SIZE = 7168
NUM_EXPERTS_CHOICES = (384,)
RMS_EPS = 1e-6


def unfused_norm_router_gemm(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Call ``_C::rms_norm`` directly (mirroring ``_dsv4_pro_norm_gate``'s
    # fallback path) so the benchmarked baseline doesn't inherit any
    # Python wrapper overhead or risk falling through to the native
    # eager-primitive ``RMSNorm.forward_native`` path.
    normed = torch.empty_like(x)
    torch.ops._C.rms_norm(normed, x, norm_weight, eps)
    logits = vllm_ops.dsv3_router_gemm(normed, gate_weight, torch.float32)
    return normed, logits


def fused_norm_router_gemm(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return vllm_ops.dsv4_norm_router_gemm(x, norm_weight, gate_weight, eps)


def _make_inputs(num_tokens: int, num_experts: int, hidden_size: int, seed: int = 0):
    torch.manual_seed(seed)
    device = "cuda"
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    norm_w = torch.randn(hidden_size, dtype=torch.bfloat16, device=device)
    gate_w = torch.randn(num_experts, hidden_size, dtype=torch.bfloat16, device=device)
    # Down-scale gate_w so the GEMV output stays in a representable range.
    gate_w = gate_w / float(hidden_size) ** 0.5
    norm_w = (norm_w * 0.1) + 1.0
    return x, norm_w, gate_w


def calculate_diff(
    num_tokens: int,
    num_experts: int,
    hidden_size: int = HIDDEN_SIZE,
    normed_atol: float = 2e-3,
    logits_atol: float = 1e-2,
    rtol: float = 1e-2,
) -> None:
    x, norm_w, gate_w = _make_inputs(num_tokens, num_experts, hidden_size)

    normed_unfused, logits_unfused = unfused_norm_router_gemm(
        x.clone(), norm_w, gate_w, RMS_EPS
    )
    normed_fused, logits_fused = fused_norm_router_gemm(
        x.clone(), norm_w, gate_w, RMS_EPS
    )

    def _max_abs(a, b):
        return (a.float() - b.float()).abs().max().item()

    print(f"\n=== M={num_tokens} E={num_experts} H={hidden_size} ===")
    print(f"normed_x  |fused - unfused| = {_max_abs(normed_fused, normed_unfused):.3e}")
    print(f"logits    |fused - unfused| = {_max_abs(logits_fused, logits_unfused):.3e}")

    ok_normed = torch.allclose(
        normed_fused.float(),
        normed_unfused.float(),
        atol=normed_atol,
        rtol=rtol,
    )
    ok_logits = torch.allclose(
        logits_fused.float(),
        logits_unfused.float(),
        atol=logits_atol,
        rtol=rtol,
    )
    if ok_normed and ok_logits:
        print(
            f"OK   fused vs unfused within "
            f"normed_atol={normed_atol:.0e} logits_atol={logits_atol:.0e} "
            f"rtol={rtol:.0e}"
        )
    else:
        print(
            f"FAIL normed_ok={ok_normed} logits_ok={ok_logits}; "
            f"see max-abs values above"
        )


def get_benchmark():
    # Only num_tokens varies (DSV4-Pro hard-codes E=384); single-axis
    # sweep yields a clean line plot with M on the x-axis.
    num_experts = NUM_EXPERTS_CHOICES[0]

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["num_tokens"],
            x_vals=list(range(1, 17)),
            line_arg="provider",
            line_vals=["unfused", "fused"],
            line_names=["unfused (rms+dsv3)", "fused (dsv4)"],
            styles=[("green", "-"), ("red", "-")],
            ylabel="us",
            plot_name=f"norm-router-gemm-E{num_experts}-H{HIDDEN_SIZE}",
            args={},
        )
    )
    def benchmark(num_tokens, provider):
        x, norm_w, gate_w = _make_inputs(num_tokens, num_experts, HIDDEN_SIZE)

        quantiles = [0.5, 0.2, 0.8]
        if provider == "unfused":
            fn = lambda: unfused_norm_router_gemm(  # noqa: E731
                x, norm_w, gate_w, RMS_EPS
            )
        else:
            fn = lambda: fused_norm_router_gemm(  # noqa: E731
                x, norm_w, gate_w, RMS_EPS
            )

        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-path",
        type=str,
        default="./configs/norm_router_gemm/",
    )
    parser.add_argument(
        "--skip-bench",
        action="store_true",
        help="Run only the correctness check, not the perf sweep.",
    )
    args = parser.parse_args()

    # Correctness sweep over the full fast-path range M=1..16.
    for m in range(1, 17):
        for e in NUM_EXPERTS_CHOICES:
            calculate_diff(num_tokens=m, num_experts=e, hidden_size=HIDDEN_SIZE)

    if args.skip_bench:
        return

    benchmark = get_benchmark()
    benchmark.run(print_data=True, save_path=args.save_path)


if __name__ == "__main__":
    main()
