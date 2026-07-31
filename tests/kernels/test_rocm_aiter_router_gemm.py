# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ROCm AITER router-gate GEMM: bf16 x bf16 -> out_dtype.

This is the ``GateLinear`` tier used on gfx950, where the MoE router gate is a
skinny GEMM (M=num_tokens, N=num_experts, K=hidden_size). Correctness baseline
is a float64 matmul; what ultimately matters is that numeric error never flips
the top-k expert selection by more than the kernel's own error.

The tier runs AITER's tuned bf16 kernel and casts, so the result carries bf16
output precision even when ``out_dtype`` is fp32, and the tuned kernels for
these shapes reduce over split-K, which contributes more error than that
rounding does. Either way the absolute error scales with the output magnitude,
which itself grows as sqrt(hidden_size) for unit-variance inputs. Tolerances
below are expressed as a fraction of the largest reference logit so they hold
across all three shapes rather than being tuned per shape.
"""

import pytest
import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms import current_platform

# Register torch.ops.vllm.rocm_aiter_router_gemm.
import vllm.model_executor.layers.fused_moe.router.gate_linear  # noqa: F401  isort: skip

# (hidden_size, num_experts): GLM-5/5.2, DeepSeek-V3 and Kimi-K2 routers.
SHAPES = [(6144, 256), (7168, 256), (7168, 384)]
NUM_TOKENS = [1, 2, 4, 8, 16, 32, 64, 128]

# Error as a fraction of the largest reference logit. Measured worst case for
# the tuned bf16 kernel across every shape/token count here is 8.8e-3; keep
# ~2x headroom for kernel-selection changes between AITER versions.
REL_TO_PEAK = 2e-2
# A top-k membership change needs error on both the promoted and demoted logit,
# so the tie window is twice the single-value tolerance.
TIE_REL_TO_PEAK = 2 * REL_TO_PEAK


def _requires_aiter_tgemm():
    if not current_platform.is_rocm():
        pytest.skip("AITER router GEMM requires ROCm")
    if not rocm_aiter_ops.is_tgemm_enabled():
        pytest.skip("AITER tuned GEMM not enabled (needs AITER linear + gfx950)")


def _run(x: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype):
    return torch.ops.vllm.rocm_aiter_router_gemm(x, weight, out_dtype)


def _assert_close_to_peak(out: torch.Tensor, ref: torch.Tensor, tol: float):
    """Compare after normalizing by the peak reference magnitude.

    ``assert_close``'s rtol is per-element, which is meaningless here: a logit
    that lands near zero is a cancelling sum of ~K large products, so its own
    magnitude says nothing about the error it accumulated.
    """
    scale = ref.abs().max().clamp(min=torch.finfo(torch.float32).tiny)
    torch.testing.assert_close(
        out.float() / scale, ref.float() / scale, atol=tol, rtol=0
    )


@pytest.mark.parametrize("hidden_dim,num_experts", SHAPES)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
def test_matches_reference(
    num_tokens: int, hidden_dim: int, num_experts: int, out_dtype: torch.dtype
):
    """bf16 activation x bf16 weight should track a float64 reference."""
    _requires_aiter_tgemm()
    torch.manual_seed(42)
    device = torch.device("cuda")
    x = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device=device)
    weight = torch.randn(num_experts, hidden_dim, dtype=torch.bfloat16, device=device)

    out = _run(x, weight, out_dtype)
    ref = x.double() @ weight.double().t()

    assert out.shape == (num_tokens, num_experts)
    assert out.dtype == out_dtype
    _assert_close_to_peak(out, ref, REL_TO_PEAK)


@pytest.mark.parametrize("hidden_dim,num_experts", SHAPES)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
def test_topk_routing_consistency(num_tokens: int, hidden_dim: int, num_experts: int):
    """The gate feeds top-k expert selection, so numeric error only matters if
    it changes the selected experts. Experts whose reference logit sits within
    the kernel's error of the k-th value are genuinely tied, so swapping them
    is acceptable; anything outside that window is a real routing bug."""
    _requires_aiter_tgemm()
    top_k = 8
    device = torch.device("cuda")
    for seed in range(5):
        torch.manual_seed(1000 + seed)
        x = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device=device)
        weight = torch.randn(
            num_experts, hidden_dim, dtype=torch.bfloat16, device=device
        )

        out = _run(x, weight, torch.float32)
        ref = x.double() @ weight.double().t()
        tie_window = TIE_REL_TO_PEAK * ref.abs().max().item()
        kernel_idx = out.topk(top_k, dim=-1).indices
        ref_vals, ref_idx = ref.topk(top_k, dim=-1)
        for t in range(num_tokens):
            got = set(kernel_idx[t].tolist())
            want = set(ref_idx[t].tolist())
            if got == want:
                continue
            kth = ref_vals[t, -1].item()
            for e in got.symmetric_difference(want):
                gap = abs(ref[t, e].item() - kth)
                assert gap < tie_window, (
                    f"top-{top_k} mismatch beyond tie tolerance: token {t}, "
                    f"expert {e}, gap {gap:.3e} > {tie_window:.3e}"
                )


@pytest.mark.parametrize("hidden_dim,num_experts", SHAPES)
@pytest.mark.parametrize("num_tokens", [1, 16, 128])
def test_matches_fp32_fallback(hidden_dim: int, num_experts: int, num_tokens: int):
    """The AITER tier must stay close to the fp32 fallback it replaces.

    When ``force_fp32_compute`` is set and no specialized kernel is available,
    ``GateLinear`` keeps fp32 weights and upcasts the activation, so the gate
    runs as an fp32 GEMM. Both operands still hold bf16-representable values, so
    the differences are in the reduction and the output: the tuned kernel
    accumulates over split-K rather than in one fp32 pass, which dominates, and
    the result is rounded to bf16 before the cast.
    """
    _requires_aiter_tgemm()
    torch.manual_seed(7)
    device = torch.device("cuda")
    x = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device=device)
    weight = torch.randn(num_experts, hidden_dim, dtype=torch.bfloat16, device=device)

    out = _run(x, weight, torch.float32)
    fp32_fallback = torch.nn.functional.linear(x.float(), weight.float())

    _assert_close_to_peak(out, fp32_fallback, REL_TO_PEAK)
