# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for cuteDSL low-latency FP8 block-scaled GEMM."""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


@pytest.fixture(autouse=True, scope="module")
def _require_sm100_and_cutedsl():
    major, _ = torch.cuda.get_device_capability()
    if major < 10:
        pytest.skip("Requires SM100+ (Blackwell)")
    from vllm.model_executor.kernels.linear.cute_dsl.ll_fp8_block import (
        is_available,
    )

    if not is_available():
        pytest.skip("cuteDSL (CUTLASS Python) not installed")


# ===== Helpers =====

SF_VEC = 128  # scale group size (matches production: K//128 groups)


def _make_fp8_tensors(M, N, K):
    """Create FP8 tensors with block scales matching production format.

    Uses deep_gemm quantization + vLLM's weight preprocessing
    (process_weights_after_loading path) to match production format.
    Returns: a_fp8, a_scale, b_fp8, b_scale (all on CUDA)
    """
    torch.manual_seed(42)
    a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    import deep_gemm

    a_fp8, a_scale = deep_gemm.per_token_cast_to_fp8(a_bf16, use_ue8m0=True)
    b_fp8, b_scale = deep_gemm.per_block_cast_to_fp8(b_bf16, use_ue8m0=True)

    # Weight preprocessing (same as process_weights_after_loading)
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
    )
    from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used

    if is_deep_gemm_e8m0_used():
        b_fp8, b_scale = deepgemm_post_process_fp8_weight_block(
            b_fp8, b_scale,
            quant_block_shape=(128, 128),
            use_e8m0=True,
        )

    return a_fp8, a_scale, b_fp8, b_scale


def _ref_deepgemm(a_fp8, a_scale, b_fp8, b_scale):
    """Reference: DeepGEMM FP8 block-scaled GEMM."""
    import deep_gemm

    M = a_fp8.shape[0]
    N = b_fp8.shape[0]
    output = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    deep_gemm.fp8_gemm_nt(
        (a_fp8, a_scale), (b_fp8, b_scale), output
    )
    return output


def _run_gemm(a_fp8, a_scale, b_fp8, b_scale):
    """Run the LL FP8 block-scaled GEMM via custom op."""
    from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used

    # Trigger custom op registration
    import vllm.model_executor.kernels.linear.cute_dsl.ll_fp8_block  # noqa: F401

    M = a_fp8.shape[0]
    N = b_fp8.shape[0]
    output = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    torch.ops.vllm.ll_fp8_block_dispatch_op(
        a_fp8, a_scale, b_fp8, b_scale, output, is_deep_gemm_e8m0_used()
    )
    return output


def _assert_close(out, ref, *, min_cos_sim=0.98, context=""):
    """Check output is close to reference via cosine similarity."""
    assert out.device.type == "cuda", f"{context}: not on CUDA"
    assert torch.isfinite(out).all(), f"{context}: NaN/Inf"
    cos = F.cosine_similarity(
        out.reshape(-1).float(), ref.reshape(-1).float(), dim=0
    ).item()
    assert cos > min_cos_sim, (
        f"{context}: cos_sim {cos:.6f} < {min_cos_sim} "
        f"(abs_err={(out.float() - ref.float()).abs().max().item():.2e})"
    )


# ===== Production shapes (after TP=4) =====

SHAPES = [
    (1536, 4096, "fused_wqa_wkv"),
    (1024, 4096, "wo_related"),
    (4096, 2048, "shared_down"),
    (4096, 512, "small_K"),
]


# =================================================================
# Correctness across M values
# =================================================================


@pytest.mark.parametrize("M", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("N,K,desc", SHAPES, ids=[s[2] for s in SHAPES])
def test_correctness(M, N, K, desc):
    a_fp8, a_scale, b_fp8, b_scale = _make_fp8_tensors(
        M, N, K
    )
    out = _run_gemm(a_fp8, a_scale, b_fp8, b_scale)
    ref = _ref_deepgemm(a_fp8, a_scale, b_fp8, b_scale)
    assert out.dtype == torch.bfloat16
    assert out.shape == (M, N)
    _assert_close(out, ref, context=f"M={M} {desc}")


# =================================================================
# Dispatch boundary: LL kernel vs DeepGEMM fallback
# =================================================================


@pytest.mark.parametrize("M", [1, 16, 17, 32])
def test_dispatch_boundary_M(M):
    """M<=16 uses LL kernel, M>16 falls back to DeepGEMM."""
    N, K = 1024, 4096
    a_fp8, a_scale, b_fp8, b_scale = _make_fp8_tensors(
        M, N, K
    )
    out = _run_gemm(a_fp8, a_scale, b_fp8, b_scale)
    ref = _ref_deepgemm(a_fp8, a_scale, b_fp8, b_scale)
    assert out.shape == (M, N)
    _assert_close(out, ref, context=f"dispatch M={M}")


# =================================================================
# Output dtype
# =================================================================


@pytest.mark.parametrize("M", [1, 8])
def test_output_bf16(M):
    a_fp8, a_scale, b_fp8, b_scale = _make_fp8_tensors(M, 1024, 4096)
    out = _run_gemm(a_fp8, a_scale, b_fp8, b_scale)
    assert out.dtype == torch.bfloat16


# =================================================================
# Numerical: no NaN/Inf
# =================================================================


@pytest.mark.parametrize("N,K,desc", SHAPES, ids=[s[2] for s in SHAPES])
def test_no_nan(N, K, desc):
    a_fp8, a_scale, b_fp8, b_scale = _make_fp8_tensors(1, N, K)
    out = _run_gemm(a_fp8, a_scale, b_fp8, b_scale)
    assert torch.isfinite(out).all(), f"NaN/Inf in {desc}"


# =================================================================
# Determinism
# =================================================================


@pytest.mark.parametrize("M", [1, 4, 8, 16])
def test_deterministic(M):
    a_fp8, a_scale, b_fp8, b_scale = _make_fp8_tensors(M, 1536, 4096)
    out1 = _run_gemm(a_fp8, a_scale, b_fp8, b_scale)
    out2 = _run_gemm(a_fp8, a_scale, b_fp8, b_scale)
    torch.testing.assert_close(out1, out2, atol=0, rtol=0)


# =================================================================
# Single token (M=1, critical decode path)
# =================================================================


@pytest.mark.parametrize(
    "N,K",
    [(1536, 4096), (1024, 4096), (4096, 2048), (4096, 512)],
    ids=["wqa", "wo", "down", "smallK"],
)
def test_single_token(N, K):
    a_fp8, a_scale, b_fp8, b_scale = _make_fp8_tensors(
        1, N, K
    )
    out = _run_gemm(a_fp8, a_scale, b_fp8, b_scale)
    assert out.shape == (1, N)
    _assert_close(out, _ref_deepgemm(a_fp8, a_scale, b_fp8, b_scale), context=f"M=1 {N}x{K}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
