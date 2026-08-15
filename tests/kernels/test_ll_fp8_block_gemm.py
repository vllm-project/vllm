# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for cuteDSL low-latency FP8 block-scaled GEMM."""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


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
    Returns: a_fp8, a_scale_packed, a_scale_fp32, b_fp8, b_scale
    """
    torch.manual_seed(42)
    a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    import deep_gemm

    # Match the production TMA-aligned packed activation-scale layout.
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8_packed_for_deepgemm,
    )

    a_fp8, a_s_packed = per_token_group_quant_fp8_packed_for_deepgemm(
        a_bf16, group_size=SF_VEC, use_ue8m0=True
    )
    _, a_s_fp32 = deep_gemm.per_token_cast_to_fp8(a_bf16, use_ue8m0=True)
    b_fp8, b_scale = deep_gemm.per_block_cast_to_fp8(b_bf16, use_ue8m0=True)

    # Keep originals for DeepGEMM reference
    b_fp8_orig, b_scale_orig = b_fp8.clone(), b_scale.clone()

    # Weight preprocessing (same as process_weights_after_loading)
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
    )
    from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used

    if is_deep_gemm_e8m0_used():
        b_fp8, b_scale = deepgemm_post_process_fp8_weight_block(
            b_fp8,
            b_scale,
            quant_block_shape=(128, 128),
            use_e8m0=True,
        )

    return a_fp8, a_s_packed, a_s_fp32, b_fp8, b_scale, b_fp8_orig, b_scale_orig


def _ref_deepgemm(a_fp8, a_scale_fp32, b_fp8_pp, b_scale_pp):
    """Reference: DeepGEMM via vLLM's fp8_gemm_nt wrapper (handles scale format)."""
    from vllm.utils.deep_gemm import fp8_gemm_nt, is_deep_gemm_e8m0_used

    M = a_fp8.shape[0]
    N = b_fp8_pp.shape[0]
    output = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    fp8_gemm_nt(
        (a_fp8, a_scale_fp32),
        (b_fp8_pp, b_scale_pp),
        output,
        is_deep_gemm_e8m0_used=is_deep_gemm_e8m0_used(),
    )
    return output


def _run_gemm(a_fp8, a_scale_packed, b_fp8, b_scale):
    """Run the LL FP8 block-scaled GEMM (uses packed int32 act scales)."""
    import vllm.model_executor.kernels.linear.cute_dsl.ll_fp8_block  # noqa: F401
    from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used

    M = a_fp8.shape[0]
    N = b_fp8.shape[0]
    output = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    torch.ops.vllm.ll_fp8_block_dispatch_op(
        a_fp8, a_scale_packed, b_fp8, b_scale, output, is_deep_gemm_e8m0_used()
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


@pytest.mark.parametrize("M", [1, 2, 4, 8, 16, 17, 32])
@pytest.mark.parametrize("N,K,desc", SHAPES, ids=[s[2] for s in SHAPES])
def test_correctness(M, N, K, desc):
    a_fp8, a_s_packed, a_s_fp32, b_fp8, b_s, b_fp8_o, b_s_o = _make_fp8_tensors(M, N, K)
    out = _run_gemm(a_fp8, a_s_packed, b_fp8, b_s)
    ref = _ref_deepgemm(a_fp8, a_s_fp32, b_fp8, b_s)
    assert out.dtype == torch.bfloat16
    assert out.shape == (M, N)
    _assert_close(out, ref, context=f"M={M} {desc}")


# =================================================================
# Dispatch boundary: LL kernel vs DeepGEMM fallback
# =================================================================


@pytest.mark.parametrize("M", [1, 16, 17, 32])
def test_dispatch_ll_kernel(M):
    """The extended-M production shape uses LL through M=32."""
    N, K = 1024, 4096
    a_fp8, a_s_packed, a_s_fp32, b_fp8, b_s, b_fp8_o, b_s_o = _make_fp8_tensors(M, N, K)
    out = _run_gemm(a_fp8, a_s_packed, b_fp8, b_s)
    ref = _ref_deepgemm(a_fp8, a_s_fp32, b_fp8, b_s)
    assert out.shape == (M, N)
    _assert_close(out, ref, context=f"dispatch M={M}")


# =================================================================
# Output dtype
# =================================================================


@pytest.mark.parametrize("M", [1, 8])
def test_output_bf16(M):
    a_fp8, a_s_packed, a_s_fp32, b_fp8, b_s, b_fp8_o, b_s_o = _make_fp8_tensors(
        M, 1024, 4096
    )
    out = _run_gemm(a_fp8, a_s_packed, b_fp8, b_s)
    assert out.dtype == torch.bfloat16


# =================================================================
# Numerical: no NaN/Inf
# =================================================================


@pytest.mark.parametrize("N,K,desc", SHAPES, ids=[s[2] for s in SHAPES])
def test_no_nan(N, K, desc):
    a_fp8, a_s_packed, a_s_fp32, b_fp8, b_s, b_fp8_o, b_s_o = _make_fp8_tensors(1, N, K)
    out = _run_gemm(a_fp8, a_s_packed, b_fp8, b_s)
    assert torch.isfinite(out).all(), f"NaN/Inf in {desc}"


# =================================================================
# Determinism
# =================================================================


@pytest.mark.parametrize("M", [1, 4, 8, 16])
def test_deterministic(M):
    a_fp8, a_s_packed, a_s_fp32, b_fp8, b_s, b_fp8_o, b_s_o = _make_fp8_tensors(
        M, 1536, 4096
    )
    out1 = _run_gemm(a_fp8, a_s_packed, b_fp8, b_s)
    out2 = _run_gemm(a_fp8, a_s_packed, b_fp8, b_s)
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
    a_fp8, a_s_packed, a_s_fp32, b_fp8, b_s, b_fp8_o, b_s_o = _make_fp8_tensors(1, N, K)
    out = _run_gemm(a_fp8, a_s_packed, b_fp8, b_s)
    assert out.shape == (1, N)
    _assert_close(
        out, _ref_deepgemm(a_fp8, a_s_fp32, b_fp8_o, b_s_o), context=f"M=1 {N}x{K}"
    )


def test_production_scale_layout():
    a_fp8, a_scale, _, b_fp8, b_scale, _, _ = _make_fp8_tensors(4, 1024, 4096)
    assert a_scale.shape == (4, 8)
    assert a_scale.stride() == (1, 4)
    assert b_scale.shape == (1024, 8)
    assert b_scale.stride() == (1, 1024)
    assert a_fp8.is_contiguous()
    assert b_fp8.is_contiguous()


@pytest.mark.parametrize(
    "M,N,K,use_e8m0,expected",
    [
        (1, 128, 512, True, "ll"),
        (16, 4096, 4096, True, "ll"),
        (17, 4096, 4096, True, "deep_gemm"),
        (17, 128, 4096, True, "deep_gemm"),
        (32, 1536, 4096, True, "ll"),
        (32, 1024, 4096, True, "ll"),
        (32, 4096, 2048, True, "ll"),
        (32, 4096, 512, True, "ll"),
        (33, 1024, 4096, True, "deep_gemm"),
        (1, 128, 768, True, "ll"),
        (1, 4104, 512, True, "ll"),
        (1, 128, 65536, True, "ll"),
        (1, 128, 65664, True, "deep_gemm"),
        (1, 4096, 16384, True, "ll"),
        (1, 4096, 16512, True, "deep_gemm"),
        (1, 8192, 2048, True, "ll"),
        (1, 8192, 2176, True, "deep_gemm"),
        (1, 16384, 1024, True, "deep_gemm"),
        (1, 128, 704, True, "deep_gemm"),
        (1, 130, 512, True, "deep_gemm"),
        (1, 128, 512, False, "deep_gemm"),
    ],
)
def test_dispatch_contract(monkeypatch, M, N, K, use_e8m0, expected):
    import vllm.model_executor.kernels.linear.cute_dsl.ll_fp8_block as module
    import vllm.utils.deep_gemm as deep_gemm

    calls = []
    monkeypatch.setattr(
        module,
        "_ll_fp8_block_gemm",
        lambda *args: calls.append("ll"),
    )
    monkeypatch.setattr(
        deep_gemm,
        "fp8_gemm_nt",
        lambda *args, **kwargs: calls.append("deep_gemm"),
    )

    q_input = torch.empty((M, K), dtype=torch.float8_e4m3fn)
    weight = torch.empty((N, K), dtype=torch.float8_e4m3fn)
    input_scale = torch.empty((M, (K + 511) // 512), dtype=torch.int32)
    weight_scale = torch.empty((N, (K + 511) // 512), dtype=torch.int32)
    output = torch.empty((M, N), dtype=torch.bfloat16)
    module._ll_fp8_block_dispatch(
        q_input,
        input_scale,
        weight,
        weight_scale,
        output,
        use_e8m0,
    )
    assert calls == [expected]


def _valid_direct_inputs():
    a_fp8, a_scale, _, b_fp8, b_scale, _, _ = _make_fp8_tensors(2, 1024, 4096)
    output = torch.empty((2, 1024), dtype=torch.bfloat16, device="cuda")
    return a_fp8, a_scale, b_fp8, b_scale, output


def test_direct_wrapper_rejects_wrong_rank():
    from vllm.model_executor.kernels.linear.cute_dsl.ll_fp8_block import (
        ll_fp8_block_gemm_kernel,
    )

    a_fp8, a_scale, b_fp8, b_scale, output = _valid_direct_inputs()
    with pytest.raises(ValueError, match="must be 2D"):
        ll_fp8_block_gemm_kernel(a_fp8.flatten(), a_scale, b_fp8, b_scale, output)


def test_direct_wrapper_rejects_wrong_input_dtype():
    from vllm.model_executor.kernels.linear.cute_dsl.ll_fp8_block import (
        ll_fp8_block_gemm_kernel,
    )

    a_fp8, a_scale, b_fp8, b_scale, output = _valid_direct_inputs()
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        ll_fp8_block_gemm_kernel(
            a_fp8.to(torch.bfloat16), a_scale, b_fp8, b_scale, output
        )


def test_direct_wrapper_rejects_wrong_scale_dtype():
    from vllm.model_executor.kernels.linear.cute_dsl.ll_fp8_block import (
        ll_fp8_block_gemm_kernel,
    )

    a_fp8, a_scale, b_fp8, b_scale, output = _valid_direct_inputs()
    with pytest.raises(ValueError, match="dtype=int32"):
        ll_fp8_block_gemm_kernel(a_fp8, a_scale.float(), b_fp8, b_scale, output)


def test_direct_wrapper_rejects_row_major_scales():
    from vllm.model_executor.kernels.linear.cute_dsl.ll_fp8_block import (
        ll_fp8_block_gemm_kernel,
    )

    a_fp8, a_scale, b_fp8, b_scale, output = _valid_direct_inputs()
    row_major_scale = a_scale.clone(memory_format=torch.contiguous_format)
    with pytest.raises(ValueError, match="packed column-major"):
        ll_fp8_block_gemm_kernel(a_fp8, row_major_scale, b_fp8, b_scale, output)


def test_direct_wrapper_rejects_wrong_output_dtype():
    from vllm.model_executor.kernels.linear.cute_dsl.ll_fp8_block import (
        ll_fp8_block_gemm_kernel,
    )

    a_fp8, a_scale, b_fp8, b_scale, output = _valid_direct_inputs()
    with pytest.raises(ValueError, match="output dtype=bfloat16"):
        ll_fp8_block_gemm_kernel(a_fp8, a_scale, b_fp8, b_scale, output.float())


def test_compile_key_dispatch_and_cache():
    from vllm.model_executor.kernels.linear.cute_dsl.ll_fp8_block import (
        ll_fp8_block_gemm_kernel,
    )

    default_key = ll_fp8_block_gemm_kernel.CompileKey()
    wo_key = ll_fp8_block_gemm_kernel.CompileKey(tile_n=8, num_stages=3)
    down_key = ll_fp8_block_gemm_kernel.CompileKey(num_dma_warps=2)
    wide_key = ll_fp8_block_gemm_kernel.CompileKey(num_stages=3, num_dma_warps=2)
    staged_key = ll_fp8_block_gemm_kernel.CompileKey(num_stages=3)
    tail_key = ll_fp8_block_gemm_kernel.CompileKey(has_k_tail=True)
    assert ll_fp8_block_gemm_kernel.dispatch(M=1, K=4096, N=1024) == wo_key
    assert ll_fp8_block_gemm_kernel.dispatch(M=16, K=2048, N=4096) == down_key
    assert ll_fp8_block_gemm_kernel.dispatch(M=10, K=512, N=4096) == down_key
    assert ll_fp8_block_gemm_kernel.dispatch(M=11, K=512, N=4096) == default_key
    assert ll_fp8_block_gemm_kernel.dispatch(M=11, K=1024, N=8192) == wide_key
    assert ll_fp8_block_gemm_kernel.dispatch(M=12, K=1024, N=8192) == down_key
    assert ll_fp8_block_gemm_kernel.dispatch(M=16, K=4096, N=1536) == staged_key
    assert ll_fp8_block_gemm_kernel.dispatch(M=1, K=768, N=128) == tail_key

    keys = ll_fp8_block_gemm_kernel.get_warmup_keys(
        shapes=((4096, 1024), (2048, 4096), (1024, 8192), (768, 128))
    )
    assert keys == [wo_key, down_key, wide_key, tail_key]

    key = default_key
    ll_fp8_block_gemm_kernel.compile(key)
    compiled = ll_fp8_block_gemm_kernel._compiled_cache[key]
    ll_fp8_block_gemm_kernel.compile(key)
    assert ll_fp8_block_gemm_kernel._compiled_cache[key] is compiled


@pytest.mark.parametrize(
    ("M", "N", "K", "message"),
    [
        (33, 1024, 4096, "M to be in"),
        (1, 1024, 704, "K to be divisible by 128"),
        (1, 4102, 4096, "N to be divisible by 8"),
    ],
)
def test_direct_wrapper_rejects_unsupported_shape(M, N, K, message):
    from vllm.model_executor.kernels.linear.cute_dsl.ll_fp8_block import (
        ll_fp8_block_gemm_kernel,
    )

    q_input = torch.empty((M, K), dtype=torch.float8_e4m3fn, device="cuda")
    weight = torch.empty((N, K), dtype=torch.float8_e4m3fn, device="cuda")
    packed_k = (K + 511) // 512
    input_scale = (
        torch.empty((M, packed_k), dtype=torch.int32, device="cuda")
        .t()
        .contiguous()
        .t()
    )
    weight_scale = (
        torch.empty((N, packed_k), dtype=torch.int32, device="cuda")
        .t()
        .contiguous()
        .t()
    )
    output = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match=message):
        ll_fp8_block_gemm_kernel(q_input, input_scale, weight, weight_scale, output)


@pytest.mark.parametrize("N,K", [(128, 128), (1024, 768), (1024, 4608), (4224, 4096)])
def test_extended_supported_shapes(N, K):
    a_fp8, a_s_packed, a_s_fp32, b_fp8, b_s, _, _ = _make_fp8_tensors(1, N, K)
    out = _run_gemm(a_fp8, a_s_packed, b_fp8, b_s)
    ref = _ref_deepgemm(a_fp8, a_s_fp32, b_fp8, b_s)
    _assert_close(out, ref, context=f"extended shape N={N} K={K}")


def test_model_warmup_finds_selected_kernel_shapes():
    from types import SimpleNamespace

    from vllm.model_executor.kernels.linear.cute_dsl.ll_fp8_block import (
        LLFp8BlockScaledMMKernel,
    )
    from vllm.model_executor.warmup.kernel_warmup import (
        _ll_fp8_block_shapes_from_model,
    )

    model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.empty((1024, 4096), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.quant_method = SimpleNamespace(
        fp8_linear=object.__new__(LLFp8BlockScaledMMKernel)
    )
    model.add_module("linear", layer)
    assert _ll_fp8_block_shapes_from_model(model) == ((4096, 1024),)

    layer.quant_method.fp8_linear = object()
    assert _ll_fp8_block_shapes_from_model(model) == ()


def test_cuda_graph_replay():
    from vllm.model_executor.kernels.linear.cute_dsl.ll_fp8_block import (
        ll_fp8_block_gemm_kernel,
    )

    a_fp8, a_scale, a_scale_fp32, b_fp8, b_scale, _, _ = _make_fp8_tensors(
        4, 1024, 4096
    )
    output = torch.empty((4, 1024), dtype=torch.bfloat16, device="cuda")
    ll_fp8_block_gemm_kernel(a_fp8, a_scale, b_fp8, b_scale, output)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ll_fp8_block_gemm_kernel(a_fp8, a_scale, b_fp8, b_scale, output)
    graph.replay()
    torch.accelerator.synchronize()

    ref = _ref_deepgemm(a_fp8, a_scale_fp32, b_fp8, b_scale)
    _assert_close(output, ref, context="CUDA graph replay")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
