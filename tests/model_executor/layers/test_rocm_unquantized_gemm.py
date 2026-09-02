# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
import torch

from vllm.platforms import current_platform

if current_platform.is_cuda():
    pytest.skip(
        "ROCm skinny GEMM tests are not supported on CUDA.",
        allow_module_level=True,
    )

from vllm.model_executor.layers import utils


def test_rocm_unquantized_gemm_gfx1x_wvsplitk_path(monkeypatch):
    x = torch.randn(1, 64, dtype=torch.float16)
    weight = torch.randn(128, 64, dtype=torch.float16)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitk_mock = MagicMock(side_effect=lambda w, x_view, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)
    llmm1_mock = MagicMock(side_effect=lambda w, x_view, _: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitk_mock.assert_called_once()
    llmm1_mock.assert_not_called()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


def test_rocm_unquantized_gemm_makes_skinny_activation_contiguous(monkeypatch):
    x = torch.randn(64, 4, dtype=torch.float16).t()
    weight = torch.randn(128, 64, dtype=torch.float16)
    assert x.shape == (4, 64)
    assert x.stride() == (1, 4)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitk_mock = MagicMock(side_effect=lambda w, x_view, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitk_mock.assert_called_once()
    x_view = wvsplitk_mock.call_args.args[1]
    assert x_view.is_contiguous()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


def test_rocm_unquantized_gemm_makes_llmm1_activation_contiguous(monkeypatch):
    x = torch.randn(1, 128, dtype=torch.float16)[:, ::2]
    weight = torch.randn(4, 64, dtype=torch.float16)
    assert x.shape == (1, 64)
    assert x.stride() == (128, 2)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    llmm1_mock = MagicMock(side_effect=lambda w, x_view, _: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    llmm1_mock.assert_called_once()
    x_view = llmm1_mock.call_args.args[1]
    assert x_view.is_contiguous()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("noncontiguous_operand", ["weight", "bias"])
def test_rocm_unquantized_gemm_rejects_unsupported_skinny_layouts(
    monkeypatch, noncontiguous_operand
):
    x = torch.randn(4, 64, dtype=torch.float16)
    weight = torch.randn(128, 64, dtype=torch.float16)
    bias = torch.randn(128, dtype=torch.float16)
    if noncontiguous_operand == "weight":
        weight = torch.randn(64, 128, dtype=torch.float16).t()
        assert not weight.is_contiguous()
    else:
        bias = torch.randn(256, dtype=torch.float16)[::2]
        assert not bias.is_contiguous()

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.rocm_aiter_ops, "is_tgemm_enabled", lambda: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitk_mock = MagicMock()
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)
    llmm1_mock = MagicMock()
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, bias)
    ref = torch.nn.functional.linear(x, weight, bias)

    wvsplitk_mock.assert_not_called()
    llmm1_mock.assert_not_called()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


def _row_stride_padded(weight: torch.Tensor, pad: int) -> torch.Tensor:
    """A weight whose row stride is offset by ``pad`` elements, as produced by
    ``maybe_pad_weight_avoid_cache_aliasing``."""
    padded = torch.nn.functional.pad(weight, (0, pad))[..., :-pad]
    assert padded.shape == weight.shape
    assert padded.stride() == (weight.shape[1] + pad, 1)
    return padded


def test_rocm_unquantized_gemm_wvsplitk_accepts_row_stride_padded_weight(monkeypatch):
    # wvSplitK reads the weight row stride, so a row-stride-padded weight must
    # still reach it instead of falling back to torch.nn.functional.linear.
    x = torch.randn(4, 64, dtype=torch.float16)
    weight = _row_stride_padded(torch.randn(128, 64, dtype=torch.float16), 64)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.rocm_aiter_ops, "is_tgemm_enabled", lambda: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitk_mock = MagicMock(side_effect=lambda w, x_view, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitk_mock.assert_called_once()
    assert wvsplitk_mock.call_args.args[0].stride() == weight.stride()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


def test_rocm_unquantized_gemm_llmm1_rejects_row_stride_padded_weight(monkeypatch):
    # LLMM1 indexes the weight by K off a raw pointer, so a padded row stride
    # must fall back even though wvSplitK would accept it.
    x = torch.randn(1, 64, dtype=torch.float16)
    weight = _row_stride_padded(torch.randn(4, 64, dtype=torch.float16), 64)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.rocm_aiter_ops, "is_tgemm_enabled", lambda: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    llmm1_mock = MagicMock()
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    llmm1_mock.assert_not_called()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "k,element_size,expected",
    [
        (4096, 2, 64),  # 8192 B row stride, aliases
        (12288, 2, 64),  # 24576 B row stride, aliases
        (1024, 2, 64),  # smallest aliasing bf16/fp16 K
        (2560, 2, 0),  # 5120 B, no aliasing
        (11008, 2, 0),  # 22016 B, no aliasing
        (2048, 1, 128),  # fp8: 2048 B, aliases, twice the pad
        (512, 4, 32),  # fp32: 2048 B, aliases, half the pad
    ],
)
def test_cache_aliasing_pad_elems(k, element_size, expected):
    assert utils.cache_aliasing_pad_elems(k, element_size, "gfx1151") == expected


@pytest.mark.parametrize("gcn_arch", ["gfx942", "gfx1201", "gfx1250"])
def test_cache_aliasing_pad_elems_skips_unlisted_arch(gcn_arch):
    # Only architectures in CACHE_ALIASING_GEOMETRY are padded, even for a K whose
    # row stride would alias on gfx11.
    assert utils.cache_aliasing_pad_elems(4096, 2, gcn_arch) == 0


def test_maybe_pad_weight_avoid_cache_aliasing(monkeypatch):
    monkeypatch.setattr(utils.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.get_gcn_arch", lambda: "gfx1151")
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_LINEAR_PADDING", True)

    weight = torch.randn(128, 4096, dtype=torch.bfloat16)
    padded = utils.maybe_pad_weight_avoid_cache_aliasing(weight)

    assert padded.shape == weight.shape
    assert padded.stride() == (4096 + 64, 1)
    assert torch.equal(padded, weight)
    # Idempotent: the padded stride no longer aliases, so a second call is a no-op
    # and preserves the pointer captured by any graph.
    repadded = utils.maybe_pad_weight_avoid_cache_aliasing(padded)
    assert repadded.data_ptr() == padded.data_ptr()


@pytest.mark.parametrize(
    "weight",
    [
        torch.randn(128, 2560, dtype=torch.bfloat16),  # no aliasing
        torch.randn(2560, 128, dtype=torch.bfloat16).t(),  # not contiguous
        torch.randn(2, 128, 4096, dtype=torch.bfloat16),  # not 2D
    ],
)
def test_maybe_pad_weight_avoid_cache_aliasing_skips(monkeypatch, weight):
    monkeypatch.setattr(utils.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.get_gcn_arch", lambda: "gfx1151")
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_LINEAR_PADDING", True)

    assert (
        utils.maybe_pad_weight_avoid_cache_aliasing(weight).data_ptr()
        == weight.data_ptr()
    )


def test_maybe_pad_weight_avoid_cache_aliasing_respects_env(monkeypatch):
    monkeypatch.setattr(utils.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.get_gcn_arch", lambda: "gfx1151")
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_LINEAR_PADDING", False)

    weight = torch.randn(128, 4096, dtype=torch.bfloat16)
    assert (
        utils.maybe_pad_weight_avoid_cache_aliasing(weight).data_ptr()
        == weight.data_ptr()
    )


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only kernel test")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rocm_unquantized_gemm_noncontiguous_activation_real_kernel(monkeypatch, dtype):
    x = torch.randn(64, 4, device="cuda", dtype=dtype).t()
    weight = torch.randn(128, 64, device="cuda", dtype=dtype)
    assert x.stride() == (1, 4)

    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    original_wvsplitk = utils.ops.wvSplitK
    wvsplitk_mock = MagicMock(side_effect=original_wvsplitk)
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitk_mock.assert_called_once()
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only kernel test")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rocm_unquantized_gemm_padded_weight_real_kernel(monkeypatch, dtype):
    x = torch.randn(4, 4096, device="cuda", dtype=dtype)
    weight = _row_stride_padded(torch.randn(128, 4096, device="cuda", dtype=dtype), 64)

    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    original_wvsplitk = utils.ops.wvSplitK
    wvsplitk_mock = MagicMock(side_effect=original_wvsplitk)
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitk_mock.assert_called_once()
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


def test_rocm_unquantized_gemm_gfx1x_n_gt_5_falls_back(monkeypatch):
    # wvSplitK skinny GEMM handles n in [1, 5] (see PR #40687); n > 5 must
    # fall back to torch.nn.functional.linear.
    x = torch.randn(6, 64, dtype=torch.float16)
    weight = torch.randn(128, 64, dtype=torch.float16)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitk_mock = MagicMock(side_effect=lambda w, x_view, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)
    llmm1_mock = MagicMock(side_effect=lambda w, x_view, _: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitk_mock.assert_not_called()
    llmm1_mock.assert_not_called()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


def test_rocm_unquantized_gemm_gfx950_wvsplitkrc_path(monkeypatch):
    x = torch.randn(1024, 16, dtype=torch.float16).t()
    weight = torch.randn(256, 1024, dtype=torch.float16)
    assert x.stride() == (1, 16)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: True)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitkrc_mock = MagicMock(side_effect=lambda x_view, w, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitKrc", wvsplitkrc_mock)
    wvsplitk_mock = MagicMock(side_effect=lambda w, x_view, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitkrc_mock.assert_called_once()
    wvsplitk_mock.assert_not_called()
    x_view = wvsplitkrc_mock.call_args.args[0]
    assert x_view.is_contiguous()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
