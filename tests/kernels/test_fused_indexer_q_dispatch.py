# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only Q dispatch/layout tests, not numerical or NVIDIA runtime validation.

Numerical GPU coverage lives in test_fused_indexer_q_rope_quant.py.
"""

import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

producer = importlib.import_module("vllm.models.deepseek_v4.common.ops.fused_indexer_q")
CUTEDSL_MODULE = "vllm.models.deepseek_v4.nvidia.ops.fused_indexer_q_cutedsl"


def _platform(monkeypatch, *, rocm=False, gfx950=True, fnuz=False):
    monkeypatch.setattr(
        producer,
        "current_platform",
        SimpleNamespace(
            is_rocm=lambda: rocm,
            is_xpu=lambda: False,
            is_cpu=lambda: False,
            fp8_dtype=lambda: torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn,
        ),
    )
    rocm_module = ModuleType("vllm.platforms.rocm")
    on_gfx950 = Mock(return_value=gfx950)
    rocm_module.on_gfx950 = on_gfx950  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm.platforms.rocm", rocm_module)
    return on_gfx950


def _inputs(*, heads=64, head_dim=128):
    # Real host tensors exercise allocation, reinterpretation and return shapes.
    # Kernel launches are mocked; no output numerical values are asserted here.
    return (
        torch.zeros(3, dtype=torch.int64),
        torch.zeros(3, heads, head_dim, dtype=torch.bfloat16),
        torch.zeros(4, 64, dtype=torch.float32),
        torch.zeros(3, heads, dtype=torch.bfloat16),
        0.125,
        0.125,
    )


@pytest.mark.parametrize("rocm,fnuz", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("explicit_fp8", [False, True])
def test_default_q_remains_platform_fp8(monkeypatch, rocm, fnuz, explicit_fp8):
    gfx_probe = _platform(monkeypatch, rocm=rocm, gfx950=False, fnuz=fnuz)
    monkeypatch.setattr(producer, "has_cutedsl", lambda: False)
    fp8_kernel = Mock()
    fp4_kernel = Mock(side_effect=AssertionError("FP8 must not dispatch FP4"))
    monkeypatch.setattr(
        producer, "_FUSED_INDEXER_Q_ROPE_QUANT_TRITON_KERNEL", fp8_kernel
    )
    monkeypatch.setattr(
        producer, "_FUSED_INDEXER_Q_ROPE_MXFP4_TRITON_KERNEL", fp4_kernel
    )
    inputs = _inputs()

    kwargs = {"use_fp4": False} if explicit_fp8 else {}
    q, weights = producer.fused_indexer_q_rope_quant(*inputs, **kwargs)

    assert q.shape == inputs[1].shape
    assert q.dtype == (torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn)
    assert weights.shape == inputs[3].shape
    assert weights.dtype == torch.float32
    fp8_kernel.assert_called_once()
    args = fp8_kernel.call_args.args
    assert all(actual is expected for actual, expected in zip(args[:4], inputs[:4]))
    assert args[4:6] == inputs[4:6]
    assert args[6] is q
    assert args[7] is weights
    assert fp8_kernel.call_args.kwargs == {
        "fp8_max": 224.0 if fnuz else 448.0,
        "use_fnuz": fnuz,
    }
    fp4_kernel.assert_not_called()
    gfx_probe.assert_not_called()


@pytest.mark.parametrize("heads", [32, 64])
def test_rocm_fp4_uses_triton_and_swizzled_scale_contract(monkeypatch, heads):
    _platform(monkeypatch, rocm=True)
    cutedsl_probe = Mock(side_effect=AssertionError("ROCm FP4 must not probe CuTeDSL"))
    monkeypatch.setattr(producer, "has_cutedsl", cutedsl_probe)
    kernel = Mock()
    monkeypatch.setattr(producer, "_FUSED_INDEXER_Q_ROPE_MXFP4_TRITON_KERNEL", kernel)

    (q, scales), weights = producer.fused_indexer_q_rope_quant(
        *_inputs(heads=heads), use_fp4=True
    )

    assert q.shape == (3, heads, 64)
    assert q.dtype == torch.uint8
    assert scales.shape == (3, 1, 4, 16, 4)
    assert scales.dtype == torch.uint8
    assert weights.shape == (3, heads)
    assert weights.dtype == torch.float32
    kernel.assert_called_once()
    assert kernel.call_args.args[6] is q
    assert kernel.call_args.args[7] is scales
    assert kernel.call_args.args[8] is weights
    cutedsl_probe.assert_not_called()


@pytest.mark.parametrize(
    "gfx950,heads,head_dim",
    [(False, 64, 128), (True, 8, 128), (True, 144, 128), (True, 64, 256)],
)
def test_rocm_fp4_rejects_unsupported_architecture_or_shape(
    monkeypatch, gfx950, heads, head_dim
):
    _platform(monkeypatch, rocm=True, gfx950=gfx950)
    kernel = Mock(side_effect=AssertionError("unsupported inputs must not launch"))
    monkeypatch.setattr(producer, "_FUSED_INDEXER_Q_ROPE_MXFP4_TRITON_KERNEL", kernel)

    with pytest.raises(ValueError, match="requires gfx950, D=128, and H in"):
        producer.fused_indexer_q_rope_quant(
            *_inputs(heads=heads, head_dim=head_dim), use_fp4=True
        )
    kernel.assert_not_called()


@pytest.mark.parametrize("use_fp4", [False, True])
@pytest.mark.parametrize("use_cutedsl", [False, True])
@pytest.mark.parametrize("heads", [32, 64])
def test_cuda_q_retains_triton_and_cutedsl_dispatch(
    monkeypatch, use_fp4, use_cutedsl, heads
):
    gfx_probe = _platform(monkeypatch)
    monkeypatch.setattr(producer, "has_cutedsl", lambda: use_cutedsl)
    triton_kernel = Mock()
    cutedsl_kernel = Mock()
    unused_kernel = Mock(side_effect=AssertionError("wrong quantization dispatch"))
    monkeypatch.setattr(
        producer,
        "_FUSED_INDEXER_Q_ROPE_MXFP4_TRITON_KERNEL",
        triton_kernel if use_fp4 else unused_kernel,
    )
    monkeypatch.setattr(
        producer,
        "_FUSED_INDEXER_Q_ROPE_QUANT_TRITON_KERNEL",
        unused_kernel if use_fp4 else triton_kernel,
    )
    cutedsl = ModuleType(CUTEDSL_MODULE)
    cutedsl._INDEXER_Q_MXFP4_KERNEL = (  # type: ignore[attr-defined]
        cutedsl_kernel if use_fp4 else unused_kernel
    )
    cutedsl._INDEXER_Q_FP8_KERNEL = (  # type: ignore[attr-defined]
        unused_kernel if use_fp4 else cutedsl_kernel
    )
    monkeypatch.setitem(sys.modules, CUTEDSL_MODULE, cutedsl)
    inputs = _inputs(heads=heads)

    quantized, weights = producer.fused_indexer_q_rope_quant(*inputs, use_fp4=use_fp4)

    selected = cutedsl_kernel if use_cutedsl else triton_kernel
    unselected = triton_kernel if use_cutedsl else cutedsl_kernel
    selected.assert_called_once()
    unselected.assert_not_called()
    unused_kernel.assert_not_called()
    gfx_probe.assert_not_called()
    assert weights.shape == (3, heads)
    assert weights.dtype == torch.float32
    if use_cutedsl:
        kwargs = selected.call_args.kwargs
        for name, expected in zip(
            ("positions", "q", "cos_sin_cache", "weights"), inputs[:4]
        ):
            assert kwargs[name] is expected
        assert kwargs["weights_softmax_scale"] == inputs[4]
        assert kwargs["weights_head_scale"] == inputs[5]
        assert kwargs["weights_out"] is weights
    if use_fp4:
        q, scales = quantized
        assert q.shape == (3, heads, 64)
        assert q.dtype == torch.uint8
        assert scales.shape == (3, heads)
        assert scales.dtype == torch.int32
        kernel_scales = (
            selected.call_args.kwargs["q_scale"]
            if use_cutedsl
            else selected.call_args.args[7]
        )
        assert kernel_scales.shape == (3, heads, 4)
        assert kernel_scales.dtype == torch.uint8
        assert kernel_scales.data_ptr() == scales.data_ptr()
    else:
        assert quantized.shape == (3, heads, 128)
        assert quantized.dtype == torch.float8_e4m3fn
        if use_cutedsl:
            assert selected.call_args.kwargs["q_fp8"].dtype == torch.uint8
            assert selected.call_args.kwargs["q_fp8"].data_ptr() == quantized.data_ptr()
