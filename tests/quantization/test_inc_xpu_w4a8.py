# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``INCXPUW4A8LinearMethod`` and the backend selector that reaches it.

The w4a8 path keeps int4 weights but dynamically quantizes activations to
per-token symmetric int8, so the GEMM stays on the int8 datapath instead of
upconverting weights to the activation dtype. It is opt-in through
``VLLM_XPU_INC_W4A16_BACKEND`` because which kernel is fastest depends on the
device: ARK can use XMX int8 on some XPUs, and the default ("auto") preference
order is deliberately left unchanged.

These tests stub out ``int4_gemm_w4a8`` so they run without an XPU: what needs
guarding is the calling convention (scale dtypes, argument order, weight reuse,
batch-size gate), which is where this path is easy to get silently wrong — the
kernel accepts bf16 scales and returns plausible-looking garbage rather than
raising.

Run `pytest tests/quantization/test_inc_xpu_w4a8.py`.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.inc.config_parser import INCLayerConfig
from vllm.model_executor.layers.quantization.inc.inc_linear import INCLinearMethod
from vllm.model_executor.layers.quantization.inc.schemes import INCWna16Scheme
from vllm.model_executor.layers.quantization.inc.schemes.inc_wna16_linear import (
    INCARKLinearMethod,
    INCXPULinearMethod,
    INCXPUW4A8LinearMethod,
)
from vllm.platforms import current_platform

_BACKEND_ENV = "VLLM_XPU_INC_W4A16_BACKEND"
_ARK_STATE = (
    "vllm.model_executor.layers.quantization.inc.schemes.inc_ark_ops.get_ark_state"
)
_QUANT_REF = "vllm._xpu_ops.xpu_ops.dynamic_per_token_int8_quant_ref"

IN_FEATURES = 128
OUT_FEATURES = 64


def make_layer_config(**overrides) -> INCLayerConfig:
    kwargs = {
        "bits": 4,
        "group_size": 128,
        "sym": True,
        "packing_format": "auto_round:auto_gptq",
        "backend": "auto",
        "data_type": "int",
        "quantized": True,
    }
    kwargs.update(overrides)
    return INCLayerConfig(**kwargs)


def _fake_xpu(monkeypatch, ark_available: bool = True) -> None:
    class DummyQuantLinear:
        pass

    monkeypatch.setattr(current_platform, "is_xpu", lambda: True)
    monkeypatch.setattr(current_platform, "is_cpu", lambda: False)
    monkeypatch.setattr(
        _ARK_STATE,
        lambda: (
            (True, None, object(), DummyQuantLinear)
            if ark_available
            else (False, "missing", None, None)
        ),
    )


def _with_w4a8_kernel(monkeypatch) -> None:
    """Present the op so the w4a8 support check passes off-XPU."""

    class FakeOps:
        int4_gemm_w4a8 = object()

    monkeypatch.setattr(torch.ops, "_xpu_C", FakeOps, raising=False)


def _dispatch(layer_config=None):
    return INCWna16Scheme().get_linear_method(
        object(), object(), "layer", layer_config or make_layer_config()
    )


@pytest.fixture
def w4a8_layer(monkeypatch):
    """A w4a8 method plus a layer whose weights are created but not processed."""
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: 1,
    )

    class DummyLayer(torch.nn.Module):
        pass

    layer = DummyLayer()
    method = INCXPUW4A8LinearMethod(make_layer_config())
    method.create_weights(
        layer=layer,
        input_size_per_partition=IN_FEATURES,
        output_partition_sizes=[OUT_FEATURES],
        input_size=IN_FEATURES,
        output_size=OUT_FEATURES,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *args, **kwargs: None,
    )
    return method, layer


def _stub_quant(monkeypatch, captured=None):
    def fake_quant(x, use_sym, bits):
        if captured is not None:
            captured["quant_args"] = (tuple(x.shape), use_sym, bits)
        m, k = x.shape
        return (
            torch.zeros(m, k, dtype=torch.int8),
            torch.ones(m, 1, dtype=x.dtype),
            torch.zeros(m, 1, dtype=torch.int32),
        )

    monkeypatch.setattr(_QUANT_REF, fake_quant, raising=False)


def _stub_gemm(monkeypatch, captured=None):
    def fake_gemm(*args):
        if captured is not None:
            captured["gemm_args"] = args
        # The real kernel always emits float16, whatever the model dtype.
        return torch.zeros(args[0].shape[0], OUT_FEATURES, dtype=torch.float16)

    monkeypatch.setattr(torch.ops._xpu_C, "int4_gemm_w4a8", fake_gemm, raising=False)


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def test_default_backend_keeps_ark_preference(monkeypatch) -> None:
    """The "auto" default must not change upstream behaviour.

    ARK still wins whenever it is importable.
    """
    monkeypatch.delenv(_BACKEND_ENV, raising=False)
    _fake_xpu(monkeypatch, ark_available=True)
    _with_w4a8_kernel(monkeypatch)

    method = _dispatch()

    assert isinstance(method, INCLinearMethod)
    assert isinstance(method.scheme, INCARKLinearMethod)


def test_w4a8_backend_overrides_available_ark(monkeypatch) -> None:
    """An explicit request wins even though ARK would otherwise be chosen."""
    monkeypatch.setenv(_BACKEND_ENV, "w4a8")
    _fake_xpu(monkeypatch, ark_available=True)
    _with_w4a8_kernel(monkeypatch)

    method = _dispatch()

    assert isinstance(method.scheme, INCXPUW4A8LinearMethod)
    assert not isinstance(method.scheme, INCARKLinearMethod)
    # w4a8 subclasses w4a16 so it can fall back for small batches.
    assert isinstance(method.scheme, INCXPULinearMethod)


def test_w4a16_backend_overrides_available_ark(monkeypatch) -> None:
    monkeypatch.setenv(_BACKEND_ENV, "w4a16")
    _fake_xpu(monkeypatch, ark_available=True)

    method = _dispatch()

    assert isinstance(method.scheme, INCXPULinearMethod)
    assert not isinstance(method.scheme, INCXPUW4A8LinearMethod)
    assert not isinstance(method.scheme, INCARKLinearMethod)


def test_ark_backend_requested_but_unavailable_raises(monkeypatch) -> None:
    """Explicitly asking for ARK must fail loudly, not fall back silently."""
    monkeypatch.setenv(_BACKEND_ENV, "ark")
    _fake_xpu(monkeypatch, ark_available=False)

    with pytest.raises(NotImplementedError, match="auto_round_kernel is unavailable"):
        _dispatch()


@pytest.mark.parametrize("backend", ["w4a16", "w4a8"])
def test_onednn_backends_reject_int2(monkeypatch, backend) -> None:
    """The oneDNN int4 GEMMs cannot serve int2; ARK is the only int2 path."""
    monkeypatch.setenv(_BACKEND_ENV, backend)
    _fake_xpu(monkeypatch, ark_available=True)
    _with_w4a8_kernel(monkeypatch)

    with pytest.raises(NotImplementedError, match="only supports int4"):
        _dispatch(make_layer_config(bits=2))


@pytest.mark.parametrize("group_size", [48, -1, 0])
def test_w4a8_rejects_unaligned_group_size(monkeypatch, group_size) -> None:
    """The kernel needs a group size that is a positive multiple of 32."""
    monkeypatch.setenv(_BACKEND_ENV, "w4a8")
    _fake_xpu(monkeypatch, ark_available=True)
    _with_w4a8_kernel(monkeypatch)

    with pytest.raises(NotImplementedError, match="multiple of 32"):
        _dispatch(make_layer_config(group_size=group_size))


@pytest.mark.parametrize("group_size", [128, 64, 32])
def test_w4a8_accepts_aligned_group_size(monkeypatch, group_size) -> None:
    monkeypatch.setenv(_BACKEND_ENV, "w4a8")
    _fake_xpu(monkeypatch, ark_available=True)
    _with_w4a8_kernel(monkeypatch)

    method = _dispatch(make_layer_config(group_size=group_size))

    assert isinstance(method.scheme, INCXPUW4A8LinearMethod)


def test_w4a8_requires_the_kernel(monkeypatch) -> None:
    """Older vllm-xpu-kernels builds lack the op entirely."""
    monkeypatch.setenv(_BACKEND_ENV, "w4a8")
    _fake_xpu(monkeypatch, ark_available=True)

    class FakeOpsNoW4A8:
        pass

    monkeypatch.setattr(torch.ops, "_xpu_C", FakeOpsNoW4A8, raising=False)

    with pytest.raises(NotImplementedError, match="int4_gemm_w4a8"):
        _dispatch()


def test_invalid_backend_value_raises(monkeypatch) -> None:
    import vllm.envs as envs

    monkeypatch.setenv(_BACKEND_ENV, "onednn")
    with pytest.raises(ValueError, match=_BACKEND_ENV):
        _ = envs.VLLM_XPU_INC_W4A16_BACKEND


# ---------------------------------------------------------------------------
# Weight processing
# ---------------------------------------------------------------------------


def test_keeps_fp16_scales(w4a8_layer) -> None:
    """The kernel reads scales as fp16 regardless of activation dtype.

    Passing bf16 scales does not raise — it silently returns wrong results — so
    an fp16 copy must exist, while the bf16 ``scales`` survive for the
    small-batch w4a16 fallback.
    """
    method, layer = w4a8_layer
    layer.scales.data.fill_(0.125)  # exactly representable in both dtypes

    method.process_weights_after_loading(layer)

    assert layer.scales_fp16.dtype is torch.float16
    assert layer.scales_fp16.shape == layer.scales.shape
    assert torch.equal(layer.scales_fp16.data.float(), layer.scales.data.float())
    # The w4a16 fallback still needs the original activation-dtype scales.
    assert layer.scales.dtype is torch.bfloat16


def test_reuses_w4a16_weight_layout(w4a8_layer) -> None:
    """w4a8 and w4a16 accept bit-identical weights, so there is no repacking."""
    method, layer = w4a8_layer
    qweight_before = layer.qweight.data.clone()

    method.process_weights_after_loading(layer)

    assert layer.qweight.dtype is torch.int32
    assert layer.qweight.shape == (IN_FEATURES // 8, OUT_FEATURES)
    assert torch.equal(layer.qweight.data, qweight_before)
    # Symmetric int4: the zero point is the constant 8, as for w4a16.
    assert layer.qzeros.dtype is torch.int8
    assert layer.qzeros.tolist() == [8]


# ---------------------------------------------------------------------------
# apply_weights
# ---------------------------------------------------------------------------


def test_falls_back_to_w4a16_for_small_batches(monkeypatch, w4a8_layer) -> None:
    """Below the token threshold, per-token quant overhead dominates."""
    method, layer = w4a8_layer
    method.process_weights_after_loading(layer)

    calls = []
    monkeypatch.setattr(
        INCXPULinearMethod,
        "apply_weights",
        lambda self, layer, x, bias=None: (
            calls.append(tuple(x.shape))
            or torch.zeros(x.shape[:-1] + (OUT_FEATURES,), dtype=x.dtype)
        ),
    )
    monkeypatch.setattr(
        torch.ops._xpu_C,
        "int4_gemm_w4a8",
        lambda *args: pytest.fail("small batches must not call the w4a8 kernel"),
        raising=False,
    )

    tokens = method._MIN_TOKENS_FOR_INT8 - 1
    x = torch.zeros(tokens, IN_FEATURES, dtype=torch.bfloat16)
    out = method.apply_weights(layer, x)

    assert calls == [(tokens, IN_FEATURES)]
    assert out.shape == (tokens, OUT_FEATURES)


def test_calls_kernel_at_threshold(monkeypatch, w4a8_layer) -> None:
    """At/above the threshold, activations are int8-quantized per token."""
    method, layer = w4a8_layer
    method.process_weights_after_loading(layer)

    monkeypatch.setattr(
        INCXPULinearMethod,
        "apply_weights",
        lambda self, layer, x, bias=None: pytest.fail(
            "large batches must not take the w4a16 fallback"
        ),
    )
    captured = {}
    _stub_quant(monkeypatch, captured)
    _stub_gemm(monkeypatch, captured)

    tokens = method._MIN_TOKENS_FOR_INT8
    x = torch.zeros(tokens, IN_FEATURES, dtype=torch.bfloat16)
    out = method.apply_weights(layer, x)

    # Symmetric 8-bit per-token quantization over the flattened activations.
    assert captured["quant_args"] == ((tokens, IN_FEATURES), True, 8)

    (
        quant_x,
        x_scale,
        x_zero,
        qweight,
        w_scale,
        w_zp,
        group_size,
        g_idx,
        bias,
    ) = captured["gemm_args"]
    assert quant_x.dtype is torch.int8
    # Both scales must reach the kernel as fp16, whatever the model dtype.
    assert x_scale.dtype is torch.float16
    assert w_scale.dtype is torch.float16
    assert w_scale is layer.scales_fp16
    assert x_zero.dtype is torch.int32
    assert qweight is layer.qweight
    assert w_zp is layer.qzeros
    assert group_size == 128
    assert g_idx is None
    assert bias is None

    # The kernel emits fp16; the result is cast back to the activation dtype.
    assert out.dtype is torch.bfloat16
    assert out.shape == (tokens, OUT_FEATURES)


@pytest.mark.parametrize("act_dtype", [torch.bfloat16, torch.float16])
def test_scales_are_fp16_for_any_activation_dtype(
    monkeypatch, w4a8_layer, act_dtype
) -> None:
    """fp16 scales are required for bf16 *and* fp16 activations alike."""
    method, layer = w4a8_layer
    layer.scales.data = layer.scales.data.to(act_dtype)
    method.process_weights_after_loading(layer)

    captured = {}
    _stub_quant(monkeypatch, captured)
    _stub_gemm(monkeypatch, captured)

    x = torch.zeros(method._MIN_TOKENS_FOR_INT8, IN_FEATURES, dtype=act_dtype)
    out = method.apply_weights(layer, x)

    assert captured["gemm_args"][1].dtype is torch.float16
    assert captured["gemm_args"][4].dtype is torch.float16
    assert out.dtype is act_dtype


def test_forwards_bias_to_kernel(monkeypatch, w4a8_layer) -> None:
    """Bias is applied by the kernel, not added afterwards."""
    method, layer = w4a8_layer
    method.process_weights_after_loading(layer)

    captured = {}
    _stub_quant(monkeypatch)
    _stub_gemm(monkeypatch, captured)

    bias = torch.zeros(OUT_FEATURES, dtype=torch.bfloat16)
    x = torch.zeros(method._MIN_TOKENS_FOR_INT8, IN_FEATURES, dtype=torch.bfloat16)
    method.apply_weights(layer, x, bias)

    assert captured["gemm_args"][8] is bias


def test_preserves_leading_dims(monkeypatch, w4a8_layer) -> None:
    """3D activations must round-trip through the flatten/reshape."""
    method, layer = w4a8_layer
    method.process_weights_after_loading(layer)

    _stub_quant(monkeypatch)
    _stub_gemm(monkeypatch)

    tokens = method._MIN_TOKENS_FOR_INT8
    x = torch.zeros(4, tokens, IN_FEATURES, dtype=torch.bfloat16)
    out = method.apply_weights(layer, x)

    assert out.shape == (4, tokens, OUT_FEATURES)
    assert out.dtype is torch.bfloat16


def test_batch_gate_uses_flattened_token_count(monkeypatch, w4a8_layer) -> None:
    """The gate counts total tokens, not the leading dimension."""
    method, layer = w4a8_layer
    method.process_weights_after_loading(layer)

    monkeypatch.setattr(
        INCXPULinearMethod,
        "apply_weights",
        lambda self, layer, x, bias=None: pytest.fail(
            "flattened token count is at the threshold; must use w4a8"
        ),
    )
    _stub_quant(monkeypatch)
    _stub_gemm(monkeypatch)

    tokens = method._MIN_TOKENS_FOR_INT8
    # Each leading slice is below the threshold, but 4 * (tokens // 4) is not.
    x = torch.zeros(4, tokens // 4, IN_FEATURES, dtype=torch.bfloat16)
    out = method.apply_weights(layer, x)

    assert out.shape == (4, tokens // 4, OUT_FEATURES)
