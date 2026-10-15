# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the zentorch MXFP4 linear kernel and scale layout.

The expected values come from vLLM's own E2M1 decode, so they check the
contract zentorch's kernel takes: E2M1 codes with the checkpoint's E8M0 bytes
as float8_e8m0fnu {G, N} scales. zentorch_dynamic_qlinear is mocked with that
arithmetic when zentorch is not installed.
"""

import pytest
import torch

from vllm.model_executor.kernels.linear.mxfp4.base import MxFp4LinearLayerConfig
from vllm.model_executor.kernels.linear.mxfp4.zentorch import (
    ZentorchMxFp4LinearKernel,
)
from vllm.model_executor.kernels.linear.zentorch_utils import (
    check_w4_group_shape,
    w4_group_scale_from_e8m0,
)
from vllm.model_executor.layers.fused_moe.experts.zentorch_moe import (
    ZentorchExpertsMxfp4,
)
from vllm.model_executor.layers.fused_moe.oracle import mxfp4 as mxfp4_oracle
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    break_fp4_bytes,
)
from vllm.platforms import current_platform

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

N, K = 64, 256


def _random_codes(*shape: int) -> torch.Tensor:
    return torch.randint(0, 256, (*shape, K // 2), dtype=torch.uint8)


def _kernel_weight(codes: torch.Tensor, group_scale: torch.Tensor) -> torch.Tensor:
    """The kernel's view of the weight: E2M1 times its {..., G, N} scales."""
    values = break_fp4_bytes(codes.reshape(-1, K // 2), torch.float32)
    scale = group_scale.float().transpose(-1, -2)
    scale = scale.repeat_interleave(K // scale.shape[-1], dim=-1)
    return values.reshape(*codes.shape[:-1], K) * scale


def _per_token_int8_matmul(x, weight, bias):
    # zentorch rounds activation ties away from zero; with E2M1 weights up to
    # ~20, one int8 step on a tie is visible at the test tolerance.
    x = x.double()
    amax = x.abs().amax(-1, keepdim=True)
    x_q = torch.trunc(x * (127 / amax) + 0.5 * x.sign())
    out = ((x_q @ weight.double().t()) * (amax / 127)).float()
    return out if bias is None else out + bias.float()


@pytest.fixture
def mock_zentorch_ops(monkeypatch):
    """Register a reference zentorch_dynamic_qlinear when zentorch is absent."""
    monkeypatch.setattr(
        "vllm.model_executor.kernels.linear.mxfp4.zentorch.has_zentorch_op",
        lambda ops: True,
    )
    monkeypatch.setattr(
        "vllm.model_executor.kernels.linear.mxfp4.zentorch.is_zentorch_e2m1_supported",
        lambda: (True, None),
    )
    if hasattr(torch.ops.zentorch, "zentorch_dynamic_qlinear"):
        yield
        return

    def _dynamic_qlinear(
        x,
        weight,
        weight_scales,
        bias=None,
        *,
        is_weight_prepacked=False,
        zentorch_op_name="",
    ):
        assert weight_scales.dtype == torch.float8_e8m0fnu
        out = _per_token_int8_matmul(
            x.reshape(-1, K), _kernel_weight(weight, weight_scales), bias
        )
        return out.to(x.dtype).reshape(*x.shape[:-1], weight.shape[0])

    lib_def = torch.library.Library("zentorch", "DEF")
    lib_def.define(
        "zentorch_dynamic_qlinear(Tensor input, Tensor weight, "
        "Tensor weight_scales, Tensor? bias=None, *, "
        "bool is_weight_prepacked=False, "
        "str zentorch_op_name='zentorch::zentorch_dynamic_qlinear') -> Tensor"
    )
    lib_impl = torch.library.Library("zentorch", "IMPL", "CPU")
    lib_impl.impl("zentorch_dynamic_qlinear", _dynamic_qlinear)
    yield
    lib_impl._destroy()
    lib_def._destroy()


def _mxfp4_dequant(codes: torch.Tensor, e8m0: torch.Tensor) -> torch.Tensor:
    values = break_fp4_bytes(codes.reshape(-1, K // 2), torch.float32)
    scale = torch.exp2(e8m0.float() - 127).reshape(-1, e8m0.shape[-1])
    values = values * scale.repeat_interleave(32, dim=-1)
    return values.reshape(*codes.shape[:-1], K)


def _check_e8m0_group_scale(group_scale, e8m0, codes):
    assert group_scale.dtype == torch.float8_e8m0fnu
    assert group_scale.is_contiguous()
    assert torch.equal(group_scale.view(torch.uint8), e8m0.transpose(-1, -2))
    torch.testing.assert_close(
        _kernel_weight(codes, group_scale),
        _mxfp4_dequant(codes, e8m0),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("experts", [None, 3])
def test_e8m0_group_scale_matches_mxfp4_dequant(experts):
    torch.manual_seed(0)
    lead = () if experts is None else (experts,)
    codes = _random_codes(*lead, N)
    e8m0 = torch.randint(118, 136, (*lead, N, K // 32), dtype=torch.uint8)
    group_scale = w4_group_scale_from_e8m0(e8m0)
    assert group_scale.shape == (*lead, K // 32, N)
    _check_e8m0_group_scale(group_scale, e8m0, codes)


def test_zen_cpu_moe_convert_emits_kernel_scales():
    torch.manual_seed(0)
    E = 3
    codes = _random_codes(E, N)
    e8m0 = torch.randint(118, 136, (E, N, K // 32), dtype=torch.uint8)
    _, _, w13_scale, _, _, _ = (
        mxfp4_oracle.convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
            mxfp4_backend=mxfp4_oracle.Mxfp4MoeBackend.ZEN_CPU,
            layer=torch.nn.Module(),
            w13_weight=codes,
            w2_weight=codes,
            w13_weight_scale=e8m0,
            w2_weight_scale=e8m0,
        )
    )
    _check_e8m0_group_scale(w13_scale, e8m0, codes)


def _make_layer(**params: torch.Tensor) -> torch.nn.Module:
    layer = torch.nn.Module()
    for name, t in params.items():
        setattr(layer, name, torch.nn.Parameter(t, requires_grad=False))
    return layer


def _process_keeping_params(kernel, layer: torch.nn.Module) -> None:
    params = dict(layer.named_parameters())
    kernel.process_weights_after_loading(layer)
    # Weight reloading writes through the loaded parameter objects.
    assert all(getattr(layer, name) is p for name, p in params.items())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mxfp4_linear_matches_dequantized_reference(mock_zentorch_ops, dtype):
    torch.manual_seed(0)
    codes = _random_codes(N)
    e8m0 = torch.randint(120, 130, (N, K // 32), dtype=torch.uint8)
    # The {G, N} scales are rewritten into the loaded parameter's own bytes.
    layer = _make_layer(weight=codes, weight_scale=e8m0.clone())
    kernel = ZentorchMxFp4LinearKernel(MxFp4LinearLayerConfig())
    _process_keeping_params(kernel, layer)

    x = torch.randn(5, K, dtype=dtype)
    bias = torch.randn(N, dtype=dtype)
    out = kernel.apply_weights(layer, x, bias)
    assert out.dtype == dtype
    # The kernel computes from bf16, whatever the model's dtype.
    x_bf16, bias_bf16 = x.to(torch.bfloat16), bias.to(torch.bfloat16)
    ref = _per_token_int8_matmul(x_bf16, _mxfp4_dequant(codes, e8m0), bias_bf16)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    "n,k,groups,ok", [(64, 256, 8, True), (48, 256, 8, False), (64, 256, 64, False)]
)
def test_w4_group_shape_gate(n, k, groups, ok):
    if ok:
        assert check_w4_group_shape(n, k, groups) == k // groups
    else:
        with pytest.raises(ValueError):
            check_w4_group_shape(n, k, groups)


def test_zentorch_without_e2m1_is_declined(monkeypatch):
    """A zentorch without E2M1 is declined during selection, which leaves the
    layer to another kernel instead of failing on the first forward."""
    declined = lambda: (False, "no E2M1")  # noqa: E731
    monkeypatch.setattr(
        "vllm.model_executor.kernels.linear.mxfp4.zentorch.has_zentorch_op",
        lambda ops: True,
    )
    monkeypatch.setattr(
        "vllm.model_executor.kernels.linear.mxfp4.zentorch.is_zentorch_e2m1_supported",
        declined,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.experts.zentorch_moe.is_zentorch_e2m1_supported",
        declined,
    )
    assert ZentorchMxFp4LinearKernel.is_supported() == (False, "no E2M1")
    # The experts leaf declines before it reads any of its arguments.
    assert ZentorchExpertsMxfp4.is_supported_config(
        ZentorchExpertsMxfp4, None, None, None, None
    ) == (False, "no E2M1")


def test_zen_cpu_backends_are_selectable(monkeypatch):
    monkeypatch.setattr(mxfp4_oracle.current_platform, "is_cpu", lambda: True)
    monkeypatch.setattr(mxfp4_oracle.current_platform, "is_xpu", lambda: False)
    assert mxfp4_oracle._get_priority_backends()[:2] == [
        mxfp4_oracle.Mxfp4MoeBackend.ZEN_CPU,
        mxfp4_oracle.Mxfp4MoeBackend.CPU,
    ]
    assert mxfp4_oracle.map_mxfp4_backend("zen_cpu") == [
        mxfp4_oracle.Mxfp4MoeBackend.ZEN_CPU
    ]
    assert mxfp4_oracle.map_mxfp4_backend("cpu") == [mxfp4_oracle.Mxfp4MoeBackend.CPU]
