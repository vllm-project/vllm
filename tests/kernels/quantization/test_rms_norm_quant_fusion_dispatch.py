# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the manual RMSNorm + input-quant fusion dispatcher
(`rms_norm_input_quant`, RFC #43224 / issue #43500).

The fused paths must produce the same quantized activation (data and scale)
as the unfused reference: RMSNorm (with optional residual add) followed by
QuantFP8, for both static per-tensor and dynamic per-token FP8.
"""

import pytest
import torch

from tests.kernels.quant_utils import FP8_DTYPE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_fusion import (
    QuantizedActivation,
    rms_norm_input_quant,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    kFp8DynamicTokenSym,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform

DTYPES = [torch.bfloat16]
NUM_TOKENS = [7, 256]
HIDDEN_SIZES = [64, 2048]
SEEDS = [0]


class _FakeLinear(torch.nn.Module):
    """Carries just the attributes the dispatcher reads."""

    def __init__(self, input_quant_key=None, input_scale=None):
        super().__init__()
        self.input_quant_key = input_quant_key
        if input_scale is not None:
            self.input_scale = input_scale


@pytest.mark.cpu_test
def test_no_quant_key_falls_back_to_plain_norm(default_vllm_config):
    """Without an input_quant_key the dispatcher must behave exactly like
    the pre-fusion code path (plain RMSNorm, optional fused-add)."""
    torch.manual_seed(0)
    norm = RMSNorm(32, eps=1e-6)
    linear = _FakeLinear()
    x = torch.randn(4, 32)

    out, residual = rms_norm_input_quant(norm, x.clone(), None, linear)
    ref = norm(x)
    assert not isinstance(out, QuantizedActivation)
    torch.testing.assert_close(out, ref)
    torch.testing.assert_close(residual, x)


@pytest.mark.cpu_test
def test_linear_none_falls_back_to_plain_norm(default_vllm_config):
    """`linear=None` (decoder-layer subclasses that swap self_attn/mlp for
    modules without the expected projection, e.g. Aria's MoE mlp) must take
    the plain-norm path, with and without residual."""
    torch.manual_seed(0)
    norm = RMSNorm(32, eps=1e-6)
    x = torch.randn(4, 32)

    out, residual = rms_norm_input_quant(norm, x.clone(), None, None)
    assert not isinstance(out, QuantizedActivation)
    torch.testing.assert_close(out, norm(x))
    torch.testing.assert_close(residual, x)

    res_in = torch.randn(4, 32)
    ref_out, ref_res = norm(x.clone(), res_in.clone())
    out2, res2 = rms_norm_input_quant(norm, x.clone(), res_in.clone(), None)
    assert not isinstance(out2, QuantizedActivation)
    torch.testing.assert_close(out2, ref_out)
    torch.testing.assert_close(res2, ref_res)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA fused ops")
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("with_residual", [False, True])
@torch.inference_mode()
def test_static_per_tensor_matches_unfused(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    with_residual: bool,
):
    torch.manual_seed(0)
    device = "cuda"
    norm = RMSNorm(hidden_size, eps=1e-6).to(device=device, dtype=dtype)
    norm.weight.data.normal_(mean=1.0, std=0.1)
    scale = torch.tensor(0.05, dtype=torch.float32, device=device)
    linear = _FakeLinear(input_quant_key=kFp8StaticTensorSym, input_scale=scale)

    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    residual = torch.randn_like(x) if with_residual else None

    # Unfused reference: norm (+ residual add) then static QuantFP8.
    if residual is not None:
        ref_normed, ref_residual = norm(x.clone(), residual.clone())
    else:
        ref_normed, ref_residual = norm(x.clone()), x
    quant_fp8 = QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR)
    ref_q, _ = quant_fp8(ref_normed, scale=scale)

    out, out_residual = rms_norm_input_quant(
        norm,
        x.clone(),
        residual.clone() if residual is not None else None,
        linear,
    )

    assert isinstance(out, QuantizedActivation)
    assert out.quant_key == kFp8StaticTensorSym
    assert out.data.dtype == FP8_DTYPE
    assert out.scale is scale
    torch.testing.assert_close(out_residual, ref_residual)
    # The fused kernel computes the residual-add + norm in fp32 while the
    # unfused reference round-trips through the activation dtype, so values
    # on a quantization boundary may round to the neighboring representable
    # value. Allow exactly one fp8e4m3 ulp: mantissa step 2^-3 = 0.125
    # relative, plus a small atol for the subnormal range.
    torch.testing.assert_close(
        out.data.to(torch.float32), ref_q.to(torch.float32), atol=0.06, rtol=0.13
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA fused ops")
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("with_residual", [False, True])
@torch.inference_mode()
def test_dynamic_per_token_matches_unfused(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    with_residual: bool,
):
    torch.manual_seed(0)
    device = "cuda"
    norm = RMSNorm(hidden_size, eps=1e-6).to(device=device, dtype=dtype)
    norm.weight.data.normal_(mean=1.0, std=0.1)
    linear = _FakeLinear(input_quant_key=kFp8DynamicTokenSym)

    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    residual = torch.randn_like(x) if with_residual else None

    # Unfused reference: norm (+ residual add) then dynamic per-token QuantFP8.
    if residual is not None:
        ref_normed, ref_residual = norm(x.clone(), residual.clone())
    else:
        ref_normed, ref_residual = norm(x.clone()), x
    quant_fp8 = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN)
    ref_q, ref_scales = quant_fp8(ref_normed)

    out, out_residual = rms_norm_input_quant(
        norm,
        x.clone(),
        residual.clone() if residual is not None else None,
        linear,
    )

    assert isinstance(out, QuantizedActivation)
    assert out.quant_key == kFp8DynamicTokenSym
    assert out.data.dtype == FP8_DTYPE
    assert out.scale.shape == (num_tokens, 1)
    torch.testing.assert_close(out_residual, ref_residual)
    torch.testing.assert_close(
        out.scale.to(torch.float32), ref_scales.to(torch.float32), atol=1e-3, rtol=0.05
    )
    # Compare dequantized values (per-token scales may differ by 1 ulp).
    deq_out = out.data.to(torch.float32) * out.scale
    deq_ref = ref_q.to(torch.float32) * ref_scales
    torch.testing.assert_close(deq_out, deq_ref, atol=0.1, rtol=0.1)
