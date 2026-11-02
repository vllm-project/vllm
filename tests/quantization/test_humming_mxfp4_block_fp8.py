# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Humming MXFP4-weight GEMM paired with a block-FP8 activation.

The Humming kernel already implements MXFP4 weight (group-32 e8m0) with a
grouped FP8 activation via WGMMA software dequant -- see the grouped-fp8
`test_mxfp4.py` cases in the humming repo, which run on SM90 (H200). Enabling
it in vLLM comes down to two things, both exercised here:

  1. A grouped FP8 activation must classify to a float32-scaled key
     (kFp8Dynamic128Sym), not a uint8 microscale key. MX microscale
     activations (group size 32) keep their e8m0 (uint8) scale.
  2. (kMxfp4Static, kFp8Dynamic128Sym) must be an allowed MoE
     (weight, activation) pair.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
    HummingExpertsBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8DynamicTokenSym,
    kMxfp4Static,
    kMxfp8Dynamic,
)
from vllm.utils.import_utils import has_humming

# make_humming_moe_quant_config stores per-sublayer humming LayerConfigs but the
# config-plumbing asserts below never read them, so a placeholder satisfies the
# required arg without pulling in the humming package.
_PLACEHOLDER_HUMMING_CONFIGS = {"w13": None, "w2": None}


def test_mxfp4_weight_with_block_fp8_activation_is_supported():
    """MXFP4 weight + block-FP8 (group-128) activation is an allowed pair."""
    assert HummingExpertsBase._supports_quant_scheme(kMxfp4Static, kFp8Dynamic128Sym)


def test_mxfp4_weight_with_per_token_fp8_activation_still_supported():
    """The pre-existing per-token FP8 pairing must keep working."""
    assert HummingExpertsBase._supports_quant_scheme(kMxfp4Static, kFp8DynamicTokenSym)


@pytest.mark.skipif(not has_humming(), reason="humming is not installed")
@pytest.mark.parametrize(
    "group_size,expected_key",
    [
        # per-token FP8 -> float32 per-token scale
        (0, kFp8DynamicTokenSym),
        # block FP8 -> float32 group-128 scale (the newly-fixed case)
        (128, kFp8Dynamic128Sym),
        # MXFP8 microscale -> e8m0 (uint8) group-32 scale (must be preserved)
        (32, kMxfp8Dynamic),
    ],
)
def test_humming_fp8_input_schema_to_quant_key(group_size, expected_key):
    from vllm.model_executor.layers.quantization.utils.humming import (
        input_schema_to_quant_key,
    )
    from vllm.utils.humming import HummingInputSchema
    from vllm.utils.humming import dtypes as humming_dtypes

    schema = HummingInputSchema(
        a_dtype=humming_dtypes.float8e4m3,
        input_scale_group_size=group_size,
    )
    assert input_schema_to_quant_key(schema, param_dtype=torch.bfloat16) == expected_key


@pytest.mark.skipif(not has_humming(), reason="humming is not installed")
def test_humming_bf16_input_schema_is_unquantized():
    from vllm.model_executor.layers.quantization.utils.humming import (
        input_schema_to_quant_key,
    )
    from vllm.utils.humming import HummingInputSchema

    # No a_dtype -> unquantized (bf16/fp16) inputs -> None.
    assert (
        input_schema_to_quant_key(HummingInputSchema(), param_dtype=torch.bfloat16)
        is None
    )


def test_block_fp8_activation_quant_config_is_block_quantized():
    """A block-FP8 activation group shape yields a block-quantized MoE config
    (FP8 dtype, [1, 128] block, not per-act-token). This is what makes the
    DeepEP prepare/finalize step quantize activations to block FP8 *before* the
    all-to-all dispatch instead of deferring to Humming -- and what makes
    HummingExpertsBase.expects_unquantized_inputs return False."""
    from vllm.model_executor.layers.quantization.utils.humming import (
        make_humming_moe_quant_config,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
    from vllm.platforms import current_platform

    qc = make_humming_moe_quant_config(
        quant_dtype=current_platform.fp8_dtype(),
        weight_dtype="float4e2m1",
        activation_group_shape=GroupShape(row=1, col=128),
        humming_configs=_PLACEHOLDER_HUMMING_CONFIGS,
    )
    assert qc.is_block_quantized
    assert qc.block_shape == [1, 128]
    assert not qc.per_act_token_quant
    assert qc.quant_dtype == current_platform.fp8_dtype()


def test_default_activation_quant_config_defers_to_humming():
    """Without a block activation shape the config stays per-token (the deferred
    path): Humming quantizes internally, preserving pre-existing behavior for
    every non-block-FP8 scheme."""
    from vllm.model_executor.layers.quantization.utils.humming import (
        make_humming_moe_quant_config,
    )
    from vllm.platforms import current_platform

    qc = make_humming_moe_quant_config(
        quant_dtype=current_platform.fp8_dtype(),
        weight_dtype="float4e2m1",
        humming_configs=_PLACEHOLDER_HUMMING_CONFIGS,
    )
    assert not qc.is_block_quantized
    assert qc.per_act_token_quant


@pytest.mark.skipif(not has_humming(), reason="humming is not installed")
@pytest.mark.parametrize(
    "capability,user_override,expected_group_size",
    [
        (90, None, 128),
        (90, kMxfp8Dynamic, 32),
        (100, None, 32),
    ],
)
def test_checkpoint_mxfp8_runs_as_block_fp8_on_hopper(
    monkeypatch, capability, user_override, expected_group_size
):
    """A checkpoint's MXFP8 MoE activations (e.g. DeepSeek V4) run as block
    FP8 on Humming's SM90 kernels; an explicit user request is honored."""
    from vllm.model_executor.layers.fused_moe.oracle import mxfp4
    from vllm.platforms import current_platform

    monkeypatch.setattr(
        current_platform, "is_device_capability_family", lambda cap: cap == capability
    )
    monkeypatch.setattr(mxfp4, "_user_moe_activation_override", lambda: user_override)
    monkeypatch.delenv("VLLM_HUMMING_INPUT_QUANT_CONFIG", raising=False)

    schema = mxfp4._humming_input_schema(kMxfp8Dynamic)

    assert str(schema.a_dtype) == "float8e4m3"
    assert schema.input_scale_group_size == expected_group_size


def _mxfp6_weight_schema():
    from vllm.utils.humming import BaseWeightSchema

    return BaseWeightSchema.from_config(
        {
            "quant_method": "humming",
            "dtype": "float6e2m3",
            "scale_dtype": "float8e8m0",
            "group_size": 32,
            "weight_scale_type": "group",
        }
    )


def _resolve_input_schema(monkeypatch, capability, activation_key, **kwargs):
    from unittest.mock import patch

    from vllm.model_executor.layers.quantization.utils.humming import (
        check_and_fallback_input_schema,
        quant_key_to_input_schema,
    )
    from vllm.platforms.interface import DeviceCapability

    class _Device:
        sm_version = capability

    with (
        patch(
            "vllm.model_executor.layers.quantization.utils.humming.schema"
            ".current_platform"
        ) as platform_mock,
        patch("humming.schema.humming.current_device", _Device()),
    ):
        platform_mock.get_device_capability.return_value = DeviceCapability(
            capability // 10, capability % 10
        )
        return check_and_fallback_input_schema(
            _mxfp6_weight_schema(),
            quant_key_to_input_schema(activation_key),
            param_dtype=torch.bfloat16,
            **kwargs,
        )


@pytest.mark.skipif(not has_humming(), reason="humming is not installed")
@pytest.mark.parametrize(
    "capability,expect_fp8",
    [
        (90, False),  # WGMMA has no block-scaled MX path: keep BF16 (wna16)
        (100, True),
        (103, True),
        (120, True),
    ],
)
def test_mxfp8_activation_honored_from_sm100(monkeypatch, capability, expect_fp8):
    """An MXFP8 activation request (FP8 E4M3, group-32, E8M0 scales) next to
    MXFP6 weights must reach the Humming kernel on SM100+ instead of being
    downgraded to BF16; SM90 keeps the BF16 (wna16) fallback."""
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp8Dynamic,
    )
    from vllm.utils.humming import dtypes as humming_dtypes

    resolved = _resolve_input_schema(monkeypatch, capability, kMxfp8Dynamic)
    if expect_fp8:
        assert resolved.a_dtype == humming_dtypes.float8e4m3
        assert resolved.input_scale_group_size == 32
    else:
        assert resolved.a_dtype is None or resolved.a_dtype.num_bits == 16


@pytest.mark.skipif(not has_humming(), reason="humming is not installed")
def test_moe_path_keeps_fp8_gate_on_sm100(monkeypatch):
    """The MoE call site passes allow_fp8_sm100=False: humming's grouped-GEMM
    path is not validated for FP8 activations on SM100-SM119, so it must keep
    the BF16 fallback even though dense linear honors FP8 there."""
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp8Dynamic,
    )

    resolved = _resolve_input_schema(
        monkeypatch, 100, kMxfp8Dynamic, allow_fp8_sm100=False
    )
    assert resolved.a_dtype is None or resolved.a_dtype.num_bits == 16


@pytest.mark.skipif(not has_humming(), reason="humming is not installed")
def test_fp4_activation_still_deprecated_on_sm100(monkeypatch):
    """Only FP8 is ungated: FP4 activations keep falling back on SM100."""
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kNvfp4Dynamic,
    )

    resolved = _resolve_input_schema(monkeypatch, 100, kNvfp4Dynamic)
    assert resolved.a_dtype is None or resolved.a_dtype.num_bits != 4
