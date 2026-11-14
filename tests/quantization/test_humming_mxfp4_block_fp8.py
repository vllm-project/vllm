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

import contextlib

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
    kNvfp4Dynamic,
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


@pytest.mark.skipif(not has_humming(), reason="humming is not installed")
@pytest.mark.parametrize(
    "capability,strict,honors",
    [
        # Hopper has no FP4 activations: falls back, and strict cannot run it.
        ((9, 0), False, False),
        ((9, 0), True, False),
        # Slow on SM10x, so it falls back unless strict mode forbids that.
        ((10, 3), False, False),
        ((10, 3), True, True),
        ((12, 1), False, True),
    ],
)
def test_humming_nvfp4_linear_declares_executed_activation(
    monkeypatch, capability, strict, honors
):
    """Humming picks its NVFP4 input schema at weight load, per device. Kernel
    selection needs the same answer up front so VLLM_STRICT_QUANT_SCHEME can
    move on to another kernel instead of failing at load."""
    import humming.schema.humming as humming_schema

    from vllm.model_executor.kernels.linear import HummingNvFp4LinearKernel
    from vllm.platforms import current_platform
    from vllm.platforms.interface import DeviceCapability

    monkeypatch.setenv("VLLM_STRICT_QUANT_SCHEME", str(int(strict)))
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda device_id=0: DeviceCapability(*capability),
    )
    monkeypatch.setattr(
        type(humming_schema.current_device),
        "sm_version",
        property(lambda self: capability[0] * 10 + capability[1]),
    )

    executed = HummingNvFp4LinearKernel.executed_activation_key(kNvfp4Dynamic)
    assert executed == (kNvfp4Dynamic if honors else None)


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


@contextlib.contextmanager
def _mock_sm_version(capability):
    from unittest.mock import patch

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
        yield


def _resolve_input_schema(capability, activation_key, **kwargs):
    from vllm.model_executor.layers.quantization.utils.humming import (
        check_and_fallback_input_schema,
        quant_key_to_input_schema,
    )

    with _mock_sm_version(capability):
        return check_and_fallback_input_schema(
            _mxfp6_weight_schema(),
            quant_key_to_input_schema(activation_key),
            param_dtype=torch.bfloat16,
            **kwargs,
        )


@pytest.mark.skipif(not has_humming(), reason="humming is not installed")
@pytest.mark.parametrize(
    "capability,activation_key,kwargs,expected",
    [
        # SM90 WGMMA has no block-scaled MX path: keep the 16-bit fallback.
        (90, kMxfp8Dynamic, {}, "16bit"),
        (100, kMxfp8Dynamic, {}, "fp8"),
        (103, kMxfp8Dynamic, {}, "fp8"),
        (120, kMxfp8Dynamic, {}, "fp8"),
        # MoE grouped-GEMM FP8 is unvalidated on SM100-SM119: keep the gate.
        (100, kMxfp8Dynamic, {"allow_fp8_sm100": False}, "16bit"),
        # FP4 requests keep their gate on SM100.
        (100, kNvfp4Dynamic, {}, "not_fp4"),
    ],
)
def test_input_schema_honors_sm_activation_gates(
    capability, activation_key, kwargs, expected
):
    """Honor MXFP8 requests while preserving architecture and MoE gates."""
    from vllm.utils.humming import dtypes as humming_dtypes

    resolved = _resolve_input_schema(capability, activation_key, **kwargs)
    if expected == "fp8":
        assert resolved.a_dtype == humming_dtypes.float8e4m3
        assert resolved.input_scale_group_size == 32
    elif expected == "16bit":
        assert resolved.a_dtype is None or resolved.a_dtype.num_bits == 16
    else:
        assert resolved.a_dtype is None or resolved.a_dtype.num_bits != 4
