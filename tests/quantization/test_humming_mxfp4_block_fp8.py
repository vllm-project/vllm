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
    from vllm.model_executor.layers.quantization.utils.humming_utils import (
        _humming_input_schema_to_quant_key,
    )
    from vllm.utils.humming import HummingInputSchema
    from vllm.utils.humming import dtypes as humming_dtypes

    schema = HummingInputSchema(
        a_dtype=humming_dtypes.float8e4m3,
        input_scale_group_size=group_size,
    )
    assert _humming_input_schema_to_quant_key(schema) == expected_key


@pytest.mark.skipif(not has_humming(), reason="humming is not installed")
def test_humming_bf16_input_schema_is_unquantized():
    from vllm.model_executor.layers.quantization.utils.humming_utils import (
        _humming_input_schema_to_quant_key,
    )
    from vllm.utils.humming import HummingInputSchema

    # No a_dtype -> unquantized (bf16/fp16) inputs -> None.
    assert _humming_input_schema_to_quant_key(HummingInputSchema()) is None


def test_block_fp8_activation_quant_config_is_block_quantized():
    """A block-FP8 activation group shape yields a block-quantized MoE config
    (FP8 dtype, [1, 128] block, not per-act-token). This is what makes the
    DeepEP prepare/finalize step quantize activations to block FP8 *before* the
    all-to-all dispatch instead of deferring to Humming -- and what makes
    HummingExpertsBase.expects_unquantized_inputs return False."""
    from vllm.model_executor.layers.quantization.utils.humming_utils import (
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
    from vllm.model_executor.layers.quantization.utils.humming_utils import (
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


def _ct_input_quant(**kwargs):
    from compressed_tensors.quantization import QuantizationArgs

    return QuantizationArgs(num_bits=8, type="float", symmetric=True, **kwargs)


@pytest.mark.skipif(not has_humming(), reason="humming is not installed")
@pytest.mark.parametrize(
    "input_quant_kwargs,expected_group_size",
    [
        # The MXFP4xFP8_BLOCK recipe: dynamic FP8 activations, group 128.
        (dict(dynamic=True, strategy="group", group_size=128), 128),
        # Per-token dynamic FP8 activations keep Humming's per-token path.
        (dict(dynamic=True, strategy="token"), 0),
    ],
)
def test_checkpoint_activations_drive_humming_input_schema(
    input_quant_kwargs, expected_group_size
):
    """The checkpoint's ``input_activations`` pick the Humming activation
    schema, keeping the group size that humming's own compressed-tensors
    schema would drop for FP8."""
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_mxfp4 import (  # noqa: E501
        _humming_input_schema,
    )
    from vllm.utils.humming import dtypes as humming_dtypes

    schema = _humming_input_schema(_ct_input_quant(**input_quant_kwargs))
    assert schema.a_dtype == humming_dtypes.float8e4m3
    assert schema.input_scale_group_size == expected_group_size


@pytest.mark.skipif(not has_humming(), reason="humming is not installed")
def test_weight_only_checkpoint_keeps_bf16_activations():
    """No ``input_activations`` means W4A16: Humming dequantizes the weights."""
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_mxfp4 import (  # noqa: E501
        _humming_input_schema,
    )

    assert _humming_input_schema(None).a_dtype is None


@pytest.mark.skipif(not has_humming(), reason="humming is not installed")
def test_env_input_quant_config_overrides_checkpoint(monkeypatch: pytest.MonkeyPatch):
    """VLLM_HUMMING_INPUT_QUANT_CONFIG wins: returning None leaves the
    conversion on its env-driven path."""
    from vllm import envs
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_mxfp4 import (  # noqa: E501
        _humming_input_schema,
    )

    monkeypatch.setattr(
        envs, "VLLM_HUMMING_INPUT_QUANT_CONFIG", {"dtype": "float8e4m3"}
    )
    input_quant = _ct_input_quant(dynamic=True, strategy="group", group_size=128)
    assert _humming_input_schema(input_quant) is None


def test_compressed_tensors_mxfp4_moe_backend_selects_humming(
    monkeypatch: pytest.MonkeyPatch,
):
    """``--moe-backend humming`` routes the compressed-tensors MXFP4 MoE
    through the oracle instead of the Cutlass/Marlin device probe."""
    from types import SimpleNamespace

    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
        compressed_tensors_moe_w4a4_mxfp4 as ct_mxfp4,
    )

    sentinel_experts = type("SentinelExperts", (), {})
    monkeypatch.setattr(
        ct_mxfp4.CutlassExpertsMxfp4, "_supports_current_device", lambda: True
    )
    monkeypatch.setattr(
        ct_mxfp4,
        "select_mxfp4_moe_backend",
        lambda moe: (Mxfp4MoeBackend.HUMMING, sentinel_experts),
    )

    method = ct_mxfp4.CompressedTensorsW4A4Mxfp4MoEMethod(
        SimpleNamespace(w13_num_shards=2, moe_backend="humming")
    )

    assert method.mxfp4_backend == Mxfp4MoeBackend.HUMMING
    assert method.experts_cls is sentinel_experts
    assert not method.use_cutlass_mxfp4
