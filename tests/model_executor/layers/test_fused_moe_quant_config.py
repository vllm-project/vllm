# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from inspect import signature

import pytest
import torch

from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
    FusedMoEQuantSchema,
    FusedMoEStaticQuantDesc,
    biased_moe_quant_config,
    fp8_w8a8_moe_quant_config,
    fp8_w8a16_moe_quant_config,
    gptq_marlin_moe_quant_config,
    int4_w4a16_moe_quant_config,
    int4_w4afp8_moe_quant_config,
    int8_w8a8_moe_quant_config,
    int8_w8a16_moe_quant_config,
    mxfp4_moe_quant_config,
    mxfp4_mxfp8_moe_quant_config,
    mxfp4_w4a8_moe_quant_config,
    mxfp4_w4a16_moe_quant_config,
    nvfp4_moe_quant_config,
    nvfp4_w4a16_moe_quant_config,
    ocp_mx_moe_quant_config,
)
from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import OCP_MX_Scheme
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    kFp8DynamicTokenSym,
    kFp8StaticTensorSym,
    kMxfp4Dynamic,
    kMxfp4Static,
    kMxfp8Dynamic,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv


def _scale() -> torch.Tensor:
    return torch.ones(1)


def _configs() -> list[FusedMoEQuantConfig]:
    fp8_dtype = current_platform.fp8_dtype()
    return [
        FusedMoEQuantConfig.make(),
        biased_moe_quant_config(_scale(), _scale()),
        fp8_w8a8_moe_quant_config(_scale(), _scale()),
        fp8_w8a8_moe_quant_config(
            _scale(), _scale(), a1_scale=_scale(), a2_scale=_scale()
        ),
        fp8_w8a8_moe_quant_config(_scale(), _scale(), per_act_token_quant=True),
        fp8_w8a8_moe_quant_config(_scale(), _scale(), per_out_ch_quant=True),
        fp8_w8a8_moe_quant_config(_scale(), _scale(), block_shape=[128, 128]),
        int8_w8a8_moe_quant_config(_scale(), _scale(), _scale(), _scale()),
        gptq_marlin_moe_quant_config(_scale(), _scale(), weight_bits=4, group_size=128),
        gptq_marlin_moe_quant_config(_scale(), _scale(), weight_bits=8, group_size=-1),
        mxfp4_w4a16_moe_quant_config(_scale(), _scale()),
        mxfp4_mxfp8_moe_quant_config(_scale(), _scale()),
        mxfp4_w4a8_moe_quant_config(
            _scale(), _scale(), a1_scale=_scale(), a2_scale=_scale()
        ),
        ocp_mx_moe_quant_config("mxfp4", _scale(), _scale()),
        nvfp4_moe_quant_config(
            _scale(), _scale(), _scale(), _scale(), _scale(), _scale()
        ),
        mxfp4_moe_quant_config(_scale(), _scale()),
        nvfp4_w4a16_moe_quant_config(_scale(), _scale(), _scale(), _scale()),
        int4_w4a16_moe_quant_config(_scale(), _scale(), block_shape=[1, 128]),
        fp8_w8a16_moe_quant_config(_scale(), _scale()),
        int8_w8a16_moe_quant_config(_scale(), _scale()),
        int4_w4afp8_moe_quant_config(
            _scale(), _scale(), _scale(), _scale(), block_shape=[128, 128]
        ),
        FusedMoEQuantConfig.make(fp8_dtype, weight_dtype="int4", block_shape=[1, 128]),
    ]


def _legacy_surface(config: FusedMoEQuantConfig) -> dict[str, object]:
    a1 = config._a1
    w1 = config._w1
    quant_dtype = a1.dtype
    weight_quant_dtype = w1.dtype
    is_quantized = quant_dtype is not None

    if a1.shape is not None and a1.shape not in (
        GroupShape.PER_TENSOR,
        GroupShape.PER_TOKEN,
    ):
        block_shape = [a1.shape.row, a1.shape.col]
    else:
        block_shape = None

    if is_quantized:
        if block_shape is not None:
            scale_shape = (32, cdiv(4096, block_shape[1]))
        elif a1.shape == GroupShape.PER_TOKEN:
            scale_shape = (32, 1)
        else:
            scale_shape = (1, 1)
    else:
        scale_shape = None

    if (quant_dtype is not None and not isinstance(quant_dtype, str)) or (
        weight_quant_dtype is not None and not isinstance(weight_quant_dtype, str)
    ):
        ocp_mx_scheme = None
    else:
        ocp_mx_scheme = OCP_MX_Scheme.from_quant_dtype(quant_dtype, weight_quant_dtype)
        ocp_mx_scheme = ocp_mx_scheme.value if ocp_mx_scheme is not None else None

    use_fp8_w8a8 = quant_dtype == current_platform.fp8_dtype()
    use_int8_w8a8 = quant_dtype == torch.int8
    use_int8_w8a16 = quant_dtype is None and weight_quant_dtype == torch.int8
    use_fp8_w8a16 = (
        quant_dtype is None and weight_quant_dtype == current_platform.fp8_dtype()
    )
    use_int4_w4a16 = quant_dtype is None and weight_quant_dtype == "int4"

    def config_name(dtype: torch.dtype) -> str | None:
        if use_fp8_w8a8:
            return "fp8_w8a8"
        if use_fp8_w8a16:
            return "fp8_w8a16"
        if use_int8_w8a16:
            return "int8_w8a16"
        if use_int4_w4a16:
            return "int4_w4a16"
        if ocp_mx_scheme is not None:
            return None
        if dtype == torch.float32:
            return "float32"
        return None

    return {
        "quant_dtype": quant_dtype,
        "weight_quant_dtype": weight_quant_dtype,
        "is_quantized": is_quantized,
        "is_per_act_token": a1.shape == GroupShape.PER_TOKEN,
        "per_act_token_quant": a1.shape == GroupShape.PER_TOKEN,
        "per_out_ch_quant": w1.shape == GroupShape.PER_TOKEN,
        "is_per_tensor": a1.shape == GroupShape.PER_TENSOR,
        "block_shape": block_shape,
        "is_block_quantized": block_shape is not None,
        "use_fp8_w8a8": use_fp8_w8a8,
        "use_int8_w8a8": use_int8_w8a8,
        "use_int8_w8a16": use_int8_w8a16,
        "use_fp8_w8a16": use_fp8_w8a16,
        "use_int4_w4a16": use_int4_w4a16,
        "use_nvfp4_w4a16": quant_dtype is None and weight_quant_dtype == "nvfp4",
        "ocp_mx_scheme": ocp_mx_scheme,
        "use_mxfp4_w4a16": quant_dtype is None and weight_quant_dtype == "mxfp4",
        "use_mxfp4_w4a4": quant_dtype == "mxfp4" and weight_quant_dtype == "mxfp4",
        "use_nvfp4_w4a4": quant_dtype == "nvfp4",
        "use_mxfp4_w4a8": quant_dtype == "fp8" and weight_quant_dtype == "mxfp4",
        "scale_shape": scale_shape,
        "batched_scale_shape": ((8, *scale_shape) if scale_shape is not None else None),
        "config_name_float16": config_name(torch.float16),
        "config_name_float32": config_name(torch.float32),
    }


def _current_surface(config: FusedMoEQuantConfig) -> dict[str, object]:
    return {
        "quant_dtype": config.quant_dtype,
        "weight_quant_dtype": config.weight_quant_dtype,
        "is_quantized": config.is_quantized,
        "is_per_act_token": config.is_per_act_token,
        "per_act_token_quant": config.per_act_token_quant,
        "per_out_ch_quant": config.per_out_ch_quant,
        "is_per_tensor": config.is_per_tensor,
        "block_shape": config.block_shape,
        "is_block_quantized": config.is_block_quantized,
        "use_fp8_w8a8": config.use_fp8_w8a8,
        "use_int8_w8a8": config.use_int8_w8a8,
        "use_int8_w8a16": config.use_int8_w8a16,
        "use_fp8_w8a16": config.use_fp8_w8a16,
        "use_int4_w4a16": config.use_int4_w4a16,
        "use_nvfp4_w4a16": config.use_nvfp4_w4a16,
        "ocp_mx_scheme": config.ocp_mx_scheme,
        "use_mxfp4_w4a16": config.use_mxfp4_w4a16,
        "use_mxfp4_w4a4": config.use_mxfp4_w4a4,
        "use_nvfp4_w4a4": config.use_nvfp4_w4a4,
        "use_mxfp4_w4a8": config.use_mxfp4_w4a8,
        "scale_shape": config.scale_shape(32, 4096),
        "batched_scale_shape": config.batched_scale_shape(8, 32, 4096),
        "config_name_float16": config.config_name(torch.float16),
        "config_name_float32": config.config_name(torch.float32),
    }


@pytest.mark.parametrize("config", _configs())
def test_schema_delegates_static_properties(config: FusedMoEQuantConfig) -> None:
    schema = config.quant_schema

    assert config.quant_dtype == schema.quant_dtype
    assert config.weight_quant_dtype == schema.weight_quant_dtype
    assert config.is_quantized == schema.is_quantized
    assert config.is_per_act_token == schema.is_per_act_token
    assert config.per_act_token_quant == schema.per_act_token_quant
    assert config.per_out_ch_quant == schema.per_out_ch_quant
    assert config.is_per_tensor == schema.is_per_tensor
    assert config.block_shape == schema.block_shape
    assert config.is_block_quantized == schema.is_block_quantized
    assert config.use_fp8_w8a8 == schema.use_fp8_w8a8
    assert config.use_int8_w8a8 == schema.use_int8_w8a8
    assert config.use_int8_w8a16 == schema.use_int8_w8a16
    assert config.use_fp8_w8a16 == schema.use_fp8_w8a16
    assert config.use_int4_w4a16 == schema.use_int4_w4a16
    assert config.use_nvfp4_w4a16 == schema.use_nvfp4_w4a16
    assert config.ocp_mx_scheme == schema.ocp_mx_scheme
    assert config.use_mxfp4_w4a16 == schema.use_mxfp4_w4a16
    assert config.use_mxfp4_w4a4 == schema.use_mxfp4_w4a4
    assert config.use_nvfp4_w4a4 == schema.use_nvfp4_w4a4
    assert config.use_mxfp4_w4a8 == schema.use_mxfp4_w4a8
    assert config.scale_shape(32, 4096) == schema.scale_shape(32, 4096)
    assert config.batched_scale_shape(8, 32, 4096) == schema.batched_scale_shape(
        8, 32, 4096
    )
    for dtype in (torch.float16, torch.float32):
        assert config.config_name(dtype) == schema.config_name(dtype)

    assert _current_surface(config) == _legacy_surface(config)


def test_schema_preserves_legacy_shape_absence() -> None:
    config = mxfp4_mxfp8_moe_quant_config(_scale(), _scale())

    assert config.quant_dtype == "mxfp8"
    assert config.block_shape is None
    assert config.is_per_tensor is False
    assert config.is_per_act_token is False
    assert config.quant_schema.a1.group_shape == GroupShape(1, 32)
    assert config.quant_schema.a1.legacy_group_shape is None
    assert config.quant_schema.w1.group_shape == GroupShape(1, 32)
    assert config.quant_schema.w1.legacy_group_shape is None


def test_schema_keeps_unquantized_alignment_for_gptq() -> None:
    config = gptq_marlin_moe_quant_config(
        _scale(), _scale(), weight_bits=4, group_size=128
    )

    assert config.quant_dtype is None
    assert config.weight_quant_dtype == "int4"
    assert config.block_shape == [1, 128]
    assert config.is_quantized is False
    assert config.use_int4_w4a16 is True
    assert config.quant_schema.a1.key is None
    assert config.quant_schema.a1.alignment_shape == GroupShape(1, 128)


def test_canonical_quant_keys() -> None:
    fp8_dtype = current_platform.fp8_dtype()

    static_fp8 = FusedMoEQuantConfig.make(
        fp8_dtype, a1_scale=_scale(), a2_scale=_scale()
    )
    assert static_fp8.quant_schema.activation_key == kFp8StaticTensorSym
    assert static_fp8.quant_schema.weight_key == kFp8StaticTensorSym
    assert static_fp8.quant_schema.a1.is_scale_static is True
    assert static_fp8.quant_schema.a1.is_scale_dynamic is False

    dynamic_fp8 = FusedMoEQuantConfig.make(fp8_dtype, per_act_token_quant=True)
    assert dynamic_fp8.quant_schema.activation_key == kFp8DynamicTokenSym
    assert dynamic_fp8.quant_schema.a1.is_scale_static is False
    assert dynamic_fp8.quant_schema.a1.is_scale_dynamic is True

    nvfp4 = nvfp4_moe_quant_config(
        _scale(), _scale(), _scale(), _scale(), _scale(), _scale()
    )
    assert nvfp4.quant_schema.activation_key == kNvfp4Dynamic
    assert nvfp4.quant_schema.weight_key == kNvfp4Static

    mxfp4 = mxfp4_moe_quant_config(_scale(), _scale())
    assert mxfp4.quant_schema.activation_key == kMxfp4Dynamic
    assert mxfp4.quant_schema.weight_key == kMxfp4Static

    mxfp4_w4a16 = mxfp4_w4a16_moe_quant_config(_scale(), _scale())
    assert mxfp4_w4a16.quant_schema.activation_key is None
    assert mxfp4_w4a16.quant_schema.weight_key == kMxfp4Static

    mxfp4_mxfp8 = mxfp4_mxfp8_moe_quant_config(_scale(), _scale())
    assert mxfp4_mxfp8.quant_schema.activation_key == kMxfp8Dynamic
    assert mxfp4_mxfp8.quant_schema.weight_key == kMxfp4Static


def test_schema_snapshots_static_abi_and_activation_fields() -> None:
    config = FusedMoEQuantConfig.make(
        current_platform.fp8_dtype(),
        is_scale_swizzled=False,
        is_scale_transposed=True,
        gemm1_alpha=1.5,
        gemm1_beta=2.5,
        gemm1_clamp_limit=3.5,
    )
    config.mx_alignment = 128

    schema = config.quant_schema
    assert schema.is_scale_swizzled is False
    assert schema.is_scale_transposed is True
    assert schema.mx_alignment == 128
    assert schema.gemm1_alpha == 1.5
    assert schema.gemm1_beta == 2.5
    assert schema.gemm1_clamp_limit == 3.5


def test_common_role_keys_require_uniform_roles() -> None:
    config = FusedMoEQuantConfig.make(current_platform.fp8_dtype())
    schema = config.quant_schema
    mismatched = replace(
        schema,
        a2=FusedMoEStaticQuantDesc(key=kFp8StaticTensorSym),
    )

    with pytest.raises(ValueError, match="activation roles"):
        _ = mismatched.activation_key


def test_opaque_legacy_weight_dtype_is_not_dropped() -> None:
    config = FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(),
        _a2=FusedMoEQuantDesc(),
        _w1=FusedMoEQuantDesc(dtype="iq2_xs"),
        _w2=FusedMoEQuantDesc(dtype="iq2_xs"),
    )

    assert config.quant_schema.weight_key is None
    assert config.weight_quant_dtype == "iq2_xs"


def test_prepacked_weight_shape_remains_static_metadata() -> None:
    group_shape = GroupShape(1, 128)
    config = FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(shape=group_shape),
        _a2=FusedMoEQuantDesc(shape=group_shape),
        _w1=FusedMoEQuantDesc(shape=group_shape),
        _w2=FusedMoEQuantDesc(shape=group_shape),
    )

    assert config.quant_schema.a1.key is None
    assert config.quant_schema.w1.key is None
    assert config.quant_schema.a1.group_shape == group_shape
    assert config.quant_schema.w1.group_shape == group_shape
    assert config.block_shape == [1, 128]
    assert config.is_quantized is False


def test_alignment_shape_cannot_duplicate_key_shape() -> None:
    with pytest.raises(AssertionError, match="must not duplicate"):
        FusedMoEStaticQuantDesc(
            key=kFp8StaticTensorSym, alignment_shape=GroupShape.PER_TENSOR
        )


def test_schema_is_immutable() -> None:
    schema = FusedMoEQuantSchema(
        a1=FusedMoEStaticQuantDesc(),
        a2=FusedMoEStaticQuantDesc(),
        w1=FusedMoEStaticQuantDesc(),
        w2=FusedMoEStaticQuantDesc(),
    )

    with pytest.raises(AttributeError):
        schema.mx_alignment = 128  # type: ignore[misc]


def test_schema_reference_does_not_change_constructor_signature() -> None:
    assert "_schema" not in signature(FusedMoEQuantConfig).parameters
