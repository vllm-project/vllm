# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Map Humming schemas and handle shared checkpoint quantization settings."""

from typing import TYPE_CHECKING, Any

import regex as re
import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    FP4_DTYPE,
    FP8_DTYPE,
    INT4_DTYPE,
    INT8_DTYPE,
    MXFP_SCALE_DTYPE,
    GroupShape,
    QuantKey,
    ScaleDesc,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_humming

if TYPE_CHECKING:
    from vllm.utils.humming import (
        BaseInputSchema,
        BaseWeightSchema,
        HummingInputSchema,
    )
    from vllm.utils.humming import dtypes as humming_dtypes


if has_humming():
    from vllm.utils.humming import dtypes as humming_dtypes

    _HUMMING_TO_QUANT_DTYPE: dict[humming_dtypes.DataType, Any] = {
        humming_dtypes.float4e0m3: FP4_DTYPE,
        humming_dtypes.float4e2m1: FP4_DTYPE,
        humming_dtypes.float8e3m4: FP8_DTYPE,
        humming_dtypes.float8e4m3: FP8_DTYPE,
        humming_dtypes.float8e5m2: torch.float8_e5m2,
        humming_dtypes.int8: torch.int8,
        humming_dtypes.uint4: INT4_DTYPE,
        humming_dtypes.uint8: INT8_DTYPE,
        humming_dtypes.uint2: torch.uint8,
        humming_dtypes.uint3: torch.uint8,
        humming_dtypes.uint5: torch.uint8,
        humming_dtypes.uint6: torch.uint8,
        humming_dtypes.uint7: torch.uint8,
    }

    _HUMMING_TO_SCALE_DTYPE: dict[humming_dtypes.DataType, torch.dtype] = {
        humming_dtypes.float8e8m0: MXFP_SCALE_DTYPE,
        humming_dtypes.float8e4m3: FP8_DTYPE,
        humming_dtypes.float16: torch.float16,
        humming_dtypes.bfloat16: torch.bfloat16,
        humming_dtypes.float32: torch.float32,
    }


logger = init_logger(__name__)


def _group_shape(group_size: int, group_size_n: int = 0) -> GroupShape:
    """
    Map humming group sizes to QuantKey GroupShape.

    group_size:   elements per group along K (col); 0 means full dimension.
    group_size_n: elements per group along N (row); 0 means 1 (per-row).

    GroupShape convention: row = N dim, col = K dim.
    """
    if group_size == 0 and group_size_n == 0:
        return GroupShape.PER_CHANNEL

    row = group_size_n if group_size_n > 0 else 1
    col = group_size if group_size > 0 else -1
    return GroupShape(row=row, col=col)


def weight_schema_to_quant_key(
    schema: "BaseWeightSchema",
    param_dtype: torch.dtype | None = None,
) -> QuantKey:
    from vllm.utils.humming import WeightScale2Type, WeightScaleType

    if param_dtype is None:
        param_dtype = torch.get_default_dtype()
        if param_dtype not in [torch.float16, torch.bfloat16]:
            param_dtype = torch.bfloat16
            if not current_platform.has_device_capability(80):
                param_dtype = torch.float16

    schema = schema.to_humming_schema(param_dtype)
    dtype = _HUMMING_TO_QUANT_DTYPE[schema.b_dtype]

    if schema.bs_dtype is not None:
        scale_dtype = _HUMMING_TO_SCALE_DTYPE[schema.bs_dtype]
    else:
        scale_dtype = torch.float32

    group_shape = _group_shape(
        schema.weight_scale_group_size,
        schema.weight_scale_group_size_n,
    )
    if schema.weight_scale_type == WeightScaleType.TENSOR:
        group_shape = GroupShape.PER_TENSOR

    scale = ScaleDesc(dtype=scale_dtype, static=True, group_shape=group_shape)

    scale2 = None
    if schema.weight_scale_2_type == WeightScale2Type.TENSOR:
        scale2 = ScaleDesc(
            dtype=torch.float32,
            static=True,
            group_shape=GroupShape.PER_TENSOR,
        )

    return QuantKey(
        dtype=dtype,
        scale=scale,
        scale2=scale2,
        symmetric=not schema.has_zero_point,
    )


def input_schema_to_quant_key(
    schema: "BaseInputSchema",
    param_dtype: torch.dtype | None = None,
) -> QuantKey | None:
    from vllm.utils.humming import InputQuantizationMode

    if param_dtype is None:
        param_dtype = torch.get_default_dtype()
        if param_dtype not in [torch.float16, torch.bfloat16]:
            param_dtype = torch.bfloat16
            if not current_platform.has_device_capability(80):
                param_dtype = torch.float16

    schema = schema.to_humming_schema(param_dtype)
    if schema.a_dtype is None or schema.a_dtype.num_bits >= 16:
        return None

    mode = schema.input_quant_mode
    if mode == InputQuantizationMode.Disabled:
        return None
    dtype = _HUMMING_TO_QUANT_DTYPE[schema.a_dtype]

    gs = schema.input_scale_group_size
    group_shape = GroupShape(row=1, col=gs) if gs > 0 else GroupShape.PER_TOKEN

    # Pick the scale dtype the Humming kernel actually consumes. An explicit
    # input_scale_dtype always wins. Otherwise infer from the grouping: MX
    # microscale activations (group size 32, e.g. MXFP8) carry an e8m0 (uint8)
    # scale, while block-FP8 (group size 128) and per-token FP8/int8 carry a
    # float32 scale. Getting this right lets a grouped FP8 activation match
    # kFp8Dynamic128Sym instead of an unmatchable uint8-scaled key.
    if schema.input_scale_dtype is not None:
        scale_dtype = _HUMMING_TO_SCALE_DTYPE[schema.input_scale_dtype]
    elif gs == 16:
        scale_dtype = FP8_DTYPE
    elif gs == 32:
        scale_dtype = MXFP_SCALE_DTYPE
    else:
        scale_dtype = torch.float32

    if mode == InputQuantizationMode.StaticTensor:
        group_shape = GroupShape.PER_TENSOR
    scale = ScaleDesc(
        dtype=scale_dtype,
        static=mode == InputQuantizationMode.StaticTensor,
        group_shape=group_shape,
    )
    scale2 = None
    has_scale2 = (
        InputQuantizationMode.StaticTensorDynamicGroup,
        InputQuantizationMode.DynamicGroupToken,
    )
    if mode in has_scale2:
        static = mode == InputQuantizationMode.StaticTensorDynamicGroup
        scale2 = ScaleDesc(
            torch.float32,
            static,
            GroupShape.PER_TENSOR if static else GroupShape.PER_TOKEN,
        )

    return QuantKey(dtype=dtype, scale=scale, scale2=scale2, symmetric=True)


def quant_key_to_input_schema(key: QuantKey | None) -> "HummingInputSchema":
    from vllm.utils.humming import HummingInputSchema, InputQuantizationMode

    if key is None:
        return HummingInputSchema(input_quant_mode=InputQuantizationMode.Disabled)

    if not key.symmetric:
        raise ValueError("Humming input quantization must be symmetric")

    quant_dtypes = {value: dtype for dtype, value in _HUMMING_TO_QUANT_DTYPE.items()}
    quant_dtypes[FP4_DTYPE] = humming_dtypes.float4e2m1
    quant_dtypes[FP8_DTYPE] = humming_dtypes.float8e4m3
    scale_dtypes = {value: dtype for dtype, value in _HUMMING_TO_SCALE_DTYPE.items()}
    if key.dtype not in quant_dtypes or key.scale.dtype not in scale_dtypes:
        raise ValueError(f"Unsupported Humming input or scale dtype: {key}")

    scale = key.scale
    group_size = 0
    if scale.static and scale.group_shape.is_per_tensor() and key.scale2 is None:
        mode = InputQuantizationMode.StaticTensor
    elif not scale.static and scale.group_shape.is_per_token() and key.scale2 is None:
        mode = InputQuantizationMode.DynamicToken
    elif not scale.static and scale.group_shape.is_per_group():
        group_size = scale.group_shape.col
        scale2 = key.scale2
        if scale2 is None:
            mode = InputQuantizationMode.DynamicGroup
        elif scale2.dtype != torch.float32:
            raise ValueError("Humming secondary input scales must be float32")
        elif scale2.static and scale2.group_shape.is_per_tensor():
            mode = InputQuantizationMode.StaticTensorDynamicGroup
        elif not scale2.static and scale2.group_shape.is_per_token():
            mode = InputQuantizationMode.DynamicGroupToken
        else:
            raise ValueError(f"Unsupported Humming secondary input scale: {scale2}")
    else:
        raise ValueError(f"Unsupported Humming input scale: {key}")

    if group_size == 0 and scale.dtype != torch.float32:
        raise ValueError("Humming tensor and token input scales must be float32")

    return HummingInputSchema(
        a_dtype=quant_dtypes[key.dtype],
        input_scale_group_size=group_size,
        input_scale_dtype=scale_dtypes[scale.dtype],
        input_quant_mode=mode,
    )


def check_and_fallback_input_schema(
    weight_schema: "BaseWeightSchema",
    input_schema: "BaseInputSchema",
    param_dtype: torch.dtype | None = None,
    allow_fallback: bool = True,
) -> "HummingInputSchema":
    from vllm.utils.humming import HummingInputSchema, InputQuantizationMode

    capability = current_platform.get_device_capability()
    assert capability is not None
    sm_version = capability.to_int()
    if param_dtype is None:
        param_dtype = torch.get_default_dtype()
        if param_dtype not in (torch.float16, torch.bfloat16):
            param_dtype = torch.bfloat16 if sm_version >= 80 else torch.float16
    assert param_dtype in (torch.float16, torch.bfloat16)

    dtype_fallback_order_map = {
        humming_dtypes.float8e5m2: (
            humming_dtypes.float8e4m3,
            humming_dtypes.float8e3m4,
            humming_dtypes.int8,
        ),
        humming_dtypes.float8e4m3: (
            humming_dtypes.float8e3m4,
            humming_dtypes.int8,
        ),
        humming_dtypes.float8e3m4: (
            humming_dtypes.float8e4m3,
            humming_dtypes.int8,
        ),
        humming_dtypes.float4e2m1: (
            humming_dtypes.float8e4m3,
            humming_dtypes.float8e3m4,
            humming_dtypes.int8,
        ),
        humming_dtypes.float4e0m3: (
            humming_dtypes.int4,
            humming_dtypes.float8e4m3,
            humming_dtypes.float8e3m4,
            humming_dtypes.int8,
        ),
        humming_dtypes.int8: (
            humming_dtypes.float8e4m3,
            humming_dtypes.float8e3m4,
        ),
        humming_dtypes.int4: (
            humming_dtypes.float4e0m3,
            humming_dtypes.int8,
            humming_dtypes.float8e4m3,
            humming_dtypes.float8e3m4,
        ),
    }

    weight_schema = weight_schema.to_humming_schema(param_dtype)
    input_schema = input_schema.to_humming_schema(param_dtype)
    input_dtype = input_schema.input_dtype
    input_bits = input_dtype.num_bits if input_dtype is not None else 16
    input_group_size = input_schema.input_scale_group_size
    input_scale_dtype = input_schema.input_scale_dtype
    input_quant_mode = input_schema.input_quant_mode

    def is_deprecated(dtype: "humming_dtypes.DataType | None") -> bool:
        is_int4_deprecated = dtype == humming_dtypes.int4 and sm_version >= 90
        is_int8_deprecated = dtype == humming_dtypes.int8 and 103 <= sm_version < 110
        return is_int4_deprecated or is_int8_deprecated

    if input_schema.is_compatible_with(weight_schema, param_dtype):
        if not allow_fallback:
            if is_deprecated(input_dtype):
                logger.warning_once(f"{input_dtype} is deprecated on SM{sm_version}")
            return input_schema

        if input_dtype is None or input_dtype.num_bits == 16:
            return input_schema

        if not is_deprecated(input_dtype):
            is_mxfp8 = (
                input_dtype.is_floating_point_type
                and input_dtype.num_bits == 8
                and input_group_size == 32
                and input_scale_dtype == humming_dtypes.float8e8m0
                and sm_version >= 120
            )
            if input_bits == 8 and input_group_size >= 0 and not is_mxfp8:
                # Prefer tokenwise fp8/int8 when the weight pairing allows it.
                new_quant_mode = InputQuantizationMode.DynamicToken
                if input_quant_mode == InputQuantizationMode.StaticTensor:
                    new_quant_mode = InputQuantizationMode.StaticTensor
                candidate = HummingInputSchema(
                    input_dtype=input_dtype,
                    input_scale_group_size=0,
                    input_scale_dtype=humming_dtypes.float32,
                    input_quant_mode=new_quant_mode,
                )
                if candidate.is_compatible_with(weight_schema, param_dtype):
                    return candidate
            return input_schema

    if allow_fallback:
        fallback_dtypes = (
            input_dtype,
            *dtype_fallback_order_map.get(input_dtype, ()),
            humming_dtypes.DataType.from_any(param_dtype),
        )
        for dtype in fallback_dtypes:
            if dtype is None or is_deprecated(dtype):
                continue

            group_size = 0
            scale_dtype = humming_dtypes.float32
            quant_mode = InputQuantizationMode.DynamicToken
            if dtype.num_bits == 16:
                scale_dtype = None
                quant_mode = InputQuantizationMode.Disabled
            elif dtype in (humming_dtypes.float8e4m3, humming_dtypes.float8e3m4):
                is_channelwise_weight = weight_schema.weight_scale_group_size == 0
                weight_group_size = weight_schema.weight_scale_group_size
                is_e8m0_scale = weight_schema.bs_dtype == humming_dtypes.float8e8m0
                is_mx_weight = weight_group_size == 32 and is_e8m0_scale
                if sm_version >= 120 and (is_channelwise_weight or is_mx_weight):
                    group_size = 32
                    scale_dtype = humming_dtypes.float8e8m0
                    quant_mode = InputQuantizationMode.DynamicGroup

            candidate = HummingInputSchema(
                input_dtype=dtype,
                input_scale_group_size=group_size,
                input_scale_dtype=scale_dtype,
                input_quant_mode=quant_mode,
            )
            if candidate.is_compatible_with(weight_schema, param_dtype):
                return candidate

    raise ValueError(
        f"No compatible Humming input schema for {input_schema} with "
        f"weight schema {weight_schema} on SM{sm_version} "
        f"({allow_fallback=})"
    )


def humming_is_layer_skipped(config: dict[str, Any], prefix: str):
    if not config:
        return True

    keys = ["ignored_layers", "ignore", "modules_to_not_convert"]
    ignored_layers: list[str] = []
    for key in keys:
        candidate = config.get(key, []) or []
        if candidate:
            ignored_layers = candidate
            break

    if any(module_name in prefix for module_name in ignored_layers):
        return True
    if "lm_head" in prefix:
        return True

    for regex in config.get("dynamic", {}):
        if regex[:1] != "-":
            continue
        if re.match(regex[2:], prefix):
            return True

    return False
