# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import regex as re
import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
)
from vllm.model_executor.layers.linear import LinearBase
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
from vllm.utils.import_utils import has_humming

if TYPE_CHECKING:
    from vllm.utils.humming import (
        AWQWeightSchema,
        BaseInputSchema,
        BaseWeightSchema,
        CompressedTensorsInputSchema,
        CompressedTensorsWeightSchema,
        Fp8WeightSchema,
        GPTQWeightSchema,
        HummingInputSchema,
        HummingWeightSchema,
        LayerConfig,
    )
    from vllm.utils.humming import dtypes as humming_dtypes

logger = init_logger(__name__)


@dataclass(kw_only=True)
class HummingMoEQuantConfig(FusedMoEQuantConfig):
    w1_humming_config: "LayerConfig"
    w2_humming_config: "LayerConfig"


if has_humming():
    from vllm.utils.humming import dtypes as humming_dtypes

    _HUMMING_TO_QUANT_DTYPE: dict[humming_dtypes.DataType, Any] = {
        humming_dtypes.float4e2m1: FP4_DTYPE,
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


def _group_shape(group_size: int, group_size_n: int = 0) -> GroupShape:
    """Map humming group sizes to QuantKey GroupShape.

    group_size:   elements per group along K (col); 0 means full dimension.
    group_size_n: elements per group along N (row); 0 means 1 (per-row).

    GroupShape convention: row = N dim, col = K dim.
    """
    if group_size == 0 and group_size_n == 0:
        return GroupShape.PER_CHANNEL

    row = group_size_n if group_size_n > 0 else 1
    col = group_size if group_size > 0 else -1
    return GroupShape(row=row, col=col)


# ---- HummingWeightSchema (post-conversion) --------------------------------


def _humming_weight_schema_to_quant_key(
    schema: "HummingWeightSchema",
) -> QuantKey:
    from vllm.utils.humming import WeightScaleType

    """Convert a HummingWeightSchema to a QuantKey."""
    dtype = _HUMMING_TO_QUANT_DTYPE[schema.b_dtype]

    if schema.bs_dtype is not None:
        scale_dtype = _HUMMING_TO_SCALE_DTYPE[schema.bs_dtype]
    else:
        scale_dtype = torch.float32

    group_shape = _group_shape(
        schema.weight_scale_group_size,
        schema.weight_scale_group_size_n,
    )

    scale = ScaleDesc(dtype=scale_dtype, static=True, group_shape=group_shape)

    scale2 = None
    if schema.weight_scale_type == WeightScaleType.GROUP_TENSOR:
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


# ---- Checkpoint-format weight schemas (pre-conversion) --------------------


def _fp8_weight_schema_to_quant_key(schema: "Fp8WeightSchema") -> QuantKey:
    if schema.weight_block_size is not None:
        gs_n, gs_k = schema.weight_block_size
        group_shape = GroupShape(row=gs_n, col=gs_k)
    else:
        group_shape = GroupShape.PER_CHANNEL

    scale = ScaleDesc(dtype=torch.float32, static=True, group_shape=group_shape)
    return QuantKey(dtype=FP8_DTYPE, scale=scale, symmetric=True)


def _awq_weight_schema_to_quant_key(schema: "AWQWeightSchema") -> QuantKey:
    group_shape = _group_shape(schema.group_size)
    scale = ScaleDesc(
        dtype=torch.float16,
        static=True,
        group_shape=group_shape,
    )
    return QuantKey(
        dtype=INT4_DTYPE,
        scale=scale,
        symmetric=not schema.zero_point,
    )


def _gptq_weight_schema_to_quant_key(schema: "GPTQWeightSchema") -> QuantKey:
    group_shape = _group_shape(schema.group_size)
    scale = ScaleDesc(
        dtype=torch.float16,
        static=True,
        group_shape=group_shape,
    )
    return QuantKey(dtype=INT4_DTYPE, scale=scale, symmetric=schema.sym)


def _compressed_tensors_weight_schema_to_quant_key(
    schema: "CompressedTensorsWeightSchema",
) -> QuantKey:
    # Determine dtype from format/type/num_bits
    fmt = schema.format
    if fmt in ("int-quantized", "float-quantized", "naive-quantized"):
        dtype = INT8_DTYPE if schema.type == "int" else FP8_DTYPE
    elif "nvfp4" in fmt or "mxfp4" in fmt:
        dtype = FP4_DTYPE
    else:
        dtype = _HUMMING_TO_QUANT_DTYPE[
            humming_dtypes.DataType.from_str(f"uint{schema.num_bits}")
        ]

    # Determine group shape from strategy
    if schema.strategy in ("group", "tensor_group"):
        group_shape = _group_shape(schema.group_size or 0)
    elif schema.strategy == "block" and schema.block_structure is not None:
        group_shape = GroupShape(
            row=schema.block_structure[0],
            col=schema.block_structure[1],
        )
    else:
        group_shape = GroupShape.PER_CHANNEL

    # Determine scale dtype
    if "mxfp" in fmt:
        scale_dtype = MXFP_SCALE_DTYPE
    elif "nvfp4" in fmt:
        scale_dtype = FP8_DTYPE
    else:
        scale_dtype = torch.float32

    scale = ScaleDesc(dtype=scale_dtype, static=True, group_shape=group_shape)

    scale2 = None
    if "nvfp4" in fmt or schema.strategy == "tensor_group":
        scale2 = ScaleDesc(
            dtype=torch.float32,
            static=True,
            group_shape=GroupShape.PER_TENSOR,
        )

    return QuantKey(
        dtype=dtype,
        scale=scale,
        scale2=scale2,
        symmetric=schema.symmetric,
    )


# ---- Dispatch for any BaseWeightSchema ------------------------------------


def weight_schema_to_quant_key(
    schema: "BaseWeightSchema",
) -> QuantKey:
    from vllm.utils.humming import (
        AWQWeightSchema,
        BitnetWeightSchema,
        CompressedTensorsWeightSchema,
        Fp8WeightSchema,
        GptOssMxfp4WeightSchema,
        GPTQWeightSchema,
        HummingWeightSchema,
        ModeloptMxfp8WeightSchema,
        ModeloptNvfp4WeightSchema,
        Mxfp4WeightSchema,
    )

    """Convert any BaseWeightSchema to a QuantKey."""
    if isinstance(schema, HummingWeightSchema):
        return _humming_weight_schema_to_quant_key(schema)

    # Schemas with fixed QuantKeys
    if isinstance(schema, (Mxfp4WeightSchema, GptOssMxfp4WeightSchema)):
        return QuantKey(
            dtype=FP4_DTYPE,
            scale=ScaleDesc(MXFP_SCALE_DTYPE, True, GroupShape(1, 32)),
        )
    if isinstance(schema, ModeloptMxfp8WeightSchema):
        return QuantKey(
            dtype=FP8_DTYPE,
            scale=ScaleDesc(MXFP_SCALE_DTYPE, True, GroupShape(1, 32)),
        )
    if isinstance(schema, ModeloptNvfp4WeightSchema):
        return QuantKey(
            dtype=FP4_DTYPE,
            scale=ScaleDesc(FP8_DTYPE, True, GroupShape(1, 16)),
            scale2=ScaleDesc(torch.float32, True, GroupShape.PER_TENSOR),
        )
    if isinstance(schema, BitnetWeightSchema):
        return QuantKey(
            dtype=torch.uint8,
            scale=ScaleDesc(torch.float32, True, GroupShape.PER_CHANNEL),
        )

    # Schemas requiring config inspection
    if isinstance(schema, Fp8WeightSchema):
        return _fp8_weight_schema_to_quant_key(schema)
    if isinstance(schema, AWQWeightSchema):
        return _awq_weight_schema_to_quant_key(schema)
    if isinstance(schema, GPTQWeightSchema):
        return _gptq_weight_schema_to_quant_key(schema)
    if isinstance(schema, CompressedTensorsWeightSchema):
        return _compressed_tensors_weight_schema_to_quant_key(schema)

    raise TypeError(f"Unsupported weight schema type: {type(schema)}")


# ---- HummingInputSchema (post-conversion) ----------------------------------


def _humming_input_schema_to_quant_key(
    schema: "HummingInputSchema",
) -> QuantKey | None:
    """Convert a HummingInputSchema to a QuantKey. Returns None if
    the schema represents unquantized (bf16/fp16) inputs."""
    if schema.a_dtype is None or schema.a_dtype.num_bits >= 16:
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
    elif gs == 32:
        scale_dtype = MXFP_SCALE_DTYPE
    else:
        scale_dtype = torch.float32

    scale = ScaleDesc(dtype=scale_dtype, static=False, group_shape=group_shape)

    return QuantKey(dtype=dtype, scale=scale, symmetric=True)


# ---- Checkpoint-format input schemas (pre-conversion) ----------------------


def _resolve_input_quant_key(
    origin_a_dtype: "humming_dtypes.DataType",
    group_size: int,
) -> QuantKey | None:
    from vllm.utils.humming import HummingInputSchema

    """Resolve the actual activation QuantKey after platform fallback."""
    a_dtype = HummingInputSchema().get_fallback_input_dtype(origin_a_dtype)
    if a_dtype is None or a_dtype.num_bits >= 16:
        return None

    dtype = _HUMMING_TO_QUANT_DTYPE[a_dtype]
    gs = group_size if a_dtype == humming_dtypes.float4e2m1 else 0
    group_shape = GroupShape(row=1, col=gs) if gs > 0 else GroupShape.PER_TOKEN
    scale_dtype = MXFP_SCALE_DTYPE if gs > 0 else torch.float32

    scale = ScaleDesc(dtype=scale_dtype, static=False, group_shape=group_shape)
    return QuantKey(dtype=dtype, scale=scale, symmetric=True)


def _compressed_tensors_input_schema_to_quant_key(
    schema: "CompressedTensorsInputSchema",
) -> QuantKey | None:
    type_bits_to_dtype = {
        ("float", 8): humming_dtypes.float8e4m3,
        ("float", 4): humming_dtypes.float4e2m1,
        ("int", 8): humming_dtypes.int8,
        ("int", 4): humming_dtypes.int4,
    }
    origin = type_bits_to_dtype.get((schema.type, schema.num_bits))
    if origin is None:
        return None
    return _resolve_input_quant_key(origin, schema.group_size)


# ---- Dispatch for any BaseInputSchema -------------------------------------


def input_schema_to_quant_key(
    schema: "BaseInputSchema",
) -> QuantKey | None:
    from vllm.utils.humming import (
        CompressedTensorsInputSchema,
        Fp8InputSchema,
        HummingInputSchema,
        ModeloptNvfp4InputSchema,
    )

    """Convert any BaseInputSchema to a QuantKey. Returns None if
    the schema represents unquantized (bf16/fp16) inputs."""
    if isinstance(schema, HummingInputSchema):
        return _humming_input_schema_to_quant_key(schema)

    if isinstance(schema, Fp8InputSchema):
        return _resolve_input_quant_key(humming_dtypes.float8e4m3, 0)

    if isinstance(schema, ModeloptNvfp4InputSchema):
        return _resolve_input_quant_key(
            humming_dtypes.float8e4m3,
            schema.group_size,
        )

    if isinstance(schema, CompressedTensorsInputSchema):
        return _compressed_tensors_input_schema_to_quant_key(schema)

    raise TypeError(f"Unsupported input schema type: {type(schema)}")


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


def convert_linear_layer_to_humming_standard(
    layer: LinearBase, name_map: dict[str, str]
):
    """Rename/reshape a linear layer's quantized params (the canonical MPLinear
    layout: ``weight_packed`` int32 + ``weight_scale``) into the parameter names
    and layout humming's weight schema expects (``weight`` / ``weight_scale``)."""
    for name, checkpoint_name in name_map.items():
        tensor = getattr(layer, checkpoint_name)
        delattr(layer, checkpoint_name)

        if name == "weight":
            input_dim = getattr(tensor, "input_dim", 1)
            output_dim = getattr(tensor, "output_dim", 0)

            if input_dim == 0 and output_dim == 1:
                tensor = tensor.transpose(1, 0).contiguous()
            else:
                assert output_dim == 0 and input_dim == 1

            tensor = tensor.view(tensor.size(0), -1).view(torch.int32)
        elif name in ["weight_scale", "zero_point"]:
            if getattr(tensor, "output_dim", 0) == 1:
                tensor = tensor.transpose(0, 1).contiguous()
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(1)

            tensor = tensor.view(torch.int32) if name == "zero_point" else tensor

        if isinstance(tensor, torch.nn.Parameter):
            param = tensor
        else:
            param = torch.nn.Parameter(tensor, requires_grad=False)

        setattr(layer, name, param)


def prepare_humming_linear_layer_config(
    layer: LinearBase,
    quant_config: dict,
    input_quant_config: dict | None = None,
) -> "LayerConfig":
    from vllm.utils.humming import (
        BaseInputSchema,
        BaseWeightSchema,
        HummingInputSchema,
        prepare_layer_config,
        transform_humming_tensors,
    )

    weight_schema = BaseWeightSchema.from_config(quant_config)
    if input_quant_config is not None:
        input_schema = BaseInputSchema.from_config(input_quant_config)
    else:
        input_schema = HummingInputSchema()

    # ReplicatedLinear has no TP partitioning and so does not set
    # input_size_per_partition; for it that is just input_size. Use hasattr
    # rather than getattr's default arg, which is evaluated eagerly and would
    # raise on layers lacking input_size (e.g. ParallelLMHead).
    if hasattr(layer, "input_size_per_partition"):
        input_size_per_partition = layer.input_size_per_partition
    else:
        input_size_per_partition = layer.input_size
    shape_k_stacks = [input_size_per_partition]
    shape_n_stacks = layer.output_partition_sizes

    # Step 1: convert weight and input schemas to humming standard format
    weight_schema, tensors = weight_schema.convert_humming(
        tensors=dict(layer.named_parameters()),
        shape_n_stacks=shape_n_stacks,
        shape_k_stacks=shape_k_stacks,
        param_dtype=layer.params_dtype,
    )
    input_schema, _ = input_schema.convert_humming(
        tensors={},
        shape_n_stacks=shape_n_stacks,
        shape_k_stacks=shape_k_stacks,
        param_dtype=layer.params_dtype,
    )

    layer.weight_schema = weight_schema

    # Step 2: transform weight (humming standard format) for forwarding.
    config = prepare_layer_config(
        shape_n=sum(layer.output_partition_sizes),
        shape_k=input_size_per_partition,
        weight_schema=weight_schema,
        input_schema=input_schema,
        pad_n_to_multiple=256,
        pad_k_to_multiple=128,
        has_bias=layer.has_bias,
        torch_dtype=layer.params_dtype,
    )
    tensors = transform_humming_tensors(config, tensors)
    for name, _ in list(layer.named_parameters()):
        delattr(layer, name)
    for name, tensor in tensors.items():
        param = torch.nn.Parameter(tensor, requires_grad=False)
        setattr(layer, name, param)

    return config


def get_humming_linear_compute_config() -> str:
    return json.dumps(
        {
            "use_batch_invariant": envs.VLLM_BATCH_INVARIANT,
            "use_f16_accum": envs.VLLM_HUMMING_USE_F16_ACCUM,
            "gemm_type": "dense",
        }
    )


def apply_humming_linear(
    layer: LinearBase,
    x: torch.Tensor,
    *,
    layer_config: "LayerConfig",
    compute_config: str,
    locks: torch.Tensor,
) -> torch.Tensor:
    from vllm.utils.humming import humming_forward

    flatten_inputs = x.reshape(-1, x.size(-1))
    output = humming_forward(
        layer_config,
        inputs=flatten_inputs,
        weight=layer.weight,
        weight_scale=getattr(layer, "weight_scale", None),
        zero_point=getattr(layer, "zero_point", None),
        bias=getattr(layer, "bias", None),
        weight_scale_2=getattr(layer, "weight_scale_2", None),
        locks=locks,
        compute_config=compute_config,
    )
    return output.view(*x.shape[:-1], output.size(-1))


def make_humming_moe_quant_config(
    quant_dtype: torch.dtype | str | None,
    weight_dtype: torch.dtype | str | None,
    weight_group_shape: GroupShape | None = None,
    activation_group_shape: GroupShape | None = None,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    w1_gscale: torch.Tensor | None = None,
    w2_gscale: torch.Tensor | None = None,
    gemm1_alpha: float | None = None,
    gemm1_beta: float | None = None,
    gemm1_clamp_limit: float | None = None,
    humming_configs: dict[str, "LayerConfig"] | None = None,
) -> HummingMoEQuantConfig:
    assert humming_configs is not None
    if quant_dtype is None:
        a_quant_desc = FusedMoEQuantDesc(dtype=None)
    elif activation_group_shape is not None:
        # Pre-dispatch quantization.
        a_quant_desc = FusedMoEQuantDesc(
            dtype=quant_dtype, shape=activation_group_shape
        )
    else:
        # Deferred path: Humming quantizes the activation internally, so the
        # descriptor only needs a non-None dtype to mark it as quantized.
        shape = GroupShape(row=1, col=-1)
        a_quant_desc = FusedMoEQuantDesc(dtype=quant_dtype, shape=shape)

    w1_quant_desc = FusedMoEQuantDesc(
        dtype=weight_dtype,
        shape=weight_group_shape,
        scale=w1_scale,
        alpha_or_gscale=w1_gscale,
        zp=w1_zp,
        bias=w1_bias,
    )

    w2_quant_desc = FusedMoEQuantDesc(
        dtype=weight_dtype,
        shape=weight_group_shape,
        scale=w2_scale,
        alpha_or_gscale=w2_gscale,
        zp=w2_zp,
        bias=w2_bias,
    )

    return HummingMoEQuantConfig(
        _a1=a_quant_desc,
        _a2=a_quant_desc,
        _w1=w1_quant_desc,
        _w2=w2_quant_desc,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
        w1_humming_config=humming_configs["w13"],
        w2_humming_config=humming_configs["w2"],
    )
