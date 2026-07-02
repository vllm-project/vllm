# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from typing import TYPE_CHECKING, Any

import regex as re
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
)
from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
    BatchedHummingGroupedExperts,
    HummingGroupedExperts,
    HummingIndexedExperts,
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
    pack_quantized_values_into_int32,
)
from vllm.scalar_type import ScalarType
from vllm.utils.import_utils import has_humming

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
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
    )
    from vllm.utils.humming import dtypes as humming_dtypes

logger = init_logger(__name__)

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
    }

    _HUMMING_TO_SCALE_DTYPE: dict[humming_dtypes.DataType, torch.dtype] = {
        humming_dtypes.float8e8m0: MXFP_SCALE_DTYPE,
        humming_dtypes.float8e4m3: FP8_DTYPE,
        humming_dtypes.float16: torch.float16,
        humming_dtypes.bfloat16: torch.bfloat16,
        humming_dtypes.float32: torch.float32,
    }


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

    scale_dtype = MXFP_SCALE_DTYPE if gs > 0 else torch.float32

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
    layer: LinearBase, name_map: dict[str, str], weight_type: ScalarType
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

            if tensor.dtype == torch.int32:
                # Already bit-packed (e.g. PackedvLLMParameter) upstream.
                tensor = tensor.view(tensor.size(0), -1).view(torch.int32)
            else:
                # `tensor` holds one signed quantized value per element (e.g.
                # an int4 value in [-8, 7] occupying a full int8 byte), not
                # bit-packed. humming's weight format stores unsigned values
                # biased by 2**(size_bits-1) (mirroring compressed-tensors'
                # own `weight + 128` conversion for its 8-bit case), so apply
                # that bias before packing `weight_type.size_bits`-wide
                # values into int32 lanes.
                bias = 1 << (weight_type.size_bits - 1)
                tensor = pack_quantized_values_into_int32(
                    tensor.to(torch.int32) + bias, weight_type, packed_dim=1
                )
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


def prepare_humming_layer(layer: LinearBase, quant_config: dict):
    from vllm.utils.humming import (
        BaseWeightSchema,
        HummingInputSchema,
        HummingMethod,
    )

    weight_schema = BaseWeightSchema.from_config(quant_config)
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

    # Step 1: convert weight to humming standard format
    weight_schema, tensors = weight_schema.convert_humming(
        tensors=dict(layer.named_parameters()),
        shape_n_stacks=shape_n_stacks,
        shape_k_stacks=shape_k_stacks,
        param_dtype=layer.params_dtype,
    )

    layer.weight_schema = weight_schema

    for name, _ in list(layer.named_parameters()):
        delattr(layer, name)

    for name, tensor in tensors.items():
        if isinstance(tensor, torch.nn.Parameter):
            tensor = tensor.data
        param = torch.nn.Parameter(tensor, requires_grad=False)
        setattr(layer, name, param)

    # Step 2: transform weight (humming standard format) for forwarding
    HummingMethod.prepare_layer_meta(
        layer=layer,
        shape_n=sum(layer.output_partition_sizes),
        shape_k=input_size_per_partition,
        weight_schema=weight_schema,
        input_schema=input_schema,
        pad_n_to_multiple=256,
        pad_k_to_multiple=128,
        has_bias=layer.has_bias,
        torch_dtype=layer.params_dtype,
    )

    HummingMethod.transform_humming_layer(layer)
    if not hasattr(layer, "locks"):
        device = layer.weight.device
        locks = torch.zeros(1024, dtype=torch.int32, device=device)
        layer.register_buffer("locks", locks)

    compute_config = {
        "use_batch_invariant": envs.VLLM_BATCH_INVARIANT,
        "use_f16_accum": envs.VLLM_HUMMING_USE_F16_ACCUM,
        "gemm_type": "dense",
    }

    layer.compute_config = json.dumps(compute_config)


def make_humming_moe_quant_config(
    quant_dtype: torch.dtype | str | None,
    weight_dtype: torch.dtype | str | None,
    weight_group_shape: GroupShape | None = None,
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
) -> FusedMoEQuantConfig:
    if quant_dtype is None:
        a_quant_desc = FusedMoEQuantDesc(dtype=None)
    else:
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

    return FusedMoEQuantConfig(
        _a1=a_quant_desc,
        _a2=a_quant_desc,
        _w1=w1_quant_desc,
        _w2=w2_quant_desc,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
    )


def get_humming_moe_quant_config(
    layer: "RoutedExperts",
    gemm1_alpha: float | None = None,
    gemm1_beta: float | None = None,
    gemm1_clamp_limit: float | None = None,
):
    input_schema = layer.input_schemas["w13"]
    weight_schema = layer.weight_schemas["w13"]

    if input_schema.a_dtype is None or input_schema.a_dtype.num_bits == 16:
        q_dtype = None
    else:
        q_dtype = str(input_schema.a_dtype)

    weight_scale_group_size = weight_schema.weight_scale_group_size
    weight_scale_group_size_n = weight_schema.weight_scale_group_size_n
    weight_group_shape: tuple[int, ...] = ()
    if weight_scale_group_size_n > 1:
        weight_group_shape = GroupShape(
            row=weight_scale_group_size,
            col=weight_scale_group_size_n,
        )
    elif weight_scale_group_size == 0:
        weight_group_shape = GroupShape(row=-1, col=1)
    else:
        weight_group_shape = GroupShape(row=weight_scale_group_size, col=1)

    return make_humming_moe_quant_config(
        quant_dtype=q_dtype,
        weight_dtype=str(weight_schema.b_dtype),
        weight_group_shape=weight_group_shape,
        w1_scale=getattr(layer, "w13_weight_scale", None),
        w1_gscale=getattr(layer, "w13_global_scale", None),
        w1_zp=getattr(layer, "w13_zero_point", None),
        w1_bias=getattr(layer, "w13_bias", None),
        w2_scale=getattr(layer, "w2_weight_scale", None),
        w2_gscale=getattr(layer, "w2_global_scale", None),
        w2_zp=getattr(layer, "w2_zero_point", None),
        w2_bias=getattr(layer, "w2_bias", None),
    )


def select_humming_moe_experts(
    config: FusedMoEConfig,
    weight_key: QuantKey | None,
    activation_key: QuantKey | None,
) -> type[mk.FusedMoEExperts] | None:
    """
    Select the primary Humming MoE Experts class
    Note: Shape-specific fallbacks may still occur at runtime.
    """

    if not has_humming():
        return None

    # NOTE: the kernels are selected in the following order.
    AVAILABLE_EXPERTS: list[type[mk.FusedMoEExperts]] = [
        BatchedHummingGroupedExperts,
        HummingGroupedExperts,
        HummingIndexedExperts,
    ]

    # NOTE(rob): We need to peak into the P/F selection to determine
    # if we are using the batched or standard expert format, which
    # if not ideal. Once we unify TP + DP/EP, we can select P/F first.
    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )

    def _make_log_backend(experts_cls: type[mk.FusedMoEExperts]):
        return f"Using {experts_cls.__name__} Humming MoE backend."

    def _make_log_unsupported(
        experts_cls: type[mk.FusedMoEExperts], reason: str | None
    ) -> str:
        if reason:
            return (
                f"Humming MoE experts {experts_cls.__name__} does not support the "
                f"deployment configuration since {reason}."
            )
        else:
            return (
                f"Humming MoE experts '{experts_cls.__name__}' does not support the "
                "deployment configuration."
            )

    for k_cls in AVAILABLE_EXPERTS:
        supported, reason = k_cls.is_supported_config(
            k_cls,
            config,
            weight_key,
            activation_key,
            activation_format,
        )
        if supported:
            logger.info_once(_make_log_backend(k_cls))
            return k_cls
        else:
            logger.debug_once(_make_log_unsupported(k_cls, reason))

    return None


def make_humming_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEExperts],
    layer: "RoutedExperts",
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> mk.FusedMoEKernel:
    # Create Prepare/Finalize.
    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=moe_quant_config,
        routing_tables=routing_tables,
        allow_new_interface=True,
        use_monolithic=issubclass(experts_cls, mk.FusedMoEExpertsMonolithic),
    )
    assert prepare_finalize is not None

    logger.info_once("Using %s", prepare_finalize.__class__.__name__)

    extra_args: dict[str, Any] = {"layer": layer}

    # Create Experts.
    if prepare_finalize.activation_format == mk.FusedMoEActivationFormat.BatchedExperts:
        max_num_tokens = prepare_finalize.max_num_tokens_per_rank()
        assert max_num_tokens is not None
        experts = experts_cls(
            moe_config=moe_config,
            quant_config=moe_quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=prepare_finalize.num_dispatchers(),
            **extra_args,
        )
    else:
        experts = experts_cls(
            moe_config=moe_config,
            quant_config=moe_quant_config,
            **extra_args,
        )

    kernel = mk.FusedMoEKernel(
        prepare_finalize,
        experts,
    )

    return kernel


def _extract_sublayer_tensors(
    layer: "RoutedExperts",
    sublayer_name: str,
) -> dict[str, torch.Tensor]:
    """Extract tensors for a specific sublayer from the layer's state dict."""
    return dict(
        (key.removeprefix(sublayer_name + "_"), value)
        for key, value in layer.state_dict().items()
        if key.startswith(sublayer_name + "_")
    )


def _replace_layer_parameters(
    layer: "RoutedExperts",
    sublayer_name: str,
    tensors: dict[str, torch.Tensor],
    preserve_bias: bool = False,
) -> None:
    """
    Replace layer parameters for a sublayer with new tensors.

    Args:
        layer: The RoutedExperts layer
        sublayer_name: Name of the sublayer (e.g., "w13", "w2")
        tensors: Dict of parameter name to tensor
        preserve_bias: If True, don't delete bias parameters
    """
    # Delete old parameters
    for name, _ in list(layer.named_parameters()):
        if not name.startswith(sublayer_name + "_"):
            continue
        if preserve_bias and name == sublayer_name + "_bias":
            continue
        delattr(layer, name)

    # Set new parameters
    for name, tensor in tensors.items():
        param_name = f"{sublayer_name}_{name}"
        param = torch.nn.Parameter(tensor, requires_grad=False)
        setattr(layer, param_name, param)


def _convert_sublayer_to_humming(
    layer: "RoutedExperts",
    sublayer_name: str,
    shape_n: int,
    shape_k: int,
    weight_schema: Any,
    input_schema: Any,
    num_experts: int,
    param_dtype: torch.dtype,
) -> tuple[Any, Any]:
    """
    Convert a sublayer's weights from checkpoint format to Humming format.

    Returns:
        Tuple of (converted_weight_schema, converted_input_schema)
    """
    from humming.schema import HummingWeightSchema

    if isinstance(weight_schema, HummingWeightSchema):
        # Already in Humming format
        return weight_schema, input_schema

    tensors = _extract_sublayer_tensors(layer, sublayer_name)

    shape_k_stacks = [shape_k]
    shape_n_stacks = [shape_n]
    if sublayer_name == "w13":
        shape_n_stacks = [shape_n // 2] * 2

    converted_weight_schema, converted_tensors = weight_schema.convert_humming(
        tensors=tensors,
        shape_n_stacks=shape_n_stacks,
        shape_k_stacks=shape_k_stacks,
        param_dtype=param_dtype,
        num_experts=num_experts,
    )

    converted_input_schema, _ = input_schema.convert_humming(
        tensors=converted_tensors,
        shape_n_stacks=shape_n_stacks,
        shape_k_stacks=shape_k_stacks,
        param_dtype=param_dtype,
        num_experts=num_experts,
    )

    _replace_layer_parameters(layer, sublayer_name, converted_tensors)

    return converted_weight_schema, converted_input_schema


def _prepare_and_transform_sublayer(
    layer: "RoutedExperts",
    sublayer_name: str,
    shape_n: int,
    shape_k: int,
    weight_schema: Any,
    input_schema: Any,
    has_bias: bool,
    num_experts: int,
    param_dtype: torch.dtype,
) -> None:
    """
    Prepare layer metadata and transform weights for a sublayer.

    This calls Humming's prepare_layer_meta and transform_humming_layer.
    """
    from humming.layer import HummingMethod

    HummingMethod.prepare_layer_meta(
        layer=layer,
        shape_n=shape_n,
        shape_k=shape_k,
        pad_n_to_multiple=256,
        pad_k_to_multiple=128,
        input_schema=input_schema,
        weight_schema=weight_schema,
        has_bias=has_bias,
        num_experts=num_experts,
        torch_dtype=param_dtype,
        sublayer_name=sublayer_name,
    )

    HummingMethod.transform_humming_layer(layer, sublayer_name=sublayer_name)


def _process_single_sublayer(
    layer: "RoutedExperts",
    sublayer_name: str,
    shape_n: int,
    shape_k: int,
    weight_schema: Any,
    input_schema: Any,
    has_bias: bool,
    num_experts: int,
    param_dtype: torch.dtype,
    force_weight_schema: Any | None = None,
) -> tuple[Any, Any]:
    """
    Process a single sublayer: convert, optionally requant, prepare, and transform.

    This combines the common logic from convert_to_humming_moe_kernel_format
    for processing a single sublayer.

    Args:
        layer: The RoutedExperts layer
        sublayer_name: Name of the sublayer (e.g., "w13", "w2")
        shape_n: Output dimension size
        shape_k: Input dimension size
        weight_schema: Initial weight quantization schema
        input_schema: Initial input quantization schema
        has_bias: Whether the layer has bias terms
        num_experts: Number of experts
        param_dtype: Parameter data type
        force_weight_schema: Optional schema to force requantization to

    Returns:
        Tuple of (final_weight_schema, final_input_schema)
    """
    from humming.schema import HummingWeightSchema

    # Step 1: Convert from checkpoint format to humming format if needed
    current_weight_schema, current_input_schema = _convert_sublayer_to_humming(
        layer=layer,
        sublayer_name=sublayer_name,
        shape_n=shape_n,
        shape_k=shape_k,
        weight_schema=weight_schema,
        input_schema=input_schema,
        num_experts=num_experts,
        param_dtype=param_dtype,
    )

    # Step 2: Force requant if needed
    assert isinstance(current_weight_schema, HummingWeightSchema)
    if force_weight_schema is not None and current_weight_schema != force_weight_schema:
        tensors = _extract_sublayer_tensors(layer, sublayer_name)

        tensors = current_weight_schema.requant_tensors(
            tensors=tensors,
            target_weight_schema=force_weight_schema,
            param_dtype=param_dtype,
        )

        current_weight_schema = force_weight_schema
        _replace_layer_parameters(layer, sublayer_name, tensors, preserve_bias=True)
        del tensors

    # Step 3: Prepare layer metadata and transform weights
    _prepare_and_transform_sublayer(
        layer=layer,
        sublayer_name=sublayer_name,
        shape_n=shape_n,
        shape_k=shape_k,
        weight_schema=current_weight_schema,
        input_schema=current_input_schema,
        has_bias=has_bias,
        num_experts=num_experts,
        param_dtype=param_dtype,
    )

    return current_weight_schema, current_input_schema


def convert_to_humming_moe_kernel_format(
    layer: "RoutedExperts",
    quant_config: dict | None = None,
    sublayer_configs: dict[str, Any] | None = None,
    weight_schema: Any | None = None,
    input_schema: Any | None = None,
    force_weight_schema: Any | None = None,
) -> None:
    """
    Convert MoE weights from checkpoint format to Humming kernel format.

    This function processes weights for each sublayer (w13, w2) by:
    1. Converting from checkpoint format to humming format if needed
    2. Force requanting if a different quantization schema is specified
    3. Preparing layer metadata for the Humming kernel
    4. Transforming weights for inference

    Args:
        layer: The RoutedExperts layer containing weights to process
        quant_config: Optional quantization config dict. Required if weight_schema
                     or input_schema are None. Used to build schemas via
                     BaseWeightSchema.from_config().
        sublayer_configs: Optional configuration dict for each sublayer (w13, w2).
                         Each config must have "shape_n" and "shape_k" keys.
                         If None, configs are built from layer.moe_config properties.
        weight_schema: Optional initial weight quantization schema.
                      If None, built from quant_config.
        input_schema: Optional initial input quantization schema.
                     If None, built from quant_config or env vars.
        force_weight_schema: Optional schema to force requantization to

    Side effects:
        - Modifies layer parameters in place
        - Sets layer.weight_schemas and layer.input_schemas
    """

    # Build schemas from quant_config if not provided
    has_bias = layer.moe_config.has_bias
    num_experts = layer.moe_config.num_local_experts
    param_dtype = layer.params_dtype

    if weight_schema is None or input_schema is None:
        if quant_config is None:
            raise ValueError(
                "Must provide either weight_schema/input_schema or quant_config"
            )

        from humming.layer import HummingInputSchema
        from humming.schema import BaseWeightSchema

        from vllm.model_executor.layers.quantization.utils.humming_utils import (
            humming_is_layer_skipped,
        )

        if weight_schema is None:
            weight_schema = BaseWeightSchema.from_config(quant_config)

        if input_schema is None:
            input_quant_config = envs.VLLM_HUMMING_INPUT_QUANT_CONFIG or {}
            if humming_is_layer_skipped(input_quant_config, layer.layer_name):
                input_schema = HummingInputSchema()
            else:
                # TODO: read input_quant_config from quant_config
                input_schema = HummingInputSchema.from_config(input_quant_config)

    # Build sublayer configs from layer properties if not provided
    if sublayer_configs is None:
        is_gated = layer.moe_config.activation.is_gated
        sublayer_configs = {
            "w13": {
                "shape_n": layer.moe_config.intermediate_size_per_partition * 2,
                "shape_k": layer.moe_config.hidden_dim,
            },
            "w2": {
                "shape_n": layer.moe_config.hidden_dim,
                "shape_k": layer.moe_config.intermediate_size_per_partition
                * (1 if is_gated else 2),
            },
        }

    layer.weight_schemas = {}
    layer.input_schemas = {}

    for sublayer_name, configs in sublayer_configs.items():
        final_weight_schema, final_input_schema = _process_single_sublayer(
            layer=layer,
            sublayer_name=sublayer_name,
            shape_n=configs["shape_n"],
            shape_k=configs["shape_k"],
            weight_schema=weight_schema,
            input_schema=input_schema,
            has_bias=has_bias,
            num_experts=num_experts,
            param_dtype=param_dtype,
            force_weight_schema=force_weight_schema,
        )

        layer.weight_schemas[sublayer_name] = final_weight_schema
        layer.input_schemas[sublayer_name] = final_input_schema

    if not hasattr(layer, "locks"):
        device = layer.w13_weight.device
        locks = torch.zeros(1024, dtype=torch.int32, device=device)
        layer.register_buffer("locks", locks)
