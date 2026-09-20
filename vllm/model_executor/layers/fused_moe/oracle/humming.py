# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert selection, weight conversion, and kernel construction for Humming MoE."""

from typing import TYPE_CHECKING, Any

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
)
from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
    BatchedHummingGroupedExperts,
    HummingGroupedExperts,
    HummingIndexedExperts,
)
from vllm.model_executor.layers.quantization.utils import humming_utils
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    FP8_DTYPE,
    GroupShape,
)
from vllm.utils.import_utils import has_humming

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.utils.humming import (
        BaseInputSchema,
        BaseWeightSchema,
        HummingInputSchema,
        HummingWeightSchema,
        LayerConfig,
    )

logger = init_logger(__name__)


def get_humming_moe_quant_config(
    layer: "RoutedExperts",
    humming_configs: dict[str, "LayerConfig"] | None = None,
    gemm1_alpha: float | None = None,
    gemm1_beta: float | None = None,
    gemm1_clamp_limit: float | None = None,
):
    if humming_configs is None:
        humming_configs = layer.humming_configs
    input_schema = layer.input_schemas["w13"]
    weight_schema = layer.weight_schemas["w13"]

    if input_schema.a_dtype is None or input_schema.a_dtype.num_bits == 16:
        q_dtype = None
    else:
        q_dtype = str(input_schema.a_dtype)

    # Block-FP8 (group-128) activations are quantized *before* the EP all-to-all
    # dispatch (so FP8 rather than BF16 crosses the interconnect) and consumed by
    # Humming as-is.
    activation_group_shape: GroupShape | None = None
    input_scale_group_size = getattr(input_schema, "input_scale_group_size", 0) or 0
    if (
        q_dtype is not None
        and q_dtype.startswith("float8")
        and input_scale_group_size == 128
    ):
        q_dtype = humming_utils._HUMMING_TO_QUANT_DTYPE.get(
            input_schema.a_dtype, FP8_DTYPE
        )
        activation_group_shape = GroupShape(row=1, col=input_scale_group_size)

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

    return humming_utils.make_humming_moe_quant_config(
        quant_dtype=q_dtype,
        weight_dtype=str(weight_schema.b_dtype),
        weight_group_shape=weight_group_shape,
        activation_group_shape=activation_group_shape,
        w1_scale=getattr(layer, "w13_weight_scale", None),
        w1_gscale=getattr(layer, "w13_weight_scale_2", None),
        w1_zp=getattr(layer, "w13_zero_point", None),
        w1_bias=getattr(layer, "w13_bias", None),
        w2_scale=getattr(layer, "w2_weight_scale", None),
        w2_gscale=getattr(layer, "w2_weight_scale_2", None),
        w2_zp=getattr(layer, "w2_zero_point", None),
        w2_bias=getattr(layer, "w2_bias", None),
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
        humming_configs=humming_configs,
    )


def select_humming_moe_experts(
    config: FusedMoEConfig,
    weight_schema: "BaseWeightSchema",
    input_schema: "BaseInputSchema",
    force_weight_schema: "HummingWeightSchema | None" = None,
    force_input_schema: "HummingInputSchema | None" = None,
) -> type[mk.FusedMoEExperts] | None:
    """Select the primary Humming MoE Experts class
    Note: Shape-specific fallbacks may still occur at runtime.
    """
    if not has_humming():
        return None

    # Select for the final schemas, including any requested requantization.
    weight_key = humming_utils.weight_schema_to_quant_key(
        force_weight_schema or weight_schema
    )
    activation_key = humming_utils.input_schema_to_quant_key(
        force_input_schema or input_schema
    )

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

    # Create Experts.
    if prepare_finalize.activation_format == mk.FusedMoEActivationFormat.BatchedExperts:
        max_num_tokens = prepare_finalize.max_num_tokens_per_rank()
        assert max_num_tokens is not None
        experts = experts_cls(
            moe_config=moe_config,
            quant_config=moe_quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=prepare_finalize.num_dispatchers(),
        )
    else:
        experts = experts_cls(
            moe_config=moe_config,
            quant_config=moe_quant_config,
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
    """Replace layer parameters for a sublayer with new tensors.

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
    """Convert a sublayer's weights from checkpoint format to Humming format.

    Returns:
        Tuple of (converted_weight_schema, converted_input_schema)

    """
    from vllm.utils.humming import HummingWeightSchema

    if isinstance(weight_schema, HummingWeightSchema):
        # Already in Humming format
        return weight_schema, input_schema

    tensors = _extract_sublayer_tensors(layer, sublayer_name)

    shape_k_stacks = [shape_k]
    shape_n_stacks = [shape_n]
    if sublayer_name == "w13" and layer.moe_config.activation.is_gated:
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
) -> "LayerConfig":
    """Prepare Humming configuration and transform one sublayer's tensors."""
    from vllm.utils.humming import (
        prepare_layer_config,
        transform_humming_tensors,
    )

    config = prepare_layer_config(
        shape_n=shape_n,
        shape_k=shape_k,
        pad_n_to_multiple=256,
        pad_k_to_multiple=128,
        input_schema=input_schema,
        weight_schema=weight_schema,
        has_bias=has_bias,
        num_experts=num_experts,
        torch_dtype=param_dtype,
    )
    tensors = transform_humming_tensors(
        config,
        _extract_sublayer_tensors(layer, sublayer_name),
    )
    _replace_layer_parameters(layer, sublayer_name, tensors)
    return config


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
) -> tuple[Any, Any, "LayerConfig"]:
    """Process a single sublayer: convert, optionally requant, prepare, and transform.

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
        Tuple of the final weight schema, input schema, and Humming layer config.

    """
    from vllm.utils.humming import HummingWeightSchema

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
    config = _prepare_and_transform_sublayer(
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

    return current_weight_schema, current_input_schema, config


def convert_to_humming_moe_kernel_format(
    layer: "RoutedExperts",
    quant_config: dict | None = None,
    sublayer_configs: dict[str, Any] | None = None,
    weight_schema: Any | None = None,
    input_schema: Any | None = None,
    force_weight_schema: Any | None = None,
) -> dict[str, "LayerConfig"]:
    """Convert MoE weights from checkpoint format to Humming kernel format.

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
        - Sets layer.humming_configs for quant config construction

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

        from vllm.model_executor.layers.quantization.utils.humming_utils import (
            humming_is_layer_skipped,
        )
        from vllm.utils.humming import BaseWeightSchema, HummingInputSchema

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
        intermediate_size = layer.moe_config.intermediate_size_per_partition
        sublayer_configs = {
            "w13": {
                "shape_n": intermediate_size * (2 if is_gated else 1),
                "shape_k": layer.moe_config.hidden_dim,
            },
            "w2": {
                "shape_n": layer.moe_config.hidden_dim,
                "shape_k": intermediate_size,
            },
        }

    layer.weight_schemas = {}
    layer.input_schemas = {}
    humming_configs = {}

    for sublayer_name, configs in sublayer_configs.items():
        final_weight_schema, final_input_schema, humming_config = (
            _process_single_sublayer(
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
        )

        layer.weight_schemas[sublayer_name] = final_weight_schema
        layer.input_schemas[sublayer_name] = final_input_schema
        humming_configs[sublayer_name] = humming_config

    layer.humming_configs = humming_configs
    return humming_configs
