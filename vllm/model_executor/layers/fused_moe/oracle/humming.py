# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config.kernel import MoEBackend
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
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
)
from vllm.utils.import_utils import has_humming

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe import RoutedExperts

logger = init_logger(__name__)


class HummingBackend(Enum):
    NONE = "NONE"
    HUMMING = "HUMMING"


def _get_priority_backends(
    moe_config: FusedMoEConfig,
    weight_key: QuantKey | None,
    activation_key: QuantKey | None,
) -> list[HummingBackend]:
    """
    Get available backends in priority order based on platform and config.

    This function can be extended to become more complex as needed.
    """
    if has_humming():
        return [HummingBackend.HUMMING]
    else:
        return []


def backend_to_kernel_cls(
    backend: HummingBackend,
) -> list[type[mk.FusedMoEExperts]]:
    if backend == HummingBackend.HUMMING:
        return [
            BatchedHummingGroupedExperts,
            HummingGroupedExperts,
            HummingIndexedExperts,
        ]
    return []


def map_humming_backend(runner_backend: MoEBackend) -> HummingBackend:
    """Map user's MoEBackend to HummingBackend."""
    mapping = {
        "humming": HummingBackend.HUMMING,
    }
    if backend := mapping.get(runner_backend):
        return backend
    raise ValueError(
        f"moe_backend='{runner_backend}' is not supported for FP8 MoE. "
        f"Expected one of {list(mapping.keys())}."
    )


def select_humming_moe_backend(
    config: FusedMoEConfig,
    weight_key: QuantKey | None,
    activation_key: QuantKey | None,
) -> tuple[HummingBackend, type[mk.FusedMoEExperts] | None]:
    """
    Select the primary FP8 MoE backend
    Note: Shape-specific fallbacks may still occur at runtime.
    """

    # NOTE: the kernels are selected in the following order.
    AVAILABLE_BACKENDS = _get_priority_backends(config, weight_key, activation_key)

    # NOTE(rob): We need to peak into the P/F selection to determine
    # if we are using the batched or standard expert format, which
    # if not ideal. Once we unify TP + DP/EP, we can select P/F first.
    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )

    def _make_log_backend(backend: HummingBackend):
        available_backend_strs = [b.value for b in AVAILABLE_BACKENDS]
        return (
            f"Using {backend.value} Humming MoE backend out "
            f"of potential backends: {available_backend_strs}."
        )

    def _make_log_unsupported(backend: HummingBackend, reason: str | None) -> str:
        if reason:
            return (
                f"Humming MoE backend {backend.value} does not support the "
                f"deployment configuration since {reason}."
            )
        else:
            return (
                f"Humming MoE backend '{backend.value}' does not support the "
                "deployment configuration."
            )

    def _return_or_raise(
        backend: HummingBackend,
        config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[HummingBackend, type[mk.FusedMoEExperts]]:
        for k_cls in backend_to_kernel_cls(backend):
            supported, reason = k_cls.is_supported_config(
                k_cls, config, weight_key, activation_key, activation_format
            )
            if supported:
                logger.info_once(_make_log_backend(backend))
                return backend, k_cls
        raise ValueError(_make_log_unsupported(backend, reason))

    # Handle explicit moe_backend from user.
    runner_backend = config.moe_backend
    if runner_backend != "auto":
        requested_backend = map_humming_backend(runner_backend)
        return _return_or_raise(
            requested_backend, config, weight_key, activation_key, activation_format
        )

    # Select kernels in order of backend.
    for backend in AVAILABLE_BACKENDS:
        for k_cls in backend_to_kernel_cls(backend):
            supported, reason = k_cls.is_supported_config(
                k_cls,
                config,
                weight_key,
                activation_key,
                activation_format,
            )
            if supported:
                logger.info_once(_make_log_backend(backend))
                return backend, k_cls
            else:
                logger.debug_once(_make_log_unsupported(backend, reason))

    return HummingBackend.NONE, None


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
    and prepare_humming_moe_layer for processing a single sublayer.

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
    num_experts = layer.moe_config.num_experts
    param_dtype = layer.param_dtype

    if weight_schema is None or input_schema is None:
        if quant_config is None:
            raise ValueError(
                "Must provide either weight_schema/input_schema or quant_config"
            )

        from humming.layer import HummingInputSchema
        from humming.schema import BaseWeightSchema

        from vllm import envs
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


def make_humming_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEExperts],
    backend: HummingBackend,
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
        inplace=not moe_config.disable_inplace,
    )

    return kernel
