# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum

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
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


class W4A8Int8MoeBackend(Enum):
    CPU_INT4 = "CPU_INT4"


def _get_priority_backends(
    moe_config: FusedMoEConfig,
) -> list[W4A8Int8MoeBackend]:
    """
    Get available backends in priority order based on platform and config.

    Currently only CPU INT4 backend is available for W4A8 INT8 MoE.
    """
    if current_platform.is_cpu():
        return [W4A8Int8MoeBackend.CPU_INT4]
    return []


def backend_to_kernel_cls(
    backend: W4A8Int8MoeBackend,
) -> list[type[mk.FusedMoEExperts]]:
    """Map W4A8Int8MoeBackend to kernel class."""
    if backend == W4A8Int8MoeBackend.CPU_INT4:
        from vllm.model_executor.layers.fused_moe.experts.cpu_int4_moe import (
            CPUExpertsInt4,
        )

        return [CPUExpertsInt4]

    else:
        raise ValueError(f"Unknown W4A8 Int8 MoE backend: {backend.value}")


def map_w4a8_int8_backend(runner_backend: MoEBackend) -> W4A8Int8MoeBackend:
    """Map user's MoEBackend to W4A8Int8MoeBackend."""
    mapping = {
        "cpu": W4A8Int8MoeBackend.CPU_INT4,
    }
    if backend := mapping.get(runner_backend):
        return backend
    raise ValueError(
        f"moe_backend='{runner_backend}' is not supported for W4A8 Int8 MoE. "
        f"Expected one of {list(mapping.keys())}."
    )


def select_w4a8_int8_moe_backend(
    config: FusedMoEConfig,
    weight_key: QuantKey | None,
    activation_key: QuantKey | None,
) -> tuple[W4A8Int8MoeBackend, type[mk.FusedMoEExperts]]:
    """
    Select the primary W4A8 Int8 MoE backend.

    Args:
        config: MoE configuration
        weight_key: Weight quantization key (should be one of kInt4W4A8Static*)
        activation_key: Activation quantization key (currently unused for W4A8)

    Returns:
        Tuple of (backend, kernel_class)
    """

    AVAILABLE_BACKENDS = _get_priority_backends(config)

    if not AVAILABLE_BACKENDS:
        raise NotImplementedError("W4A8 Int8 MoE is only supported on CPU platforms")

    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )

    def _make_log_backend(backend: W4A8Int8MoeBackend) -> str:
        available_backend_strs = [b.value for b in AVAILABLE_BACKENDS]
        return (
            f"Using {backend.value} W4A8 Int8 MoE backend out "
            f"of potential backends: {available_backend_strs}."
        )

    def _make_log_unsupported(backend: W4A8Int8MoeBackend, reason: str | None) -> str:
        if reason:
            return (
                f"W4A8 Int8 MoE backend {backend.value} does not support the "
                f"deployment configuration since {reason}."
            )
        else:
            return (
                f"W4A8 Int8 MoE backend '{backend.value}' does not support the "
                "deployment configuration."
            )

    def _return_or_raise(
        backend: W4A8Int8MoeBackend,
    ) -> tuple[W4A8Int8MoeBackend, type[mk.FusedMoEExperts]]:
        reason = None
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
        requested_backend = map_w4a8_int8_backend(runner_backend)
        return _return_or_raise(requested_backend)

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

    raise NotImplementedError(
        "No W4A8 Int8 MoE backend supports the deployment configuration."
    )


def make_w4a8_int8_moe_quant_config(
    block_shape: tuple[int, int] | None = None,
) -> FusedMoEQuantConfig:
    """
    Create FusedMoEQuantConfig for W4A8 Int8 MoE.

    Args:
        block_shape: Quantization block shape (row, col).
                    For channel-wise: (-1, 1) or None
                    For group-wise: (1, group_size)

    Returns:
        FusedMoEQuantConfig with appropriate settings for W4A8 Int8
    """
    # W4A8 Int8 uses static weight quantization, dynamic activation quantization
    # Weights are 4-bit (stored as int8, packed to uint8),
    # activations are dynamically quantized to 8-bit in kernel

    group_shape = GroupShape(*block_shape) if block_shape is not None else None

    return FusedMoEQuantConfig(
        # Activations: unquantized (FP/BF16), dynamically quantized in kernel
        _a1=FusedMoEQuantDesc(shape=group_shape),
        _a2=FusedMoEQuantDesc(shape=group_shape),
        # Weights: INT8 (4-bit values), pre-packed with scales
        # dtype=None means already quantized/packed
        _w1=FusedMoEQuantDesc(dtype=None, shape=group_shape),
        _w2=FusedMoEQuantDesc(dtype=None, shape=group_shape),
    )


def pack_int4_weights_for_kleidi(
    int4_as_int8: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None,
    in_features: int,
    out_features: int,
    group_size: int,
) -> torch.Tensor:
    """
    Pack INT4 weights (stored as int8 in [-8,7]) to KleidiAI format.

    Args:
        int4_as_int8: [out, in] int8 tensor with values in [-8, 7]
        scales: [out, in//group_size] or [out, 1] for channel-wise
        bias: [out] optional bias
        in_features: Input dimension
        out_features: Output dimension
        group_size: Quantization group size (-1 for channel-wise)

    Returns:
        Packed weight tensor in KleidiAI format
    """
    # Shift to unsigned nibble [0, 15]
    tmp = int4_as_int8.add(8)
    # Pack pairs along input dimension
    uint8_nibbles = ((tmp[:, 1::2] << 4) | tmp[:, ::2]).to(torch.uint8)

    # Determine scale dtype based on group_size
    # KleidiAI groupwise kernels accept bfloat16 scales
    # KleidiAI channelwise kernels accept float32 scales
    scale_dtype = torch.float32 if group_size == -1 else torch.bfloat16
    scales_typed = scales.to(scale_dtype)
    bias_typed = None if bias is None else bias.to(torch.float32)

    # Pack using KleidiAI op
    actual_group_size = in_features if group_size == -1 else group_size
    return torch.ops.aten._dyn_quant_pack_4bit_weight(
        uint8_nibbles,
        scales_typed,
        bias_typed,
        actual_group_size,
        in_features,
        out_features,
    )


def convert_to_w4a8_int8_moe_format(
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    group_size: int,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """
    Pack INT4 MoE weights to KleidiAI format.

    This function packs the INT4 weights (stored as int8 values) into
    the format expected by the KleidiAI dynamic_4bit_int_moe kernel.

    Args:
        w13_weight: [E, 2*IN, H] int8 tensor (int4 values in [-8,7])
        w2_weight: [E, H, IN] int8 tensor (int4 values in [-8,7])
        w13_weight_scale: [E, 2*IN, H/g or 1] scale tensor
        w2_weight_scale: [E, H, IN/g or 1] scale tensor
        group_size: Quantization group size (-1 for channel-wise)
        w13_bias: Optional [E, 2*IN] bias tensor
        w2_bias: Optional [E, H] bias tensor

    Returns:
        Tuple of (w13_packed, w2_packed) tensors
    """
    # Derive dimensions from tensor shapes
    E = w13_weight.shape[0]  # num_experts
    I2 = w13_weight.shape[1]  # w13_out_features (2*IN)
    H = w13_weight.shape[2]  # w13_in_features (hidden_size)
    IN = w2_weight.shape[2]  # w2_in_features (intermediate_size)
    w2_out_features = w2_weight.shape[1]  # Should equal H

    # Pack per expert
    w13_packed_list = []
    w2_packed_list = []

    for e in range(E):
        w13_packed_list.append(
            pack_int4_weights_for_kleidi(
                w13_weight[e],  # [2I, H]
                w13_weight_scale[e],  # [2I, H/g or 1]
                w13_bias[e] if w13_bias is not None else None,  # [2I]
                H,
                I2,
                group_size,
            )
        )
        w2_packed_list.append(
            pack_int4_weights_for_kleidi(
                w2_weight[e],  # [H, IN]
                w2_weight_scale[e],  # [H, IN/g or 1]
                w2_bias[e] if w2_bias is not None else None,  # [H]
                IN,
                w2_out_features,  # in_features=IN, out_features=H
                group_size,
            )
        )

    # Stack all experts
    w13_packed = torch.stack(w13_packed_list, dim=0)
    w2_packed = torch.stack(w2_packed_list, dim=0)
    empty = torch.empty(0)

    return w13_packed, w2_packed, empty, empty, empty, empty


def make_w4a8_int8_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEExperts],
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> mk.FusedMoEKernel:
    """
    Create FusedMoEKernel for W4A8 Int8 MoE.

    Args:
        moe_quant_config: Quantization configuration
        moe_config: MoE configuration
        experts_cls: Expert kernel class (should be CPUExpertsInt4)
        routing_tables: Optional routing tables for expert parallelism

    Returns:
        Configured FusedMoEKernel instance
    """
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
    # W4A8 Int8 currently only supports monolithic interface
    if not issubclass(experts_cls, mk.FusedMoEExpertsMonolithic):
        raise ValueError(
            f"W4A8 Int8 MoE only supports monolithic experts, "
            f"but got {experts_cls.__name__}"
        )

    experts = experts_cls(
        moe_config=moe_config,
        quant_config=moe_quant_config,
    )

    kernel = mk.FusedMoEKernel(
        prepare_finalize,
        experts,
        inplace=not moe_config.disable_inplace,
    )

    return kernel
