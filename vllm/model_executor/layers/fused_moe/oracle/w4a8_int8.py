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
    weight_key: QuantKey | None = None,
    activation_key: QuantKey | None = None,
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
