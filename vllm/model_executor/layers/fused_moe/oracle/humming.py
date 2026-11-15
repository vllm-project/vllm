# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum
from typing import TYPE_CHECKING, Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config.kernel import MoEBackend
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.oracle.base import (
    MoEKernelOracle,
    make_moe_kernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.utils.humming import LayerConfig

logger = init_logger(__name__)


class HummingMoeBackend(Enum):
    HUMMING = "HUMMING"
    # No compute backend: requantization sources only dequantize weights.
    NONE = "NONE"


def _get_priority_backends(
    moe_config: FusedMoEConfig,
) -> list[HummingMoeBackend]:
    """Get available backends in priority order based on platform and config."""
    return [HummingMoeBackend.HUMMING]


def backend_to_kernel_cls(
    backend: HummingMoeBackend,
) -> list[type[mk.FusedMoEExperts]]:
    if backend == HummingMoeBackend.HUMMING:
        from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
            BatchedHummingGroupedExperts,
            HummingGroupedExperts,
            HummingIndexedExperts,
        )

        return [
            BatchedHummingGroupedExperts,
            HummingGroupedExperts,
            HummingIndexedExperts,
        ]
    else:
        raise ValueError(f"Unknown Humming MoE backend: {backend.value}")


def map_humming_backend(runner_backend: MoEBackend) -> HummingMoeBackend:
    """Map user's MoEBackend to HummingMoeBackend."""
    mapping = {
        "humming": HummingMoeBackend.HUMMING,
    }
    if backend := mapping.get(runner_backend):
        return backend
    raise ValueError(
        f"moe_backend='{runner_backend}' is not supported for Humming MoE. "
        f"Expected one of {list(mapping.keys())}."
    )


class HummingMoEKernelOracle(MoEKernelOracle[HummingMoeBackend]):
    """Humming MoE backend selection for one deployment configuration."""

    name = "Humming"
    placeholder_backend = HummingMoeBackend.NONE

    def backend_enum_cls(self) -> type[HummingMoeBackend]:
        return HummingMoeBackend

    def get_priority_backends(
        self, moe_config: FusedMoEConfig
    ) -> list[HummingMoeBackend]:
        return _get_priority_backends(moe_config)

    def backend_to_kernel_cls(
        self, backend: HummingMoeBackend
    ) -> list[type[mk.FusedMoEExperts]]:
        return backend_to_kernel_cls(backend)

    def map_backend(self, runner_backend: MoEBackend) -> HummingMoeBackend:
        return map_humming_backend(runner_backend)

    def select_backend(
        self,
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None = None,
        activation_key: QuantKey | None = None,
    ) -> tuple[HummingMoeBackend, type[mk.FusedMoEExperts]]:
        return select_humming_moe_backend(
            moe_config, weight_key=weight_key, activation_key=activation_key
        )

    def make_kernel(
        self,
        quant_config: FusedMoEQuantConfig,
        moe_config: FusedMoEConfig,
        backend: HummingMoeBackend,
        experts_cls: type[mk.FusedMoEExperts],
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> mk.FusedMoEKernel:
        return make_humming_moe_kernel(
            quant_config, moe_config, experts_cls, routing_tables
        )

    def convert_to_kernel_format(
        self,
        backend: HummingMoeBackend,
        moe_config: FusedMoEConfig,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        layer: torch.nn.Module | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if backend == HummingMoeBackend.HUMMING and layer is not None:
            convert_to_humming_moe_kernel_format(layer=layer, **kwargs)
            return (
                getattr(layer, "w13_weight", w13_weight),
                getattr(layer, "w2_weight", w2_weight),
            )
        return w13_weight, w2_weight

    def make_quant_config(
        self,
        layer: "RoutedExperts",
        humming_configs: dict[str, Any] | None = None,
        gemm1_alpha: float | None = None,
        gemm1_beta: float | None = None,
        gemm1_clamp_limit: float | None = None,
    ) -> FusedMoEQuantConfig:
        return get_humming_moe_quant_config(
            layer=layer,
            humming_configs=humming_configs,
            gemm1_alpha=gemm1_alpha,
            gemm1_beta=gemm1_beta,
            gemm1_clamp_limit=gemm1_clamp_limit,
        )


def select_humming_moe_backend(
    config: FusedMoEConfig,
    weight_key: QuantKey | None = None,
    activation_key: QuantKey | None = None,
) -> tuple[HummingMoeBackend, type[mk.FusedMoEExperts]]:
    """Select the primary Humming MoE backend.
    Note: Shape-specific fallbacks may still occur at runtime.
    Returns `NONE` and a placeholder experts class for requantization sources.
    """
    source = HummingMoEKernelOracle.maybe_skip_for_requantization_source()
    if source is not None:
        return source

    AVAILABLE_BACKENDS = _get_priority_backends(config)

    oracle = HummingMoEKernelOracle(config, weight_key, activation_key)

    # Handle explicit moe_backend from user.
    runner_backend = config.moe_backend
    if runner_backend != "auto":
        requested_backend = map_humming_backend(runner_backend)
        return oracle.return_or_raise(requested_backend, AVAILABLE_BACKENDS)

    # Select kernels in order of backend.
    for backend in AVAILABLE_BACKENDS:
        kernel_cls, _ = oracle.try_backend(backend)
        if kernel_cls is not None:
            logger.info_once(oracle.make_log_backend(backend, AVAILABLE_BACKENDS))
            return backend, kernel_cls

    raise NotImplementedError(
        "No Humming MoE backend supports the deployment configuration "
        f"(weight_key={weight_key}, activation_key={activation_key}). "
        "Set `VLLM_LOGGING_LEVEL=DEBUG` to see per-backend unsupported reasons."
    )


def make_humming_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEExperts],
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> mk.FusedMoEKernel:
    return make_moe_kernel(moe_quant_config, moe_config, experts_cls, routing_tables)


def convert_to_humming_moe_kernel_format(
    layer: "RoutedExperts",
    quant_config: dict | None = None,
    sublayer_configs: dict[str, Any] | None = None,
    weight_schema: Any | None = None,
    input_schema: Any | None = None,
    force_weight_schema: Any | None = None,
    allow_input_schema_fallback: bool = True,
) -> dict[str, "LayerConfig"]:
    from vllm.model_executor.layers.quantization.utils.humming.moe import (
        convert_to_humming_moe_kernel_format as _convert_format,
    )

    return _convert_format(
        layer=layer,
        quant_config=quant_config,
        sublayer_configs=sublayer_configs,
        weight_schema=weight_schema,
        input_schema=input_schema,
        force_weight_schema=force_weight_schema,
        allow_input_schema_fallback=allow_input_schema_fallback,
    )


def get_humming_moe_quant_config(
    layer: "RoutedExperts",
    humming_configs: dict[str, "LayerConfig"] | None = None,
    gemm1_alpha: float | None = None,
    gemm1_beta: float | None = None,
    gemm1_clamp_limit: float | None = None,
):
    from vllm.model_executor.layers.quantization.utils.humming.moe import (
        get_humming_moe_quant_config as _get_config,
    )

    return _get_config(
        layer=layer,
        humming_configs=humming_configs,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
    )
