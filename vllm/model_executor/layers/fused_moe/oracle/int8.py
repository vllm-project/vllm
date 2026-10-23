# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum
from typing import Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config.kernel import MoEBackend
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    int8_w8a8_moe_quant_config,
    int8_w8a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.oracle.base import (
    MoEKernelOracle,
    make_moe_kernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
)
from vllm.model_executor.utils import replace_parameter

logger = init_logger(__name__)


class Int8MoeBackend(Enum):
    TRITON = "TRITON"
    HUMMING = "HUMMING"
    CPU = "CPU"


def _get_priority_backends(
    moe_config: FusedMoEConfig,
) -> list[Int8MoeBackend]:
    """Get available backends in priority order based on platform and config."""
    return [
        Int8MoeBackend.TRITON,
        Int8MoeBackend.HUMMING,
        Int8MoeBackend.CPU,
    ]


def backend_to_kernel_cls(
    backend: Int8MoeBackend,
) -> list[type[mk.FusedMoEExperts]]:
    if backend == Int8MoeBackend.TRITON:
        from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
            TritonExperts,
        )

        return [TritonExperts]

    elif backend == Int8MoeBackend.HUMMING:
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
    elif backend == Int8MoeBackend.CPU:
        from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
            ArmCPUExpertsInt8,
            CPUExpertsInt8,
            PowerCPUExpertsInt8,
            ZenCPUExpertsInt8,
        )

        return [
            ZenCPUExpertsInt8,
            PowerCPUExpertsInt8,
            ArmCPUExpertsInt8,
            CPUExpertsInt8,
        ]
    else:
        raise ValueError(f"Unknown Int8 MoE backend: {backend.value}")


def map_int8_backend(runner_backend: MoEBackend) -> Int8MoeBackend:
    """Map user's MoEBackend to Int8MoeBackend."""
    mapping = {
        "triton": Int8MoeBackend.TRITON,
        "humming": Int8MoeBackend.HUMMING,
    }
    if backend := mapping.get(runner_backend):
        return backend
    raise ValueError(
        f"moe_backend='{runner_backend}' is not supported for Int8 MoE. "
        f"Expected one of {list(mapping.keys())}."
    )


class Int8MoEKernelOracle(MoEKernelOracle[Int8MoeBackend]):
    """INT8 MoE backend selection for one deployment configuration."""

    name = "Int8"
    backend_to_kernel_cls = staticmethod(backend_to_kernel_cls)


def select_int8_moe_backend(
    config: FusedMoEConfig,
    weight_key: QuantKey | None = kInt8StaticChannelSym,
    activation_key: QuantKey | None = kInt8DynamicTokenSym,
) -> tuple[Int8MoeBackend, type[mk.FusedMoEExperts]]:
    """Select the primary Int8 MoE backend.
    Note: Shape-specific fallbacks may still occur at runtime.
    """
    AVAILABLE_BACKENDS = _get_priority_backends(config)

    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )
    oracle = Int8MoEKernelOracle(config, weight_key, activation_key, activation_format)

    # Handle explicit moe_backend from user.
    runner_backend = config.moe_backend
    if runner_backend != "auto":
        requested_backend = map_int8_backend(runner_backend)
        return oracle.return_or_raise(requested_backend, AVAILABLE_BACKENDS)

    # Select kernels in order of backend.
    for backend in AVAILABLE_BACKENDS:
        kernel_cls, reason = oracle.try_backend(backend)
        if kernel_cls is not None:
            logger.info_once(oracle.make_log_backend(backend, AVAILABLE_BACKENDS))
            return backend, kernel_cls
        logger.debug_once(oracle.make_log_unsupported(backend, reason))

    raise NotImplementedError(
        "No Int8 MoE backend supports the deployment configuration "
        f"(weight_key={weight_key}, activation_key={activation_key}). "
        "Set `VLLM_LOGGING_LEVEL=DEBUG` to see per-backend unsupported reasons."
    )


def make_int8_moe_quant_config(
    int8_backend: Int8MoeBackend,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    per_act_token_quant: bool = False,
    layer: torch.nn.Module | None = None,
) -> FusedMoEQuantConfig:
    if (a1_scale is None) != (a2_scale is None):
        raise ValueError("a1_scale and a2_scale must both be provided or both be None")

    scales_absent = a1_scale is None and a2_scale is None

    if int8_backend == Int8MoeBackend.HUMMING:
        from vllm.model_executor.layers.fused_moe import RoutedExperts
        from vllm.model_executor.layers.quantization.utils.humming import (
            get_humming_moe_quant_config,
        )

        assert isinstance(layer, RoutedExperts)
        return get_humming_moe_quant_config(
            layer,
            gemm1_alpha=getattr(layer, "swiglu_alpha", None),
            gemm1_beta=getattr(layer, "swiglu_beta", None),
            gemm1_clamp_limit=getattr(layer, "swiglu_limit", None),
        )

    if scales_absent and not per_act_token_quant:
        return int8_w8a16_moe_quant_config(
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_zp=None,
            w2_zp=None,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
        )

    return int8_w8a8_moe_quant_config(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        per_act_token_quant=per_act_token_quant,
    )


def _humming_int8_weight_schema(
    weight: torch.Tensor, weight_scale: torch.Tensor
) -> dict[str, Any]:
    """Build the humming compressed-tensors int8 schema from the canonical
    on-device tensors; humming does the signed-int8 -> native conversion."""
    config: dict[str, Any] = {
        "quant_method": "compressed-tensors",
        "format": "int-quantized",
        "type": "int",
        "num_bits": 8,
        "symmetric": True,
        "strategy": "channel",
    }
    num_experts, num_output = weight.shape[0], weight.shape[-2]
    if weight_scale.numel() < num_experts * num_output:
        config["strategy"] = "tensor"
    return config


def convert_to_int8_moe_kernel_format(
    int8_backend: Int8MoeBackend,
    w13: torch.Tensor,
    w2: torch.Tensor,
    layer: torch.nn.Module | None = None,
    w13_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert INT8 MoE weights to backend-specific kernel format."""
    if int8_backend == Int8MoeBackend.HUMMING:
        from vllm.model_executor.layers.quantization.utils.humming import (
            convert_to_humming_moe_kernel_format,
        )

        assert layer is not None
        # The Humming conversion expects canonical CT scale parameter names.
        # Online int8 produces per-channel (E, N) w*_scale; expose them as
        # (E, N, 1) w*_weight_scale tensors before conversion.
        for sub in ("w13", "w2"):
            if hasattr(layer, f"{sub}_weight_scale"):
                continue
            scale = getattr(layer, f"{sub}_scale").data
            if scale.dim() < 3:
                scale = scale.unsqueeze(-1)
            replace_parameter(layer, f"{sub}_weight_scale", scale)
            delattr(layer, f"{sub}_scale")
        convert_to_humming_moe_kernel_format(
            layer,
            quant_config=_humming_int8_weight_schema(w13, layer.w13_weight_scale),
        )
        return layer.w13_weight, layer.w2_weight
    elif int8_backend not in (Int8MoeBackend.TRITON, Int8MoeBackend.CPU):
        raise ValueError(f"Unsupported Int8 MoE backend: {int8_backend.value}")

    return w13, w2


def make_int8_moe_kernel(
    int8_backend: Int8MoeBackend,
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEExperts],
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> mk.FusedMoEKernel:
    return make_moe_kernel(moe_quant_config, moe_config, experts_cls, routing_tables)
