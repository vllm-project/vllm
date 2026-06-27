# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum
from typing import Any

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
    int8_w8a8_moe_quant_config,
    int8_w8a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.oracle.base import MoEKernelOracle
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

logger = init_logger(__name__)


class Int8MoeBackend(Enum):
    TRITON = "TRITON"
    HUMMING = "HUMMING"
    CPU = "CPU"


class Int8MoEKernelOracle(MoEKernelOracle[Int8MoeBackend]):
    """Int8 MoE kernel oracle.

    Int8 MoE has three backends (Triton, Humming, CPU); the CPU backend is
    moved to the front of the priority list on CPU platforms. ``select_backend``
    is inherited from ``MoEKernelOracle`` — the generic priority loop handles
    all three. The Humming backend threads ``layer`` through weight conversion,
    quant-config construction, and kernel construction, so ``make_quant_config``,
    ``convert_to_kernel_format`` and ``make_kernel`` are overridden with the
    Int8-specific signatures (delegating to the module-level helpers, which
    keep the backend-branching bodies).
    """

    def backend_enum_cls(self) -> type[Int8MoeBackend]:
        return Int8MoeBackend

    def get_priority_backends(self, moe_config: FusedMoEConfig) -> list[Int8MoeBackend]:
        backends = [
            Int8MoeBackend.TRITON,
            Int8MoeBackend.HUMMING,
            Int8MoeBackend.CPU,
        ]
        if current_platform.is_cpu():
            backends.insert(0, backends.pop(backends.index(Int8MoeBackend.CPU)))
        return backends

    def backend_to_kernel_cls(
        self, backend: Int8MoeBackend
    ) -> list[type[mk.FusedMoEExperts]]:
        if backend == Int8MoeBackend.TRITON:
            from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
                TritonExperts,
            )

            return [TritonExperts]
        if backend == Int8MoeBackend.HUMMING:
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
        if backend == Int8MoeBackend.CPU:
            from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
                CPUExpertsInt8,
            )

            return [CPUExpertsInt8]
        raise ValueError(f"Unknown Int8 MoE backend: {backend.value}")

    def map_backend(self, runner_backend: MoEBackend) -> Int8MoeBackend:
        mapping = {
            "triton": Int8MoeBackend.TRITON,
            "humming": Int8MoeBackend.HUMMING,
        }
        if backend := mapping.get(runner_backend):
            return backend
        raise ValueError(
            f"moe_backend={runner_backend!r} is not supported for Int8 MoE. "
            f"Expected one of {list(mapping.keys())}."
        )

    def select_backend(
        self,
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None = kInt8StaticChannelSym,
        activation_key: QuantKey | None = kInt8DynamicTokenSym,
    ) -> tuple[Int8MoeBackend, type[mk.FusedMoEExperts] | None]:
        return super().select_backend(moe_config, weight_key, activation_key)

    def make_quant_config(  # type: ignore[override]
        self,
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
        return make_int8_moe_quant_config(
            int8_backend,
            w1_scale,
            w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            per_act_token_quant=per_act_token_quant,
            layer=layer,
        )

    def convert_to_kernel_format(  # type: ignore[override]
        self,
        int8_backend: Int8MoeBackend,
        w13: torch.Tensor,
        w2: torch.Tensor,
        layer: torch.nn.Module | None = None,
        w13_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return convert_to_int8_moe_kernel_format(
            int8_backend, w13, w2, layer=layer, w13_scale=w13_scale
        )

    def make_kernel(  # type: ignore[override]
        self,
        int8_backend: Int8MoeBackend,
        moe_quant_config: FusedMoEQuantConfig,
        moe_config: FusedMoEConfig,
        experts_cls: type[mk.FusedMoEExperts],
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        layer: torch.nn.Module | None = None,
    ) -> mk.FusedMoEKernel:
        return make_int8_moe_kernel(
            int8_backend,
            moe_quant_config,
            moe_config,
            experts_cls,
            routing_tables=routing_tables,
            layer=layer,
        )


_ORACLE = Int8MoEKernelOracle()


# ---------------------------------------------------------------------------
# Module-level function wrappers / helpers
#
# The data-declaration functions below delegate to the oracle singleton so the
# public API stays bit-identical while the logic lives on the class. The
# backend-branching helpers (make_quant_config / convert / make_kernel and the
# humming schema) keep their bodies here — the Humming backend threads a
# ``layer`` through them, which does not fit the generic ABC signatures — and
# the oracle methods above delegate to them.
# ---------------------------------------------------------------------------


def _get_priority_backends(moe_config: FusedMoEConfig) -> list[Int8MoeBackend]:
    """Wrapper preserved for backward compat."""
    return _ORACLE.get_priority_backends(moe_config)


def backend_to_kernel_cls(
    backend: Int8MoeBackend,
) -> list[type[mk.FusedMoEExperts]]:
    """Wrapper preserved for backward compat."""
    return _ORACLE.backend_to_kernel_cls(backend)


def map_int8_backend(runner_backend: MoEBackend) -> Int8MoeBackend:
    """Wrapper preserved for backward compat."""
    return _ORACLE.map_backend(runner_backend)


def select_int8_moe_backend(
    config: FusedMoEConfig,
    weight_key: QuantKey | None = kInt8StaticChannelSym,
    activation_key: QuantKey | None = kInt8DynamicTokenSym,
) -> tuple[Int8MoeBackend, type[mk.FusedMoEExperts] | None]:
    """Wrapper preserved for backward compat.

    Note: Shape-specific fallbacks may still occur at runtime.
    """
    return _ORACLE.select_backend(config, weight_key, activation_key)


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
    assert (a1_scale is None and a2_scale is None) or (
        a1_scale is not None and a2_scale is not None
    ), "a1_scale and a2_scale must both be provided or both be None"

    if int8_backend == Int8MoeBackend.HUMMING:
        from vllm.model_executor.layers.fused_moe import RoutedExperts
        from vllm.model_executor.layers.quantization.utils.humming_utils import (
            get_humming_moe_quant_config,
        )

        assert isinstance(layer, RoutedExperts)
        return get_humming_moe_quant_config(layer)

    if a1_scale is None or a2_scale is None:
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
        from vllm.model_executor.layers.quantization.utils.humming_utils import (
            convert_to_humming_moe_kernel_format,
        )

        assert layer is not None
        # Humming reads canonical CT scales (w*_weight_scale) from the layer.
        # Online int8 produces per-channel (E, N) w*_scale; expose them as the
        # (E, N, 1) w*_weight_scale humming's loader expects.
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
    elif int8_backend == Int8MoeBackend.CPU:
        from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
            prepare_int8_moe_layer_for_cpu,
        )

        w13, w2 = prepare_int8_moe_layer_for_cpu(w13, w2)
    elif int8_backend != Int8MoeBackend.TRITON:
        raise ValueError(f"Unsupported Int8 MoE backend: {int8_backend.value}")

    return w13, w2


def make_int8_moe_kernel(
    int8_backend: Int8MoeBackend,
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEExperts],
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    layer: torch.nn.Module | None = None,
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

    extra_kwargs = {}
    if int8_backend == Int8MoeBackend.HUMMING:
        assert layer is not None
        extra_kwargs = {"layer": layer}

    # Create Experts.
    if prepare_finalize.activation_format == mk.FusedMoEActivationFormat.BatchedExperts:
        max_num_tokens = prepare_finalize.max_num_tokens_per_rank()
        assert max_num_tokens is not None
        experts = experts_cls(
            moe_config=moe_config,
            quant_config=moe_quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=prepare_finalize.num_dispatchers(),
            **extra_kwargs,
        )
    else:
        experts = experts_cls(
            moe_config=moe_config,
            quant_config=moe_quant_config,
            **extra_kwargs,
        )

    kernel = mk.FusedMoEKernel(
        prepare_finalize,
        experts,
    )

    return kernel
