# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config.kernel import MoEBackend
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
from vllm.platforms import current_platform


class Int8MoeBackend(Enum):
    TRITON = "TRITON"
    CPU = "CPU"


class Int8MoEKernelOracle(MoEKernelOracle[Int8MoeBackend]):
    """Int8 MoE kernel oracle.

    Int8 MoE has two backends (Triton and CPU). The only platform quirk
    is that the CPU backend is moved to the front of the priority list on
    CPU platforms; otherwise selection follows the generic priority loop,
    so ``select_backend`` and ``make_kernel`` are inherited from
    ``MoEKernelOracle``. ``make_quant_config`` and
    ``convert_to_kernel_format`` are overridden with the Int8-specific
    behaviour.
    """

    def backend_enum_cls(self) -> type[Int8MoeBackend]:
        return Int8MoeBackend

    def get_priority_backends(self, moe_config: FusedMoEConfig) -> list[Int8MoeBackend]:
        backends = [Int8MoeBackend.TRITON, Int8MoeBackend.CPU]
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
        if backend == Int8MoeBackend.CPU:
            from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
                CPUExpertsInt8,
            )

            return [CPUExpertsInt8]
        raise ValueError(f"Unknown Int8 MoE backend: {backend.value}")

    def map_backend(self, runner_backend: MoEBackend) -> Int8MoeBackend:
        mapping = {"triton": Int8MoeBackend.TRITON}
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
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        a1_scale: torch.Tensor | None = None,
        a2_scale: torch.Tensor | None = None,
        w1_bias: torch.Tensor | None = None,
        w2_bias: torch.Tensor | None = None,
        per_act_token_quant: bool = False,
    ) -> FusedMoEQuantConfig:
        return make_int8_moe_quant_config(
            w1_scale,
            w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            per_act_token_quant=per_act_token_quant,
        )

    def convert_to_kernel_format(
        self,
        backend: Int8MoeBackend,
        moe_config: FusedMoEConfig,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return convert_to_int8_moe_kernel_format(backend, w13_weight, w2_weight)


_ORACLE = Int8MoEKernelOracle()


# ---------------------------------------------------------------------------
# Module-level function wrappers
#
# These preserve the existing public API used throughout the codebase
# (e.g. by ``compressed_tensors_moe_int8``). Their bodies delegate to the
# oracle singleton above so the behaviour is bit-identical with the
# pre-refactor code path. The plan in the #37753 series is to migrate
# callers off the module-level functions over time, after which these
# wrappers can be removed.
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
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    per_act_token_quant: bool = False,
) -> FusedMoEQuantConfig:
    """Construct ``FusedMoEQuantConfig`` for Int8 MoE."""
    assert (a1_scale is None and a2_scale is None) or (
        a1_scale is not None and a2_scale is not None
    ), "a1_scale and a2_scale must both be provided or both be None"

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


def convert_to_int8_moe_kernel_format(
    int8_backend: Int8MoeBackend,
    w13: torch.Tensor,
    w2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert INT8 MoE weights to backend-specific kernel format."""
    if int8_backend == Int8MoeBackend.CPU:
        from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
            prepare_int8_moe_layer_for_cpu,
        )

        w13, w2 = prepare_int8_moe_layer_for_cpu(w13, w2)
    elif int8_backend != Int8MoeBackend.TRITON:
        raise ValueError(f"Unsupported Int8 MoE backend: {int8_backend.value}")

    return w13, w2


def make_int8_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEExperts],
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> mk.FusedMoEKernel:
    """Wrapper preserved for backward compat.

    Note: the ``backend`` arg is not consumed by the generic
    ``make_kernel``; ``Int8MoeBackend.TRITON`` is passed as a placeholder.
    """
    return _ORACLE.make_kernel(
        moe_quant_config,
        moe_config,
        experts_cls,
        Int8MoeBackend.TRITON,
        routing_tables,
    )
