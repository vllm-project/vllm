# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config.kernel import MoEBackend
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
)
from vllm.model_executor.layers.fused_moe.oracle.base import MoEKernelOracle
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
)
from vllm.platforms import current_platform


class W4A8Int8MoeBackend(Enum):
    CPU_INT4 = "CPU_INT4"


class W4A8Int8MoEKernelOracle(MoEKernelOracle[W4A8Int8MoeBackend]):
    """W4A8 Int8 MoE kernel oracle.

    W4A8 Int8 MoE has a single backend (CPU_INT4) and is CPU-only.
    The subclass declares the backend data
    ``select_backend`` is inherited from ``MoEKernelOracle`` but wrapped to raise
    a CPU-only message early on non-CPU platforms.
    ``make_kernel`` is overridden to enforce monolithic-only experts.
    ``make_quant_config`` and ``convert_to_kernel_format`` are overridden
    with their full W4A8-specific signatures.
    """

    def backend_enum_cls(self) -> type[W4A8Int8MoeBackend]:
        return W4A8Int8MoeBackend

    def get_priority_backends(
        self, moe_config: FusedMoEConfig
    ) -> list[W4A8Int8MoeBackend]:
        if current_platform.is_cpu():
            return [W4A8Int8MoeBackend.CPU_INT4]
        return []

    def backend_to_kernel_cls(
        self, backend: W4A8Int8MoeBackend
    ) -> list[type[mk.FusedMoEExperts]]:
        if backend == W4A8Int8MoeBackend.CPU_INT4:
            from vllm.model_executor.layers.fused_moe.experts.cpu_int4_moe import (
                CPUExpertsInt4,
            )

            return [CPUExpertsInt4]
        raise ValueError(f"Unknown W4A8 Int8 MoE backend: {backend.value}")

    def map_backend(self, runner_backend: MoEBackend) -> W4A8Int8MoeBackend:
        mapping = {"cpu": W4A8Int8MoeBackend.CPU_INT4}
        if backend := mapping.get(runner_backend):
            return backend
        raise ValueError(
            f"moe_backend={runner_backend!r} is not supported for W4A8 Int8 "
            f"MoE. Expected one of {list(mapping.keys())}."
        )

    def select_backend(
        self,
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None = None,
        activation_key: QuantKey | None = None,
    ) -> tuple[W4A8Int8MoeBackend, type[mk.FusedMoEExperts] | None]:
        if not current_platform.is_cpu():
            raise NotImplementedError(
                "W4A8 Int8 MoE is only supported on CPU platforms"
            )
        return super().select_backend(moe_config, weight_key, activation_key)

    def make_kernel(
        self,
        quant_config: FusedMoEQuantConfig,
        moe_config: FusedMoEConfig,
        experts_cls: type[mk.FusedMoEExperts],
        backend: W4A8Int8MoeBackend,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> mk.FusedMoEKernel:
        if not issubclass(experts_cls, mk.FusedMoEExpertsMonolithic):
            raise ValueError(
                f"W4A8 Int8 MoE only supports monolithic experts, "
                f"but got {experts_cls.__name__}"
            )
        return super().make_kernel(
            quant_config, moe_config, experts_cls, backend, routing_tables
        )

    def convert_to_kernel_format(  # type: ignore[override]
        self,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        w13_weight_scale: torch.Tensor,
        w2_weight_scale: torch.Tensor,
        group_size: int,
        w13_bias: torch.Tensor | None = None,
        w2_bias: torch.Tensor | None = None,
    ):
        return convert_to_w4a8_int8_moe_format(
            w13_weight,
            w2_weight,
            w13_weight_scale,
            w2_weight_scale,
            group_size,
            w13_bias=w13_bias,
            w2_bias=w2_bias,
        )

    def make_quant_config(
        self,
        block_shape: tuple[int, int] | None = None,
    ) -> FusedMoEQuantConfig:
        return make_w4a8_int8_moe_quant_config(block_shape=block_shape)


_ORACLE = W4A8Int8MoEKernelOracle()


# ---------------------------------------------------------------------------
# Module-level function wrappers preserved for backward compat.
# These delegate to the oracle singleton above so the behaviour is
# bit-identical with the pre-refactor code path.
# ---------------------------------------------------------------------------


def _get_priority_backends(
    moe_config: FusedMoEConfig,
) -> list[W4A8Int8MoeBackend]:
    """Wrapper preserved for backward compat."""
    return _ORACLE.get_priority_backends(moe_config)


def backend_to_kernel_cls(
    backend: W4A8Int8MoeBackend,
) -> list[type[mk.FusedMoEExperts]]:
    """Wrapper preserved for backward compat."""
    return _ORACLE.backend_to_kernel_cls(backend)


def map_w4a8_int8_backend(runner_backend: MoEBackend) -> W4A8Int8MoeBackend:
    """Wrapper preserved for backward compat."""
    return _ORACLE.map_backend(runner_backend)


def select_w4a8_int8_moe_backend(
    config: FusedMoEConfig,
    weight_key: QuantKey | None,
    activation_key: QuantKey | None,
) -> tuple[W4A8Int8MoeBackend, type[mk.FusedMoEExperts] | None]:
    """Wrapper preserved for backward compat."""
    return _ORACLE.select_backend(config, weight_key, activation_key)


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
    w13_out_features = w13_weight.shape[1]  # 2 * intermediate_size
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
                w13_out_features,
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
    """Wrapper preserved for backward compat."""
    return _ORACLE.make_kernel(
        moe_quant_config,
        moe_config,
        experts_cls,
        W4A8Int8MoeBackend.CPU_INT4,
        routing_tables,
    )
