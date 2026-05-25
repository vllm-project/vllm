# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from enum import Enum
from typing import TYPE_CHECKING

import torch

import vllm._custom_ops as ops
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
    BatchedMarlinExperts,
    MarlinExperts,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_act_int8_process_scales,
    marlin_moe_permute_scales,
    marlin_permute_bias,
    moe_awq_to_marlin_zero_points,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
)
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig
    from vllm.model_executor.layers.quantization.awq_marlin import AWQMarlinConfig

logger = init_logger(__name__)


class WNA16MoEBackend(Enum):
    MARLIN = "MARLIN"
    BATCHED_MARLIN = "BATCHED_MARLIN"
    XPU = "XPU"


def backend_to_kernel_cls(
    backend: WNA16MoEBackend,
) -> list[type[mk.FusedMoEExperts]]:
    """Return the experts class for the given backend, or None for NONE."""
    if backend == WNA16MoEBackend.MARLIN:
        from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
            MarlinExperts,
        )

        return [MarlinExperts]

    elif backend == WNA16MoEBackend.BATCHED_MARLIN:
        from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
            BatchedMarlinExperts,
        )

        return [BatchedMarlinExperts]

    elif backend == WNA16MoEBackend.XPU:
        from vllm.model_executor.layers.fused_moe.experts.xpu_moe import (
            XPUExpertsWNA16,
        )

        return [XPUExpertsWNA16]

    else:
        raise ValueError(f"Unknown WNA16 MoE backend: {backend.value}")


def _get_priority_backends() -> list[WNA16MoEBackend]:
    """
    Get available backends in priority order based on platform and config.
    """
    if current_platform.is_xpu():
        return [WNA16MoEBackend.XPU]
    return [
        WNA16MoEBackend.MARLIN,
        WNA16MoEBackend.BATCHED_MARLIN,
    ]


def select_wna16_moe_backend(
    config: FusedMoEConfig,
    weight_key: QuantKey,
    weight_bits: int,
) -> tuple[WNA16MoEBackend, type[mk.FusedMoEExperts]]:
    """Select the WNA16 MoE backend.

    Args:
        config: the shared ``FusedMoEConfig`` for this layer.
        weight_bits: quantization bit-width (4 or 8). 8-bit weights are not
            supported by the modular Marlin kernel, so ``NONE`` is returned.

    Returns:
        A tuple of (``WNA16MoEBackend``, experts class or ``None``).
    """

    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )

    def _make_log_backend(backend: WNA16MoEBackend):
        return f"Using '{backend.value}' WNA16 MoE backend."

    def _make_log_unsupported(backend: WNA16MoEBackend, reason: str | None) -> str:
        if reason:
            return (
                f"WNA16 MoE backend '{backend.value}' does not support the "
                f"deployment configuration since {reason}."
            )
        return (
            f"WNA16 MoE backend '{backend.value}' does not support the "
            "deployment configuration."
        )

    def _return_or_raise(
        backend: WNA16MoEBackend,
        config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[WNA16MoEBackend, type[mk.FusedMoEExperts]]:
        reason: str | None = None
        for k_cls in backend_to_kernel_cls(backend):
            supported, reason = k_cls.is_supported_config(
                k_cls, config, weight_key, activation_key, activation_format
            )
            if supported:
                logger.info_once(_make_log_backend(backend), scope="local")
                return backend, k_cls
        raise ValueError(_make_log_unsupported(backend, reason))

    # Select kernels in order of backend.
    AVAILABLE_BACKENDS = _get_priority_backends()

    for backend in AVAILABLE_BACKENDS:
        activation_key = None  # always BF16 activation for WNA16 MoE
        for k_cls in backend_to_kernel_cls(backend):
            supported, reason = k_cls.is_supported_config(
                k_cls, config, weight_key, activation_key, activation_format
            )
            if supported:
                logger.info_once(_make_log_backend(backend), scope="local")
                return backend, k_cls
            else:
                logger.debug_once(_make_log_unsupported(backend, reason), scope="local")

    raise NotImplementedError(
        "No WNA16 MoE backend supports the deployment configuration."
    )


def make_wna16_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEExperts] | None,
    is_k_full: bool,
    w13_g_idx: torch.Tensor | None,
    w2_g_idx: torch.Tensor | None,
    w13_g_idx_sort_indices: torch.Tensor | None,
    w2_g_idx_sort_indices: torch.Tensor | None,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> mk.FusedMoEKernel:
    from vllm.model_executor.layers.fused_moe.all2all_utils import (
        maybe_make_prepare_finalize,
    )
    from vllm.model_executor.layers.fused_moe.experts.xpu_moe import (
        XPUExpertsWNA16,
    )

    assert experts_cls in (MarlinExperts, BatchedMarlinExperts, XPUExpertsWNA16)

    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=moe_quant_config,
        routing_tables=routing_tables,
        allow_new_interface=True,
    )
    assert prepare_finalize is not None
    assert isinstance(prepare_finalize, mk.FusedMoEPrepareAndFinalizeModular)

    if experts_cls is XPUExpertsWNA16:
        assert (
            prepare_finalize.activation_format == mk.FusedMoEActivationFormat.Standard
        ), (
            "XPUExpertsWNA16 only supports the Standard activation format; "
            "xpu_fused_moe(is_int4=True) does not implement BatchedExperts."
        )
        experts: mk.FusedMoEExperts = XPUExpertsWNA16(
            moe_config=moe_config,
            quant_config=moe_quant_config,
        )
    elif (
        prepare_finalize.activation_format == mk.FusedMoEActivationFormat.BatchedExperts
    ):
        assert experts_cls == BatchedMarlinExperts
        max_num_tokens = prepare_finalize.max_num_tokens_per_rank()
        assert max_num_tokens is not None
        experts = BatchedMarlinExperts(
            max_num_tokens=max_num_tokens,
            num_dispatchers=prepare_finalize.num_dispatchers(),
            moe_config=moe_config,
            quant_config=moe_quant_config,
            w13_g_idx=w13_g_idx,
            w2_g_idx=w2_g_idx,
            w13_g_idx_sort_indices=w13_g_idx_sort_indices,
            w2_g_idx_sort_indices=w2_g_idx_sort_indices,
            is_k_full=is_k_full,
        )
    else:
        assert experts_cls == MarlinExperts
        experts = MarlinExperts(
            moe_config=moe_config,
            quant_config=moe_quant_config,
            w13_g_idx=w13_g_idx,
            w2_g_idx=w2_g_idx,
            w13_g_idx_sort_indices=w13_g_idx_sort_indices,
            w2_g_idx_sort_indices=w2_g_idx_sort_indices,
            is_k_full=is_k_full,
        )

    return mk.FusedMoEKernel(
        prepare_finalize,
        experts,
        inplace=not moe_config.disable_inplace,
    )


# ---------------------------------------------------------------------------
# Per-backend weight post-processing
# ---------------------------------------------------------------------------


def _process_weights_marlin(
    layer: torch.nn.Module,
    quant_config: "AutoGPTQConfig",
    input_dtype: torch.dtype | None,
    w13_qweight: torch.Tensor,
    w2_qweight: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_scales: torch.Tensor,
    w13_g_idx: torch.Tensor,
    w2_g_idx: torch.Tensor,
    w13_qzeros: torch.Tensor | None = None,
    w2_qzeros: torch.Tensor | None = None,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,  # w13_qweight
    torch.Tensor,  # w2_qweight
    torch.Tensor,  # w13_scales
    torch.Tensor,  # w2_scales
    torch.Tensor,  # w13_g_idx
    torch.Tensor,  # w2_g_idx
    torch.Tensor,  # w13_g_idx_sort_indices
    torch.Tensor,  # w2_g_idx_sort_indices
    torch.Tensor | None,  # w13_qzeros
    torch.Tensor | None,  # w2_qzeros
    torch.Tensor | None,  # w13_input_global_scale
    torch.Tensor | None,  # w2_input_global_scale
    torch.Tensor | None,  # w13_bias
    torch.Tensor | None,  # w2_bias
]:
    """Standard Marlin weight post-processing shared by MARLIN and
    BATCHED_MARLIN backends.

    Steps
    -----
    1. Optional FP8 preprocessing of packed weights / scales.
    2. Sort / reset g_idx tensors for act-order handling.
    3. Repack weights via ``gptq_marlin_moe_repack``.
    4. Permute scales (and optionally extract INT8 global scales).
    5. Permute bias tensors.
    """
    is_a_8bit = input_dtype is not None and input_dtype.itemsize == 1

    marlin_w13_qweight: torch.Tensor
    marlin_w2_qweight: torch.Tensor
    marlin_w13_scales: torch.Tensor
    marlin_w2_scales: torch.Tensor
    w13_g_idx_sort_indices: torch.Tensor | None = None
    w2_g_idx_sort_indices: torch.Tensor | None = None
    w13_input_global_scale: torch.Tensor | None = None
    w2_input_global_scale: torch.Tensor | None = None
    w13_bias_out: torch.Tensor | None = None
    w2_bias_out: torch.Tensor | None = None

    # --- FP8 weight / scale adjustment ---
    if input_dtype == torch.float8_e4m3fn:
        marlin_w13_qweight = ops.marlin_int4_fp8_preprocess(w13_qweight, inplace=False)
        marlin_w2_qweight = ops.marlin_int4_fp8_preprocess(w2_qweight, inplace=False)
        marlin_w13_scales = w13_scales.data * 512
        marlin_w2_scales = w2_scales.data * 512
    else:
        marlin_w13_qweight = w13_qweight
        marlin_w2_qweight = w2_qweight
        marlin_w13_scales = w13_scales
        marlin_w2_scales = w2_scales

    # --- Process act_order (g_idx) ---
    if quant_config.desc_act:
        num_experts = w13_g_idx.shape[0]
        w13_g_idx_sort_indices = torch.empty_like(w13_g_idx)
        w2_g_idx_sort_indices = torch.empty_like(w2_g_idx)
        w13_sorted_g_idx = torch.empty_like(w13_g_idx)
        w2_sorted_g_idx = torch.empty_like(w2_g_idx)
        for e in range(num_experts):
            w13_g_idx_sort_indices[e] = torch.argsort(w13_g_idx[e]).to(torch.int32)
            w2_g_idx_sort_indices[e] = torch.argsort(w2_g_idx[e]).to(torch.int32)
            w13_sorted_g_idx[e] = w13_g_idx[e][w13_g_idx_sort_indices[e]]
            w2_sorted_g_idx[e] = w2_g_idx[e][w2_g_idx_sort_indices[e]]
    else:
        num_experts = w13_g_idx.shape[0]
        device = w13_g_idx.device
        w13_g_idx = torch.nn.Parameter(
            torch.empty((num_experts, 0), dtype=torch.int32, device=device),
            requires_grad=False,
        )
        w2_g_idx = torch.nn.Parameter(
            torch.empty((num_experts, 0), dtype=torch.int32, device=device),
            requires_grad=False,
        )
        w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty((num_experts, 0), dtype=torch.int32, device=device),
            requires_grad=False,
        )
        w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty((num_experts, 0), dtype=torch.int32, device=device),
            requires_grad=False,
        )

    # --- Repack weights ---
    marlin_w13_qweight = ops.gptq_marlin_moe_repack(
        marlin_w13_qweight,
        w13_g_idx_sort_indices,
        marlin_w13_qweight.shape[1] * quant_config.pack_factor,
        marlin_w13_qweight.shape[2],
        quant_config.quant_type.size_bits,
        is_a_8bit=is_a_8bit,
    )
    marlin_w2_qweight = ops.gptq_marlin_moe_repack(
        marlin_w2_qweight,
        w2_g_idx_sort_indices,
        marlin_w2_qweight.shape[1] * quant_config.pack_factor,
        marlin_w2_qweight.shape[2],
        quant_config.quant_type.size_bits,
        is_a_8bit=is_a_8bit,
    )

    # --- Permute scales ---
    marlin_w13_scales = marlin_moe_permute_scales(
        s=marlin_w13_scales,
        size_k=layer.intermediate_size_per_partition,
        size_n=marlin_w13_scales.shape[2],
        group_size=quant_config.group_size,
        is_a_8bit=is_a_8bit,
    )
    marlin_w2_scales = marlin_moe_permute_scales(
        s=marlin_w2_scales,
        size_k=marlin_w2_scales.shape[1]
        * (
            quant_config.group_size
            if quant_config.group_size != -1
            else quant_config.pack_factor
        ),
        size_n=marlin_w2_scales.shape[2],
        group_size=quant_config.group_size,
        is_a_8bit=is_a_8bit,
    )

    if input_dtype == torch.int8:
        if layer.num_groups_w13 > 1:
            marlin_w13_scales, w13_input_global_scale = marlin_act_int8_process_scales(
                marlin_w13_scales
            )
        if layer.num_groups_w2 > 1:
            marlin_w2_scales, w2_input_global_scale = marlin_act_int8_process_scales(
                marlin_w2_scales
            )

    # --- Permute bias ---
    if w13_bias is not None:
        w13_bias_out = marlin_permute_bias(w13_bias)
    if w2_bias is not None:
        w2_bias_out = marlin_permute_bias(w2_bias)

    return (
        marlin_w13_qweight,
        marlin_w2_qweight,
        marlin_w13_scales,
        marlin_w2_scales,
        w13_g_idx,
        w2_g_idx,
        w13_g_idx_sort_indices,
        w2_g_idx_sort_indices,
        w13_qzeros,
        w2_qzeros,
        w13_input_global_scale,
        w2_input_global_scale,
        w13_bias_out,
        w2_bias_out,
    )


def _process_awq_weights_marlin(
    layer: torch.nn.Module,
    quant_config: "AWQMarlinConfig",
    input_dtype: torch.dtype | None,
    w13_qweight: torch.Tensor,
    w2_qweight: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_scales: torch.Tensor,
    w13_qzeros: torch.Tensor,
    w2_qzeros: torch.Tensor,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,  # w13_qweight
    torch.Tensor,  # w2_qweight
    torch.Tensor,  # w13_scales
    torch.Tensor,  # w2_scales
    torch.Tensor | None,  # w13_g_idx
    torch.Tensor | None,  # w2_g_idx
    torch.Tensor | None,  # w13_g_idx_sort_indices
    torch.Tensor | None,  # w2_g_idx_sort_indices
    torch.Tensor | None,  # w13_qzeros
    torch.Tensor | None,  # w2_qzeros
    torch.Tensor | None,  # w13_input_global_scale
    torch.Tensor | None,  # w2_input_global_scale
    torch.Tensor | None,  # w13_bias
    torch.Tensor | None,  # w2_bias
]:
    """AWQ-specific Marlin weight post-processing.

    AWQ checkpoints use a different packing order than GPTQ, so they need
    AWQ-specific weight repacking and zero-point conversion before Marlin runs.
    """
    num_experts = w13_qweight.shape[0]
    device = w13_qweight.device
    is_a_8bit = input_dtype is not None and input_dtype.itemsize == 1
    w13_input_global_scale: torch.Tensor | None = None
    w2_input_global_scale: torch.Tensor | None = None
    w13_bias_out: torch.Tensor | None = None
    w2_bias_out: torch.Tensor | None = None

    if input_dtype == torch.float8_e4m3fn:
        ops.marlin_int4_fp8_preprocess(
            w13_qweight.view(-1, w13_qweight.size(2)),
            w13_qzeros.view(-1, w13_qzeros.size(2)),
            inplace=True,
        )
        ops.marlin_int4_fp8_preprocess(
            w2_qweight.view(-1, w2_qweight.size(2)),
            w2_qzeros.view(-1, w2_qzeros.size(2)),
            inplace=True,
        )
        w13_scales = w13_scales.data * 512
        w2_scales = w2_scales.data * 512

    w13_g_idx_sort_indices = torch.nn.Parameter(
        torch.empty((num_experts, 0), dtype=torch.int32, device=device),
        requires_grad=False,
    )
    w2_g_idx_sort_indices = torch.nn.Parameter(
        torch.empty((num_experts, 0), dtype=torch.int32, device=device),
        requires_grad=False,
    )

    marlin_w13_qweight = ops.awq_marlin_moe_repack(
        w13_qweight,
        w13_g_idx_sort_indices,
        size_k=w13_qweight.shape[1],
        size_n=w13_qweight.shape[2] * quant_config.pack_factor,
        num_bits=quant_config.weight_bits,
        is_a_8bit=is_a_8bit,
    )
    marlin_w2_qweight = ops.awq_marlin_moe_repack(
        w2_qweight,
        w2_g_idx_sort_indices,
        size_k=w2_qweight.shape[1],
        size_n=w2_qweight.shape[2] * quant_config.pack_factor,
        num_bits=quant_config.weight_bits,
        is_a_8bit=is_a_8bit,
    )

    marlin_w13_scales = marlin_moe_permute_scales(
        s=w13_scales,
        size_k=layer.intermediate_size_per_partition,
        size_n=w13_scales.shape[2],
        group_size=quant_config.group_size,
        is_a_8bit=is_a_8bit,
    )
    if input_dtype == torch.int8 and layer.num_groups_w13 > 1:
        marlin_w13_scales, w13_input_global_scale = marlin_act_int8_process_scales(
            marlin_w13_scales
        )

    marlin_w2_scales = marlin_moe_permute_scales(
        s=w2_scales,
        size_k=layer.intermediate_size_per_partition,
        size_n=w2_scales.shape[2],
        group_size=quant_config.group_size,
        is_a_8bit=is_a_8bit,
    )
    if input_dtype == torch.int8 and layer.num_groups_w2 > 1:
        marlin_w2_scales, w2_input_global_scale = marlin_act_int8_process_scales(
            marlin_w2_scales
        )

    marlin_w13_qzeros = moe_awq_to_marlin_zero_points(
        w13_qzeros,
        size_k=w13_qzeros.shape[1],
        size_n=w13_qzeros.shape[2] * quant_config.pack_factor,
        num_bits=quant_config.weight_bits,
        is_a_8bit=is_a_8bit,
    )
    marlin_w2_qzeros = moe_awq_to_marlin_zero_points(
        w2_qzeros,
        size_k=w2_qzeros.shape[1],
        size_n=w2_qzeros.shape[2] * quant_config.pack_factor,
        num_bits=quant_config.weight_bits,
        is_a_8bit=is_a_8bit,
    )

    if w13_bias is not None:
        w13_bias_out = marlin_permute_bias(w13_bias)
    if w2_bias is not None:
        w2_bias_out = marlin_permute_bias(w2_bias)

    return (
        marlin_w13_qweight,
        marlin_w2_qweight,
        marlin_w13_scales,
        marlin_w2_scales,
        None,
        None,
        w13_g_idx_sort_indices,
        w2_g_idx_sort_indices,
        marlin_w13_qzeros,
        marlin_w2_qzeros,
        w13_input_global_scale,
        w2_input_global_scale,
        w13_bias_out,
        w2_bias_out,
    )


def _process_weights_xpu(
    layer: torch.nn.Module,
    quant_config: QuantizationConfig,
    w13_qweight: torch.Tensor,
    w2_qweight: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_scales: torch.Tensor,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,  # w13_qweight
    torch.Tensor,  # w2_qweight
    torch.Tensor,  # w13_scales
    torch.Tensor,  # w2_scales
    torch.Tensor | None,  # w13_bias
    torch.Tensor | None,  # w2_bias
]:
    """Repack GPTQ-format INT4 MoE weights into the layout
    `vllm_xpu_kernels.fused_moe_interface.xpu_fused_moe(is_int4=True)` expects:

        w13: [E, 2*N, K] int4 (uint8 storage [E, 2*N, K // 2])
        w13_scales: [E, 2*N, K // group_size] params_dtype
        w2:  [E, K, N]   int4 (uint8 storage [E, K, N // 2])
        w2_scales:  [E, K, N // group_size]   params_dtype

    Input GPTQ layout from FusedMoE.weight_loader:
        w13: [E, K // 8, 2*N] int32 (8 nibbles per int32 along the input dim)
        w13_scales: [E, K // group_size, 2*N] params_dtype
        w2:  [E, N // 8, K] int32
        w2_scales:  [E, N // group_size, K] params_dtype

    Transpose dim 1 ↔ dim 2 then view int32 → uint8 to recover sequential
    int4-packed bytes along the input dim. Each packed int32 holds 8 nibbles
    `(n7<<28)|(n6<<24)|...|(n1<<4)|n0` in ascending K order; on a
    little-endian host the int32→uint8 view exposes them as bytes
    `[n1<<4|n0, n3<<4|n2, n5<<4|n4, n7<<4|n6]`, i.e. two nibbles per byte
    with the lower nibble = lower input-K index. xpu_fused_moe(is_int4=True)
    expects this convention; on a big-endian host the byte order reverses
    and the kernel would silently miscompute, so we hard-fail.
    """
    del layer, quant_config  # unused — kept for parity with the marlin helper

    if sys.byteorder != "little":
        raise NotImplementedError(
            "_process_weights_xpu requires a little-endian host: the GPTQ "
            "int32 → uint8 nibble repack relies on LE byte ordering."
        )

    w13_xpu = w13_qweight.transpose(1, 2).contiguous().view(torch.uint8)
    w2_xpu = w2_qweight.transpose(1, 2).contiguous().view(torch.uint8)
    w13_scales_xpu = w13_scales.transpose(1, 2).contiguous()
    w2_scales_xpu = w2_scales.transpose(1, 2).contiguous()

    return (
        w13_xpu,
        w2_xpu,
        w13_scales_xpu,
        w2_scales_xpu,
        w13_bias,
        w2_bias,
    )


def convert_to_wna16_moe_kernel_format(
    backend: WNA16MoEBackend,
    layer: torch.nn.Module,
    quant_config: QuantizationConfig,
    input_dtype: torch.dtype | None,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_g_idx: torch.Tensor | None = None,
    w2_g_idx: torch.Tensor | None = None,
    w13_qzeros: torch.Tensor | None = None,
    w2_qzeros: torch.Tensor | None = None,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,  # w13_qweight
    torch.Tensor,  # w2_qweight
    torch.Tensor,  # w13_scales
    torch.Tensor,  # w2_scales
    torch.Tensor | None,  # w13_g_idx
    torch.Tensor | None,  # w2_g_idx
    torch.Tensor | None,  # w13_g_idx_sort_indices
    torch.Tensor | None,  # w2_g_idx_sort_indices
    torch.Tensor | None,  # w13_qzeros
    torch.Tensor | None,  # w2_qzeros
    torch.Tensor | None,  # w13_input_global_scale
    torch.Tensor | None,  # w2_input_global_scale
    torch.Tensor | None,  # w13_bias
    torch.Tensor | None,  # w2_bias
]:
    """Dispatch weight post-processing to the appropriate per-backend handler.

    To add a new backend, implement a ``_process_weights_<name>`` helper and
    add a branch here.

    Args:
        backend: the selected ``WNA16MoEBackend``.
        layer: the ``FusedMoE`` layer whose parameters are being prepared.
        quant_config: the ``QuantizationConfig`` for this layer.
        input_dtype: optional activation dtype, usually should be 16 bit.
    """
    if backend in (
        WNA16MoEBackend.MARLIN,
        WNA16MoEBackend.BATCHED_MARLIN,
    ):
        from vllm.model_executor.layers.quantization.auto_gptq import (
            AutoGPTQConfig,
        )
        from vllm.model_executor.layers.quantization.awq_marlin import (
            AWQMarlinConfig,
        )

        if isinstance(quant_config, AWQMarlinConfig):
            if w13_qzeros is None or w2_qzeros is None:
                raise ValueError("AWQ Marlin MoE requires zero-point tensors.")
            return _process_awq_weights_marlin(
                layer,
                quant_config,
                input_dtype,
                w13,
                w2,
                w13_scale,
                w2_scale,
                w13_qzeros,
                w2_qzeros,
                w13_bias,
                w2_bias,
            )

        if not isinstance(quant_config, AutoGPTQConfig):
            raise TypeError(
                "Marlin WNA16 MoE backend requires AutoGPTQConfig or "
                "AWQMarlinConfig, got "
                f"{type(quant_config).__name__}."
            )
        if w13_g_idx is None or w2_g_idx is None:
            raise ValueError("GPTQ Marlin MoE requires g_idx tensors.")
        return _process_weights_marlin(
            layer,
            quant_config,
            input_dtype,
            w13,
            w2,
            w13_scale,
            w2_scale,
            w13_g_idx,
            w2_g_idx,
            w13_qzeros,
            w2_qzeros,
            w13_bias,
            w2_bias,
        )
    elif backend == WNA16MoEBackend.XPU:
        (
            w13_xpu,
            w2_xpu,
            w13_scale_xpu,
            w2_scale_xpu,
            w13_bias_out,
            w2_bias_out,
        ) = _process_weights_xpu(
            layer,
            quant_config,
            w13,
            w2,
            w13_scale,
            w2_scale,
            w13_bias,
            w2_bias,
        )
        empty = torch.empty((0,), dtype=torch.int32, device=w13.device)
        return (
            w13_xpu,
            w2_xpu,
            w13_scale_xpu,
            w2_scale_xpu,
            empty,  # w13_g_idx
            empty,  # w2_g_idx
            empty,  # w13_g_idx_sort_indices
            empty,  # w2_g_idx_sort_indices
            None,  # w13_qzeros — sym int4 on XPU has none; kernel does uint4b8→s4
            None,  # w2_qzeros
            None,  # w13_input_global_scale
            None,  # w2_input_global_scale
            w13_bias_out,
            w2_bias_out,
        )
    else:
        raise ValueError(f"Unsupported wna16 MoE backend: {backend.value}")
