# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import Enum
from typing import TYPE_CHECKING

import torch

import vllm._custom_ops as ops
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_act_int8_process_scales,
    marlin_moe_permute_scales,
    marlin_permute_bias,
)
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.gptq_marlin import GPTQMarlinConfig

logger = init_logger(__name__)


class GptqMarlinMoeBackend(Enum):
    # No modular-kernel support (e.g. 8-bit weights).
    NONE = "None"
    # Standard (TP / no EP) path – uses MarlinExperts.
    MARLIN = "MARLIN"
    # Expert-Parallel batched path – uses BatchedMarlinExperts.
    BATCHED_MARLIN = "BATCHED_MARLIN"


def backend_to_kernel_cls(
    backend: GptqMarlinMoeBackend,
) -> type[mk.FusedMoEExperts] | None:
    """Return the experts class for the given backend, or None for NONE."""
    if backend == GptqMarlinMoeBackend.MARLIN:
        from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
            MarlinExperts,
        )

        return MarlinExperts

    elif backend == GptqMarlinMoeBackend.BATCHED_MARLIN:
        from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
            BatchedMarlinExperts,
        )

        return BatchedMarlinExperts

    elif backend == GptqMarlinMoeBackend.NONE:
        return None

    else:
        raise ValueError(f"Unknown GPTQ-Marlin MoE backend: {backend.value}")


def select_gptq_marlin_moe_backend(
    config: FusedMoEConfig,
    weight_bits: int,
) -> tuple[GptqMarlinMoeBackend, type[mk.FusedMoEExperts] | None]:
    """Select the GPTQ-Marlin MoE backend.

    Returns a ``(backend, experts_cls)`` pair analogous to
    ``select_mxfp4_moe_backend()``.

    Args:
        config: the shared ``FusedMoEConfig`` for this layer.
        weight_bits: quantization bit-width (4 or 8). 8-bit weights are not
            supported by the modular Marlin kernel, so ``NONE`` is returned.

    Returns:
        A tuple of (``GptqMarlinMoeBackend``, experts class or ``None``).
    """
    # 8-bit GPTQ is not supported by the modular Marlin kernel.
    if weight_bits == 8:
        logger.debug_once(
            "GPTQ-Marlin 8-bit MoE modular kernel is not supported. "
            "Falling back to legacy fused_marlin_moe().",
            scope="local",
        )
        return GptqMarlinMoeBackend.NONE, None

    if not current_platform.is_cuda():
        logger.debug_once(
            "GPTQ-Marlin MoE modular kernel is only supported on CUDA. "
            "Falling back to legacy fused_marlin_moe().",
            scope="local",
        )
        return GptqMarlinMoeBackend.NONE, None

    # Choose BATCHED vs standard based on activation format (Expert Parallel).
    if config.moe_parallel_config.use_batched_activation_format:
        backend = GptqMarlinMoeBackend.BATCHED_MARLIN
    else:
        backend = GptqMarlinMoeBackend.MARLIN

    experts_cls = backend_to_kernel_cls(backend)
    logger.info_once(
        "Using '%s' GPTQ-Marlin MoE backend.", backend.value, scope="local"
    )
    return backend, experts_cls


# ---------------------------------------------------------------------------
# Per-backend weight post-processing
# ---------------------------------------------------------------------------


def _process_weights_marlin(
    layer: torch.nn.Module,
    quant_config: "GPTQMarlinConfig",
    input_dtype: torch.dtype | None,
) -> None:
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

    # --- FP8 weight / scale adjustment ---
    if input_dtype == torch.float8_e4m3fn:
        ops.marlin_int4_fp8_preprocess(layer.w13_qweight, inplace=True)
        ops.marlin_int4_fp8_preprocess(layer.w2_qweight, inplace=True)
        layer.w13_scales.data = layer.w13_scales.data * 512
        layer.w2_scales.data = layer.w2_scales.data * 512

    # --- Process act_order (g_idx) ---
    if quant_config.desc_act:
        num_experts = layer.w13_g_idx.shape[0]
        w13_g_idx_sort_indices = torch.empty_like(layer.w13_g_idx)
        w2_g_idx_sort_indices = torch.empty_like(layer.w2_g_idx)
        w13_sorted_g_idx = torch.empty_like(layer.w13_g_idx)
        w2_sorted_g_idx = torch.empty_like(layer.w2_g_idx)
        for e in range(num_experts):
            w13_g_idx_sort_indices[e] = torch.argsort(layer.w13_g_idx[e]).to(
                torch.int32
            )
            w2_g_idx_sort_indices[e] = torch.argsort(layer.w2_g_idx[e]).to(torch.int32)
            w13_sorted_g_idx[e] = layer.w13_g_idx[e][w13_g_idx_sort_indices[e]]
            w2_sorted_g_idx[e] = layer.w2_g_idx[e][w2_g_idx_sort_indices[e]]
        replace_parameter(layer, "w13_g_idx", w13_sorted_g_idx)
        replace_parameter(layer, "w2_g_idx", w2_sorted_g_idx)
        replace_parameter(layer, "w13_g_idx_sort_indices", w13_g_idx_sort_indices)
        replace_parameter(layer, "w2_g_idx_sort_indices", w2_g_idx_sort_indices)
    else:
        num_experts = layer.w13_g_idx.shape[0]
        device = layer.w13_g_idx.device
        layer.w13_g_idx = torch.nn.Parameter(
            torch.empty((num_experts, 0), dtype=torch.int32, device=device),
            requires_grad=False,
        )
        layer.w2_g_idx = torch.nn.Parameter(
            torch.empty((num_experts, 0), dtype=torch.int32, device=device),
            requires_grad=False,
        )
        layer.w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty((num_experts, 0), dtype=torch.int32, device=device),
            requires_grad=False,
        )
        layer.w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty((num_experts, 0), dtype=torch.int32, device=device),
            requires_grad=False,
        )

    # --- Repack weights ---
    marlin_w13_qweight = ops.gptq_marlin_moe_repack(
        layer.w13_qweight,
        layer.w13_g_idx_sort_indices,
        layer.w13_qweight.shape[1] * quant_config.pack_factor,
        layer.w13_qweight.shape[2],
        quant_config.quant_type.size_bits,
        is_a_8bit=is_a_8bit,
    )
    replace_parameter(layer, "w13_qweight", marlin_w13_qweight)
    marlin_w2_qweight = ops.gptq_marlin_moe_repack(
        layer.w2_qweight,
        layer.w2_g_idx_sort_indices,
        layer.w2_qweight.shape[1] * quant_config.pack_factor,
        layer.w2_qweight.shape[2],
        quant_config.quant_type.size_bits,
        is_a_8bit=is_a_8bit,
    )
    replace_parameter(layer, "w2_qweight", marlin_w2_qweight)

    # Alias for modular kernel (expects w13_weight / w2_weight)
    layer.w13_weight = layer.w13_qweight
    layer.w2_weight = layer.w2_qweight

    # --- Permute scales ---
    marlin_w13_scales = marlin_moe_permute_scales(
        s=layer.w13_scales,
        size_k=layer.intermediate_size_per_partition,
        size_n=layer.w13_scales.shape[2],
        group_size=quant_config.group_size,
        is_a_8bit=is_a_8bit,
    )
    if input_dtype == torch.int8 and layer.num_groups_w13 > 1:
        marlin_w13_scales, w13_input_global_scale = marlin_act_int8_process_scales(
            marlin_w13_scales
        )
        layer.register_parameter(
            "w13_input_global_scale",
            torch.nn.Parameter(w13_input_global_scale, requires_grad=False),
        )
    replace_parameter(layer, "w13_scales", marlin_w13_scales)

    marlin_w2_scales = marlin_moe_permute_scales(
        s=layer.w2_scales,
        size_k=layer.w2_scales.shape[1]
        * (
            quant_config.group_size
            if quant_config.group_size != -1
            else quant_config.pack_factor
        ),
        size_n=layer.w2_scales.shape[2],
        group_size=quant_config.group_size,
        is_a_8bit=is_a_8bit,
    )
    if input_dtype == torch.int8 and layer.num_groups_w2 > 1:
        marlin_w2_scales, w2_input_global_scale = marlin_act_int8_process_scales(
            marlin_w2_scales
        )
        layer.register_parameter(
            "w2_input_global_scale",
            torch.nn.Parameter(w2_input_global_scale, requires_grad=False),
        )
    replace_parameter(layer, "w2_scales", marlin_w2_scales)

    # --- Permute bias ---
    if hasattr(layer, "w13_bias") and layer.w13_bias is not None:
        layer.w13_bias.data = marlin_permute_bias(layer.w13_bias)
    if hasattr(layer, "w2_bias") and layer.w2_bias is not None:
        layer.w2_bias.data = marlin_permute_bias(layer.w2_bias)


def process_weights_for_marlin_backend(
    backend: GptqMarlinMoeBackend,
    layer: torch.nn.Module,
    quant_config: "GPTQMarlinConfig",
    input_dtype: torch.dtype | None,
) -> None:
    """Dispatch weight post-processing to the appropriate per-backend handler.

    To add a new backend, implement a ``_process_weights_<name>`` helper and
    add a branch here.

    Args:
        backend: the selected ``GptqMarlinMoeBackend``.
        layer: the ``FusedMoE`` layer whose parameters are being prepared.
        quant_config: the ``GPTQMarlinConfig`` for this layer.
        input_dtype: optional activation dtype (e.g. ``torch.int8``,
            ``torch.float8_e4m3fn``).
    """
    if backend == GptqMarlinMoeBackend.NONE:
        # No modular-kernel support; weights are used as-is by the legacy
        # fused_marlin_moe() path, which handles all transforms internally.
        return

    elif backend in (
        GptqMarlinMoeBackend.MARLIN,
        GptqMarlinMoeBackend.BATCHED_MARLIN,
    ):
        _process_weights_marlin(layer, quant_config, input_dtype)

    else:
        raise ValueError(f"Unsupported GPTQ-Marlin MoE backend: {backend.value}")
