# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.gptq_marlin import GPTQMarlinConfig

logger = init_logger(__name__)


class WNA16MoEBackend(Enum):
    MARLIN = "MARLIN"
    BATCHED_MARLIN = "BATCHED_MARLIN"


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

    else:
        raise ValueError(f"Unknown WNA16 MoE backend: {backend.value}")


def _get_priority_backends() -> list[WNA16MoEBackend]:
    """
    Get available backends in priority order based on platform and config.
    """
    _AVAILABLE_BACKENDS = [
        WNA16MoEBackend.MARLIN,
        WNA16MoEBackend.BATCHED_MARLIN,
    ]
    return _AVAILABLE_BACKENDS


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
    layer: torch.nn.Module,
    is_k_full: bool,
    w13_g_idx: torch.Tensor | None,
    w2_g_idx: torch.Tensor | None,
    w13_g_idx_sort_indices: torch.Tensor | None,
    w2_g_idx_sort_indices: torch.Tensor | None,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    shared_experts: torch.nn.Module | None = None,
) -> mk.FusedMoEKernel:
    # Currently, we only support MarlinExperts and BatchedMarlinExperts
    assert experts_cls in (MarlinExperts, BatchedMarlinExperts)

    from vllm.model_executor.layers.fused_moe.all2all_utils import (
        maybe_make_prepare_finalize,
    )

    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=moe_quant_config,
        routing_tables=routing_tables,
        allow_new_interface=True,
    )
    assert prepare_finalize is not None
    assert isinstance(prepare_finalize, mk.FusedMoEPrepareAndFinalizeModular)

    if prepare_finalize.activation_format == mk.FusedMoEActivationFormat.BatchedExperts:
        assert experts_cls == BatchedMarlinExperts
        max_num_tokens = prepare_finalize.max_num_tokens_per_rank()
        assert max_num_tokens is not None
        experts: mk.FusedMoEExperts = BatchedMarlinExperts(
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
        shared_experts=shared_experts,
        inplace=not moe_config.disable_inplace,
    )


# ---------------------------------------------------------------------------
# Per-backend weight post-processing
# ---------------------------------------------------------------------------


def _process_weights_marlin(
    layer: torch.nn.Module,
    quant_config: "GPTQMarlinConfig",
    input_dtype: torch.dtype | None,
    w13_qweight: torch.Tensor,
    w2_qweight: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_scales: torch.Tensor,
    w13_g_idx: torch.Tensor,
    w2_g_idx: torch.Tensor,
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
        w13_input_global_scale,
        w2_input_global_scale,
        w13_bias_out,
        w2_bias_out,
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
    w13_g_idx: torch.Tensor,
    w2_g_idx: torch.Tensor,
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
        from vllm.model_executor.layers.quantization.gptq_marlin import (
            GPTQMarlinConfig,
        )

        if not isinstance(quant_config, GPTQMarlinConfig):
            raise TypeError(
                "Marlin WNA16 MoE backend requires GPTQMarlinConfig, got "
                f"{type(quant_config).__name__}."
            )
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
            w13_bias,
            w2_bias,
        )
    else:
        raise ValueError(f"Unsupported wna16 MoE backend: {backend.value}")
