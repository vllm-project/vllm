# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from enum import Enum
from typing import Any

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
)

import vllm._custom_ops as ops
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config.kernel import MoEBackend
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    int4_w4a16_moe_quant_config,
    int8_w8a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
    BatchedMarlinExperts,
    MarlinExperts,
    MarlinExpertsBase,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_mxint4_moe import (
    TrtLlmMxint4ExpertsMonolithic,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_act_int8_process_scales,
    marlin_moe_padded_intermediate,
    marlin_moe_permute_scales,
    marlin_permute_bias,
    moe_awq_to_marlin_zero_points,
    moe_packed_to_marlin_zero_points,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


class WNA16MoEBackend(Enum):
    MARLIN = "MARLIN"
    BATCHED_MARLIN = "BATCHED_MARLIN"
    HUMMING = "HUMMING"
    CPU = "CPU"
    FLASHINFER_TRTLLM = "FLASHINFER_TRTLLM"
    XPU = "XPU"


def backend_to_kernel_cls(
    backend: WNA16MoEBackend,
) -> list[type[mk.FusedMoEExperts]]:
    """Return the experts class for the given backend, or None for NONE."""
    if backend == WNA16MoEBackend.HUMMING:
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
    elif backend == WNA16MoEBackend.MARLIN:
        return [MarlinExperts]
    elif backend == WNA16MoEBackend.BATCHED_MARLIN:
        return [BatchedMarlinExperts]
    elif backend == WNA16MoEBackend.FLASHINFER_TRTLLM:
        return [TrtLlmMxint4ExpertsMonolithic]
    elif backend == WNA16MoEBackend.XPU:
        from vllm.model_executor.layers.fused_moe.experts.xpu_moe import (
            XPUExpertsWNA16,
        )

        return [XPUExpertsWNA16]
    elif backend == WNA16MoEBackend.CPU:
        from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
            CPUExpertsInt4,
        )

        return [CPUExpertsInt4]
    else:
        raise ValueError(f"Unknown WNA16 MoE backend: {backend.value}")


def _get_priority_backends() -> list[WNA16MoEBackend]:
    """
    Get available backends in priority order based on platform and config.
    """
    if current_platform.is_cpu():
        return [WNA16MoEBackend.CPU]
    if current_platform.is_xpu():
        return [WNA16MoEBackend.XPU]

    _AVAILABLE_BACKENDS = [
        WNA16MoEBackend.FLASHINFER_TRTLLM,
        WNA16MoEBackend.MARLIN,
        WNA16MoEBackend.BATCHED_MARLIN,
        WNA16MoEBackend.HUMMING,
    ]
    return _AVAILABLE_BACKENDS


def map_wna16_backend(runner_backend: MoEBackend) -> WNA16MoEBackend:
    """Map user's MoEBackend to WNA16MoEBackend."""
    mapping = {
        "marlin": WNA16MoEBackend.MARLIN,
        "humming": WNA16MoEBackend.HUMMING,
        "flashinfer_trtllm": WNA16MoEBackend.FLASHINFER_TRTLLM,
    }
    if backend := mapping.get(runner_backend):
        return backend
    raise ValueError(
        f"moe_backend='{runner_backend}' is not supported for WNA16 MoE. "
        f"Expected one of {list(mapping.keys())}."
    )


def select_wna16_moe_backend(
    config: FusedMoEConfig,
    weight_key: QuantKey,
) -> tuple[WNA16MoEBackend, type[mk.FusedMoEExperts]]:
    """Select the WNA16 MoE backend.

    Args:
        config: the shared ``FusedMoEConfig`` for this layer.
        weight_key: The QuantKey describing the weight quantization.
                    Must have int4 or int8 type.

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

    # Handle explicit moe_backend from user.
    runner_backend = config.moe_backend
    if runner_backend != "auto":
        requested_backend = map_wna16_backend(runner_backend)
        return _return_or_raise(
            requested_backend, config, weight_key, None, activation_format
        )

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


def make_wna16_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    group_size: int,
    num_bits: int,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    a1_gscale: torch.Tensor | None = None,
    a2_gscale: torch.Tensor | None = None,
) -> FusedMoEQuantConfig:
    """Create the FusedMoEQuantConfig for 4 or 8-bit WNA16 MoE."""
    if num_bits == 4:
        return int4_w4a16_moe_quant_config(
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_zp=w1_zp,
            w2_zp=w2_zp,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            block_shape=[0, group_size],
            a1_gscale=a1_gscale,
            a2_gscale=a2_gscale,
        )
    else:
        assert num_bits == 8
        return int8_w8a16_moe_quant_config(
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_zp=w1_zp,
            w2_zp=w2_zp,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            block_shape=[0, group_size],
            a1_gscale=a1_gscale,
            a2_gscale=a2_gscale,
        )


def make_wna16_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEExperts],
    backend: WNA16MoEBackend = WNA16MoEBackend.MARLIN,
    layer: torch.nn.Module | None = None,
    is_k_full: bool = False,
    w13_g_idx: torch.Tensor | None = None,
    w2_g_idx: torch.Tensor | None = None,
    w13_g_idx_sort_indices: torch.Tensor | None = None,
    w2_g_idx_sort_indices: torch.Tensor | None = None,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> mk.FusedMoEKernel:
    from vllm.model_executor.layers.fused_moe.all2all_utils import (
        maybe_make_prepare_finalize,
    )
    from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
        CPUExpertsInt4,
    )
    from vllm.model_executor.layers.fused_moe.experts.xpu_moe import (
        XPUExpertsWNA16,
    )

    # Currently, we only support TrtLlmMxint4ExpertsMonolithic, MarlinExperts,
    # BatchedMarlinExperts, XPUExpertsWNA16, CPUExpertsInt4, and the Humming
    # grouped/indexed experts.
    allowed_experts: tuple[type[mk.FusedMoEExperts], ...] = (
        MarlinExperts,
        BatchedMarlinExperts,
        TrtLlmMxint4ExpertsMonolithic,
        XPUExpertsWNA16,
        CPUExpertsInt4,
    )
    if backend == WNA16MoEBackend.HUMMING:
        allowed_experts += tuple(backend_to_kernel_cls(WNA16MoEBackend.HUMMING))
    assert experts_cls in allowed_experts

    is_monolithic = experts_cls.is_monolithic()

    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=moe_quant_config,
        routing_tables=routing_tables,
        allow_new_interface=True,
        use_monolithic=is_monolithic,
    )
    assert prepare_finalize is not None

    logger.info_once("Using %s", prepare_finalize.__class__.__name__, scope="local")

    extra_args: dict[str, Any] = {}
    if backend == WNA16MoEBackend.HUMMING:
        assert layer is not None
        extra_args = {"layer": layer}
    elif issubclass(experts_cls, MarlinExpertsBase):
        extra_args = {
            "w13_g_idx": w13_g_idx,
            "w2_g_idx": w2_g_idx,
            "w13_g_idx_sort_indices": w13_g_idx_sort_indices,
            "w2_g_idx_sort_indices": w2_g_idx_sort_indices,
            "is_k_full": is_k_full,
        }

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
        max_num_tokens = prepare_finalize.max_num_tokens_per_rank()
        assert max_num_tokens is not None
        experts = experts_cls(
            max_num_tokens=max_num_tokens,
            num_dispatchers=prepare_finalize.num_dispatchers(),
            moe_config=moe_config,
            quant_config=moe_quant_config,
            **extra_args,
        )
    else:
        experts = experts_cls(
            moe_config=moe_config,
            quant_config=moe_quant_config,
            **extra_args,
        )

    return mk.FusedMoEKernel(
        prepare_finalize,
        experts,
    )


# ---------------------------------------------------------------------------
# Per-backend weight post-processing
# ---------------------------------------------------------------------------


def _process_weights_flashinfer(
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
    torch.Tensor | None,  # w13_g_idx_sort_indices
    torch.Tensor | None,  # w2_g_idx_sort_indices
    torch.Tensor | None,  # w13_qzeros
    torch.Tensor | None,  # w2_qzeros
    torch.Tensor | None,  # w13_input_global_scale
    torch.Tensor | None,  # w2_input_global_scale
    torch.Tensor | None,  # w13_bias
    torch.Tensor | None,  # w2_bias
]:
    """Flashinfer (TRT-LLM MXINT4) weight post-processing.

    Steps
    -----
    1. Transform weights/scales via ``prepare_static_weights_for_trtllm_mxint4_moe``.
    2. Return transformed tensors, passing through g_idx/bias unchanged.
    """
    from vllm.model_executor.layers.quantization.utils.flashinfer_mxint4_moe import (
        prepare_static_weights_for_trtllm_mxint4_moe,
    )

    dict_weights_mxint4 = prepare_static_weights_for_trtllm_mxint4_moe(
        w13_qweight,
        w13_scales,
        w2_qweight,
        w2_scales,
    )

    return (
        dict_weights_mxint4["gemm1_weights"],
        dict_weights_mxint4["gemm2_weights"],
        dict_weights_mxint4["gemm1_scales"],
        dict_weights_mxint4["gemm2_scales"],
        w13_g_idx,
        w2_g_idx,
        None,
        None,
        None,
        None,
        None,
        None,
        w13_bias,
        w2_bias,
    )


def _pad_w13_shard_cols(x: torch.Tensor, unit: int, padded_unit: int) -> torch.Tensor:
    """Zero-pad each of the two gate/up shards of a ``(E, rows, 2 * unit)``
    tensor along its last dim, from ``unit`` to ``padded_unit`` columns."""
    if padded_unit == unit:
        return x
    e, rows, _ = x.shape
    x = x.view(e, rows, 2, unit)
    x = torch.nn.functional.pad(x, (0, padded_unit - unit))
    return x.reshape(e, rows, 2 * padded_unit).contiguous()


def _pad_rows(x: torch.Tensor, padded_rows: int) -> torch.Tensor:
    """Zero-pad a ``(E, rows, cols)`` tensor to ``padded_rows`` rows."""
    if padded_rows == x.size(1):
        return x
    return torch.nn.functional.pad(x, (0, 0, 0, padded_rows - x.size(1)))


def _pad_w13_bias(bias: torch.Tensor, n: int, padded_n: int) -> torch.Tensor:
    """Zero-pad each gate/up shard of a ``(E, 2 * n)`` bias to ``padded_n``."""
    if padded_n == n:
        return bias
    e = bias.size(0)
    bias = bias.view(e, 2, n)
    bias = torch.nn.functional.pad(bias, (0, padded_n - n))
    return bias.reshape(e, 2 * padded_n).contiguous()


def _process_weights_marlin(
    layer: torch.nn.Module,
    input_dtype: torch.dtype | None,
    num_bits: int,
    pack_factor: int,
    group_size: int,
    actorder: str | None,
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
        # NOTE: for non-zp quantization format only
        marlin_w13_qweight = ops.marlin_int4_fp8_preprocess(w13_qweight, inplace=False)
        marlin_w2_qweight = ops.marlin_int4_fp8_preprocess(w2_qweight, inplace=False)
        marlin_w13_scales = w13_scales.data * 512
        marlin_w2_scales = w2_scales.data * 512
    else:
        marlin_w13_qweight = w13_qweight
        marlin_w2_qweight = w2_qweight
        marlin_w13_scales = w13_scales
        marlin_w2_scales = w2_scales

    # --- Pad the intermediate size to a valid Marlin thread tile ---
    # GPTQ packs along K: w13's N is in the (shard) columns, w2's N in the rows.
    # Act-order keeps the strict shape and is never padded.
    N = layer.intermediate_size_per_partition
    padded_N = marlin_moe_padded_intermediate(N, group_size)
    if padded_N != N:
        assert actorder != "group", (
            "Marlin MoE thread-tile padding is unsupported with act-order"
        )
        marlin_w13_qweight = _pad_w13_shard_cols(marlin_w13_qweight, N, padded_N)
        marlin_w2_qweight = _pad_rows(marlin_w2_qweight, padded_N // pack_factor)
        marlin_w13_scales = _pad_w13_shard_cols(marlin_w13_scales, N, padded_N)
        if group_size > 0:
            marlin_w2_scales = _pad_rows(marlin_w2_scales, padded_N // group_size)
        if w13_qzeros is not None:
            w13_qzeros = _pad_w13_shard_cols(
                w13_qzeros, N // pack_factor, padded_N // pack_factor
            )
        if w2_qzeros is not None and group_size > 0:
            w2_qzeros = _pad_rows(w2_qzeros, padded_N // group_size)
        if w13_bias is not None:
            w13_bias = _pad_w13_bias(w13_bias, N, padded_N)

    # --- Process act_order (g_idx) ---
    if actorder == "group":
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
        w13_g_idx = w13_sorted_g_idx
        w2_g_idx = w2_sorted_g_idx
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
        marlin_w13_qweight.shape[1] * pack_factor,
        marlin_w13_qweight.shape[2],
        num_bits,
        is_a_8bit=is_a_8bit,
    )
    marlin_w2_qweight = ops.gptq_marlin_moe_repack(
        marlin_w2_qweight,
        w2_g_idx_sort_indices,
        marlin_w2_qweight.shape[1] * pack_factor,
        marlin_w2_qweight.shape[2],
        num_bits,
        is_a_8bit=is_a_8bit,
    )

    # --- Permute scales ---
    marlin_w13_scales = marlin_moe_permute_scales(
        s=marlin_w13_scales,
        size_k=layer.intermediate_size_per_partition,
        size_n=marlin_w13_scales.shape[2],
        group_size=group_size,
        is_a_8bit=is_a_8bit,
    )
    group_size_or_pack_factor = group_size if group_size != -1 else pack_factor
    marlin_w2_scales = marlin_moe_permute_scales(
        s=marlin_w2_scales,
        size_k=marlin_w2_scales.shape[1] * group_size_or_pack_factor,
        size_n=marlin_w2_scales.shape[2],
        group_size=group_size,
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

    # --- Permute zero points ---
    if w13_qzeros is not None and w2_qzeros is not None:
        w13_qzeros = moe_packed_to_marlin_zero_points(
            w13_qzeros,
            size_k=w13_qzeros.shape[1],
            size_n=w13_qzeros.shape[2] * pack_factor,
            num_bits=num_bits,
            is_a_8bit=is_a_8bit,
        )
        w2_qzeros = moe_packed_to_marlin_zero_points(
            w2_qzeros,
            size_k=w2_qzeros.shape[1],
            size_n=w2_qzeros.shape[2] * pack_factor,
            num_bits=num_bits,
            is_a_8bit=is_a_8bit,
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
    weight_bits: int,
    pack_factor: int,
    group_size: int,
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

    # --- Pad the intermediate size to a valid Marlin thread tile ---
    # AWQ packs along N: w13's N is in the (shard) columns, w2's N in the rows.
    N = layer.intermediate_size_per_partition
    padded_N = marlin_moe_padded_intermediate(N, group_size)
    if padded_N != N:
        w13_qweight = _pad_w13_shard_cols(
            w13_qweight, N // pack_factor, padded_N // pack_factor
        )
        w2_qweight = _pad_rows(w2_qweight, padded_N)
        w13_scales = _pad_w13_shard_cols(w13_scales, N, padded_N)
        w13_qzeros = _pad_w13_shard_cols(
            w13_qzeros, N // pack_factor, padded_N // pack_factor
        )
        if group_size > 0:
            w2_scales = _pad_rows(w2_scales, padded_N // group_size)
            w2_qzeros = _pad_rows(w2_qzeros, padded_N // group_size)
        if w13_bias is not None:
            w13_bias = _pad_w13_bias(w13_bias, N, padded_N)

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
        size_n=w13_qweight.shape[2] * pack_factor,
        num_bits=weight_bits,
        is_a_8bit=is_a_8bit,
    )
    marlin_w2_qweight = ops.awq_marlin_moe_repack(
        w2_qweight,
        w2_g_idx_sort_indices,
        size_k=w2_qweight.shape[1],
        size_n=w2_qweight.shape[2] * pack_factor,
        num_bits=weight_bits,
        is_a_8bit=is_a_8bit,
    )

    marlin_w13_scales = marlin_moe_permute_scales(
        s=w13_scales,
        size_k=layer.intermediate_size_per_partition,
        size_n=w13_scales.shape[2],
        group_size=group_size,
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
        group_size=group_size,
        is_a_8bit=is_a_8bit,
    )
    if input_dtype == torch.int8 and layer.num_groups_w2 > 1:
        marlin_w2_scales, w2_input_global_scale = marlin_act_int8_process_scales(
            marlin_w2_scales
        )

    marlin_w13_qzeros = moe_awq_to_marlin_zero_points(
        w13_qzeros,
        size_k=w13_qzeros.shape[1],
        size_n=w13_qzeros.shape[2] * pack_factor,
        num_bits=weight_bits,
        is_a_8bit=is_a_8bit,
    )
    marlin_w2_qzeros = moe_awq_to_marlin_zero_points(
        w2_qzeros,
        size_k=w2_qzeros.shape[1],
        size_n=w2_qzeros.shape[2] * pack_factor,
        num_bits=weight_bits,
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


def _process_weights_cpu(
    quant_config: QuantizationConfig | QuantizationArgs | None,
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
    """CPU INT4 W4A16 weight post-processing."""
    from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
        prepare_int4_moe_layer_for_cpu,
    )
    from vllm.model_executor.layers.quantization.auto_awq import (
        AutoAWQConfig,
    )
    from vllm.model_executor.layers.quantization.auto_gptq import (
        AutoGPTQConfig,
    )

    # Detect packing format.
    # AWQ: qweight is [E, K, 2*N//8] (packed along output/N dim).
    # GPTQ: qweight is [E, K//8, 2*N] (packed along input/K dim).
    # compressed-tensors: qweight is [E, K//8, 2*N] (packed along input/K dim).
    if isinstance(quant_config, AutoAWQConfig):
        # AWQ: K is stored unpacked in dim 1.
        cpu_quant_algo = ops.CPUQuantAlgo.AWQ
    elif isinstance(quant_config, (AutoGPTQConfig, QuantizationArgs)):
        # GPTQ / compressed-tensors: K//8 is stored packed in dim 1.
        if isinstance(quant_config, AutoGPTQConfig) and quant_config.desc_act:
            raise NotImplementedError(
                "CPU WNA16 MoE backend does not support GPTQ with "
                "desc_act=True. The fused MoE kernel has no g_idx "
                "reordering support."
            )
        cpu_quant_algo = ops.CPUQuantAlgo.GPTQ
    else:
        raise TypeError(
            "CPU WNA16 MoE backend requires AutoAWQConfig, AutoGPTQConfig "
            f"or QuantizationArgs, got {type(quant_config).__name__}."
        )

    # Determine zero points for repacking.
    w13_zeros: torch.Tensor | None = None
    w2_zeros: torch.Tensor | None = None
    if w13_qzeros is not None:
        w13_zeros = (
            w13_qzeros.data.view(torch.int32)
            if w13_qzeros.dtype != torch.int32
            else w13_qzeros.data
        )
    if w2_qzeros is not None:
        w2_zeros = (
            w2_qzeros.data.view(torch.int32)
            if w2_qzeros.dtype != torch.int32
            else w2_qzeros.data
        )

    (
        blocked_w13,
        blocked_w2,
        blocked_s13,
        blocked_s2,
        blocked_z13,
        blocked_z2,
    ) = prepare_int4_moe_layer_for_cpu(
        w13,
        w2,
        w13_scale,
        w2_scale,
        quant_algo=cpu_quant_algo,
        w13_zeros=w13_zeros,
        w2_zeros=w2_zeros,
    )
    return (
        blocked_w13,
        blocked_w2,
        blocked_s13,
        blocked_s2,
        w13_g_idx,
        w2_g_idx,
        None,  # w13_g_idx_sort_indices (unused on CPU)
        None,  # w2_g_idx_sort_indices (unused on CPU)
        blocked_z13,
        blocked_z2,
        None,  # w13_input_global_scale
        None,  # w2_input_global_scale
        w13_bias.to(torch.float32) if w13_bias is not None else None,
        w2_bias.to(torch.float32) if w2_bias is not None else None,
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


def _humming_wna16_weight_schema(
    quant_config: QuantizationConfig | QuantizationArgs | None,
) -> dict[str, Any]:
    """Humming weight schema for a WNA16 checkpoint, derived from the quant
    config rather than the running kernel."""
    from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig
    from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig

    if isinstance(quant_config, AutoAWQConfig):
        return {
            "quant_method": "awq",
            "bits": quant_config.weight_bits,
            "group_size": quant_config.group_size,
            "zero_point": quant_config.zero_point,
        }
    if isinstance(quant_config, AutoGPTQConfig):
        return {
            "quant_method": "gptq",
            "bits": quant_config.weight_bits,
            "group_size": quant_config.group_size,
            "desc_act": quant_config.desc_act,
            "sym": quant_config.is_sym,
        }
    raise TypeError(
        "Humming WNA16 MoE requires AutoAWQConfig or AutoGPTQConfig, "
        f"got {type(quant_config).__name__}."
    )


def convert_to_wna16_moe_kernel_format(
    backend: WNA16MoEBackend,
    layer: torch.nn.Module,
    quant_config: QuantizationConfig | QuantizationArgs | None,
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
) -> (
    tuple[
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
    ]
    | None
):
    """Dispatch weight post-processing to the appropriate per-backend handler.

    To add a new backend, implement a ``_process_weights_<name>`` helper and
    add a branch here. Backends that rewrite the layer's parameters in place
    (e.g. Humming) return ``None``; the caller then skips the param scatter.

    Args:
        backend: the selected ``WNA16MoEBackend``.
        layer: the ``FusedMoE`` layer whose parameters are being prepared.
        quant_config: the ``QuantizationConfig`` for this layer.
        input_dtype: optional activation dtype, usually should be 16 bit.
    """
    if backend == WNA16MoEBackend.HUMMING:
        from vllm.model_executor.layers.quantization.utils.humming_utils import (
            convert_to_humming_moe_kernel_format,
        )

        convert_to_humming_moe_kernel_format(
            layer, quant_config=_humming_wna16_weight_schema(quant_config)
        )
        return None

    if backend in (
        WNA16MoEBackend.MARLIN,
        WNA16MoEBackend.BATCHED_MARLIN,
    ):
        from vllm.model_executor.layers.quantization.auto_awq import (
            AutoAWQConfig,
        )
        from vllm.model_executor.layers.quantization.auto_gptq import (
            AutoGPTQConfig,
        )

        if isinstance(quant_config, AutoAWQConfig):
            if w13_qzeros is None or w2_qzeros is None:
                raise ValueError("AWQ Marlin MoE requires zero-point tensors.")

            weight_bits = quant_config.weight_bits
            pack_factor = quant_config.pack_factor
            group_size = quant_config.group_size

            return _process_awq_weights_marlin(
                layer,
                weight_bits,
                pack_factor,
                group_size,
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
        elif isinstance(quant_config, AutoGPTQConfig):
            num_bits = quant_config.quant_type.size_bits
            pack_factor = quant_config.pack_factor
            group_size = quant_config.group_size
            actorder = "group" if quant_config.desc_act else None
        elif isinstance(quant_config, QuantizationArgs):
            num_bits = quant_config.num_bits
            pack_factor = 32 // quant_config.num_bits
            group_size = quant_config.group_size
            actorder = quant_config.actorder
        else:
            raise TypeError(
                "Marlin WNA16 MoE backend requires AutoAWQConfig, AutoGPTQConfig or "
                f"QuantizationArgs, got {type(quant_config).__name__}."
            )
        if w13_g_idx is None or w2_g_idx is None:
            raise ValueError("GPTQ Marlin MoE requires g_idx tensors.")
        return _process_weights_marlin(
            layer,
            input_dtype,
            num_bits,
            pack_factor,
            group_size,
            actorder,
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
    elif backend == WNA16MoEBackend.CPU:
        return _process_weights_cpu(
            quant_config,
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
    elif backend == WNA16MoEBackend.FLASHINFER_TRTLLM:
        return _process_weights_flashinfer(
            w13,
            w2,
            w13_scale,
            w2_scale,
            w13_g_idx,
            w2_g_idx,
            w13_bias,
            w2_bias,
        )
    elif backend == WNA16MoEBackend.XPU:
        assert quant_config is not None
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
