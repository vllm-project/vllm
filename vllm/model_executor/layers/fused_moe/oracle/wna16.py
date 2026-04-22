# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    int4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
    BatchedMarlinExperts,
    MarlinExperts,
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_mxint4_moe import (
    is_flashinfer_mxint4_moe_available,
)

logger = init_logger(__name__)


class WNA16MoeBackend(Enum):
    MARLIN = "MARLIN"
    FLASHINFER = "FLASHINFER"


def select_wna16_moe_backend(
    config: FusedMoEConfig,
    num_bits: int,
    group_size: int,
) -> tuple[WNA16MoeBackend, type[mk.FusedMoEExperts] | None]:
    """
    Select the WNA16 MoE backend and experts class.

    Returns (backend, experts_cls) where experts_cls may be None for
    non-4-bit Marlin which falls back to the direct fused_marlin_moe() path.
    """
    if (
        is_flashinfer_mxint4_moe_available()
        and num_bits == 4
        and group_size == 32
    ):
        from vllm.model_executor.layers.fused_moe.experts.trtllm_mxint4_moe import (  # noqa: E501
            TrtLlmMxint4ExpertsMonolithic,
        )

        logger.info_once(
            "Using Flashinfer backend for WNA16 MoE "
            f"(group_size={group_size}, num_bits={num_bits})",
            scope="local",
        )
        return WNA16MoeBackend.FLASHINFER, TrtLlmMxint4ExpertsMonolithic

    if num_bits != 4:
        # Non-4-bit Marlin falls back to the direct fused_marlin_moe() path.
        logger.info_once(
            "Using Marlin backend for WNA16 MoE "
            f"(group_size={group_size}, num_bits={num_bits})",
            scope="local",
        )
        return WNA16MoeBackend.MARLIN, None

    # 4-bit Marlin: select batched vs standard based on activation format.
    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )
    experts_cls: type[mk.FusedMoEExperts] = (
        BatchedMarlinExperts
        if activation_format == mk.FusedMoEActivationFormat.BatchedExperts
        else MarlinExperts
    )
    logger.info_once(
        "Using Marlin backend for WNA16 MoE "
        f"(group_size={group_size}, num_bits={num_bits})",
        scope="local",
    )
    return WNA16MoeBackend.MARLIN, experts_cls


def make_wna16_moe_quant_config(
    layer: torch.nn.Module,
    group_size: int,
) -> FusedMoEQuantConfig:
    """Create the FusedMoEQuantConfig for 4-bit WNA16 MoE."""
    return int4_w4a16_moe_quant_config(
        w1_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
        w1_zp=None,
        w2_zp=None,
        block_shape=[0, group_size],
    )


def make_wna16_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    backend: WNA16MoeBackend,
    experts_cls: type[mk.FusedMoEExperts],
    w13_g_idx: torch.Tensor | None = None,
    w2_g_idx: torch.Tensor | None = None,
    w13_g_idx_sort_indices: torch.Tensor | None = None,
    w2_g_idx_sort_indices: torch.Tensor | None = None,
    is_k_full: bool = True,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | None = None,
    shared_experts: SharedExperts | None = None,
) -> mk.FusedMoEKernel:
    """
    Create the FusedMoEKernel for WNA16 MoE.

    For the Flashinfer (mxint4) backend, a monolithic kernel is created.
    For the Marlin backend, a modular kernel is created using MarlinExperts
    or BatchedMarlinExperts depending on the activation format.
    """
    is_monolithic = issubclass(experts_cls, mk.FusedMoEExpertsMonolithic)

    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=moe_quant_config,
        routing_tables=routing_tables,
        allow_new_interface=True,
        use_monolithic=is_monolithic,
    )
    assert prepare_finalize is not None

    logger.info_once(
        "Using %s", prepare_finalize.__class__.__name__, scope="local"
    )

    if is_monolithic:
        experts = experts_cls(
            moe_config=moe_config,
            quant_config=moe_quant_config,
        )
    elif (
        prepare_finalize.activation_format
        == mk.FusedMoEActivationFormat.BatchedExperts
    ):
        max_num_tokens = prepare_finalize.max_num_tokens_per_rank()
        assert max_num_tokens is not None
        experts = experts_cls(
            moe_config=moe_config,
            quant_config=moe_quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=prepare_finalize.num_dispatchers(),
            w13_g_idx=w13_g_idx,
            w2_g_idx=w2_g_idx,
            w13_g_idx_sort_indices=w13_g_idx_sort_indices,
            w2_g_idx_sort_indices=w2_g_idx_sort_indices,
            is_k_full=is_k_full,
        )
    else:
        experts = experts_cls(
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
        inplace=(not moe_config.disable_inplace and not is_monolithic),
    )
