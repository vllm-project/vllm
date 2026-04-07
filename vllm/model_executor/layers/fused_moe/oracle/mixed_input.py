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
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
)

logger = init_logger(__name__)


class MixedInputMoeBackend(Enum):
    MARLIN = "MARLIN"


def select_mixed_input_moe_backend(
    config: FusedMoEConfig,
) -> tuple[MixedInputMoeBackend, type[mk.FusedMoEExperts]]:
    """
    Select the MoE backend for mixed-input (W4A16) quantization.

    Mixed-input refers to quantization schemes where weights are quantized
    (e.g., int4 group quantized) but activations are in floating point
    (fp16/bf16), or optionally in int8/fp8. Currently only the MARLIN
    backend is supported for these weight-only quantization schemes.
    """
    from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
        BatchedMarlinExperts,
        MarlinExperts,
    )

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
        "Using MARLIN Mixed-Input MoE backend with %s.",
        experts_cls.__name__,
        scope="local",
    )

    return MixedInputMoeBackend.MARLIN, experts_cls


def make_mixed_input_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEExperts],
    backend: MixedInputMoeBackend,
    w13_g_idx: torch.Tensor | None = None,
    w2_g_idx: torch.Tensor | None = None,
    w13_g_idx_sort_indices: torch.Tensor | None = None,
    w2_g_idx_sort_indices: torch.Tensor | None = None,
    is_k_full: bool = True,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | None = None,
    shared_experts: SharedExperts | None = None,
) -> mk.FusedMoEKernel:
    """Create a FusedMoEKernel for mixed-input (W4A16) quantization."""

    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=moe_quant_config,
        routing_tables=routing_tables,
        allow_new_interface=True,
        use_monolithic=False,
    )
    assert prepare_finalize is not None

    logger.info_once("Using %s", prepare_finalize.__class__.__name__, scope="local")

    common_kwargs: dict = {
        "moe_config": moe_config,
        "quant_config": moe_quant_config,
        "w13_g_idx": w13_g_idx,
        "w2_g_idx": w2_g_idx,
        "w13_g_idx_sort_indices": w13_g_idx_sort_indices,
        "w2_g_idx_sort_indices": w2_g_idx_sort_indices,
        "is_k_full": is_k_full,
    }

    if prepare_finalize.activation_format == mk.FusedMoEActivationFormat.BatchedExperts:
        max_num_tokens = prepare_finalize.max_num_tokens_per_rank()
        assert max_num_tokens is not None
        experts = experts_cls(
            max_num_tokens=max_num_tokens,
            num_dispatchers=prepare_finalize.num_dispatchers(),
            **common_kwargs,
        )
    else:
        experts = experts_cls(**common_kwargs)

    return mk.FusedMoEKernel(
        prepare_finalize,
        experts,
        shared_experts=shared_experts,
        inplace=not moe_config.disable_inplace,
    )
