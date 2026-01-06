# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm.distributed import (
    get_ep_group,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEPrepareAndFinalize,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_deep_ep, has_pplx

if current_platform.is_cuda_alike():
    if has_pplx():
        from .pplx_prepare_finalize import (
            PplxPrepareAndFinalize,
            pplx_hidden_dim_scale_bytes,
        )
    if has_deep_ep():
        from .deepep_ht_prepare_finalize import DeepEPHTPrepareAndFinalize
        from .deepep_ll_prepare_finalize import (
            DEEPEP_QUANT_BLOCK_SHAPE,
            DeepEPLLPrepareAndFinalize,
        )


def maybe_roundup_layer_hidden_size(
    hidden_size: int,
    act_dtype: torch.dtype,
    moe_parallel_config: FusedMoEParallelConfig,
) -> int:
    """
    Given layer hidden size and MoE configurations, round up hidden_size
    if necessary.

    Args:
        hidden_size: Layer hidden-size
        act_dtype: Data type of the layer activations.
        moe_parallel_config: Fused MoE parallelization strategy configuration.

    Return:
        Rounded up hidden_size if rounding up is required based on the configs
        and all2all backend.
        Original hidden size otherwise.
    """
    if moe_parallel_config.use_deepep_ht_kernels:
        hidden_size = DeepEPHTPrepareAndFinalize.maybe_roundup_layer_hidden_size(
            hidden_size, act_dtype
        )

    if moe_parallel_config.use_deepep_ll_kernels:
        hidden_size = DeepEPLLPrepareAndFinalize.maybe_roundup_layer_hidden_size(
            hidden_size
        )

    return hidden_size


def maybe_make_prepare_finalize(
    moe: FusedMoEConfig,
    quant_config: FusedMoEQuantConfig | None,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> FusedMoEPrepareAndFinalize | None:
    if not moe.moe_parallel_config.use_all2all_kernels:
        return None

    all2all_manager = get_ep_group().device_communicator.all2all_manager
    assert all2all_manager is not None

    prepare_finalize: FusedMoEPrepareAndFinalize | None = None

    # TODO(rob): update this as part of the MoE refactor.
    assert not moe.use_flashinfer_cutlass_kernels, (
        "Must be created in modelopt.py or fp8.py"
    )

    if moe.use_pplx_kernels:
        assert quant_config is not None

        hidden_dim_bytes, hidden_scale_bytes = pplx_hidden_dim_scale_bytes(
            moe.max_num_tokens,
            moe.hidden_dim,
            moe.in_dtype,
            quant_config.quant_dtype,
            per_act_token_quant=quant_config.per_act_token_quant,
            block_shape=quant_config.block_shape,
        )

        all_to_all_args = dict(
            max_num_tokens=moe.max_num_tokens,
            num_experts=moe.num_experts,
            experts_per_token=moe.experts_per_token,  # topk
            rank=all2all_manager.rank,
            world_size=all2all_manager.world_size,
            # dp_size actually means tp_size, bug in pplx kernels
            dp_size=all2all_manager.tp_group.world_size,
            hidden_dim=moe.hidden_dim,
            hidden_dim_bytes=hidden_dim_bytes,
            hidden_dim_scale_bytes=hidden_scale_bytes,
        )

        num_dispatchers = (
            all2all_manager.world_size // all2all_manager.tp_group.world_size
        )

        # Intranode pplx a2a takes a group name while internode does not.
        if not all2all_manager.internode:
            all_to_all_args["group_name"] = all2all_manager.cpu_group.group_name

        handle = all2all_manager.get_handle(all_to_all_args)

        prepare_finalize = PplxPrepareAndFinalize(
            handle,
            max_num_tokens=moe.max_num_tokens,
            num_local_experts=moe.num_local_experts,
            num_dispatchers=num_dispatchers,
        )
    elif moe.use_deepep_ht_kernels:
        assert moe.dp_size == all2all_manager.dp_world_size

        all_to_all_args = dict()
        handle = all2all_manager.get_handle(all_to_all_args)
        prepare_finalize = DeepEPHTPrepareAndFinalize(
            handle,
            num_dispatchers=all2all_manager.world_size,
            dp_size=all2all_manager.dp_world_size,
            rank_expert_offset=all2all_manager.rank * moe.num_local_experts,
        )

    elif moe.use_deepep_ll_kernels:
        assert quant_config is not None
        global_to_physical = physical_to_global = local_expert_global_ids = None
        if routing_tables is not None:
            (
                global_to_physical,
                physical_to_global,
                local_expert_global_ids,
            ) = routing_tables
        all_to_all_args = dict(
            max_num_tokens_per_dp_rank=moe.max_num_tokens,
            token_hidden_size=moe.hidden_dim,
            num_ep_ranks=all2all_manager.world_size,
            num_global_experts=moe.num_experts,
            num_local_experts=moe.num_experts // all2all_manager.world_size,
        )
        handle = all2all_manager.get_handle(all_to_all_args)

        # Note: We may want to use FP8 dispatch just to reduce
        # data movement.
        use_fp8_dispatch = (
            quant_config.quant_dtype == current_platform.fp8_dtype()
            and quant_config.block_shape == DEEPEP_QUANT_BLOCK_SHAPE
        )

        prepare_finalize = DeepEPLLPrepareAndFinalize(
            handle,
            max_tokens_per_rank=moe.max_num_tokens,
            num_dispatchers=all2all_manager.world_size,
            use_fp8_dispatch=use_fp8_dispatch,
            global_to_physical=global_to_physical,
            physical_to_global=physical_to_global,
            local_expert_global_ids=local_expert_global_ids,
        )

    return prepare_finalize
