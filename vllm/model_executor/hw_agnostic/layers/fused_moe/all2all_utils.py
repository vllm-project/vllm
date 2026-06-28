# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""All2All transport selection for the hw-agnostic FusedMoE.

The vendored path keeps a single transport — ``allgather_reducescatter``,
the naive AllGather/ReduceScatter pair implemented over
``torch.distributed`` collectives. Vendor-fused transports (DeepEP, Mori,
NIXL, FlashInfer-NVL) are out of scope here; users that request them are
rejected by ``MoERunner._validate_supported_settings``.
"""

from vllm.distributed import get_ep_group
from vllm.logger import init_logger
from vllm.model_executor.hw_agnostic.layers.fused_moe.config import (
    FusedMoEConfig,
)
from vllm.model_executor.hw_agnostic.layers.fused_moe.modular_kernel import (
    FusedMoEPrepareAndFinalizeModular,
)
from vllm.model_executor.hw_agnostic.layers.fused_moe.prepare_finalize import (  # noqa: E501
    make_moe_prepare_and_finalize_naive_dp_ep,
    make_moe_prepare_and_finalize_no_dp_ep,
)

logger = init_logger(__name__)


def maybe_roundup_layer_hidden_size(
    hidden_size: int,
    *_,
    **__,
) -> int:
    # The naive AG/RS transport places no hidden-size alignment constraint;
    # round-ups were vendor-specific (DeepEP/NIXL).
    return hidden_size


def maybe_make_prepare_finalize(
    moe: FusedMoEConfig,
) -> FusedMoEPrepareAndFinalizeModular:
    """Pick the vendored P/F for this deployment.

    - ``dp_size > 1``: AllGather/ReduceScatter naive DP/EP P/F. Covers both
      DP-without-EP (broadcast tokens, gather output) and DP+EP with the
      ``allgather_reducescatter`` transport.
    - ``dp_size == 1``: NoDPEP P/F (single-rank or TP-only or TP+EP without
      DP; ``expert_map`` masks remote experts inside the kernel).
    """
    if moe.moe_parallel_config.dp_size > 1:
        all2all_manager = get_ep_group().device_communicator.all2all_manager
        assert all2all_manager is not None
        return make_moe_prepare_and_finalize_naive_dp_ep(
            is_sequence_parallel=moe.moe_parallel_config.is_sequence_parallel,
            num_dispatchers=all2all_manager.world_size,
        )
    return make_moe_prepare_and_finalize_no_dp_ep()
