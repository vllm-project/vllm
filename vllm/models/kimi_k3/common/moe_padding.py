# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared MoE intermediate sizing for the Kimi K3 model."""

from vllm.config import VllmConfig


def effective_moe_tp_size(vllm_config: VllmConfig) -> int:
    """Number of ways the MoE intermediate dimension is actually sharded.

    Mirrors ``FusedMoEParallelConfig.make()``: the experts are sharded over
    ``dp_size * pcp_size * tp_size`` devices, unless expert parallelism is
    enabled, in which case each device owns whole experts and the intermediate
    is not split at all.
    """
    parallel_config = vllm_config.parallel_config
    flatten_tp_size = (
        parallel_config.data_parallel_size
        * parallel_config.prefill_context_parallel_size
        * parallel_config.tensor_parallel_size
    )
    if flatten_tp_size > 1 and parallel_config.enable_expert_parallel:
        return 1
    return flatten_tp_size


def padded_moe_intermediate_size(
    moe_intermediate_size: int,
    moe_tp_size: int,
    min_moe_intermediate_per_partition: int,
) -> int:
    """Round the MoE intermediate size up to the minimum shard width.

    The MoE kernels want at least ``min_moe_intermediate_per_partition``
    columns per shard, so a narrow intermediate is padded before it is split.
    ``moe_tp_size`` must be the effective MoE shard count from
    :func:`effective_moe_tp_size`, not the tensor-parallel world size: with
    expert parallelism the intermediate is never split, so padding it would
    allocate the padding in full on every rank (3072 -> 4096 at TP16,
    3072 -> 8192 at TP32) for no benefit.

    Args:
        moe_intermediate_size: Unpadded MoE intermediate size from the config.
        moe_tp_size: Number of shards the intermediate is split into.
        min_moe_intermediate_per_partition: Minimum columns per shard.

    Returns:
        The intermediate size the expert weights are allocated with.
    """
    if moe_tp_size <= 1:
        return moe_intermediate_size
    if moe_intermediate_size // moe_tp_size >= min_moe_intermediate_per_partition:
        return moe_intermediate_size
    return min_moe_intermediate_per_partition * moe_tp_size
