# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.distributed import divide


def extra_groups_for_head_shards(ngroups: int, tp_size: int):
    """Compute the increase in group numbers to account for
    replication in order to accompany the head shards."""

    # in the case ngoups % tp_size == 0, this will be zero
    if ngroups % tp_size == 0:
        return 0

    # for n_groups == 1, this is exactly tp_size - n_groups
    return tp_size - ngroups


def get_mamba_state_shape(
    intermediate_size: int,
    tp_world_size: int,
    n_groups: int,
    num_heads: int,
    head_dim: int,
    state_size: int,
    conv_kernel: int,
    use_v1: bool = True,
) -> tuple[tuple[int, int], tuple[int, int, int]]:
    """ Get the shape of mamba state."""

    # if n_groups is not divisible by world_size, need to extend the shards
    # to ensure all groups needed by a head is sharded along with it
    n_groups = (n_groups +
                extra_groups_for_head_shards(n_groups, tp_world_size))

    # - heads and n_groups are TP-ed
    conv_dim = (intermediate_size + 2 * n_groups * state_size)
    # contiguous along 'dim' axis
    conv_state_shape = (
        conv_kernel - 1,
        divide(conv_dim, tp_world_size),
    )

    if not use_v1:
        conv_state_shape = (conv_state_shape[1], conv_state_shape[0])

    # These are not TP-ed as they depend on A, dt_bias, D
    # - they are typically small
    #   e.g., (h_heads, head_dim, state_size) = (128, 64, 128)
    temporal_state_shape = (
        divide(num_heads, tp_world_size),
        head_dim,
        state_size,
    )

    return conv_state_shape, temporal_state_shape
