# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense (non-expert) weight P2P between active and waking ranks.

Used by flash_epscale L2 wake to refill dense weights on a rank that
just remapped its CuMem memory (vaddrs preserved, contents = garbage).
Mirrors the filter used by elastic_ep's batch_transfer_weights so the
two paths cover the same parameter set.
"""

from collections.abc import Iterable, Sequence

import torch
import torch.nn as nn
from torch.distributed import P2POp

import vllm.distributed.parallel_state as parallel_state
from vllm.distributed.parallel_state import GroupCoordinator


def get_max_dp_group() -> GroupCoordinator:
    """Return the always-preserved max DP communicator.

    `parallel_state._DP_MAX` is set on the first NCCL split and kept alive
    even when `_DP` is shrunk via `_sync_dp_group_to_active_ep_size`. Read
    it directly to do P2P across all original DP ranks without mutating
    any DP global state. Falls back to `_DP` if no split ever happened.
    """
    if parallel_state._DP_MAX is not None:
        return parallel_state._DP_MAX
    assert parallel_state._DP is not None, "DP group not initialized"
    return parallel_state._DP


def _collect_dense_params(
    model: nn.Module,
    expert_weights: Sequence[Iterable[torch.Tensor]],
) -> list[torch.Tensor]:
    expert_ptrs: set[int] = set()
    for group in expert_weights:
        for w in group:
            expert_ptrs.add(w.data_ptr())

    params: list[torch.Tensor] = []
    for name, param in model.state_dict().items():
        if name.endswith("expert_map") or "._shared_experts" in name:
            continue
        if param.data_ptr() in expert_ptrs:
            continue
        params.append(param.data)
    assert params, "no dense params found to transfer"
    return params


def transfer_dense_to_waking_ranks(
    model: nn.Module,
    expert_weights: Sequence[Iterable[torch.Tensor]],
    dp_group: GroupCoordinator,
    waking_dp_ranks: list[int],
) -> None:
    """Refill dense weights on waking ranks from one active rank in the DP group.

    Must be called on every rank of `dp_group` in lockstep — active
    non-sender ranks return early after posting nothing. NCCL pairs the
    isend/irecv ops by (sender, receiver, op_index).
    """
    my_rank = dp_group.rank_in_group
    dp_size = dp_group.world_size
    waking = set(waking_dp_ranks)
    is_waking = my_rank in waking
    # Lowest-numbered active rank acts as sender. With suffix-only sleeping
    # this is always rank 0; kept general in case the convention relaxes.
    sender = next(r for r in range(dp_size) if r not in waking)
    if my_rank != sender and not is_waking:
        return

    params = _collect_dense_params(model, expert_weights)

    ops: list[P2POp] = []
    if my_rank == sender:
        for w in params:
            for peer in waking_dp_ranks:
                op = object.__new__(P2POp)
                op.op = torch.distributed.isend
                op.tensor = w
                op.group_peer = peer
                ops.append(op)
    else:
        for w in params:
            op = object.__new__(P2POp)
            op.op = torch.distributed.irecv
            op.tensor = w
            op.group_peer = sender
            ops.append(op)

    device_comm = dp_group.device_communicator
    if device_comm is None:
        raise RuntimeError("dp_group has no device_communicator")
    device_comm.batch_isend_irecv(ops)
    torch.accelerator.synchronize()
