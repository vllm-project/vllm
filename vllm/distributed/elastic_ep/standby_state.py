# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.distributed.parallel_state import (
    _init_stateless_group,
    _node_count,
    get_pp_group,
    get_tp_group,
    get_world_group,
)
from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator

_STANDBY_WORLD: StatelessGroupCoordinator | None = None
_STANDBY_WORLD_NODE_COUNT: int | None = None
_STANDBY_DP: StatelessGroupCoordinator | None = None
_STANDBY_EP: StatelessGroupCoordinator | None = None
_STANDBY_EPLB: StatelessGroupCoordinator | None = None


def get_standby_dp_group() -> StatelessGroupCoordinator | None:
    return _STANDBY_DP


def get_standby_ep_group() -> StatelessGroupCoordinator | None:
    return _STANDBY_EP


def get_standby_eplb_group() -> StatelessGroupCoordinator | None:
    return _STANDBY_EPLB


def get_standby_world_group() -> StatelessGroupCoordinator | None:
    return _STANDBY_WORLD


def _build_surviving_rank_tensor(
    new_dp_size: int,
    pp_size: int,
    tp_size: int,
    dead_dp_ranks: set[int],
    current_world_ranks: list[int] | None = None,
) -> torch.Tensor:
    """Build a rank tensor containing only the surviving world ranks.

    Returns a tensor of shape (*, new_dp_size, pp_size, tp_size) with the
    actual world-rank values of the surviving engines, preserving the
    original ordering and skipping dead DP ranks.

    Args:
        current_world_ranks: The actual world ranks of the current group.
            If None, fetched from get_world_group().ranks. After previous
            scale-downs these may be non-contiguous (e.g. [0, 2, 3]).
    """
    old_dp_size = new_dp_size + len(dead_dp_ranks)
    if current_world_ranks is None:
        current_world_ranks = get_world_group().ranks
    full = torch.tensor(current_world_ranks).reshape(
        -1, old_dp_size, pp_size, tp_size
    )
    surviving_dp_indices = [
        i for i in range(old_dp_size) if i not in dead_dp_ranks
    ]
    assert len(surviving_dp_indices) == new_dp_size
    return full[:, surviving_dp_indices, :, :]


def create_standby_groups(
    new_dp_size: int,
    new_world_size_across_dp: int,
    master_ip: str,
    coord_store_port: int,
    enable_eplb: bool = True,
    backend: str | None = None,
    dead_dp_ranks: set[int] | None = None,
) -> None:
    global \
        _STANDBY_WORLD, \
        _STANDBY_WORLD_NODE_COUNT, \
        _STANDBY_DP, \
        _STANDBY_EP, \
        _STANDBY_EPLB

    from vllm.distributed.utils import get_cached_tcp_store_client

    world_group = get_world_group()
    assert isinstance(world_group, StatelessGroupCoordinator)
    backend = backend or world_group.backend

    coord_store = get_cached_tcp_store_client(master_ip, coord_store_port)

    tp_size = get_tp_group().world_size
    pp_size = get_pp_group().world_size

    if dead_dp_ranks:
        # Fault-triggered: surviving world ranks are non-contiguous.
        all_ranks = _build_surviving_rank_tensor(
            new_dp_size, pp_size, tp_size, dead_dp_ranks
        )
        surviving_world_ranks = sorted(all_ranks.flatten().tolist())
    else:
        assert new_world_size_across_dp == (
            torch.distributed.get_world_size() * new_dp_size
        )
        all_ranks = torch.arange(new_world_size_across_dp).reshape(
            -1, new_dp_size, pp_size, tp_size
        )
        surviving_world_ranks = list(range(new_world_size_across_dp))

    standby_world_ranks = [surviving_world_ranks]
    _STANDBY_WORLD = _init_stateless_group(
        standby_world_ranks,
        "world",
        master_ip,
        backend,
        use_device_communicator=False,
        coord_store=coord_store,
    )
    _STANDBY_WORLD_NODE_COUNT = _node_count(_STANDBY_WORLD.tcp_store_group)

    standby_dp_ranks = all_ranks.transpose(1, 3).reshape(-1, new_dp_size).unbind(0)
    standby_dp_ranks = [x.tolist() for x in standby_dp_ranks]
    _STANDBY_DP = _init_stateless_group(
        standby_dp_ranks, "dp", master_ip, backend, coord_store=coord_store
    )

    standby_ep_ranks = (
        all_ranks.transpose(1, 2).reshape(-1, new_dp_size * tp_size).unbind(0)
    )
    standby_ep_ranks = [x.tolist() for x in standby_ep_ranks]
    _STANDBY_EP = _init_stateless_group(
        standby_ep_ranks, "ep", master_ip, backend, coord_store=coord_store
    )

    if enable_eplb:
        _STANDBY_EPLB = _init_stateless_group(
            standby_ep_ranks,
            "eplb",
            master_ip,
            backend,
            coord_store=coord_store,
        )


def pop_standby_groups() -> dict:
    """Return all standby groups and clear the standby state."""
    global \
        _STANDBY_WORLD, \
        _STANDBY_WORLD_NODE_COUNT, \
        _STANDBY_DP, \
        _STANDBY_EP, \
        _STANDBY_EPLB

    result = dict(
        world=_STANDBY_WORLD,
        dp=_STANDBY_DP,
        ep=_STANDBY_EP,
        eplb=_STANDBY_EPLB,
        node_count=_STANDBY_WORLD_NODE_COUNT,
    )
    _STANDBY_WORLD = None
    _STANDBY_WORLD_NODE_COUNT = None
    _STANDBY_DP = None
    _STANDBY_EP = None
    _STANDBY_EPLB = None
    return result
