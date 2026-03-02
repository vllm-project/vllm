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


def create_standby_groups(
    new_dp_size: int,
    new_world_size_across_dp: int,
    master_ip: str,
    world_group_ports: list[list[int]],
    dp_group_ports: list[list[int]],
    ep_group_ports: list[list[int]],
    eplb_group_ports: list[list[int]] | None = None,
    backend: str | None = None,
) -> None:
    global \
        _STANDBY_WORLD, \
        _STANDBY_WORLD_NODE_COUNT, \
        _STANDBY_DP, \
        _STANDBY_EP, \
        _STANDBY_EPLB

    assert new_world_size_across_dp == torch.distributed.get_world_size() * new_dp_size
    world_group = get_world_group()
    assert isinstance(world_group, StatelessGroupCoordinator)
    backend = backend or world_group.backend

    standby_world_ranks = [list(range(new_world_size_across_dp))]
    _STANDBY_WORLD = _init_stateless_group(
        standby_world_ranks,
        "world",
        world_group_ports,
        master_ip,
        backend,
        use_device_communicator=False,
    )
    _STANDBY_WORLD_NODE_COUNT = _node_count(_STANDBY_WORLD.tcp_store_group)

    tp_size = get_tp_group().world_size
    pp_size = get_pp_group().world_size

    all_ranks = torch.arange(new_world_size_across_dp).reshape(
        -1, new_dp_size, pp_size, tp_size
    )
    standby_dp_ranks = all_ranks.transpose(1, 3).reshape(-1, new_dp_size).unbind(0)
    standby_dp_ranks = [x.tolist() for x in standby_dp_ranks]
    _STANDBY_DP = _init_stateless_group(
        standby_dp_ranks, "dp", dp_group_ports, master_ip, backend
    )

    standby_ep_ranks = (
        all_ranks.transpose(1, 2).reshape(-1, new_dp_size * tp_size).unbind(0)
    )
    standby_ep_ranks = [x.tolist() for x in standby_ep_ranks]
    _STANDBY_EP = _init_stateless_group(
        standby_ep_ranks, "ep", ep_group_ports, master_ip, backend
    )

    if eplb_group_ports is not None:
        _STANDBY_EPLB = _init_stateless_group(
            standby_ep_ranks, "eplb", eplb_group_ports, master_ip, backend
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
