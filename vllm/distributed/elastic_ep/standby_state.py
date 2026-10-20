# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from typing import TYPE_CHECKING

import torch

from vllm.distributed.device_communicators.pynccl import defer_comm_warmup_on_rocm
from vllm.distributed.parallel_state import (
    _init_stateless_group,
    _node_count,
    get_pcp_group,
    get_pp_group,
    get_tp_group,
    get_world_group,
)
from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

logger = init_logger(__name__)

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


def _shrink_source(
    ranks: list[list[int]],
) -> "tuple[PyNcclCommunicator, list[int]] | None":
    """Find a live communicator these ranks are a subset of, if asked.

    Only for the way down, and only under a flag. A reconfiguration that drops
    ranks builds the smaller groups out of the ranks that stay, which is
    exactly the shape ncclCommShrink serves and the shape ncclCommSplit cannot,
    since split is collective over every rank of the parent and the ranks on
    their way out never call this.

    The reason to bother: standing up a fresh communicator makes RCCL build a
    new P2P transport, and on MI355 the IPC export that goes with it is refused
    outright once a process has reconfigured once, which is where this group
    fails. Deriving from a live parent inherits the transport. Measured
    locally, a two rank child of a four rank parent allocates 15 shareable
    buffers this way against 64 either fresh or shrunk without
    NCCL_COMM_SHRINK_SHARE_RESOURCES=1, so the variable is the whole point and
    the flag is worth nothing without it.

    It is 15 and not 0, and those 15 are the 6 MiB and 2 MiB buffers that get
    refused, so this may well not be enough. What a local box cannot answer is
    whether they are exported in the same context as the refused ones, since
    the refusal lands inside ncclCommInitRank during a bootstrap this path does
    not perform.
    """
    if os.getenv("VLLM_EEP_USE_COMM_SHRINK") != "1":
        return None
    try:
        return _find_shrink_source(ranks)
    except Exception:
        # A lever that answers one question must not be the reason a pass fails
        # for an unrelated one, and the caller has a working path either way.
        logger.warning("[EEP-DEBUG] shrink source lookup failed", exc_info=True)
        return None


def _find_shrink_source(
    ranks: list[list[int]],
) -> "tuple[PyNcclCommunicator, list[int]] | None":
    from vllm.distributed.parallel_state import get_ep_group

    parent_group = get_ep_group()
    parent = getattr(
        getattr(parent_group, "device_communicator", None), "pynccl_comm", None
    )
    if parent is None or not parent.available or parent.disabled:
        return None
    if not parent.nccl.has_symbol("ncclCommShrink"):
        logger.warning(
            "[EEP-DEBUG] VLLM_EEP_USE_COMM_SHRINK is set but the loaded "
            "RCCL does not export ncclCommShrink, building fresh instead"
        )
        return None

    world_group = get_world_group()
    mine = next((group for group in ranks if world_group.rank in group), None)
    if mine is None:
        return None

    parent_ranks = list(parent_group.ranks)
    if not set(mine).issubset(parent_ranks):
        return None
    exclude = [parent_ranks.index(rank) for rank in parent_ranks if rank not in mine]
    if not exclude:
        return None
    logger.warning(
        "[EEP-DEBUG] shrinking a live %d rank communicator to %d by dropping %s "
        "instead of building a fresh one",
        parent.world_size,
        parent.world_size - len(exclude),
        exclude,
    )
    return parent, exclude


def create_standby_groups(
    new_dp_size: int,
    new_world_size_across_dp: int,
    master_ip: str,
    coord_store_port: int,
    use_all2all: bool,
    enable_eplb: bool = True,
    backend: str | None = None,
) -> None:
    global \
        _STANDBY_WORLD, \
        _STANDBY_WORLD_NODE_COUNT, \
        _STANDBY_DP, \
        _STANDBY_EP, \
        _STANDBY_EPLB

    from vllm.distributed.utils import get_cached_tcp_store_client

    assert new_world_size_across_dp == torch.distributed.get_world_size() * new_dp_size
    world_group = get_world_group()
    assert isinstance(world_group, StatelessGroupCoordinator)
    backend = backend or world_group.backend

    coord_store = get_cached_tcp_store_client(master_ip, coord_store_port)

    standby_world_ranks = [list(range(new_world_size_across_dp))]
    _STANDBY_WORLD = _init_stateless_group(
        standby_world_ranks,
        "world",
        master_ip,
        backend,
        use_device_communicator=False,
        coord_store=coord_store,
    )
    _STANDBY_WORLD_NODE_COUNT = _node_count(_STANDBY_WORLD.tcp_store_group)

    tp_size = get_tp_group().world_size
    pcp_size = get_pcp_group().world_size
    pp_size = get_pp_group().world_size

    all_ranks = torch.arange(new_world_size_across_dp).reshape(
        -1, new_dp_size, pp_size, pcp_size, tp_size
    )
    standby_dp_ranks = all_ranks.transpose(1, 4).reshape(-1, new_dp_size).unbind(0)
    standby_dp_ranks = [x.tolist() for x in standby_dp_ranks]

    # Deferred to commit so the warm-up runs while the engine is paused.
    with defer_comm_warmup_on_rocm():
        _STANDBY_DP = _init_stateless_group(
            standby_dp_ranks,
            "dp",
            master_ip,
            backend,
            coord_store=coord_store,
            shrink_from=_shrink_source(standby_dp_ranks),
        )

        standby_ep_ranks = (
            all_ranks.transpose(1, 2)
            .reshape(-1, new_dp_size * pcp_size * tp_size)
            .unbind(0)
        )
        standby_ep_ranks = [x.tolist() for x in standby_ep_ranks]
        _STANDBY_EP = _init_stateless_group(
            standby_ep_ranks,
            "ep",
            master_ip,
            backend,
            coord_store,
            use_all2all=use_all2all,
            shrink_from=_shrink_source(standby_ep_ranks),
        )

        if enable_eplb:
            _STANDBY_EPLB = _init_stateless_group(
                standby_ep_ranks,
                "eplb",
                master_ip,
                backend,
                coord_store=coord_store,
                shrink_from=_shrink_source(standby_ep_ranks),
            )


def probe_stateless_bootstrap(
    new_dp_size: int,
    new_world_size_across_dp: int,
    master_ip: str,
    coord_store_port: int,
    backend: str | None = None,
) -> None:
    """Build one throwaway stateless group and tear it down immediately.

    Diagnostic only. Creating the standby groups succeeds 22 times out of 22
    for the first reconfiguration a process performs, and is refused 10 times
    out of 10 for the second. But in this test the second reconfiguration is
    also the only one that scales down and the only one that asks for a
    smaller group, so the logs cannot say which of those three decides it.

    This spends a second bootstrap right after the standby groups are built,
    on the way up and at the larger size, so the direction and the size are
    held fixed and only the count changes. If the refusal follows the number
    of bootstraps a process has done, this one is refused while the standby
    groups a moment earlier were not.

    It runs after create_standby_groups rather than before it, because every
    rank has to take part and the newly started ranks only arrive in time for
    that call. Placed ahead of it, this waits for ranks that are still coming
    up and the scale command passes its deadline.

    The group is built exactly as the standby EP group is, same ranks, same
    backend, same store, deferred warm-up and all, and differs only in its
    name, which scopes its rendezvous keys.
    """
    from vllm.distributed.utils import get_cached_tcp_store_client

    world_group = get_world_group()
    assert isinstance(world_group, StatelessGroupCoordinator)
    backend = backend or world_group.backend
    coord_store = get_cached_tcp_store_client(master_ip, coord_store_port)

    tp_size = get_tp_group().world_size
    pcp_size = get_pcp_group().world_size
    pp_size = get_pp_group().world_size
    all_ranks = torch.arange(new_world_size_across_dp).reshape(
        -1, new_dp_size, pp_size, pcp_size, tp_size
    )
    probe_ranks = (
        all_ranks.transpose(1, 2)
        .reshape(-1, new_dp_size * pcp_size * tp_size)
        .unbind(0)
    )
    probe_ranks = [x.tolist() for x in probe_ranks]

    with defer_comm_warmup_on_rocm():
        group = _init_stateless_group(
            probe_ranks,
            "eepprobe",
            master_ip,
            backend,
            coord_store=coord_store,
        )
    group.destroy()


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
