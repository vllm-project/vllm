# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Placement of an engine's offloading workers across nodes."""

import multiprocessing as mp
import traceback
from multiprocessing.queues import Queue

import pytest
import torch

from vllm.utils.network_utils import get_open_port
from vllm.v1.kv_offload.cpu.node_topology import replica_placement

RESULT_TIMEOUT_S = 120


def _locate(
    rank: int, group_size: int, world_size: int, port: int, result: Queue
) -> None:
    """Join a real gloo world group and locate this worker in it."""
    try:
        from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            get_world_group,
            init_distributed_environment,
        )
        from vllm.v1.kv_offload.cpu.node_topology import multi_node_topology

        with set_current_vllm_config(VllmConfig(device_config=DeviceConfig("cpu"))):
            init_distributed_environment(
                world_size=group_size,
                rank=rank,
                distributed_init_method=f"tcp://127.0.0.1:{port}",
                local_rank=rank,
                backend="gloo",
            )
        try:
            located = multi_node_topology(world_size)
            # Every rank reached the same collectives: the group still works.
            get_world_group().barrier()
        finally:
            destroy_distributed_environment()
        result.put((rank, located, None))
    except Exception:
        result.put((rank, None, traceback.format_exc()))


def _locate_on_every_rank(group_size: int, world_size: int) -> list:
    ctx = mp.get_context("spawn")
    result: Queue = ctx.Queue()
    port = get_open_port()
    procs = [
        ctx.Process(target=_locate, args=(rank, group_size, world_size, port, result))
        for rank in range(group_size)
    ]
    for proc in procs:
        proc.start()
    try:
        reports = sorted(result.get(timeout=RESULT_TIMEOUT_S) for _ in procs)
    finally:
        for proc in procs:
            proc.join(timeout=RESULT_TIMEOUT_S)
            if proc.is_alive():
                proc.kill()
    errors = [error for _, _, error in reports if error is not None]
    assert not errors, "\n".join(errors)
    assert all(proc.exitcode == 0 for proc in procs)
    return [located for _, located, _ in reports]


@pytest.mark.skipif(
    not torch.distributed.is_gloo_available(), reason="gloo backend required"
)
@pytest.mark.parametrize(
    "group_size, world_size, spans_nodes",
    [
        # The world group is the engine's group.
        (2, 2, False),
        # The world group holds two data parallel replicas of two workers.
        (4, 2, False),
        # A group that is not made of whole replicas cannot be identified.
        (3, 2, None),
    ],
)
def test_topology_of_a_single_node_engine(
    group_size: int, world_size: int, spans_nodes: bool | None
):
    located = _locate_on_every_rank(group_size, world_size)

    assert located == [(None, spans_nodes)] * group_size


def _nodes(*node_of_rank: int):
    """same_node_as of a group whose rank r runs on node node_of_rank[r]."""
    calls: list[int] = []

    def same_node_as(source_rank: int) -> list[bool]:
        calls.append(source_rank)
        return [node == node_of_rank[source_rank] for node in node_of_rank]

    return same_node_as, calls


def test_placement_within_an_engine_spanning_nodes():
    same_node_as, calls = _nodes(0, 1, 1)

    assert replica_placement(2, 3, 3, same_node_as) == (False, frozenset({1, 2}))
    assert calls == [0]


@pytest.mark.parametrize("rank", range(6))
def test_placement_ignores_single_node_replicas(rank: int):
    # The world group holds two single-node replicas of three workers each.
    same_node_as, calls = _nodes(0, 0, 0, 1, 1, 1)

    assert replica_placement(rank, 3, 6, same_node_as) is None
    assert calls == [0, 3]


def test_placement_of_a_replica_spanning_nodes():
    # Replica 1 (ranks 3-5) puts rank 5 on another node.
    same_node_as, calls = _nodes(0, 0, 0, 1, 1, 2)

    assert replica_placement(4, 3, 6, same_node_as) == (True, frozenset({2}))
    assert replica_placement(1, 3, 6, same_node_as) is None
    # Every rank checks every replica, in order.
    assert calls == [0, 3, 0, 3]
