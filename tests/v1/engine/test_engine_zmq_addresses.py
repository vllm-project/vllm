# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import socket
from unittest.mock import MagicMock

import pytest

from vllm.config import ParallelConfig, VllmConfig
from vllm.utils.network_utils import split_zmq_path
from vllm.v1.engine.utils import get_engine_zmq_addresses


def test_multi_api_server_zmq_addresses_are_unique_and_bindable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each independent ``get_open_port`` call observes a port as free but
    does not hold it, so concurrent allocations can return duplicates. The
    address allocator must hand each API server a distinct, bindable port
    that also stays clear of the DP master's reserved range."""
    monkeypatch.setenv("VLLM_PORT", "30000")
    # Place DP master inside the allocation window (start_port = 31000) so
    # the test fails if the allocator forgets to skip the reserved range.
    monkeypatch.setenv("VLLM_DP_MASTER_PORT", "31005")

    parallel_config = MagicMock(spec=ParallelConfig)
    # dp_size != dp_size_local forces client_local_only=False (TCP path).
    parallel_config.data_parallel_size = 4
    parallel_config.data_parallel_size_local = 1
    parallel_config.data_parallel_rank_local = None
    parallel_config.data_parallel_master_ip = "127.0.0.1"
    parallel_config.local_engines_only = False
    parallel_config.enable_elastic_ep = False
    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.parallel_config = parallel_config

    addresses = get_engine_zmq_addresses(vllm_config, num_api_servers=8)

    ports = [int(split_zmq_path(a)[2]) for a in addresses.inputs + addresses.outputs]
    assert len(ports) == 16
    assert len(set(ports)) == 16
    assert all(not 31005 <= port < 31015 for port in ports)

    sockets: list[socket.socket] = []
    try:
        for port in ports:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", port))
            sockets.append(s)
    finally:
        for s in sockets:
            s.close()


def test_multi_api_server_zmq_addresses_local_only_uses_ipc() -> None:
    """When the client only talks to colocated engines, addresses must be
    unique IPC paths. No TCP port allocation needed."""
    parallel_config = MagicMock(spec=ParallelConfig)
    # dp_size == dp_size_local forces client_local_only=True.
    parallel_config.data_parallel_size = 4
    parallel_config.data_parallel_size_local = 4
    parallel_config.data_parallel_rank_local = None
    parallel_config.data_parallel_master_ip = "127.0.0.1"
    parallel_config.local_engines_only = False
    parallel_config.enable_elastic_ep = False
    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.parallel_config = parallel_config

    addresses = get_engine_zmq_addresses(vllm_config, num_api_servers=4)

    all_addresses = addresses.inputs + addresses.outputs
    assert len(all_addresses) == 8
    assert len(set(all_addresses)) == 8
    assert all(split_zmq_path(addr)[0] == "ipc" for addr in all_addresses)
