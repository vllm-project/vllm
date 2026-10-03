# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared fixtures for the ECZmqConnector tests.

The connector is built from a `VllmConfig`, so the helpers here assemble the
smallest config that carries what it reads, and hand out workers bound to real
sockets on free ports (so the send/receive path is exercised for real, without
a GPU).
"""

import time
from typing import Any
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.config import VllmConfig
from vllm.config.ec_transfer import ECTransferConfig
from vllm.config.parallel import ParallelConfig
from vllm.utils.network_utils import get_open_port

WORKER_MODULE = "vllm.distributed.ec_transfer.ec_connector.zmq.worker"


def make_vllm_config(
    role: str = "ec_both",
    *,
    ec_port: int | None = None,
    ec_ip: str = "127.0.0.1",
    tensor_parallel_size: int = 1,
    data_parallel_rank: int = 0,
    extra_config: dict[str, Any] | None = None,
) -> Mock:
    """A config carrying only what the ZMQ connector reads."""
    config = Mock(spec=VllmConfig)
    config.parallel_config = ParallelConfig(
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_rank=data_parallel_rank,
    )
    config.ec_transfer_config = ECTransferConfig(
        ec_connector="ECZmqConnector",
        ec_role=role,
        ec_ip=ec_ip,
        ec_port=ec_port if ec_port is not None else get_open_port(),
        ec_connector_extra_config=extra_config or {},
    )
    config.model_config = Mock()
    return config


def wait_until(predicate, timeout: float = 5.0) -> bool:
    """Poll `predicate` until it holds; the transfer is asynchronous."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


@pytest.fixture
def vllm_config():
    """Factory for configs; see `make_vllm_config`."""
    return make_vllm_config


@pytest.fixture
def until():
    """Poll a predicate until it holds; see `wait_until`."""
    return wait_until


@pytest.fixture
def make_worker():
    """Factory for `ECZmqWorker`s that stages on the CPU.

    Rank lookups are patched because the tests run without a distributed
    environment; the device is CPU so the load path can be asserted on any
    host.
    """
    from vllm.distributed.ec_transfer.ec_connector.zmq.worker import ECZmqWorker

    workers: list[ECZmqWorker] = []

    def factory(
        role: str = "ec_both",
        *,
        tp_rank: int = 0,
        pcp_rank: int = 0,
        tp_size: int = 1,
        vllm_config: Mock | None = None,
        **config_kwargs: Any,
    ) -> ECZmqWorker:
        pcp_group = Mock()
        pcp_group.rank_in_group = pcp_rank
        with (
            patch(
                f"{WORKER_MODULE}.get_tensor_model_parallel_rank",
                return_value=tp_rank,
            ),
            patch(
                f"{WORKER_MODULE}.get_tensor_model_parallel_world_size",
                return_value=tp_size,
            ),
            patch(f"{WORKER_MODULE}.get_pcp_group", return_value=pcp_group),
        ):
            worker = ECZmqWorker(
                vllm_config or make_vllm_config(role, **config_kwargs),
                device=torch.device("cpu"),
            )
        workers.append(worker)
        return worker

    yield factory

    for worker in workers:
        worker.shutdown()


@pytest.fixture
def make_connector():
    """Factory for `ECZmqConnector`s, staging on the CPU.

    Same patching as `make_worker`: no distributed environment, and the load
    path targets the CPU so it can be asserted on any host.
    """
    from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
    from vllm.distributed.ec_transfer.ec_connector.zmq.connector import (
        ECZmqConnector,
    )

    connectors: list[ECZmqConnector] = []

    def factory(
        role: ECConnectorRole,
        vllm_config: Mock,
        *,
        tp_rank: int = 0,
        pcp_rank: int = 0,
        tp_size: int = 1,
    ) -> ECZmqConnector:
        pcp_group = Mock()
        pcp_group.rank_in_group = pcp_rank
        platform = Mock()
        platform.device_type = "cpu"
        with (
            patch(
                f"{WORKER_MODULE}.get_tensor_model_parallel_rank",
                return_value=tp_rank,
            ),
            patch(
                f"{WORKER_MODULE}.get_tensor_model_parallel_world_size",
                return_value=tp_size,
            ),
            patch(f"{WORKER_MODULE}.get_pcp_group", return_value=pcp_group),
            patch(f"{WORKER_MODULE}.current_platform", platform),
        ):
            connector = ECZmqConnector(vllm_config, role)
        connectors.append(connector)
        return connector

    yield factory

    for connector in connectors:
        connector.shutdown()
