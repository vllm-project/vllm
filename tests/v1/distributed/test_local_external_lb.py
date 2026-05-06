# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse

from vllm.entrypoints.openai.local_external_lb import (
    MultiPortExternalLBSupervisor,
    build_multi_port_external_lb_child_args,
    infer_multi_port_external_lb_start_rank,
)


def _make_args(**overrides) -> argparse.Namespace:
    base = {
        "host": None,
        "port": 8000,
        "data_parallel_multi_port_external_lb": True,
        "data_parallel_supervisor_port": 9256,
        "data_parallel_size": 8,
        "data_parallel_size_local": 4,
        "data_parallel_start_rank": None,
        "data_parallel_rank": None,
        "data_parallel_external_lb": False,
        "data_parallel_hybrid_lb": False,
        "api_server_count": None,
        "headless": False,
        "grpc": False,
        "uds": None,
        "ssl_keyfile": None,
        "ssl_certfile": None,
        "ssl_ca_certs": None,
        "node_rank": 1,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_infer_multi_port_external_lb_start_rank_uses_node_rank():
    args = _make_args()
    assert infer_multi_port_external_lb_start_rank(args) == 4


def test_build_multi_port_external_lb_child_args_sets_external_rank_server():
    args = _make_args(data_parallel_start_rank=8, api_server_count=None)
    child_args = build_multi_port_external_lb_child_args(args, local_rank=2)

    assert child_args.port == 8002
    assert child_args.data_parallel_rank == 10
    assert child_args.data_parallel_size_local == 1
    assert child_args.data_parallel_external_lb is True
    assert child_args.data_parallel_hybrid_lb is False
    assert child_args.data_parallel_multi_port_external_lb is False
    assert child_args.api_server_count == 1


def test_multi_port_external_lb_supervisor_aggregates_health():
    supervisor = MultiPortExternalLBSupervisor(_make_args())

    supervisor.children_healthy = True

    assert supervisor.is_healthy() is True


def test_multi_port_external_lb_supervisor_is_unhealthy_after_shutdown_requested():
    supervisor = MultiPortExternalLBSupervisor(_make_args())
    supervisor.children_healthy = True
    supervisor._stop_requested.set()

    assert supervisor.is_healthy() is False
