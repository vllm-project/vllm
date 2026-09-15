# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import socket

import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.distributed.utils import StatelessProcessGroup
from vllm.utils.network_utils import get_ip


def _create_stateless_process_group() -> StatelessProcessGroup:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    listen_socket: socket.socket | None = None
    endpoint: list[str | int | None] = [None, None]

    if rank == 0:
        ip = get_ip()
        listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listen_socket.bind((ip, 0))
        listen_socket.listen()
        endpoint[:] = [ip, listen_socket.getsockname()[1]]

    dist.broadcast_object_list(endpoint, src=0)
    ip, port = endpoint
    assert isinstance(ip, str)
    assert isinstance(port, int)
    return StatelessProcessGroup.create(
        ip,
        port,
        rank,
        world_size,
        listen_socket=listen_socket,
    )


def _run_test(pg):
    test_result = all(in_the_same_node_as(pg, source_rank=0))

    expected = os.environ.get("VLLM_TEST_SAME_HOST", "1") == "1"
    assert test_result == expected, f"Expected {expected}, got {test_result}"
    if pg == dist.group.WORLD:
        print("Same node test passed! when using torch distributed!")
    else:
        print("Same node test passed! when using StatelessProcessGroup!")


if __name__ == "__main__":
    dist.init_process_group(backend="gloo")
    stateless_pg = _create_stateless_process_group()

    for pg in [dist.group.WORLD, stateless_pg]:
        if os.environ.get("VLLM_TEST_WITH_DEFAULT_DEVICE_SET", "0") == "1":
            default_devices = ["cpu"]
            if torch.cuda.is_available():
                default_devices.append("cuda")
            for device in default_devices:
                torch.set_default_device(device)
                _run_test(pg)
        else:
            _run_test(pg)
