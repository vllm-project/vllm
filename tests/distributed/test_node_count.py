# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import socket

import torch.distributed as dist

from vllm.distributed.parallel_state import _node_count
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


if __name__ == "__main__":
    dist.init_process_group(backend="gloo")
    stateless_pg = _create_stateless_process_group()

    for pg in [dist.group.WORLD, stateless_pg]:
        test_result = _node_count(pg)

        # Expected node count based on environment variable)
        expected = int(os.environ.get("NUM_NODES", "1"))

        assert test_result == expected, f"Expected {expected} nodes, got {test_result}"

        if pg == dist.group.WORLD:
            print(
                f"Node count test passed! Got {test_result} nodes "
                f"when using torch distributed!"
            )
        else:
            print(
                f"Node count test passed! Got {test_result} nodes "
                f"when using StatelessProcessGroup!"
            )
