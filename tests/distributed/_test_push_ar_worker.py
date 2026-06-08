# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Worker helper for push allreduce unit tests.
Run via torch.multiprocessing.spawn from test_push_all_reduce.py.

Provides init/teardown helpers that create separate gloo (CPU) and
nccl (device) process groups for PushAllReduce (which needs gloo for
IPC handle exchange) and NCCL reference reduction (which needs nccl).
"""

import os
import socket

import torch
import torch.distributed as dist


def find_free_port() -> int:
    """Find a free TCP port for distributed init."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# Global references to process groups created by init_groups
_cpu_group = None
_nccl_group = None


def init_groups(rank: int, world_size: int, port: int):
    """Initialize gloo (CPU) and nccl process groups.

    PushAllReduce uses the gloo group for IPC handle exchange.
    NCCL group is used for reference allreduce.
    """
    global _cpu_group, _nccl_group

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    torch.cuda.set_device(rank)

    dist.init_process_group(
        backend="gloo", rank=rank, world_size=world_size
    )
    _cpu_group = dist.group.WORLD

    # Create a separate NCCL group for reference allreduce
    _nccl_group = dist.new_group(backend="nccl")


def get_cpu_group():
    return _cpu_group


def get_nccl_group():
    return _nccl_group


def teardown():
    """Clean up distributed groups."""
    dist.destroy_process_group()
