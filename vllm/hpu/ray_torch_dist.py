"""This file is modeled after ray/python/ray/train/torch/config.py
The logics are duplicated right now to allow maximum flexibility for
setting up PyTorch DDP process groups outside the context of Ray Train.
Eventually, these use cases should be consolidated.
"""

from abc import ABC
from collections import defaultdict
from datetime import timedelta
import os
import torch
import torch.distributed as dist
from typing import Callable, List, T

import ray
from ray.actor import ActorHandle
from ray.train._internal.utils import get_address_and_port
from ray.train.constants import DEFAULT_NCCL_SOCKET_IFNAME
from ray.air._internal.torch_utils import get_device


class TorchDistributedWorker(ABC):
    """Defines the interfaces required by the init_torch_dist_process_group().
    This is modeled after RayTrainerWorker, which allows arbitrary functions
    to be executed on a remote DDP worker.
    """

    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Executes the input function and returns the output.
        Args:
            func: The function to execute.
            args, kwargs: The arguments to pass into func.
        """
        return func(*args, **kwargs)


def _init_torch_distributed(
    init_method: str,
    backend: str,
    rank: int,
    world_size: int,
    local_rank: int,
    local_world_size: int,
    master_addr: str,
    master_port: str,
    gpu_ids: List[int],
    **init_process_group_kwargs,
):
    """Initialize torch distributed backend"""
    if init_method == "env":
        os.environ["MASTER_ADDR"] = str(master_addr)
        os.environ["MASTER_PORT"] = str(master_port)
        url = "env://"
    elif init_method == "tcp":
        url = f"tcp://{master_addr}:{master_port}"
    else:
        raise ValueError(
            f"The provided init_method ("
            f"{init_method}) is not supported. Must "
            f"be either 'env' or 'tcp'."
        )

    if backend == "nccl":
        # Same as in Ray Train
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
        # All workers on a same node should share the same set of
        # visible GPUs. Otherwise they can't talk among themselves.
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gid) for gid in gpu_ids)
        if "NCCL_SOCKET_IFNAME" not in os.environ:
            os.environ["NCCL_SOCKET_IFNAME"] = DEFAULT_NCCL_SOCKET_IFNAME
    if backend == 'hccl':
        import torch
        import habana_frameworks.torch.core as htcore 
        import habana_frameworks.torch.distributed.hccl as hccl
        import habana_frameworks.torch.gpu_migration

    init_process_group_kwargs.update(
        dict(
            backend=backend,
            init_method=url,
            rank=rank,
            world_size=world_size,
        )
    )
    init_process_group_kwargs.setdefault("timeout", timedelta(seconds=1800))

    dist.init_process_group(**init_process_group_kwargs)

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_WORLD_SIZE"] = str(local_world_size)


def _get_node_and_gpu_ids():
    """Returns the node_id and gpu_ids for this worker."""
    node_id = ray.get_runtime_context().get_node_id()
    gpu_ids = ray.get_gpu_ids()
    return node_id, gpu_ids


def init_torch_dist_process_group(
    workers: List[ActorHandle],
    backend: str = "gloo",
    init_method: str = "env",
    **init_process_group_kwargs,
) -> List[int]:
    """Initialize a torch distributed process group.
    Note: this util assumes that the order of the workers passed in
    are their global ranks.
    Args:
        workers: A list of TorchDistributedWorker actors.
        backend: The torch distributed backend to use,
            possible choices are "gloo" or "nccl".
        init_method: The initialization method to use,
            possible choices are "env" or "tcp".
        init_process_group_kwargs: Additional kwargs to pass to the call to
            :meth:`torch.distributed.init_process_group`.
    Returns:
        Local ranks on their respective nodes for the list of workers.
    """
    if not dist.is_available():
        raise RuntimeError("Distributed torch is not available.")

    # Build a map from node_id to workers on that node.
    node_and_gpu_ids = ray.get(
        [w.execute.remote(_get_node_and_gpu_ids) for w in workers]
    )
    # All the workers on a specific node.
    node_to_workers = defaultdict(list)
    # All the gpu ids visible to all the workers on a specific node.
    node_to_gpu_ids = defaultdict(set)
    for i, (node_id, gpu_ids) in enumerate(node_and_gpu_ids):
        node_to_workers[node_id].append(i)
        # Force list.
        if not isinstance(gpu_ids, list):
            gpu_ids = [gpu_ids]
        # It is possible for a worker to have access to multiple GPUs.
        for gpu_id in gpu_ids:
            node_to_gpu_ids[node_id].add(gpu_id)

    # Assume the first worker is the master.
    master_addr, master_port = ray.get(workers[0].execute.remote(get_address_and_port))

    setup_futures = []
    world_size = len(workers)
    local_ranks = []
    for rank, worker in enumerate(workers):
        node_id = node_and_gpu_ids[rank][0]
        local_rank = node_to_workers[node_id].index(rank)
        local_world_size = len(node_to_workers[node_id])
        setup_futures.append(
            worker.execute.remote(
                _init_torch_distributed,
                init_method=init_method,
                backend=backend,
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
                local_world_size=local_world_size,
                master_addr=master_addr,
                master_port=master_port,
                # list(set) will sort the gpu ids, so VISIBLE_CUDA_DEVICES
                # is always sorted.
                gpu_ids=list(node_to_gpu_ids[node_id]),
                **init_process_group_kwargs,
            )
        )
        local_ranks.append(local_rank)

    # Wait for all workers to join the process group.
    ray.get(setup_futures)

    return local_ranks


def _shutdown_torch_distributed():
    """Shutdown torch distributed backend"""
    dist.destroy_process_group()

    if not torch.cuda.is_available():
        return

    # Clean up cuda memory.
    devices = get_device()
    if not isinstance(devices, list):
        devices = [devices]
    for device in devices:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def shutdown_torch_dist_process_group(workers: List[ActorHandle]):
    ray.get([w.execute.remote(_shutdown_torch_distributed) for w in workers])

