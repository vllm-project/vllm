"""Test the communication operators.

Run `pytest tests/distributed/test_comm_ops.py --forked`.
"""
from multiprocessing import Process, set_start_method
import torch

from vllm.config import ParallelConfig
from vllm.engine.ray_utils import get_open_port
from vllm.worker.worker import _init_distributed_environment


def init_test_distributed_environment(pipeline_parallel_size: int,
                                      tensor_parallel_size: int, rank: int,
                                      distributed_init_port: str):
    parallel_config = ParallelConfig(pipeline_parallel_size,
                                     tensor_parallel_size,
                                     worker_use_ray=True)
    distributed_init_method = f"tcp://localhost:{distributed_init_port}"
    torch.cuda.set_device(rank)
    _init_distributed_environment(parallel_config, rank,
                                  distributed_init_method)


def multi_process_tensor_parallel(tensor_parallel_size, test_target):
    set_start_method("spawn", force=True)
    distributed_init_port = get_open_port()
    processes = []
    for rank in range(tensor_parallel_size):
        p = Process(target=test_target,
                    args=(tensor_parallel_size, rank, distributed_init_port))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    assert all(p.exitcode == 0 for p in processes)
