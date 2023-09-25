from multiprocessing import Process

import torch

from vllm.config import ParallelConfig
from vllm.engine.ray_utils import get_open_port
from vllm.worker.worker import _init_distributed_environment
from vllm.model_executor.parallel_utils.tensor_parallel.communication_op import (
    tenosr_model_parallel_all_reduce,
    tensor_model_parallel_all_gather,
)

def init_test_distributed_environment(pipeline_parallel_size: int,
                                 tensor_parallel_size: int,
                                 rank: int,
                                 distributed_init_port: str):
    parallel_config = ParallelConfig(pipeline_parallel_size,
                                     tensor_parallel_size,
                                     worker_use_ray=True)
    distributed_init_method = f"tcp://localhost:{distributed_init_port}"
    # All test workers use GPU 0
    torch.cuda.set_device(0)
    _init_distributed_environment(parallel_config, rank, distributed_init_method)


def all_reduce_test_worker(tensor_parallel_size: int,
                           rank: int,
                           distributed_init_port: str):
    init_test_distributed_environment(1,
                                 tensor_parallel_size,
                                 rank,
                                 distributed_init_port)
    num_elements = 4
    t = torch.ones(rank, dtype=torch.float32, device="cuda") * rank
    t = tenosr_model_parallel_all_reduce(t)
    assert t.size() == (num_elements,)
    expected = (torch.ones(num_elements, dtype=torch.float32, device="cuda")
                * (tensor_parallel_size - 1) * tensor_parallel_size / 2)
    assert torch.allclose(t, expected)

def test_all_reduce():
    distributed_init_port = get_open_port()
    tensor_parallel_size = 2
    processes = []
    for rank in range(tensor_parallel_size):
        p = Process(target=all_reduce_test_worker,
                    args=(tensor_parallel_size,
                            rank,
                            distributed_init_port))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    assert all(p.exitcode == 0 for p in processes)
