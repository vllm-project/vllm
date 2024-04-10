import ray

from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.utils import get_open_port


def init_test_distributed_environment(
    pipeline_parallel_size: int,
    tensor_parallel_size: int,
    rank: int,
    distributed_init_port: str,
    local_rank: int = -1,
) -> None:
    distributed_init_method = f"tcp://localhost:{distributed_init_port}"
    init_distributed_environment(
        world_size=pipeline_parallel_size * tensor_parallel_size,
        rank=rank,
        distributed_init_method=distributed_init_method,
        local_rank=local_rank)
    ensure_model_parallel_initialized(tensor_parallel_size,
                                      pipeline_parallel_size)


def multi_process_tensor_parallel(
    tensor_parallel_size: int,
    test_target,
) -> None:
    # Using ray helps debugging the error when it failed
    # as compared to multiprocessing.
    ray.init()

    distributed_init_port = get_open_port()
    refs = []
    for rank in range(tensor_parallel_size):
        refs.append(
            test_target.remote(tensor_parallel_size, rank,
                               distributed_init_port))
    ray.get(refs)

    ray.shutdown()
