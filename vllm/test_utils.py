import ray

from vllm.config import ParallelConfig
from vllm.utils import get_open_port
from vllm.worker.worker import init_distributed_environment


def init_test_distributed_environment(
    pipeline_parallel_size: int,
    tensor_parallel_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    parallel_config = ParallelConfig(pipeline_parallel_size,
                                     tensor_parallel_size,
                                     worker_use_ray=True)
    distributed_init_method = f"tcp://localhost:{distributed_init_port}"
    init_distributed_environment(
        parallel_config,
        rank,
        cupy_port=None,
        distributed_init_method=distributed_init_method)


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
