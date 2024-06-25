import os
import subprocess
import sys
import time
import warnings
from contextlib import contextmanager

import ray
import requests

from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.utils import get_open_port

# Path to root of repository so that utilities can be imported by ray workers
VLLM_PATH = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))


@ray.remote(num_gpus=1)
class ServerRunner:
    MAX_SERVER_START_WAIT_S = 600  # wait for server to start for 60 seconds

    def __init__(self, args):
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "vllm.entrypoints.openai.api_server"] +
            args,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        self._wait_for_server()

    def ready(self):
        return True

    def _wait_for_server(self):
        # run health check
        start = time.time()
        while True:
            try:
                if requests.get(
                        "http://localhost:8000/health").status_code == 200:
                    break
            except Exception as err:
                if self.proc.poll() is not None:
                    raise RuntimeError("Server exited unexpectedly.") from err

                time.sleep(0.5)
                if time.time() - start > self.MAX_SERVER_START_WAIT_S:
                    raise RuntimeError(
                        "Server failed to start in time.") from err

    def __del__(self):
        if hasattr(self, "proc"):
            self.proc.terminate()


def init_test_distributed_environment(
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
    local_rank: int = -1,
) -> None:
    distributed_init_method = f"tcp://localhost:{distributed_init_port}"
    init_distributed_environment(
        world_size=pp_size * tp_size,
        rank=rank,
        distributed_init_method=distributed_init_method,
        local_rank=local_rank)
    ensure_model_parallel_initialized(tp_size, pp_size)


def multi_process_tensor_parallel(
    tp_size: int,
    pp_size: int,
    test_target,
) -> None:
    # Using ray helps debugging the error when it failed
    # as compared to multiprocessing.
    ray.init(runtime_env={"working_dir": VLLM_PATH})

    distributed_init_port = get_open_port()
    refs = []
    for rank in range(tp_size * pp_size):
        refs.append(
            test_target.remote(tp_size, pp_size, rank, distributed_init_port))
    ray.get(refs)

    ray.shutdown()


@contextmanager
def error_on_warning():
    """
    Within the scope of this context manager, tests will fail if any warning
    is emitted.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        yield
