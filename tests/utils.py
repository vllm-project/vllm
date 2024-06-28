import os
import subprocess
import sys
import time
import warnings
from contextlib import contextmanager
from typing import Dict, List

import openai
import ray
import requests

from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import get_open_port, is_hip

if is_hip():
    from amdsmi import (amdsmi_get_gpu_vram_usage,
                        amdsmi_get_processor_handles, amdsmi_init,
                        amdsmi_shut_down)

    @contextmanager
    def _nvml():
        try:
            amdsmi_init()
            yield
        finally:
            amdsmi_shut_down()
else:
    from pynvml import (nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo,
                        nvmlInit, nvmlShutdown)

    @contextmanager
    def _nvml():
        try:
            nvmlInit()
            yield
        finally:
            nvmlShutdown()


# Path to root of repository so that utilities can be imported by ray workers
VLLM_PATH = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))


class RemoteOpenAIServer:
    DUMMY_API_KEY = "token-abc123"  # vLLM's OpenAI server does not need API key
    MAX_SERVER_START_WAIT_S = 600  # wait for server to start for 60 seconds

    @ray.remote(num_gpus=1)
    class _RemoteRunner:

        def __init__(self, cli_args: List[str], *, wait_url: str,
                     wait_timeout: float) -> None:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            self.proc = subprocess.Popen(
                [
                    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                    *cli_args
                ],
                env=env,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )

            self._wait_for_server(url=wait_url, timeout=wait_timeout)

        def ready(self):
            return True

        def _wait_for_server(self, *, url: str, timeout: float):
            # run health check
            start = time.time()
            while True:
                try:
                    if requests.get(url).status_code == 200:
                        break
                except Exception as err:
                    if self.proc.poll() is not None:
                        raise RuntimeError(
                            "Server exited unexpectedly.") from err

                    time.sleep(0.5)
                    if time.time() - start > timeout:
                        raise RuntimeError(
                            "Server failed to start in time.") from err

        def __del__(self):
            if hasattr(self, "proc"):
                self.proc.terminate()

    def __init__(self, cli_args: List[str], *, auto_port: bool = True) -> None:
        if auto_port:
            if "-p" in cli_args or "--port" in cli_args:
                raise ValueError("You have manually specified the port"
                                 "when `auto_port=True`.")

            cli_args = cli_args + ["--port", str(get_open_port())]

        parser = make_arg_parser()
        args = parser.parse_args(cli_args)
        self.host = str(args.host or 'localhost')
        self.port = int(args.port)

        self._runner = self._RemoteRunner.remote(  # type: ignore
            cli_args,
            wait_url=self.url_for("health"),
            wait_timeout=self.MAX_SERVER_START_WAIT_S)

        self._wait_until_ready()

    @property
    def url_root(self) -> str:
        return f"http://{self.host}:{self.port}"

    def url_for(self, *parts: str) -> str:
        return self.url_root + "/" + "/".join(parts)

    def _wait_until_ready(self) -> None:
        ray.get(self._runner.ready.remote())

    def get_client(self):
        return openai.OpenAI(
            base_url=self.url_for("v1"),
            api_key=self.DUMMY_API_KEY,
        )

    def get_async_client(self):
        return openai.AsyncOpenAI(
            base_url=self.url_for("v1"),
            api_key=self.DUMMY_API_KEY,
        )


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


def multi_process_parallel(
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


@_nvml()
def wait_for_gpu_memory_to_clear(devices: List[int],
                                 threshold_bytes: int,
                                 timeout_s: float = 120) -> None:
    # Use nvml instead of pytorch to reduce measurement error from torch cuda
    # context.
    start_time = time.time()
    while True:
        output: Dict[int, str] = {}
        output_raw: Dict[int, float] = {}
        for device in devices:
            if is_hip():
                dev_handle = amdsmi_get_processor_handles()[device]
                mem_info = amdsmi_get_gpu_vram_usage(dev_handle)
                gb_used = mem_info["vram_used"] / 2**10
            else:
                dev_handle = nvmlDeviceGetHandleByIndex(device)
                mem_info = nvmlDeviceGetMemoryInfo(dev_handle)
                gb_used = mem_info.used / 2**30
            output_raw[device] = gb_used
            output[device] = f'{gb_used:.02f}'

        print('gpu memory used (GB): ', end='')
        for k, v in output.items():
            print(f'{k}={v}; ', end='')
        print('')

        dur_s = time.time() - start_time
        if all(v <= (threshold_bytes / 2**30) for v in output_raw.values()):
            print(f'Done waiting for free GPU memory on devices {devices=} '
                  f'({threshold_bytes/2**30=}) {dur_s=:.02f}')
            break

        if dur_s >= timeout_s:
            raise ValueError(f'Memory of devices {devices=} not free after '
                             f'{dur_s=:.02f} ({threshold_bytes/2**30=})')

        time.sleep(5)
