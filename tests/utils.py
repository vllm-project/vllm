import functools
import os
import signal
import subprocess
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import openai
import pytest
import requests
from openai.types.completion import Completion
from transformers import AutoTokenizer
from typing_extensions import ParamSpec

from tests.models.utils import TextTextLogprobs
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.model_executor.model_loader.loader import get_model_loader
from vllm.platforms import current_platform
from vllm.utils import (FlexibleArgumentParser, cuda_device_count_stateless,
                        get_open_port, is_hip)

if current_platform.is_rocm():
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
elif current_platform.is_cuda():
    from pynvml import (nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo,
                        nvmlInit, nvmlShutdown)

    @contextmanager
    def _nvml():
        try:
            nvmlInit()
            yield
        finally:
            nvmlShutdown()
else:

    @contextmanager
    def _nvml():
        yield


VLLM_PATH = Path(__file__).parent.parent
"""Path to root of the vLLM repository."""


class RemoteOpenAIServer:
    DUMMY_API_KEY = "token-abc123"  # vLLM's OpenAI server does not need API key

    def __init__(self,
                 model: str,
                 vllm_serve_args: List[str],
                 *,
                 env_dict: Optional[Dict[str, str]] = None,
                 auto_port: bool = True,
                 max_wait_seconds: Optional[float] = None) -> None:
        if auto_port:
            if "-p" in vllm_serve_args or "--port" in vllm_serve_args:
                raise ValueError("You have manually specified the port "
                                 "when `auto_port=True`.")

            # Don't mutate the input args
            vllm_serve_args = vllm_serve_args + [
                "--port", str(get_open_port())
            ]

        parser = FlexibleArgumentParser(
            description="vLLM's remote OpenAI server.")
        parser = make_arg_parser(parser)
        args = parser.parse_args(["--model", model, *vllm_serve_args])
        self.host = str(args.host or 'localhost')
        self.port = int(args.port)

        # download the model before starting the server to avoid timeout
        is_local = os.path.isdir(model)
        if not is_local:
            engine_args = AsyncEngineArgs.from_cli_args(args)
            model_config = engine_args.create_model_config()
            load_config = engine_args.create_load_config()

            model_loader = get_model_loader(load_config)
            model_loader.download_model(model_config)

        env = os.environ.copy()
        # the current process might initialize cuda,
        # to be safe, we should use spawn method
        env['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        if env_dict is not None:
            env.update(env_dict)
        self.proc = subprocess.Popen(
            ["vllm", "serve", model, *vllm_serve_args],
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        max_wait_seconds = max_wait_seconds or 240
        self._wait_for_server(url=self.url_for("health"),
                              timeout=max_wait_seconds)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.proc.terminate()
        try:
            self.proc.wait(3)
        except subprocess.TimeoutExpired:
            # force kill if needed
            self.proc.kill()

    def _wait_for_server(self, *, url: str, timeout: float):
        # run health check
        start = time.time()
        while True:
            try:
                if requests.get(url).status_code == 200:
                    break
            except Exception as err:
                result = self.proc.poll()
                if result is not None and result != 0:
                    raise RuntimeError("Server exited unexpectedly.") from err

                time.sleep(0.5)
                if time.time() - start > timeout:
                    raise RuntimeError(
                        "Server failed to start in time.") from err

    @property
    def url_root(self) -> str:
        return f"http://{self.host}:{self.port}"

    def url_for(self, *parts: str) -> str:
        return self.url_root + "/" + "/".join(parts)

    def get_client(self):
        return openai.OpenAI(
            base_url=self.url_for("v1"),
            api_key=self.DUMMY_API_KEY,
        )

    def get_async_client(self):
        return openai.AsyncOpenAI(
            base_url=self.url_for("v1"),
            api_key=self.DUMMY_API_KEY,
            max_retries=0,
        )


def compare_two_settings(model: str,
                         arg1: List[str],
                         arg2: List[str],
                         env1: Optional[Dict[str, str]] = None,
                         env2: Optional[Dict[str, str]] = None,
                         max_wait_seconds: Optional[float] = None) -> None:
    """
    Launch API server with two different sets of arguments/environments
    and compare the results of the API calls.

    Args:
        model: The model to test.
        arg1: The first set of arguments to pass to the API server.
        arg2: The second set of arguments to pass to the API server.
        env1: The first set of environment variables to pass to the API server.
        env2: The second set of environment variables to pass to the API server.
    """

    trust_remote_code = "--trust-remote-code"
    if trust_remote_code in arg1 or trust_remote_code in arg2:
        tokenizer = AutoTokenizer.from_pretrained(model,
                                                  trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model)

    prompt = "Hello, my name is"
    token_ids = tokenizer(prompt)["input_ids"]
    results = []
    for args, env in ((arg1, env1), (arg2, env2)):
        with RemoteOpenAIServer(model,
                                args,
                                env_dict=env,
                                max_wait_seconds=max_wait_seconds) as server:
            client = server.get_client()

            # test models list
            models = client.models.list()
            models = models.data
            served_model = models[0]
            results.append({
                "test": "models_list",
                "id": served_model.id,
                "root": served_model.root,
            })

            # test with text prompt
            completion = client.completions.create(model=model,
                                                   prompt=prompt,
                                                   max_tokens=5,
                                                   temperature=0.0)

            results.append({
                "test": "single_completion",
                "text": completion.choices[0].text,
                "finish_reason": completion.choices[0].finish_reason,
                "usage": completion.usage,
            })

            # test using token IDs
            completion = client.completions.create(
                model=model,
                prompt=token_ids,
                max_tokens=5,
                temperature=0.0,
            )

            results.append({
                "test": "token_ids",
                "text": completion.choices[0].text,
                "finish_reason": completion.choices[0].finish_reason,
                "usage": completion.usage,
            })

            # test seeded random sampling
            completion = client.completions.create(model=model,
                                                   prompt=prompt,
                                                   max_tokens=5,
                                                   seed=33,
                                                   temperature=1.0)

            results.append({
                "test": "seeded_sampling",
                "text": completion.choices[0].text,
                "finish_reason": completion.choices[0].finish_reason,
                "usage": completion.usage,
            })

            # test seeded random sampling with multiple prompts
            completion = client.completions.create(model=model,
                                                   prompt=[prompt, prompt],
                                                   max_tokens=5,
                                                   seed=33,
                                                   temperature=1.0)

            results.append({
                "test":
                "seeded_sampling",
                "text": [choice.text for choice in completion.choices],
                "finish_reason":
                [choice.finish_reason for choice in completion.choices],
                "usage":
                completion.usage,
            })

            # test simple list
            batch = client.completions.create(
                model=model,
                prompt=[prompt, prompt],
                max_tokens=5,
                temperature=0.0,
            )

            results.append({
                "test": "simple_list",
                "text0": batch.choices[0].text,
                "text1": batch.choices[1].text,
            })

            # test streaming
            batch = client.completions.create(
                model=model,
                prompt=[prompt, prompt],
                max_tokens=5,
                temperature=0.0,
                stream=True,
            )
            texts = [""] * 2
            for chunk in batch:
                assert len(chunk.choices) == 1
                choice = chunk.choices[0]
                texts[choice.index] += choice.text
            results.append({
                "test": "streaming",
                "texts": texts,
            })

    n = len(results) // 2
    arg1_results = results[:n]
    arg2_results = results[n:]
    for arg1_result, arg2_result in zip(arg1_results, arg2_results):
        assert arg1_result == arg2_result, (
            f"Results for {model=} are not the same with {arg1=} and {arg2=}. "
            f"{arg1_result=} != {arg2_result=}")


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
    test_target: Any,
) -> None:
    import ray

    # Using ray helps debugging the error when it failed
    # as compared to multiprocessing.
    # NOTE: We need to set working_dir for distributed tests,
    # otherwise we may get import errors on ray workers
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


def get_physical_device_indices(devices):
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        return devices

    visible_indices = [int(x) for x in visible_devices.split(",")]
    index_mapping = {i: physical for i, physical in enumerate(visible_indices)}
    return [index_mapping[i] for i in devices if i in index_mapping]


@_nvml()
def wait_for_gpu_memory_to_clear(devices: List[int],
                                 threshold_bytes: int,
                                 timeout_s: float = 120) -> None:
    # Use nvml instead of pytorch to reduce measurement error from torch cuda
    # context.
    devices = get_physical_device_indices(devices)
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


_P = ParamSpec("_P")


def fork_new_process_for_each_test(
        f: Callable[_P, None]) -> Callable[_P, None]:
    """Decorator to fork a new process for each test function.
    See https://github.com/vllm-project/vllm/issues/7053 for more details.
    """

    @functools.wraps(f)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> None:
        # Make the process the leader of its own process group
        # to avoid sending SIGTERM to the parent process
        os.setpgrp()
        from _pytest.outcomes import Skipped
        pid = os.fork()
        print(f"Fork a new process to run a test {pid}")
        if pid == 0:
            try:
                f(*args, **kwargs)
            except Skipped as e:
                # convert Skipped to exit code 0
                print(str(e))
                os._exit(0)
            except Exception:
                import traceback
                traceback.print_exc()
                os._exit(1)
            else:
                os._exit(0)
        else:
            pgid = os.getpgid(pid)
            _pid, _exitcode = os.waitpid(pid, 0)
            # ignore SIGTERM signal itself
            old_signal_handler = signal.signal(signal.SIGTERM, signal.SIG_IGN)
            # kill all child processes
            os.killpg(pgid, signal.SIGTERM)
            # restore the signal handler
            signal.signal(signal.SIGTERM, old_signal_handler)
            assert _exitcode == 0, (f"function {f} failed when called with"
                                    f" args {args} and kwargs {kwargs}")

    return wrapper


def multi_gpu_test(*, num_gpus: int):
    """
    Decorate a test to be run only when multiple GPUs are available.
    """
    test_selector = getattr(pytest.mark, f"distributed_{num_gpus}_gpus")
    test_skipif = pytest.mark.skipif(
        cuda_device_count_stateless() < num_gpus,
        reason=f"Need at least {num_gpus} GPUs to run the test.",
    )

    def wrapper(f: Callable[_P, None]) -> Callable[_P, None]:
        return test_selector(test_skipif(fork_new_process_for_each_test(f)))

    return wrapper


async def completions_with_server_args(
    prompts: List[str],
    model_name: str,
    server_cli_args: List[str],
    num_logprobs: Optional[int],
    max_wait_seconds: int = 240,
) -> Completion:
    '''Construct a remote OpenAI server, obtain an async client to the
    server & invoke the completions API to obtain completions.

    Args:
      prompts: test prompts
      model_name: model to spin up on the vLLM server
      server_cli_args: CLI args for starting the server
      num_logprobs: Number of logprobs to report (or `None`)
      max_wait_seconds: timeout interval for bringing up server.
                        Default: 240sec

    Returns:
      OpenAI Completion instance
    '''

    outputs = None
    with RemoteOpenAIServer(model_name,
                            server_cli_args,
                            max_wait_seconds=max_wait_seconds) as server:
        client = server.get_async_client()
        outputs = await client.completions.create(model=model_name,
                                                  prompt=prompts,
                                                  temperature=0,
                                                  stream=False,
                                                  max_tokens=5,
                                                  logprobs=num_logprobs)
    assert outputs is not None

    return outputs


def get_client_text_generations(completions: Completion) -> List[str]:
    '''Extract generated tokens from the output of a
    request made to an Open-AI-protocol completions endpoint.
    '''
    return [x.text for x in completions.choices]


def get_client_text_logprob_generations(
        completions: Completion) -> List[TextTextLogprobs]:
    '''Operates on the output of a request made to an Open-AI-protocol
    completions endpoint; obtains top-rank logprobs for each token in
    each :class:`SequenceGroup`
    '''
    text_generations = get_client_text_generations(completions)
    text = ''.join(text_generations)
    return [(text_generations, text,
             (None if x.logprobs is None else x.logprobs.top_logprobs))
            for x in completions.choices]
