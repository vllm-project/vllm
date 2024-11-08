import asyncio
import copy
import functools
import os
import signal
import subprocess
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

import openai
import pytest
import requests
import torch
import torch.nn.functional as F
from openai.types.completion import Completion
from typing_extensions import ParamSpec

import vllm.envs as envs
from tests.models.utils import TextTextLogprobs
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.model_executor.model_loader.loader import get_model_loader
from vllm.platforms import current_platform
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.utils import (FlexibleArgumentParser, GB_bytes,
                        cuda_device_count_stateless, get_open_port)

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
            self.proc.wait(8)
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
            except Exception:
                # this exception can only be raised by requests.get,
                # which means the server is not ready yet.
                # the stack trace is not useful, so we suppress it
                # by using `raise from None`.
                result = self.proc.poll()
                if result is not None and result != 0:
                    raise RuntimeError("Server exited unexpectedly.") from None

                time.sleep(0.5)
                if time.time() - start > timeout:
                    raise RuntimeError(
                        "Server failed to start in time.") from None

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


def _test_completion(
    client: openai.OpenAI,
    model: str,
    prompt: str,
    token_ids: List[int],
):
    results = []

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

    return results


def _test_completion_close(
    client: openai.OpenAI,
    model: str,
    prompt: str,
):
    results = []

    # test with text prompt
    completion = client.completions.create(model=model,
                                           prompt=prompt,
                                           max_tokens=1,
                                           logprobs=5,
                                           temperature=0.0)

    logporbs = completion.choices[0].logprobs.top_logprobs[0]
    logporbs = {k: round(v, 2) for k, v in logporbs.items()}

    results.append({
        "test": "completion_close",
        "logprobs": logporbs,
    })

    return results


def _test_embeddings(
    client: openai.OpenAI,
    model: str,
    text: str,
):
    results = []

    # test with text input
    embeddings = client.embeddings.create(
        model=model,
        input=text,
        encoding_format="float",
    )

    results.append({
        "test": "single_embedding",
        "embedding": embeddings.data[0].embedding,
        "usage": embeddings.usage,
    })

    return results


def _test_image_text(
    client: openai.OpenAI,
    model_name: str,
    image_url: str,
):
    results = []

    # test pure text input
    messages = [{
        "role":
        "user",
        "content": [
            {
                "type": "text",
                "text": "How do you feel today?"
            },
        ],
    }]

    chat_completion = client.chat.completions.create(model=model_name,
                                                     messages=messages,
                                                     temperature=0.0,
                                                     max_tokens=1,
                                                     logprobs=True,
                                                     top_logprobs=5)
    top_logprobs = chat_completion.choices[0].logprobs.content[0].top_logprobs

    for x in top_logprobs:
        x.logprob = round(x.logprob, 2)

    results.append({
        "test": "pure_text",
        "logprobs": top_logprobs,
    })

    messages = [{
        "role":
        "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            },
            {
                "type": "text",
                "text": "What's in this image?"
            },
        ],
    }]

    chat_completion = client.chat.completions.create(model=model_name,
                                                     messages=messages,
                                                     temperature=0.0,
                                                     max_tokens=1,
                                                     logprobs=True,
                                                     top_logprobs=5)
    top_logprobs = chat_completion.choices[0].logprobs.content[0].top_logprobs

    results.append({
        "test": "text_image",
        "logprobs": top_logprobs,
    })

    return results


def compare_two_settings(model: str,
                         arg1: List[str],
                         arg2: List[str],
                         env1: Optional[Dict[str, str]] = None,
                         env2: Optional[Dict[str, str]] = None,
                         *,
                         method: str = "generate",
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

    compare_all_settings(
        model,
        [arg1, arg2],
        [env1, env2],
        method=method,
        max_wait_seconds=max_wait_seconds,
    )


def compare_all_settings(model: str,
                         all_args: List[List[str]],
                         all_envs: List[Optional[Dict[str, str]]],
                         *,
                         method: str = "generate",
                         max_wait_seconds: Optional[float] = None) -> None:
    """
    Launch API server with several different sets of arguments/environments
    and compare the results of the API calls with the first set of arguments.
    Args:
        model: The model to test.
        all_args: A list of argument lists to pass to the API server.
        all_envs: A list of environment dictionaries to pass to the API server.
    """

    trust_remote_code = False
    for args in all_args:
        if "--trust-remote-code" in args:
            trust_remote_code = True
            break

    tokenizer_mode = "auto"
    for args in all_args:
        if "--tokenizer-mode" in args:
            tokenizer_mode = args[args.index("--tokenizer-mode") + 1]
            break

    tokenizer = get_tokenizer(
        model,
        trust_remote_code=trust_remote_code,
        tokenizer_mode=tokenizer_mode,
    )

    can_force_load_format = True

    for args in all_args:
        if "--load-format" in args:
            can_force_load_format = False
            break

    prompt = "Hello, my name is"
    token_ids = tokenizer(prompt).input_ids
    ref_results: List = []
    for i, (args, env) in enumerate(zip(all_args, all_envs)):
        if can_force_load_format:
            # we are comparing the results and
            # usually we don't need real weights.
            # we force to use dummy weights by default,
            # and it should work for most of the cases.
            # if not, we can use VLLM_TEST_FORCE_LOAD_FORMAT
            # environment variable to force the load format,
            # e.g. in quantization tests.
            args = args + ["--load-format", envs.VLLM_TEST_FORCE_LOAD_FORMAT]
        compare_results: List = []
        results = ref_results if i == 0 else compare_results
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

            if method == "generate":
                results += _test_completion(client, model, prompt, token_ids)
            elif method == "generate_close":
                results += _test_completion_close(client, model, prompt)
            elif method == "generate_with_image":
                results += _test_image_text(
                    client, model,
                    "https://upload.wikimedia.org/wikipedia/commons/0/0b/RGBA_comp.png"
                )
            elif method == "encode":
                results += _test_embeddings(client, model, prompt)
            else:
                raise ValueError(f"Unknown method: {method}")

            if i > 0:
                # if any setting fails, raise an error early
                ref_args = all_args[0]
                ref_envs = all_envs[0]
                compare_args = all_args[i]
                compare_envs = all_envs[i]
                for ref_result, compare_result in zip(ref_results,
                                                      compare_results):
                    ref_result = copy.deepcopy(ref_result)
                    compare_result = copy.deepcopy(compare_result)
                    if "embedding" in ref_result and method == "encode":
                        sim = F.cosine_similarity(
                            torch.tensor(ref_result["embedding"]),
                            torch.tensor(compare_result["embedding"]),
                            dim=0,
                        )
                        assert sim >= 0.999, (
                            f"Embedding for {model=} are not the same.\n"
                            f"cosine_similarity={sim}\n")
                        del ref_result["embedding"]
                        del compare_result["embedding"]
                    assert ref_result == compare_result, (
                        f"Results for {model=} are not the same.\n"
                        f"{ref_args=} {ref_envs=}\n"
                        f"{compare_args=} {compare_envs=}\n"
                        f"{ref_result=}\n"
                        f"{compare_result=}\n")


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
def error_on_warning(category: Type[Warning] = Warning):
    """
    Within the scope of this context manager, tests will fail if any warning
    of the given category is emitted.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=category)

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
            if current_platform.is_rocm():
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


def large_gpu_mark(min_gb: int) -> pytest.MarkDecorator:
    """Gets a pytest skipif mark, which triggers ig the the device doesn't have
    meet a minimum memory requirement in gb; can be leveraged via 
    @large_gpu_test to skip tests in environments without enough resources, or
    called when filtering tests to run directly.
    """
    try:
        if current_platform.is_cpu():
            memory_gb = 0
        else:
            memory_gb = current_platform.get_device_total_memory() / GB_bytes
    except Exception as e:
        warnings.warn(
            f"An error occurred when finding the available memory: {e}",
            stacklevel=2,
        )
        memory_gb = 0

    return pytest.mark.skipif(
        memory_gb < min_gb,
        reason=f"Need at least {min_gb}GB GPU memory to run the test.",
    )


def large_gpu_test(*, min_gb: int):
    """
    Decorate a test to be skipped if no GPU is available or it does not have
    sufficient memory.

    Currently, the CI machine uses L4 GPU which has 24 GB VRAM.
    """
    test_skipif = large_gpu_mark(min_gb)

    def wrapper(f: Callable[_P, None]) -> Callable[_P, None]:
        return test_skipif(f)

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
    max_tokens: Union[int, list] = 5,
) -> List[Completion]:
    '''Construct a remote OpenAI server, obtain an async client to the
    server & invoke the completions API to obtain completions.

    Args:
      prompts: test prompts
      model_name: model to spin up on the vLLM server
      server_cli_args: CLI args for starting the server
      num_logprobs: Number of logprobs to report (or `None`)
      max_wait_seconds: timeout interval for bringing up server.
                        Default: 240sec
      max_tokens: max_tokens value for each of the given input prompts.
        if only one max_token value is given, the same value is used
        for all the prompts.

    Returns:
      OpenAI Completion instance
    '''

    if isinstance(max_tokens, int):
        max_tokens = [max_tokens] * len(prompts)

    assert len(max_tokens) == len(prompts)

    outputs = None
    max_wait_seconds = 240 * 3  # 240 is default
    with RemoteOpenAIServer(model_name,
                            server_cli_args,
                            max_wait_seconds=max_wait_seconds) as server:
        client = server.get_async_client()
        outputs = [ client.completions.create(model=model_name,
                                              prompt=[p],
                                              temperature=0,
                                              stream=False,
                                              max_tokens=max_tok,
                                              logprobs=num_logprobs) \
                    for p, max_tok in zip(prompts, max_tokens) ]
        outputs = await asyncio.gather(*outputs)

    assert outputs is not None, "Completion API call failed."

    return outputs


def get_client_text_generations(completions: List[Completion]) -> List[str]:
    '''Extract generated tokens from the output of a
    request made to an Open-AI-protocol completions endpoint.
    '''
    assert all([len(x.choices) == 1 for x in completions])
    return [x.choices[0].text for x in completions]


def get_client_text_logprob_generations(
        completions: List[Completion]) -> List[TextTextLogprobs]:
    '''Operates on the output of a request made to an Open-AI-protocol
    completions endpoint; obtains top-rank logprobs for each token in
    each :class:`SequenceGroup`
    '''
    text_generations = get_client_text_generations(completions)
    text = ''.join(text_generations)
    return [(text_generations, text,
             (None if x.logprobs is None else x.logprobs.top_logprobs))
            for completion in completions for x in completion.choices]
