# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import contextlib
import copy
import functools
import importlib
import itertools
import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import time
import warnings
from collections.abc import Callable, Iterable
from contextlib import ExitStack, contextmanager, suppress
from multiprocessing import Process
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import anthropic
import cloudpickle
import httpx
import openai
import psutil
import pytest
import requests
import torch
import torch.nn.functional as F
from openai.types.completion import Completion
from typing_extensions import ParamSpec

import vllm.envs as envs
from tests.models.utils import TextTextLogprobs
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.cli.serve import ServeSubcommand
from vllm.model_executor.layers.quantization.kernels.scaled_mm import (
    init_fp8_linear_kernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.ScaledMMLinearKernel import (  # noqa: E501
    FP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import W8A8BlockFp8LinearOp
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
)
from vllm.model_executor.model_loader import get_model_loader
from vllm.platforms import current_platform
from vllm.tokenizers import get_tokenizer
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.mem_constants import GB_bytes
from vllm.utils.network_utils import get_open_port
from vllm.utils.torch_utils import (
    cuda_device_count_stateless,
    set_random_seed,  # noqa: F401 - re-exported for use in test files
)

FP8_DTYPE = current_platform.fp8_dtype()

if current_platform.is_rocm():
    from amdsmi import (
        amdsmi_get_gpu_vram_usage,
        amdsmi_get_processor_handles,
        amdsmi_init,
        amdsmi_shut_down,
    )

    @contextmanager
    def _nvml():
        try:
            amdsmi_init()
            yield
        finally:
            amdsmi_shut_down()
elif current_platform.is_cuda():
    from vllm.third_party.pynvml import (
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlInit,
        nvmlShutdown,
    )

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

    def _start_server(
        self, model: str, vllm_serve_args: list[str], env_dict: dict[str, str] | None
    ) -> None:
        """Subclasses override this method to customize server process launch"""
        env = os.environ.copy()
        # the current process might initialize cuda,
        # to be safe, we should use spawn method
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        if env_dict is not None:
            env.update(env_dict)
        serve_cmd = ["vllm", "serve", model, *vllm_serve_args]
        print(f"Launching RemoteOpenAIServer with: {' '.join(serve_cmd)}")
        print(f"Environment variables: {env}")
        self.proc: subprocess.Popen = subprocess.Popen(
            serve_cmd,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

    def __init__(
        self,
        model: str,
        vllm_serve_args: list[str],
        *,
        env_dict: dict[str, str] | None = None,
        seed: int = 0,
        auto_port: bool = True,
        max_wait_seconds: float | None = None,
        override_hf_configs: dict[str, Any] | None = None,
    ) -> None:
        if auto_port:
            if "-p" in vllm_serve_args or "--port" in vllm_serve_args:
                raise ValueError(
                    "You have manually specified the port when `auto_port=True`."
                )

            # No need for a port if using unix sockets
            if "--uds" not in vllm_serve_args:
                # Don't mutate the input args
                vllm_serve_args = vllm_serve_args + ["--port", str(get_open_port())]
        if seed is not None:
            if "--seed" in vllm_serve_args:
                raise ValueError(
                    f"You have manually specified the seed when `seed={seed}`."
                )

            vllm_serve_args = vllm_serve_args + ["--seed", str(seed)]

        if override_hf_configs is not None:
            vllm_serve_args = vllm_serve_args + [
                "--hf-overrides",
                json.dumps(override_hf_configs),
            ]

        parser = FlexibleArgumentParser(description="vLLM's remote OpenAI server.")
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        parser = ServeSubcommand().subparser_init(subparsers)
        args = parser.parse_args(["--model", model, *vllm_serve_args])
        self.uds = args.uds
        if args.uds:
            self.host = None
            self.port = None
        else:
            self.host = str(args.host or "127.0.0.1")
            self.port = int(args.port)

        self.show_hidden_metrics = args.show_hidden_metrics_for_version is not None

        # download the model before starting the server to avoid timeout
        is_local = os.path.isdir(model)
        if not is_local:
            engine_args = AsyncEngineArgs.from_cli_args(args)
            model_config = engine_args.create_model_config()
            load_config = engine_args.create_load_config()

            model_loader = get_model_loader(load_config)
            model_loader.download_model(model_config)

        self._start_server(model, vllm_serve_args, env_dict)
        max_wait_seconds = max_wait_seconds or 240
        self._wait_for_server(url=self.url_for("health"), timeout=max_wait_seconds)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Kill all child processes (including Ray workers) before terminating
        # the main process to avoid orphaned processes holding onto ports
        try:
            parent = psutil.Process(self.proc.pid)
            children = parent.children(recursive=True)
            for child in children:
                with suppress(psutil.NoSuchProcess):
                    child.terminate()
            # Wait for children to terminate
            psutil.wait_procs(children, timeout=5)
            # Force kill any remaining children
            for child in children:
                with suppress(psutil.NoSuchProcess):
                    child.kill()
        except psutil.NoSuchProcess:
            pass  # Main process already exited

        self.proc.terminate()
        try:
            self.proc.wait(8)
        except subprocess.TimeoutExpired:
            # force kill if needed
            self.proc.kill()

    def _poll(self) -> int | None:
        """Subclasses override this method to customize process polling"""
        return self.proc.poll()

    def _wait_for_server(self, *, url: str, timeout: float):
        # run health check
        start = time.time()
        client = (
            httpx.Client(transport=httpx.HTTPTransport(uds=self.uds))
            if self.uds
            else requests
        )
        while True:
            try:
                if client.get(url).status_code == 200:
                    break
            except Exception:
                # this exception can only be raised by requests.get,
                # which means the server is not ready yet.
                # the stack trace is not useful, so we suppress it
                # by using `raise from None`.
                result = self._poll()
                if result is not None and result != 0:
                    raise RuntimeError("Server exited unexpectedly.") from None

                time.sleep(0.5)
                if time.time() - start > timeout:
                    raise RuntimeError("Server failed to start in time.") from None

    @property
    def url_root(self) -> str:
        return (
            f"http://{self.uds.split('/')[-1]}"
            if self.uds
            else f"http://{self.host}:{self.port}"
        )

    def url_for(self, *parts: str) -> str:
        return self.url_root + "/" + "/".join(parts)

    def get_client(self, **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = 600
        return openai.OpenAI(
            base_url=self.url_for("v1"),
            api_key=self.DUMMY_API_KEY,
            max_retries=0,
            **kwargs,
        )

    def get_async_client(self, **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = 600
        return openai.AsyncOpenAI(
            base_url=self.url_for("v1"),
            api_key=self.DUMMY_API_KEY,
            max_retries=0,
            **kwargs,
        )

    def get_client_anthropic(self, **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = 600
        return anthropic.Anthropic(
            base_url=self.url_for(),
            api_key=self.DUMMY_API_KEY,
            max_retries=0,
            **kwargs,
        )

    def get_async_client_anthropic(self, **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = 600
        return anthropic.AsyncAnthropic(
            base_url=self.url_for(), api_key=self.DUMMY_API_KEY, max_retries=0, **kwargs
        )


class RemoteOpenAIServerCustom(RemoteOpenAIServer):
    """Launch test server with custom child process"""

    def _start_server(
        self, model: str, vllm_serve_args: list[str], env_dict: dict[str, str] | None
    ) -> None:
        self.proc: Process = Process(
            target=self.child_process_fxn, args=(env_dict, model, vllm_serve_args)
        )  # type: ignore[assignment]
        self.proc.start()

    def __init__(
        self,
        model: str,
        vllm_serve_args: list[str],
        child_process_fxn: Callable[[dict[str, str] | None, str, list[str]], None],
        *,
        env_dict: dict[str, str] | None = None,
        seed: int = 0,
        auto_port: bool = True,
        max_wait_seconds: float | None = None,
    ) -> None:
        """Store custom child process function then invoke superclass
        constructor which will indirectly launch it."""
        self.child_process_fxn = child_process_fxn
        super().__init__(
            model=model,
            vllm_serve_args=vllm_serve_args,
            env_dict=env_dict,
            seed=seed,
            auto_port=auto_port,
            max_wait_seconds=max_wait_seconds,
        )

    def _poll(self) -> int | None:
        return self.proc.exitcode

    def __exit__(self, exc_type, exc_value, traceback):
        self.proc.terminate()
        self.proc.join(8)
        if self.proc.is_alive():
            # force kill if needed
            self.proc.kill()


def _test_completion(
    client: openai.OpenAI,
    model: str,
    prompt: str,
    token_ids: list[int],
):
    results = []

    # test with text prompt
    completion = client.completions.create(
        model=model, prompt=prompt, max_tokens=5, temperature=0.0
    )

    results.append(
        {
            "test": "single_completion",
            "text": completion.choices[0].text,
            "finish_reason": completion.choices[0].finish_reason,
            "usage": completion.usage,
        }
    )

    # test using token IDs
    completion = client.completions.create(
        model=model,
        prompt=token_ids,
        max_tokens=5,
        temperature=0.0,
    )

    results.append(
        {
            "test": "token_ids",
            "text": completion.choices[0].text,
            "finish_reason": completion.choices[0].finish_reason,
            "usage": completion.usage,
        }
    )

    # test seeded random sampling
    completion = client.completions.create(
        model=model, prompt=prompt, max_tokens=5, seed=33, temperature=1.0
    )

    results.append(
        {
            "test": "seeded_sampling",
            "text": completion.choices[0].text,
            "finish_reason": completion.choices[0].finish_reason,
            "usage": completion.usage,
        }
    )

    # test seeded random sampling with multiple prompts
    completion = client.completions.create(
        model=model, prompt=[prompt, prompt], max_tokens=5, seed=33, temperature=1.0
    )

    results.append(
        {
            "test": "seeded_sampling",
            "text": [choice.text for choice in completion.choices],
            "finish_reason": [choice.finish_reason for choice in completion.choices],
            "usage": completion.usage,
        }
    )

    # test simple list
    batch = client.completions.create(
        model=model,
        prompt=[prompt, prompt],
        max_tokens=5,
        temperature=0.0,
    )

    results.append(
        {
            "test": "simple_list",
            "text0": batch.choices[0].text,
            "text1": batch.choices[1].text,
        }
    )

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

    results.append(
        {
            "test": "streaming",
            "texts": texts,
        }
    )

    return results


def _test_completion_close(
    client: openai.OpenAI,
    model: str,
    prompt: str,
):
    results = []

    # test with text prompt
    completion = client.completions.create(
        model=model, prompt=prompt, max_tokens=1, logprobs=5, temperature=0.0
    )

    logprobs = completion.choices[0].logprobs.top_logprobs[0]
    logprobs = {k: round(v, 2) for k, v in logprobs.items()}

    results.append(
        {
            "test": "completion_close",
            "logprobs": logprobs,
        }
    )

    return results


def _test_chat(
    client: openai.OpenAI,
    model: str,
    prompt: str,
):
    results = []

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    # test with text prompt
    chat_response = client.chat.completions.create(
        model=model, messages=messages, max_tokens=5, temperature=0.0
    )

    results.append(
        {
            "test": "completion_close",
            "text": chat_response.choices[0].message.content,
            "finish_reason": chat_response.choices[0].finish_reason,
            "usage": chat_response.usage,
        }
    )

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

    results.append(
        {
            "test": "single_embedding",
            "embedding": embeddings.data[0].embedding,
            "usage": embeddings.usage,
        }
    )

    return results


def _test_image_text(
    client: openai.OpenAI,
    model_name: str,
    image_url: str,
):
    results = []

    # test pure text input
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "How do you feel today?"},
            ],
        }
    ]

    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
        max_tokens=1,
        logprobs=True,
        top_logprobs=5,
    )
    top_logprobs = chat_completion.choices[0].logprobs.content[0].top_logprobs

    for x in top_logprobs:
        x.logprob = round(x.logprob, 2)

    results.append(
        {
            "test": "pure_text",
            "logprobs": top_logprobs,
        }
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "What's in this image?"},
            ],
        }
    ]

    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
        max_tokens=1,
        logprobs=True,
        top_logprobs=5,
    )
    top_logprobs = chat_completion.choices[0].logprobs.content[0].top_logprobs

    results.append(
        {
            "test": "text_image",
            "logprobs": top_logprobs,
        }
    )

    return results


def compare_two_settings(
    model: str,
    arg1: list[str],
    arg2: list[str],
    env1: dict[str, str] | None = None,
    env2: dict[str, str] | None = None,
    *,
    method: str = "generate",
    max_wait_seconds: float | None = None,
) -> None:
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


def compare_all_settings(
    model: str,
    all_args: list[list[str]],
    all_envs: list[dict[str, str] | None],
    *,
    method: str = "generate",
    max_wait_seconds: float | None = None,
) -> None:
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
    ref_results: list = []
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
        compare_results: list = []
        results = ref_results if i == 0 else compare_results
        with RemoteOpenAIServer(
            model, args, env_dict=env, max_wait_seconds=max_wait_seconds
        ) as server:
            client = server.get_client()

            # test models list
            models = client.models.list()
            models = models.data
            served_model = models[0]
            results.append(
                {
                    "test": "models_list",
                    "id": served_model.id,
                    "root": served_model.root,
                }
            )

            if method == "generate":
                results += _test_completion(client, model, prompt, token_ids)
            elif method == "generate_close":
                results += _test_completion_close(client, model, prompt)
            elif method == "generate_chat":
                results += _test_chat(client, model, prompt)
            elif method == "generate_with_image":
                results += _test_image_text(
                    client,
                    model,
                    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/RGBA_comp.png",
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
                for ref_result, compare_result in zip(ref_results, compare_results):
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
                            f"cosine_similarity={sim}\n"
                        )
                        del ref_result["embedding"]
                        del compare_result["embedding"]
                    assert ref_result == compare_result, (
                        f"Results for {model=} are not the same.\n"
                        f"{ref_args=} {ref_envs=}\n"
                        f"{compare_args=} {compare_envs=}\n"
                        f"{ref_result=}\n"
                        f"{compare_result=}\n"
                    )


@contextmanager
def ensure_current_vllm_config():
    """Context manager that ensures a vllm config is set for the duration of the context.

    If a config is already set, this is a no-op. Otherwise, it creates a default
    VllmConfig and sets it for the duration of the context.

    This is useful for tests that need to call functions which require a vllm config
    (like init_distributed_environment or ensure_model_parallel_initialized) but don't
    need a specific config.

    Example:
        with ensure_current_vllm_config():
            init_distributed_environment(...)
            ensure_model_parallel_initialized(...)
    """
    from vllm.config import (
        VllmConfig,
        get_current_vllm_config_or_none,
        set_current_vllm_config,
    )

    if get_current_vllm_config_or_none() is not None:
        # Config already set, just yield
        yield
    else:
        # No config set, create a default one for the duration
        with set_current_vllm_config(VllmConfig()):
            yield


def init_test_distributed_environment(
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
    local_rank: int = -1,
) -> None:
    # Note: This function is often called from Ray worker processes, so we
    # can't rely on pytest fixtures to set the config. We check if the config
    # is already set and only create a default one if needed.
    from vllm.config import (
        VllmConfig,
        get_current_vllm_config_or_none,
        set_current_vllm_config,
    )

    distributed_init_method = f"tcp://localhost:{distributed_init_port}"

    if get_current_vllm_config_or_none() is not None:
        # Config already set, use it directly
        init_distributed_environment(
            world_size=pp_size * tp_size,
            rank=rank,
            distributed_init_method=distributed_init_method,
            local_rank=local_rank,
        )
        ensure_model_parallel_initialized(tp_size, pp_size)
    else:
        # No config set, create a default one for the test
        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment(
                world_size=pp_size * tp_size,
                rank=rank,
                distributed_init_method=distributed_init_method,
                local_rank=local_rank,
            )
            ensure_model_parallel_initialized(tp_size, pp_size)


def multi_process_parallel(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    test_target: Any,
) -> None:
    import ray

    # Using ray helps debugging the error when it failed
    # as compared to multiprocessing.
    # NOTE: We need to set working_dir for distributed tests,
    # otherwise we may get import errors on ray workers
    # NOTE: Force ray not to use gitignore file as excluding, otherwise
    # it will not move .so files to working dir.
    # So we have to manually add some of large directories
    os.environ["RAY_RUNTIME_ENV_IGNORE_GITIGNORE"] = "1"
    ray.init(
        runtime_env={
            "working_dir": VLLM_PATH,
            "excludes": [
                "build",
                ".git",
                "cmake-build-*",
                "shellcheck",
                "dist",
                "ep_kernels_workspace",
            ],
        }
    )

    distributed_init_port = get_open_port()
    refs = []
    for rank in range(tp_size * pp_size):
        refs.append(
            test_target.remote(
                monkeypatch,
                tp_size,
                pp_size,
                rank,
                distributed_init_port,
            ),
        )
    ray.get(refs)

    ray.shutdown()


@contextmanager
def error_on_warning(category: type[Warning] = Warning):
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
def wait_for_gpu_memory_to_clear(
    *,
    devices: list[int],
    threshold_bytes: int | None = None,
    threshold_ratio: float | None = None,
    timeout_s: float = 120,
) -> None:
    assert threshold_bytes is not None or threshold_ratio is not None
    # Use nvml instead of pytorch to reduce measurement error from torch cuda
    # context.
    devices = get_physical_device_indices(devices)
    start_time = time.time()
    while True:
        output: dict[int, str] = {}
        output_raw: dict[int, tuple[float, float]] = {}
        for device in devices:
            if current_platform.is_rocm():
                dev_handle = amdsmi_get_processor_handles()[device]
                mem_info = amdsmi_get_gpu_vram_usage(dev_handle)
                gb_used = mem_info["vram_used"] / 2**10
                gb_total = mem_info["vram_total"] / 2**10
            else:
                dev_handle = nvmlDeviceGetHandleByIndex(device)
                mem_info = nvmlDeviceGetMemoryInfo(dev_handle)
                gb_used = mem_info.used / 2**30
                gb_total = mem_info.total / 2**30
            output_raw[device] = (gb_used, gb_total)
            output[device] = f"{gb_used:.02f}/{gb_total:.02f}"

        print("gpu memory used/total (GiB): ", end="")
        for k, v in output.items():
            print(f"{k}={v}; ", end="")
        print("")

        if threshold_bytes is not None:
            is_free = lambda used, total: used <= threshold_bytes / 2**30
            threshold = f"{threshold_bytes / 2**30} GiB"
        else:
            is_free = lambda used, total: used / total <= threshold_ratio
            threshold = f"{threshold_ratio:.2f}"

        dur_s = time.time() - start_time
        if all(is_free(used, total) for used, total in output_raw.values()):
            print(
                f"Done waiting for free GPU memory on devices {devices=} "
                f"({threshold=}) {dur_s=:.02f}"
            )
            break

        if dur_s >= timeout_s:
            raise ValueError(
                f"Memory of devices {devices=} not free after "
                f"{dur_s=:.02f} ({threshold=})"
            )

        time.sleep(5)


_P = ParamSpec("_P")


def fork_new_process_for_each_test(func: Callable[_P, None]) -> Callable[_P, None]:
    """Decorator to fork a new process for each test function.
    See https://github.com/vllm-project/vllm/issues/7053 for more details.
    """

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> None:
        # Make the process the leader of its own process group
        # to avoid sending SIGTERM to the parent process
        os.setpgrp()
        from _pytest.outcomes import Skipped

        # Create a unique temporary file to store exception info from child
        # process. Use test function name and process ID to avoid collisions.
        with (
            tempfile.NamedTemporaryFile(
                delete=False,
                mode="w+b",
                prefix=f"vllm_test_{func.__name__}_{os.getpid()}_",
                suffix=".exc",
            ) as exc_file,
            ExitStack() as delete_after,
        ):
            exc_file_path = exc_file.name
            delete_after.callback(os.remove, exc_file_path)

            pid = os.fork()
            print(f"Fork a new process to run a test {pid}")
            if pid == 0:
                # Parent process responsible for deleting, don't delete
                # in child.
                delete_after.pop_all()
                try:
                    func(*args, **kwargs)
                except Skipped as e:
                    # convert Skipped to exit code 0
                    print(str(e))
                    os._exit(0)
                except Exception as e:
                    import traceback

                    tb_string = traceback.format_exc()

                    # Try to serialize the exception object first
                    exc_to_serialize: dict[str, Any]
                    try:
                        # First, try to pickle the actual exception with
                        # its traceback.
                        exc_to_serialize = {"pickled_exception": e}
                        # Test if it can be pickled
                        cloudpickle.dumps(exc_to_serialize)
                    except (Exception, KeyboardInterrupt):
                        # Fall back to string-based approach.
                        exc_to_serialize = {
                            "exception_type": type(e).__name__,
                            "exception_msg": str(e),
                            "traceback": tb_string,
                        }
                    try:
                        with open(exc_file_path, "wb") as f:
                            cloudpickle.dump(exc_to_serialize, f)
                    except Exception:
                        # Fallback: just print the traceback.
                        print(tb_string)
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
                if _exitcode != 0:
                    # Try to read the exception from the child process
                    exc_info = {}
                    if os.path.exists(exc_file_path):
                        with (
                            contextlib.suppress(Exception),
                            open(exc_file_path, "rb") as f,
                        ):
                            exc_info = cloudpickle.load(f)

                    if (
                        original_exception := exc_info.get("pickled_exception")
                    ) is not None:
                        # Re-raise the actual exception object if it was
                        # successfully pickled.
                        assert isinstance(original_exception, Exception)
                        raise original_exception

                    if (original_tb := exc_info.get("traceback")) is not None:
                        # Use string-based traceback for fallback case
                        raise AssertionError(
                            f"Test {func.__name__} failed when called with"
                            f" args {args} and kwargs {kwargs}"
                            f" (exit code: {_exitcode}):\n{original_tb}"
                        ) from None

                    # Fallback to the original generic error
                    raise AssertionError(
                        f"function {func.__name__} failed when called with"
                        f" args {args} and kwargs {kwargs}"
                        f" (exit code: {_exitcode})"
                    ) from None

    return wrapper


def spawn_new_process_for_each_test(f: Callable[_P, None]) -> Callable[_P, None]:
    """Decorator to spawn a new process for each test function."""

    @functools.wraps(f)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> None:
        # Check if we're already in a subprocess
        if os.environ.get("RUNNING_IN_SUBPROCESS") == "1":
            # If we are, just run the function directly
            return f(*args, **kwargs)

        import torch.multiprocessing as mp

        with suppress(RuntimeError):
            mp.set_start_method("spawn")

        # Get the module
        module_name = f.__module__

        # Create a process with environment variable set
        env = os.environ.copy()
        env["RUNNING_IN_SUBPROCESS"] = "1"

        with tempfile.TemporaryDirectory() as tempdir:
            output_filepath = os.path.join(tempdir, "new_process.tmp")

            # `cloudpickle` allows pickling complex functions directly
            input_bytes = cloudpickle.dumps((f, output_filepath))

            repo_root = str(VLLM_PATH.resolve())

            env = dict(env or os.environ)
            env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

            cmd = [sys.executable, "-m", f"{module_name}"]

            returned = subprocess.run(
                cmd, input=input_bytes, capture_output=True, env=env
            )

            # check if the subprocess is successful
            try:
                returned.check_returncode()
            except Exception as e:
                # wrap raised exception to provide more information
                raise RuntimeError(
                    f"Error raised in subprocess:\n{returned.stderr.decode()}"
                ) from e

    return wrapper


def create_new_process_for_each_test(
    method: Literal["spawn", "fork"] | None = None,
) -> Callable[[Callable[_P, None]], Callable[_P, None]]:
    """Creates a decorator that runs each test function in a new process.

    Args:
        method: The process creation method. Can be either "spawn" or "fork".
               If not specified, it defaults to "spawn" on ROCm and XPU
               platforms and "fork" otherwise.

    Returns:
        A decorator to run test functions in separate processes.
    """
    if method is None:
        use_spawn = current_platform.is_rocm() or current_platform.is_xpu()
        method = "spawn" if use_spawn else "fork"

    assert method in ["spawn", "fork"], "Method must be either 'spawn' or 'fork'"

    if method == "fork":
        return fork_new_process_for_each_test

    return spawn_new_process_for_each_test


def large_gpu_mark(min_gb: int) -> pytest.MarkDecorator:
    """
    Get a pytest mark, which skips the test if the GPU doesn't meet
    a minimum memory requirement in GB.

    This can be leveraged via `@large_gpu_test` to skip tests in environments
    without enough resources, or called when filtering tests to run directly.
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


requires_fp8 = pytest.mark.skipif(
    not current_platform.supports_fp8(),
    reason="FP8 is not supported on this GPU (requires Hopper or "
    "Ada architecture, compute capability 8.9+)",
)


def large_gpu_test(*, min_gb: int):
    """
    Decorate a test to be skipped if no GPU is available or it does not have
    sufficient memory.

    Currently, the CI machine uses L4 GPU which has 24 GB VRAM.
    """
    mark = large_gpu_mark(min_gb)

    def wrapper(f: Callable[_P, None]) -> Callable[_P, None]:
        return mark(f)

    return wrapper


def multi_gpu_marks(*, num_gpus: int):
    """Get a collection of pytest marks to apply for `@multi_gpu_test`."""
    test_selector = pytest.mark.distributed(num_gpus=num_gpus)
    test_skipif = pytest.mark.skipif(
        cuda_device_count_stateless() < num_gpus,
        reason=f"Need at least {num_gpus} GPUs to run the test.",
    )

    return [test_selector, test_skipif]


def multi_gpu_test(*, num_gpus: int):
    """
    Decorate a test to be run only when multiple GPUs are available.
    """
    marks = multi_gpu_marks(num_gpus=num_gpus)

    def wrapper(f: Callable[_P, None]) -> Callable[_P, None]:
        func = create_new_process_for_each_test()(f)
        for mark in reversed(marks):
            func = mark(func)

        return func

    return wrapper


async def completions_with_server_args(
    prompts: list[str],
    model_name: str,
    server_cli_args: list[str],
    num_logprobs: int | None,
    max_wait_seconds: int = 240,
    max_tokens: int | list = 5,
) -> list[Completion]:
    """Construct a remote OpenAI server, obtain an async client to the
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
    """

    if isinstance(max_tokens, int):
        max_tokens = [max_tokens] * len(prompts)

    assert len(max_tokens) == len(prompts)

    outputs = None
    with RemoteOpenAIServer(
        model_name, server_cli_args, max_wait_seconds=max_wait_seconds
    ) as server:
        client = server.get_async_client()
        outputs = [
            client.completions.create(
                model=model_name,
                prompt=[p],
                temperature=0,
                stream=False,
                max_tokens=max_tok,
                logprobs=num_logprobs,
            )
            for p, max_tok in zip(prompts, max_tokens)
        ]
        outputs = await asyncio.gather(*outputs)

    assert outputs is not None, "Completion API call failed."

    return outputs


def get_client_text_generations(completions: list[Completion]) -> list[str]:
    """Extract generated tokens from the output of a
    request made to an Open-AI-protocol completions endpoint.
    """
    assert all([len(x.choices) == 1 for x in completions])
    return [x.choices[0].text for x in completions]


def get_client_text_logprob_generations(
    completions: list[Completion],
) -> list[TextTextLogprobs]:
    """Operates on the output of a request made to an Open-AI-protocol
    completions endpoint; obtains top-rank logprobs for each token in
    each {class}`SequenceGroup`
    """
    text_generations = get_client_text_generations(completions)
    text = "".join(text_generations)
    return [
        (
            text_generations,
            text,
            (None if x.logprobs is None else x.logprobs.top_logprobs),
        )
        for completion in completions
        for x in completion.choices
    ]


def has_module_attribute(module_name, attribute_name):
    """
    Helper function to check if a module has a specific attribute.
    """
    try:
        module = importlib.import_module(module_name)
        return hasattr(module, attribute_name)
    except ImportError:
        return False


def get_attn_backend_list_based_on_platform() -> list[str]:
    if current_platform.is_cuda():
        return ["FLASH_ATTN", "TRITON_ATTN", "TREE_ATTN"]
    elif current_platform.is_rocm():
        attn_backend_list = ["TRITON_ATTN"]
        try:
            import aiter  # noqa: F401

            attn_backend_list.append("ROCM_AITER_FA")
        except Exception:
            print("Skip ROCM_AITER_FA on ROCm as aiter is not installed")

        return attn_backend_list
    elif current_platform.is_xpu():
        return ["FLASH_ATTN", "TRITON_ATTN"]
    else:
        raise ValueError("Unsupported platform")


@contextmanager
def override_cutlass_fp8_supported(value: bool):
    with patch(
        "vllm.model_executor.layers.quantization.utils.w8a8_utils.cutlass_fp8_supported",
        return_value=value,
    ):
        yield


def prep_prompts(batch_size: int, ln_range: tuple[int, int] = (800, 1100)):
    """
    Generate prompts which a bunch of assignments,
    then asking for the value of one of them.
    The prompt is just under 10k tokens; sliding window is 4k
    so the answer is outside sliding window, but should still be correct.
    Args:
        batch_size: number of prompts to generate
        ln_range: an argument to control the length of the prompt
    """
    prompts: list[str] = []
    answer: list[int] = []
    indices: list[int] = []
    random.seed(1)
    for _ in range(batch_size):
        idx = random.randint(30, 90)
        indices.append(idx)
        prompt = (
            "```python\n# We set a number of variables, "
            f"x{idx} will be important later\n"
        )
        ln = random.randint(*ln_range)
        for k in range(30, ln):
            v = random.randint(10, 99)
            if k == idx:
                answer.append(v)
            prompt += f"x{k} = {v}\n"
        prompt += f"# Now, we check the value of x{idx}:\n"
        prompt += f"assert x{idx} == "
        prompts.append(prompt)
    return prompts, answer, indices


def check_answers(
    indices: list[int], answer: list[int], outputs: list[str], accept_rate: float = 0.7
):
    answer2 = [int(text[0:2].strip()) for text in outputs]
    print(list(zip(indices, zip(answer, answer2))))
    numok = 0
    for a1, a2 in zip(answer, answer2):
        if a1 == a2:
            numok += 1
    frac_ok = numok / len(answer)
    print(f"Num OK: {numok}/{len(answer)} {frac_ok}")
    assert frac_ok >= accept_rate


def flat_product(*iterables: Iterable[Any]):
    """
    Flatten lists of tuples of the cartesian product.
    Useful when we want to avoid nested tuples to allow
    test params to be unpacked directly from the decorator.

    Example:
    flat_product([(1, 2), (3, 4)], ["a", "b"]) ->
    [
      (1, 2, "a"),
      (1, 2, "b"),
      (3, 4, "a"),
      (3, 4, "b"),
    ]
    """
    for element in itertools.product(*iterables):
        normalized = (e if isinstance(e, tuple) else (e,) for e in element)
        yield tuple(itertools.chain(*normalized))


class TestFP8Layer(torch.nn.Module):
    """
    Test helper for FP8 linear operations. Creates random weights and scales
    based on quantization configuration.

    Args:
        weight_shape: Shape of the weight tensor (out_features, in_features).
        activation_quant_key: Activation quantization configuration.
        weight_quant_key: Weight quantization configuration.
        out_dtype: Output dtype. Defaults to current default dtype.
        force_kernel: Optional kernel to force use of specific implementation.
    """

    def __init__(
        self,
        weight_shape: tuple[int, int],
        activation_quant_key: QuantKey,
        weight_quant_key: QuantKey,
        out_dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        force_kernel: FP8ScaledMMLinearKernel | None = None,
    ):
        super().__init__()
        per_tensor_weights = weight_quant_key.scale.group_shape.is_per_tensor()
        is_static_activation_scale = activation_quant_key.scale.static
        weight_scale_shape = (1,) if per_tensor_weights else (weight_shape[0], 1)

        self.weight_scale = torch.rand(
            weight_scale_shape, dtype=torch.float32, device=device
        )
        self.input_scale = (
            torch.rand(1, dtype=torch.float32, device=device)
            if is_static_activation_scale
            else None
        )
        self.weight = torch.rand(weight_shape, device=device).to(dtype=FP8_DTYPE).t()
        self.input_scale_ub = None

        out_dtype = torch.get_default_dtype() if out_dtype is None else out_dtype

        self.kernel = init_fp8_linear_kernel(
            activation_quant_key=activation_quant_key,
            weight_quant_key=weight_quant_key,
            out_dtype=out_dtype,
            force_kernel=force_kernel,
        )

    def is_quant_fp8_enabled(self) -> bool:
        return self.kernel.quant_fp8.enabled()

    def forward(
        self, y: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.kernel.apply_weights(self, y, bias)


# TODO: Drop TestBlockFP8Layer in favour of a unified TestFP8Layer
# after refactoring W8A8BlockFp8LinearOp.
# https://github.com/vllm-project/vllm/issues/31818
class TestBlockFP8Layer:
    """
    Test helper for blockwise FP8 linear operations. Creates random weights
    and scales for W8A8BlockFp8LinearOp.

    This is a workaround until W8A8BlockFp8LinearOp implements the kernel
    abstraction (ScaledMMLinearKernel) for blockwise quantization.

    Args:
        weight_shape: Shape of the weight tensor (out_features, in_features).
        group_shape: Blockwise quantization group shape.
        cutlass_block_fp8_supported: Whether CUTLASS blockwise FP8 is available.
        use_aiter_and_is_supported: Whether to use aiter quantization ops.
        transpose_weights: Whether to transpose weights after creation.
    """

    def __init__(
        self,
        weight_shape: tuple[int, int],
        group_shape: GroupShape,
        cutlass_block_fp8_supported: bool = False,
        use_aiter_and_is_supported: bool = False,
        transpose_weights: bool = False,
    ):
        weight_scale_shape = weight_shape[0] // group_shape[1]
        self.weight_scale = torch.rand(
            (weight_scale_shape, weight_scale_shape), dtype=torch.float32
        )
        self.weight = torch.rand(weight_shape).to(dtype=FP8_DTYPE)
        self.input_scale = None
        if transpose_weights:
            self.weight = self.weight.t()

        self.linear_op = W8A8BlockFp8LinearOp(
            weight_group_shape=GroupShape(group_shape[1], group_shape[1]),
            act_quant_group_shape=group_shape,
            cutlass_block_fp8_supported=cutlass_block_fp8_supported,
            use_aiter_and_is_supported=use_aiter_and_is_supported,
        )

    def __call__(
        self, y: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.linear_op.apply(
            input=y,
            weight=self.weight,
            weight_scale=self.weight_scale,
            input_scale=self.input_scale,
            bias=bias,
        )

    def is_quant_fp8_enabled(self) -> bool:
        return self.linear_op.input_quant_op.enabled()
