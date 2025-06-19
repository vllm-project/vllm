# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import os
import weakref
from contextlib import ExitStack

import pytest

from tests.utils import wait_for_gpu_memory_to_clear
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig
from vllm.platforms import current_platform


@contextlib.contextmanager
def temporary_environ(env_vars):
    """
    Temporarily set environment variables and restore them afterward.
    We have to do this vs monkeypatch because monkeypatch doesn't work
    with "module" scoped fixtures.
    """
    original_env = {k: os.environ.get(k) for k in env_vars}
    try:
        os.environ.update(env_vars)
        yield
    finally:
        for k, v in original_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(scope="class")
def llm_pair(request):
    model = request.param

    with temporary_environ({
            "VLLM_USE_V1": "1",
            "VLLM_FLASH_ATTN_VERSION": "3"
    }):
        full = LLM(
            model=model,
            gpu_memory_utilization=0.45,
            trust_remote_code=True,
            max_model_len=1024,
            compilation_config=CompilationConfig(full_cuda_graph=True),
        )
        piecewise = LLM(
            model=model,
            gpu_memory_utilization=0.45,
            trust_remote_code=True,
            max_model_len=1024,
            compilation_config=CompilationConfig(),
        )

    # PyTest caches the fixture values so we use weakref.proxy to enable GC
    yield weakref.proxy(full), weakref.proxy(piecewise)
    del full
    del piecewise

    wait_for_gpu_memory_to_clear(
        devices=[0],
        threshold_ratio=0.1,
    )


@pytest.mark.parametrize(
    "llm_pair",
    [
        # Model names for the llm_pair fixture
        "deepseek-ai/DeepSeek-V2-Lite",
        "Qwen/Qwen2-1.5B-Instruct"
    ],
    indirect=True)
@pytest.mark.skipif(current_platform.get_device_capability() != (9, 0),
                    reason="Only Hopper GPUs support FA3 and FlashMLA")
class TestFullCUDAGraph:
    """
    Use a class such that an llm pair is constructed once for all
    batch_size/max_tokens combinations and released immediately after.

    Module-scope fixtures would stick around the whole time,
    meaning there would be multiple LLM instances hogging memory simultaneously.
    """

    @pytest.mark.parametrize(("batch_size", "max_tokens"), [
        (1, 10),
        (7, 10),
        (16, 10),
        (25, 10),
        (32, 10),
        (45, 10),
        (64, 10),
        (123, 10),
        (8, 5),
        (8, 30),
    ])
    def test_full_cudagraph(self, batch_size, max_tokens,
                            llm_pair: tuple[LLM, LLM]):
        """
        Test various batch sizes and max_tokens to ensure that the
        full cudagraph compilation works for padded cases too.
        """

        piecewise_llm, full_cudagraph_llm = llm_pair

        prompts = ["Hello, my name is"] * batch_size
        sampling_params = SamplingParams(temperature=0.0,
                                         max_tokens=max_tokens,
                                         top_p=0.95)

        piecewise_responses = piecewise_llm.generate(prompts, sampling_params)
        full_responses = full_cudagraph_llm.generate(prompts, sampling_params)

        # Check that all responses are the same
        for piecewise_res, full_res in zip(piecewise_responses,
                                           full_responses):
            assert piecewise_res.outputs[0].text == full_res.outputs[0].text


@pytest.mark.parametrize(
    "model, supported",
    [
        ("Qwen/Qwen2-1.5B-Instruct", True),
        # MLA does not support capturing CUDA Graphs with size > max_num_seqs
        ("deepseek-ai/DeepSeek-V2-Lite", False),
    ])
@pytest.mark.skipif(current_platform.get_device_capability() != (9, 0),
                    reason="Only Hopper GPUs support FA3 and FlashMLA")
def test_lower_max_num_seqs(model, supported):
    with temporary_environ({
            "VLLM_USE_V1": "1",
            "VLLM_FLASH_ATTN_VERSION": "3"
    }), ExitStack() as stack:
        if not supported:
            stack.enter_context(pytest.raises(RuntimeError))

        llm = LLM(model=model,
                  max_num_seqs=256,
                  trust_remote_code=True,
                  max_model_len=1024,
                  compilation_config=CompilationConfig(
                      full_cuda_graph=True,
                      cudagraph_capture_sizes=[64, 256, 512]))
        llm.generate(["Hello, my name is"] * 10)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Skip if not cuda")
def test_full_cudagraph_with_invalid_backend():
    with temporary_environ({
            "VLLM_USE_V1": "1",
            "VLLM_FLASH_ATTN_VERSION":
            "2"  #FA2 not supported with full_cuda_graph
    }), pytest.raises(RuntimeError):
        LLM(model="Qwen/Qwen2-1.5B-Instruct",
            compilation_config=CompilationConfig(full_cuda_graph=True))
