# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import os
import weakref
from dataclasses import dataclass
from typing import Optional

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


@dataclass
class BackendConfig:
    name: str
    env_vars: dict
    comp_config: dict
    specific_gpu_arch: Optional[tuple] = None


# Define all backend configurations of full cudagraph to be tested
backend_configs = {
    # FA3 on Hopper
    "FA3":
    BackendConfig(name="FA3",
                  env_vars={"VLLM_FLASH_ATTN_VERSION": "3"},
                  comp_config={
                      "cudagraph_mode": "FULL",
                  },
                  specific_gpu_arch=(9, 0)),
    # FlashMLA on Hopper
    "FlashMLA":
    BackendConfig(name="FlashMLA",
                  env_vars={
                      "VLLM_ATTENTION_BACKEND": "FLASHMLA",
                  },
                  comp_config={
                      "cudagraph_mode": "FULL_AND_PIECEWISE",
                  },
                  specific_gpu_arch=(9, 0)),
    # FlashAttention MLA on Hopper
    "FlashAttentionMLA":
    BackendConfig(name="FlashAttentionMLA",
                  env_vars={
                      "VLLM_ATTENTION_BACKEND": "FLASH_ATTN_MLA",
                  },
                  comp_config={
                      "cudagraph_mode": "FULL_DECODE_ONLY",
                  },
                  specific_gpu_arch=(9, 0)),
    # Cutlass MLA on Blackwell
    "CutlassMLA":
    BackendConfig(
        name="CutlassMLA",
        env_vars={
            "VLLM_USE_V1": "1",
            "VLLM_ATTENTION_BACKEND": "CUTLASS_MLA",
            "FORCE_NUM_KV_SPLITS":
            "1",  # TODO: remove this when hang issue is fixed
        },
        comp_config={
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "cudagraph_capture_sizes": [16, 32, 64, 128, 256, 512],
        },
        specific_gpu_arch=(10, 0)),
    # FA2
    "FA2":
    BackendConfig(name="FA2",
                  env_vars={"VLLM_FLASH_ATTN_VERSION": "2"},
                  comp_config={
                      "cudagraph_mode": "FULL",
                  }),
    # Triton Attention
    "TritonAttn":
    BackendConfig(name="TritonAttn",
                  env_vars={"VLLM_ATTENTION_BACKEND": "TRITON_ATTN_VLLM_V1"},
                  comp_config={
                      "cudagraph_mode": "FULL",
                  }),
    # FlashInfer
    "FlashInfer":
    BackendConfig(name="FlashInfer",
                  env_vars={"VLLM_ATTENTION_BACKEND": "FLASHINFER"},
                  comp_config={
                      "cudagraph_mode": "FULL_AND_PIECEWISE",
                  }),
}

test_params_full_cudagraph = []

# deepseek-ai/DeepSeek-V2-Lite with MLA
MLA_backends = ["FlashMLA", "FlashAttentionMLA", "CutlassMLA"]
for mla_backend in MLA_backends:
    test_params_full_cudagraph.append(
        pytest.param(
            ("deepseek-ai/DeepSeek-V2-Lite", backend_configs[mla_backend])))

# Qwen/Qwen2-1.5B-Instruct with other backends
other_backend_configs = [
    backend_configs[c] for c in backend_configs if c not in MLA_backends
]
for backend_config in other_backend_configs:
    test_params_full_cudagraph.append(
        pytest.param(("Qwen/Qwen2-1.5B-Instruct", backend_config)))


@pytest.fixture(scope="class")
def llm_pair(request):
    model, backend_config = request.param

    # Dynamically skip test if GPU capability is not met
    if backend_config.specific_gpu_arch and backend_config.specific_gpu_arch\
        != current_platform.get_device_capability():
        if backend_config.specific_gpu_arch == (9, 0):
            pytest.skip("Only Hopper GPUs support FA3 and FlashMLA")
        elif backend_config.specific_gpu_arch == (10, 0):
            pytest.skip("Only Blackwell GPUs support Cutlass MLA")

    env_vars = {
        "VLLM_USE_V1": "1",
        # Force native sampler to avoid potential nondeterminism in FlashInfer
        # when per-request generators are not used in V1.
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        **backend_config.env_vars,
    }
    with temporary_environ(env_vars):
        full = LLM(
            model=model,
            gpu_memory_utilization=0.43,
            trust_remote_code=True,
            max_model_len=1024,
            max_num_seqs=128,
            compilation_config=\
                CompilationConfig(**backend_config.comp_config),
            generation_config="vllm",
            seed=42,
        )
        piecewise = LLM(
            model=model,
            gpu_memory_utilization=0.43,
            trust_remote_code=True,
            max_model_len=1024,
            max_num_seqs=128,
            compilation_config=CompilationConfig(cudagraph_mode="PIECEWISE"),
            generation_config="vllm",
            seed=42,
        )

    # PyTest caches the fixture values so we use weakref.proxy to enable GC
    yield weakref.proxy(full), weakref.proxy(piecewise)
    del full
    del piecewise

    wait_for_gpu_memory_to_clear(
        devices=[0],
        threshold_ratio=0.1,
    )


@pytest.mark.parametrize("llm_pair", test_params_full_cudagraph, indirect=True)
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

        full_cudagraph_llm, piecewise_llm = llm_pair

        prompts = ["the quick brown fox"] * batch_size
        # Use purely greedy decoding to avoid top-p truncation sensitivity
        # that can amplify tiny numeric differences across runtimes.
        sampling_params = SamplingParams(temperature=0.0,
                                         max_tokens=max_tokens,
                                         top_p=1.0)

        piecewise_responses = piecewise_llm.generate(prompts, sampling_params)
        full_responses = full_cudagraph_llm.generate(prompts, sampling_params)

        # Check that all responses are the same
        for piecewise_res, full_res in zip(piecewise_responses,
                                           full_responses):
            assert piecewise_res.outputs[0].text.lower() == \
                full_res.outputs[0].text.lower()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Skip if not cuda")
def test_full_cudagraph_with_invalid_backend():
    with temporary_environ({
            "VLLM_USE_V1": "1",
            "VLLM_ATTENTION_BACKEND": "FLEX_ATTENTION"
            # Flex_Attention is not supported with full cuda graph
    }), pytest.raises(RuntimeError):
        LLM(model="Qwen/Qwen2-1.5B-Instruct",
            compilation_config=CompilationConfig(cudagraph_mode="FULL"))
