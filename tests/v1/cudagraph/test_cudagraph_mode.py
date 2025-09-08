# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import os
import weakref
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Optional

import pytest

from tests.utils import wait_for_gpu_memory_to_clear
from vllm import LLM
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
    # FA2
    "FA2":
    BackendConfig(name="FA2",
                  env_vars={"VLLM_FLASH_ATTN_VERSION": "2"},
                  comp_config={
                      "cudagraph_mode": "FULL_AND_PIECEWISE",
                  }),
    # Triton Attention
    "TritonAttn":
    BackendConfig(name="TritonAttn",
                  env_vars={"VLLM_ATTENTION_BACKEND": "TRITON_ATTN_VLLM_V1"},
                  comp_config={
                      "cudagraph_mode": "FULL_AND_PIECEWISE",
                  }),
    # FlashInfer
    "FlashInfer":
    BackendConfig(name="FlashInfer",
                  env_vars={"VLLM_ATTENTION_BACKEND": "FLASHINFER"},
                  comp_config={
                      "cudagraph_mode": "FULL_AND_PIECEWISE",
                  }),
}

# test attention backend and cudagraph_mode combo
# (backend_name, cudagraph_mode, supported)
combo_cases_1 = [
    ("FA3", "FULL", True),
    ("FA3", "FULL_AND_PIECEWISE", True),
    ("FA2", "FULL", True),  # Should fallback to FULL_AND_PIECEWISE
    ("FA2", "FULL_AND_PIECEWISE", True),
    ("FlashInfer", "FULL", True),  # Should fallback to FULL_AND_PIECEWISE
    ("FlashInfer", "FULL_AND_PIECEWISE", True),
]


@pytest.mark.parametrize("combo_case", combo_cases_1)
def test_backend_and_cudagraph_mode_combo(combo_case):
    backend_name, cudagraph_mode, supported = combo_case
    if backend_name == "FlashInfer":
        try:
            import flashinfer  # noqa: F401
        except ImportError:
            pytest.skip("FlashInfer is not installed")
    backend_config = backend_configs[backend_name]
    # Dynamically skip test if GPU capability is not met
    if backend_config.specific_gpu_arch and backend_config.specific_gpu_arch\
        != current_platform.get_device_capability():
        pytest.skip("Only Hopper GPUs support FA3 and FlashMLA")

    env_vars = {"VLLM_USE_V1": "1", **backend_configs[backend_name].env_vars}

    with temporary_environ(env_vars), ExitStack() as stack:
        if not supported:
            stack.enter_context(pytest.raises(Exception))

        llm = LLM(model="Qwen/Qwen2-1.5B-Instruct",
                  max_num_seqs=256,
                  trust_remote_code=True,
                  gpu_memory_utilization=0.45,
                  max_model_len=1024,
                  compilation_config=CompilationConfig(
                      level=3, cudagraph_mode=cudagraph_mode))
        llm.generate(["Hello, my name is"] * 10)

    try:
        llm = weakref.proxy(llm)
        del llm
    except UnboundLocalError:
        pass

    wait_for_gpu_memory_to_clear(
        devices=[0],
        threshold_ratio=0.1,
    )


# test cudagraph_mode with different compilation level.
# (backend_name, cudagraph_mode, compilation_level, supported)
combo_cases_2 = [
    ("FA2", "FULL", 0, True),  # no compilation + full cudagraph
    ("FA2", "FULL", 3, True),  # piecewise compilation + full cudagraph
    ("FA2", "PIECEWISE", 0, False),  # no compilation + piecewise cudagraph
    ("FA2", "PIECEWISE", 3,
     True),  # piecewise compilation + piecewise cudagraph
    ("FA2", "FULL_AND_PIECEWISE", 0,
     False),  # piecewise cudagraph not supported without piecewise compilation
    ("FA2", "FULL_AND_PIECEWISE", 3, True),
    ("FA2", "FULL_DECODE_ONLY", 0, True),
    ("FA2", "FULL_DECODE_ONLY", 3, True),
    ("FA2", "NONE", 0, True),  # no compilation + no cudagraph
    ("FA2", "NONE", 3, True),  # piecewise compilation + no cudagraph
]


@pytest.mark.parametrize("combo_case", combo_cases_2)
def test_cudagraph_compilation_combo(combo_case):
    backend_name, cudagraph_mode, compilation_level, supported\
        = combo_case

    env_vars = {"VLLM_USE_V1": "1", **backend_configs[backend_name].env_vars}

    with temporary_environ(env_vars), ExitStack() as stack:
        if not supported:
            stack.enter_context(pytest.raises(Exception))

        llm = LLM(model="Qwen/Qwen2-1.5B-Instruct",
                  max_num_seqs=256,
                  trust_remote_code=True,
                  gpu_memory_utilization=0.45,
                  max_model_len=1024,
                  compilation_config=CompilationConfig(
                      level=compilation_level, cudagraph_mode=cudagraph_mode))
        llm.generate(["Hello, my name is"] * 10)
    try:
        llm = weakref.proxy(llm)
        del llm
    except UnboundLocalError:
        pass
    finally:
        wait_for_gpu_memory_to_clear(
            devices=[0],
            threshold_ratio=0.1,
        )
