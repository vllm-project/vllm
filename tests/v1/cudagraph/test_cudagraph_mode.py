# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import weakref
from contextlib import ExitStack

import pytest

from tests.utils import wait_for_gpu_memory_to_clear
from tests.v1.attention.utils import full_cg_backend_configs as backend_configs
from vllm import LLM
from vllm.config import CompilationConfig, CompilationMode
from vllm.platforms import current_platform

# test attention backend and cudagraph_mode combo
# (backend_name, cudagraph_mode, supported)
if current_platform.is_rocm():
    combo_cases_1 = [
        ("RocmAttn", "FULL", True),
        ("RocmAttn", "FULL_AND_PIECEWISE", True),
        ("TritonAttn", "FULL", True),
        ("TritonAttn", "FULL_AND_PIECEWISE", True),
    ]
else:
    combo_cases_1 = [
        ("FA3", "FULL", True),
        ("FA3", "FULL_AND_PIECEWISE", True),
        ("FA2", "FULL", True),  # Should fallback to FULL_AND_PIECEWISE
        ("FA2", "FULL_AND_PIECEWISE", True),
        ("FlashInfer", "FULL", True),  # Should fallback to FULL_AND_PIECEWISE
        ("FlashInfer", "FULL_AND_PIECEWISE", True),
    ]


@pytest.mark.parametrize("backend_name, cudagraph_mode, supported", combo_cases_1)
def test_backend_and_cudagraph_mode_combo(backend_name, cudagraph_mode, supported):
    if backend_name == "FlashInfer":
        try:
            import flashinfer  # noqa: F401
        except ImportError:
            pytest.skip("FlashInfer is not installed")
    backend_config = backend_configs[backend_name]
    # Dynamically skip test if GPU capability is not met
    if (
        backend_config.specific_gpu_arch
        and backend_config.specific_gpu_arch != current_platform.get_device_capability()
    ):
        pytest.skip("Only Hopper GPUs support FA3 and FlashMLA")

    attention_config = backend_config.attention_config

    with ExitStack() as stack:
        if not supported:
            stack.enter_context(pytest.raises(Exception))

        llm = LLM(
            model="Qwen/Qwen2-1.5B-Instruct",
            max_num_seqs=256,
            trust_remote_code=True,
            gpu_memory_utilization=0.45,
            max_model_len=1024,
            attention_config=attention_config,
            compilation_config=CompilationConfig(
                mode=CompilationMode.VLLM_COMPILE, cudagraph_mode=cudagraph_mode
            ),
        )
        llm.generate(["Hello, my name is"] * 10)
    # when above code raises, `llm` may be undefined, so we need to catch that
    try:
        llm = weakref.proxy(llm)
        del llm
    except UnboundLocalError:
        pass

    wait_for_gpu_memory_to_clear(
        devices=[0],
        threshold_ratio=0.1,
    )


# test cudagraph_mode with different compilation mode.
# (backend_name, cudagraph_mode, compilation_mode, supported)
attn_backend = "RocmAttn" if current_platform.is_rocm() else "FA2"

combo_cases_2 = [
    (attn_backend, "FULL", CompilationMode.NONE, True),
    (attn_backend, "FULL", CompilationMode.VLLM_COMPILE, True),
    (attn_backend, "PIECEWISE", CompilationMode.NONE, True),
    (attn_backend, "PIECEWISE", CompilationMode.VLLM_COMPILE, True),
    (attn_backend, "FULL_AND_PIECEWISE", CompilationMode.NONE, True),
    (attn_backend, "FULL_AND_PIECEWISE", CompilationMode.VLLM_COMPILE, True),
    (attn_backend, "FULL_DECODE_ONLY", CompilationMode.NONE, True),
    (attn_backend, "FULL_DECODE_ONLY", CompilationMode.VLLM_COMPILE, True),
    (attn_backend, "NONE", CompilationMode.NONE, True),
    (attn_backend, "NONE", CompilationMode.VLLM_COMPILE, True),
]


@pytest.mark.parametrize(
    "backend_name,cudagraph_mode,compilation_mode,supported", combo_cases_2
)
def test_cudagraph_compilation_combo(
    backend_name, cudagraph_mode, compilation_mode, supported
):
    backend_config = backend_configs[backend_name]
    attention_config = backend_config.attention_config

    with ExitStack() as stack:
        if not supported:
            stack.enter_context(pytest.raises(Exception))

        llm = LLM(
            model="Qwen/Qwen2-1.5B-Instruct",
            max_num_seqs=256,
            trust_remote_code=True,
            gpu_memory_utilization=0.45,
            max_model_len=1024,
            attention_config=attention_config,
            compilation_config=CompilationConfig(
                mode=compilation_mode, cudagraph_mode=cudagraph_mode
            ),
        )
        llm.generate(["Hello, my name is"] * 10)
    # when above code raises, `llm` may be undefined, so we need to catch that
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
