# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm._aiter_ops import is_aiter_found, rocm_aiter_ops
from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode
from vllm.envs import disable_envs_cache
from vllm.platforms import current_platform

from ....conftest import VllmRunner
from ....utils import (
    assert_rocm_custom_allreduce_backend_state_on_worker,
    multi_gpu_test,
)

PROMPTS = ["Hello, my name is", "The capital of France is"]


def _run_generation(
    vllm_runner: type[VllmRunner],
    monkeypatch: pytest.MonkeyPatch,
    compilation_config: CompilationConfig,
    *,
    model: str,
    max_tokens: int,
    use_aiter_custom_ar: bool,
    quick_reduce_quantization: str,
) -> list[tuple[list[int], str]]:
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        m.setenv("VLLM_ROCM_USE_AITER", "1")
        m.setenv(
            "VLLM_ROCM_USE_AITER_CUSTOM_AR",
            "1" if use_aiter_custom_ar else "0",
        )
        m.setenv("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION", quick_reduce_quantization)
        disable_envs_cache()
        rocm_aiter_ops.refresh_env_variables()

        with vllm_runner(
            model,
            dtype="half",
            tensor_parallel_size=2,
            compilation_config=compilation_config,
            max_model_len=256,
            max_num_seqs=len(PROMPTS),
            gpu_memory_utilization=0.7,
        ) as llm:
            llm.get_llm().collective_rpc(
                assert_rocm_custom_allreduce_backend_state_on_worker,
                args=(use_aiter_custom_ar, quick_reduce_quantization),
            )

            return llm.generate_greedy(PROMPTS, max_tokens)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only")
@pytest.mark.skipif(not is_aiter_found(), reason="AITER is not installed")
@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "quick_reduce_quantization",
    [
        pytest.param("FP", id="quick-reduce-on"),
        pytest.param("NONE", id="quick-reduce-off"),
    ],
)
@pytest.mark.parametrize(
    "cudagraph_mode",
    [
        pytest.param(CUDAGraphMode.NONE, id="cudagraph-none"),
        pytest.param(CUDAGraphMode.FULL, id="cudagraph-full"),
    ],
)
@pytest.mark.parametrize(
    "model,max_tokens",
    [
        pytest.param("facebook/opt-125m", 8, id="opt-125m"),
    ],
)
def test_rocm_aiter_custom_ar_e2e(
    vllm_runner: type[VllmRunner],
    monkeypatch: pytest.MonkeyPatch,
    cudagraph_mode: CUDAGraphMode,
    quick_reduce_quantization: str,
    model: str,
    max_tokens: int,
):
    compilation_mode = (
        CompilationMode.NONE
        if cudagraph_mode == CUDAGraphMode.NONE
        else CompilationMode.VLLM_COMPILE
    )
    compilation_config = CompilationConfig(
        mode=compilation_mode,
        cudagraph_mode=cudagraph_mode,
    )

    baseline_generations = _run_generation(
        vllm_runner,
        monkeypatch,
        compilation_config,
        model=model,
        max_tokens=max_tokens,
        use_aiter_custom_ar=False,
        quick_reduce_quantization=quick_reduce_quantization,
    )
    aiter_custom_ar_generations = _run_generation(
        vllm_runner,
        monkeypatch,
        compilation_config,
        model=model,
        max_tokens=max_tokens,
        use_aiter_custom_ar=True,
        quick_reduce_quantization=quick_reduce_quantization,
    )

    assert aiter_custom_ar_generations == baseline_generations
