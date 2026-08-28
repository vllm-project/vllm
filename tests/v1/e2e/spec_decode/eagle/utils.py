# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest

from vllm import SamplingParams
from vllm.config import CompilationConfig
from vllm.platforms import current_platform

from ..utils import (
    _skip_if_insufficient_gpus_for_tp,
    assert_request_outputs_match,
    evaluate_llm_for_gsm8k,
    get_test_prompts,
)


def _run_eagle_correctness(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_setup: tuple[str, str, str, int],
    mm_enabled: bool,
    expected_accuracy_threshold: float,
    enable_chunked_prefill: bool,
    model_impl: str,
    attn_backend: str,
    vllm_runner,
):
    """
    Compare the outputs of an original LLM and a speculative LLM
    which should be the same when using eagle speculative decoding.
    """
    method, model_name, spec_model_name, tp_size = model_setup
    _skip_if_insufficient_gpus_for_tp(tp_size)

    test_prompts = get_test_prompts(mm_enabled)

    extra_kwargs: dict[str, Any] = {}
    if not mm_enabled and "Qwen3-VL" in model_name:
        # These cases only exercise text generation. Avoid profiling an unused
        # vision tower, which adds substantial memory to both reference runs.
        extra_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}

    if "Llama-4-Scout" in model_name and attn_backend == "FLASH_ATTN":
        if current_platform.is_rocm():
            print(
                "FLASH_ATTN for spec_decode not supported on "
                "ROCm currently. Changing to FLEX_ATTENTION backend."
            )
            attention_config = {"backend": "FLEX_ATTENTION"}
        else:
            attention_config = None
    else:
        attention_config = {"backend": attn_backend}

    if attn_backend == "TRITON_ATTN" and not current_platform.is_rocm():
        pytest.skip(
            "TRITON_ATTN does not support "
            "multi-token eagle spec decode on current platform"
        )

    with monkeypatch.context() as m:
        m.setenv("VLLM_MLA_DISABLE", "1")

        if attn_backend == "ROCM_AITER_FA" and current_platform.is_rocm():
            if "deepseek" in model_name.lower():
                m.setenv("VLLM_ROCM_USE_AITER", "1")
                m.delenv("VLLM_MLA_DISABLE", raising=False)
                attention_config = {"backend": "ROCM_AITER_MLA"}
            else:
                m.setenv("VLLM_ROCM_USE_AITER", "1")

        max_model_len = 2048
        max_num_batched_tokens = 128 if enable_chunked_prefill else max_model_len

        with vllm_runner(
            model_name,
            block_size=None,
            trust_remote_code=False,
            max_model_len=max_model_len,
            tensor_parallel_size=tp_size,
            attention_config=attention_config,
            enable_chunked_prefill=None,
            compilation_config=CompilationConfig(),
            **extra_kwargs,
        ) as ref_runner:
            evaluate_llm_for_gsm8k(
                ref_runner.llm,
                expected_accuracy_threshold=expected_accuracy_threshold,
            )
            ref_outputs = ref_runner.llm.chat(test_prompts, sampling_config)

        with vllm_runner(
            model_name,
            block_size=None,
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            speculative_config={
                "method": method,
                "model": spec_model_name,
                "num_speculative_tokens": 3,
                "max_model_len": max_model_len,
            },
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=enable_chunked_prefill,
            model_impl=model_impl,
            attention_config=attention_config,
            compilation_config=CompilationConfig(),
            **extra_kwargs,
        ) as spec_runner:
            # EAGLE/EAGLE3 supports async scheduling by default.
            has_async = (
                spec_runner.llm.llm_engine.vllm_config.scheduler_config.async_scheduling
            )
            assert has_async, (
                f"Expected async scheduling for {method}: target={model_name}, "
                f"draft={spec_model_name}, backend={attn_backend}; got {has_async}"
            )
            evaluate_llm_for_gsm8k(
                spec_runner.llm,
                expected_accuracy_threshold=expected_accuracy_threshold,
            )
            spec_outputs = spec_runner.llm.chat(test_prompts, sampling_config)

        assert_request_outputs_match(
            ref_outputs,
            spec_outputs,
            required_matches=int(0.6 * len(ref_outputs)) + 1,
            context=(
                f"{method} target={model_name}, draft={spec_model_name}, "
                f"backend={attn_backend}"
            ),
        )
