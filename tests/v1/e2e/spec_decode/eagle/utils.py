# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest

from vllm import SamplingParams
from vllm.config import CompilationConfig
from vllm.platforms import current_platform

from ..utils import (
    _skip_if_insufficient_gpus_for_tp,
    check_spec_decode_matches_reference,
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
    """Compare the outputs of an original LLM and a speculative LLM
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

        check_spec_decode_matches_reference(
            vllm_runner,
            sampling_config,
            test_prompts,
            target_model=model_name,
            target_engine_kwargs=dict(
                block_size=None,
                trust_remote_code=False,
                max_model_len=max_model_len,
                tensor_parallel_size=tp_size,
                attention_config=attention_config,
                enable_chunked_prefill=None,
                compilation_config=CompilationConfig(),
                **extra_kwargs,
            ),
            spec_model=model_name,
            spec_engine_kwargs=dict(
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
            ),
            # Heuristic: expect at least 60% of the prompts to match exactly
            # Upon failure, inspect the outputs to check for inaccuracy.
            prompts_required_matches=int(0.6 * len(test_prompts)) + 1,
            gsm8k_spec_accuracy_threshold=expected_accuracy_threshold,
            gsm8k_target_accuracy_threshold=expected_accuracy_threshold,
        )
