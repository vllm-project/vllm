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


def check_mtp_correctness(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_setup: tuple[str, str, int, str | None],
    mm_enabled: bool,
    expected_accuracy_threshold: float,
    vllm_runner,
):
    """Compare the outputs of a original LLM and a speculative LLM
    which should be the same when using MTP speculative decoding. Due to some variance
    in the engine, it is possible for some outputs to differ, so we expect that at least
    6/10 output tokens match exactly, and that the GSM8k accuracy is above a precomputed
    reference threshold for each model.
    """
    method, model_name, tp_size, draft_model = model_setup
    _skip_if_insufficient_gpus_for_tp(tp_size)

    # Generate test prompts inside the function instead of using fixture
    test_prompts = get_test_prompts(mm_enabled)
    with monkeypatch.context() as m:
        m.setenv("VLLM_MLA_DISABLE", "1")

        attn_backend = "TRITON_ATTN" if current_platform.is_rocm() else "auto"

        # Skip multimodal profiling for models that don't need it in this test.
        extra_kwargs: dict[str, Any] = {}
        if "Qwen3.5" in model_name:
            extra_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        elif "gemma-4" in model_name:
            extra_kwargs["limit_mm_per_prompt"] = {"image": 0, "audio": 0}

        speculative_config: dict[str, Any] = {
            "method": method,
            "num_speculative_tokens": 1,
            "max_model_len": 2048,
        }
        if draft_model is not None:
            speculative_config["model"] = draft_model
            speculative_config["num_speculative_tokens"] = 2

        engine_kwargs: dict[str, Any] = dict(
            block_size=None,
            max_model_len=2048,
            tensor_parallel_size=tp_size,
            trust_remote_code=True,
            attention_backend=attn_backend,
            enable_chunked_prefill=None,
            compilation_config=CompilationConfig(),
            **extra_kwargs,
        )
        check_spec_decode_matches_reference(
            vllm_runner,
            sampling_config,
            test_prompts,
            target_model=model_name,
            target_engine_kwargs=engine_kwargs,
            spec_model=model_name,
            spec_engine_kwargs={
                **engine_kwargs,
                "speculative_config": speculative_config,
            },
            # Heuristic: expect at least 80% of the prompts to match exactly
            # Upon failure, inspect the outputs to check for inaccuracy.
            prompts_required_matches=int(0.8 * len(test_prompts)) + 1,
            gsm8k_spec_accuracy_threshold=expected_accuracy_threshold,
            gsm8k_target_accuracy_threshold=expected_accuracy_threshold,
        )
