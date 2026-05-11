# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for vLLM `vllm/v1/engine/processor.Processor._validate_model_input()`
handling of maximum context length for decoder models.

This test ensures:
- A prompt that is one token shorter than the model's maximum context length
  can be processed successfully when requesting one additional token.
- A prompt that reaches the model's maximum context length throws a
  `ValueError` when requesting at least one additional token.
"""

import pytest

from tests.conftest import VllmRunner
from tests.utils import create_new_process_for_each_test
from vllm.exceptions import VLLMValidationError


@create_new_process_for_each_test()
@pytest.mark.parametrize("model, max_model_len", [("JackFram/llama-160m", 2048)])
@pytest.mark.parametrize(
    "prompt_len, max_tokens",
    [
        (2047, 1),  # prompt_len = max_model_len - 1 -> allowed
        (2048, 1),  # prompt_len = max_model_len -> not allowed
    ],
)
def test_decoder_max_context_length_validation(
    model: str,
    max_model_len: int,
    vllm_runner: type[VllmRunner],
    prompt_len: int,
    max_tokens: int,
) -> None:
    """Check vLLM decoder model input validation for edge cases where
    the prompt length is (almost) equal to the max model length."""

    prompt_ids = [[43] * prompt_len]

    with vllm_runner(
        model_name=model,
        tokenizer_name=model,
        max_model_len=max_model_len,
        max_num_seqs=1,
        tensor_parallel_size=1,
    ) as vllm_model:
        if prompt_len + max_tokens <= max_model_len:
            # Should succeed as constraints are met
            vllm_model.generate_greedy(prompt_ids, max_tokens)
        else:
            # Should raise the ValueError defined in
            # vllm/v1/engine/processor.Processor_validate_model_input()
            expected_msg = (
                f"The decoder prompt (length {prompt_len}) plus the number of "
                f"requested output tokens (at least 1) is longer than "
                f"the maximum model length of {max_model_len}. "
                "Make sure that `max_model_len` is no smaller than the number of "
                "text tokens (prompt + requested output tokens)."
            )
            with pytest.raises(ValueError) as excinfo:
                vllm_model.generate_greedy(prompt_ids, max_tokens)
            assert expected_msg in str(excinfo.value)


@create_new_process_for_each_test()
@pytest.mark.parametrize("model", ["JackFram/llama-160m"])
def test_auto_fit_max_model_len_rejects_oversized_input(
    model: str,
    vllm_runner: type[VllmRunner],
) -> None:
    """When max_model_len='auto' and KV cache memory is very limited,
    the engine auto-fits max_model_len to a small value. The frontend
    must see this reduced value and reject prompts that exceed it,
    rather than accepting them and hanging."""

    # Use a tiny KV cache budget to force auto-fit to a very small
    # max_model_len (e.g. ~16 tokens).
    kv_cache_bytes = 1_000_000  # 1 MB

    with vllm_runner(
        model_name=model,
        max_model_len=-1,
        max_num_seqs=1,
        enforce_eager=True,
        kv_cache_memory_bytes=kv_cache_bytes,
        load_format="dummy",
    ) as vllm_model:
        auto_fitted_len = (
            vllm_model.llm.llm_engine.vllm_config.model_config.max_model_len
        )
        # Sanity check: auto-fit should have reduced it well below the
        # model's native context length.
        assert auto_fitted_len < 2048, (
            f"Expected auto-fit to reduce max_model_len significantly, "
            f"but got {auto_fitted_len}"
        )

        # A prompt longer than the auto-fitted length must be rejected.
        oversized_prompt = [[43] * (auto_fitted_len + 10)]
        with pytest.raises(VLLMValidationError, match="Please reduce the length"):
            vllm_model.generate_greedy(oversized_prompt, max_tokens=4)
