# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.exceptions import VLLMValidationError


def test_empty_prompt(vllm_runner):
    with (
        vllm_runner("openai-community/gpt2", enforce_eager=True) as runner,
        pytest.raises(VLLMValidationError, match="decoder prompt cannot be empty"),
    ):
        runner.llm.generate([""])


def test_out_of_vocab_token(vllm_runner):
    with (
        vllm_runner("openai-community/gpt2", enforce_eager=True) as runner,
        pytest.raises(VLLMValidationError, match="out of vocabulary"),
    ):
        runner.llm.generate({"prompt_token_ids": [999999]})


def test_out_of_vocab_token_skip_tokenizer_init(vllm_runner):
    # Without a tokenizer the model vocab size is the only bound; an
    # out-of-range id must not reach the embedding lookup.
    with (
        vllm_runner(
            "openai-community/gpt2",
            enforce_eager=True,
            skip_tokenizer_init=True,
        ) as runner,
        pytest.raises(VLLMValidationError, match="out of vocabulary"),
    ):
        runner.llm.generate({"prompt_token_ids": [999999]})


def test_raw_input_only_model_requires_mm_data(vllm_runner):
    # A prompt without multimodal data would give such a model nothing to run on,
    # so it must be rejected before it reaches the engine.
    with (
        vllm_runner(
            "nvidia/Nemotron-3-Diarization",
            runner="pooling",
            skip_tokenizer_init=True,
        ) as runner,
        pytest.raises(VLLMValidationError, match="must include `multi_modal_data`"),
    ):
        runner.llm.encode({"prompt_token_ids": [0]}, pooling_task="plugin")


def test_require_mm_embeds(vllm_runner):
    with (
        vllm_runner(
            "llava-hf/llava-1.5-7b-hf",
            enforce_eager=True,
            enable_mm_embeds=False,
        ) as runner,
        pytest.raises(ValueError, match="--enable-mm-embeds"),
    ):
        runner.llm.generate(
            {
                "prompt": "<image>",
                "multi_modal_data": {"image": torch.empty(1, 1, 1)},
            }
        )
