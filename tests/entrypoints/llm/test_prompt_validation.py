# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from transformers import GPT2Config

from vllm import LLM
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


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_model_seed_out_of_numpy_range_fails_during_initialization(tmp_path, seed):
    GPT2Config(
        n_layer=1,
        n_head=1,
        n_embd=8,
        vocab_size=16,
        bos_token_id=1,
        eos_token_id=2,
    ).save_pretrained(tmp_path)

    with pytest.raises(ValueError, match="seed"):
        LLM(
            model=str(tmp_path),
            seed=seed,
            skip_tokenizer_init=True,
            load_format="dummy",
            max_model_len=16,
            enforce_eager=True,
        )
