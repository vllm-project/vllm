# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from transformers import GPT2Config

from vllm import LLM, SamplingParams
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


def test_token_past_model_vocab(vllm_runner):
    # A text-only model whose tokenizer is larger than its vocab cannot embed
    # the extra ids; they must not reach the embedding lookup.
    with (
        vllm_runner(
            "openai-community/gpt2",
            enforce_eager=True,
            load_format="dummy",
            hf_overrides={"vocab_size": 50000},
        ) as runner,
        pytest.raises(VLLMValidationError, match="out of vocabulary"),
    ):
        runner.llm.generate({"prompt_token_ids": [50100]})


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


def test_truncation_preserves_reusable_prompt(vllm_runner):
    """Request-local truncation must not shorten a later reuse of the input."""
    with vllm_runner(
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        enforce_eager=True,
        max_model_len=128,
        max_num_seqs=1,
        gpu_memory_utilization=0.1,
    ) as runner:
        llm = runner.llm
        original_token_ids = llm.get_tokenizer().encode(
            "Remember this secret word: pineapple. What was the secret word?"
        )
        sampling_params = SamplingParams(temperature=0, max_tokens=1)
        for side in ("left", "right"):
            prompt = {"prompt_token_ids": original_token_ids.copy()}
            truncated = llm.generate(
                prompt,
                sampling_params,
                tokenization_kwargs={
                    "truncate_prompt_tokens": 6,
                    "truncation_side": side,
                },
                use_tqdm=False,
            )[0]
            expected_token_ids = (
                original_token_ids[-6:] if side == "left" else original_token_ids[:6]
            )
            assert truncated.prompt_token_ids == expected_token_ids
            assert prompt == {"prompt_token_ids": original_token_ids}
            reused = llm.generate(prompt, sampling_params, use_tqdm=False)[0]
            assert reused.prompt_token_ids == original_token_ids
