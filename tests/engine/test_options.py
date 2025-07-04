# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import nullcontext

import pytest

from vllm.entrypoints.llm import LLM
from vllm.sampling_params import SamplingParams


@pytest.mark.parametrize("model", ["distilbert/distilgpt2"])
def test_skip_tokenizer_initialization(model: str):
    # This test checks if the flag skip_tokenizer_init skips the initialization
    # of tokenizer and detokenizer. The generated output is expected to contain
    # token ids.
    llm = LLM(
        model=model,
        skip_tokenizer_init=True,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(prompt_logprobs=True, detokenize=True)

    with pytest.raises(ValueError, match="cannot pass text prompts when"):
        llm.generate("abc", sampling_params)

    outputs = llm.generate({"prompt_token_ids": [1, 2, 3]},
                           sampling_params=sampling_params)
    assert len(outputs) > 0
    completions = outputs[0].outputs
    assert len(completions) > 0
    assert completions[0].text == ""
    assert completions[0].token_ids


@pytest.mark.parametrize("model", ["distilbert/distilgpt2"])
@pytest.mark.parametrize("enable_prompt_embeds", [True, False])
def test_enable_prompt_embeds(hf_runner, model: str,
                              enable_prompt_embeds: bool):
    prompt = "abc"

    with hf_runner(model) as hf_model:
        token_ids = hf_model.tokenizer(prompt, return_tensors="pt").input_ids
        token_ids = token_ids.to(hf_model.model.device)

        embed_layer = hf_model.model.get_input_embeddings()
        prompt_embeds = embed_layer(token_ids).squeeze(0)

    ctx = (nullcontext() if enable_prompt_embeds else pytest.raises(
        ValueError, match="set `--enable-prompt-embeds`"))

    # This test checks if the flag skip_tokenizer_init skips the initialization
    # of tokenizer and detokenizer. The generated output is expected to contain
    # token ids.
    llm = LLM(
        model=model,
        enable_prompt_embeds=enable_prompt_embeds,
        enforce_eager=True,
    )

    with ctx:
        llm.generate({"prompt_embeds": prompt_embeds})
