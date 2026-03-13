# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm import TokensPrompt


@pytest.mark.parametrize(
    "model",
    ["Qwen/Qwen3-0.6B"],
)
@torch.inference_mode
def test_extract_hidden_states(hf_runner, vllm_runner, model: str):
    n_prompt_tokens = [55, 56, 57]
    token_prompts = [[1024 + i for i in range(n)] for n in n_prompt_tokens]

    with vllm_runner(
        model,
        max_model_len=128,
        enforce_eager=True,
        runner="pooling",
        enable_prefix_caching=True,
    ) as vllm_model:
        pooling_outputs = vllm_model.llm.encode(
            [TokensPrompt(prompt_token_ids=t) for t in token_prompts],
            pooling_task="token_embed",
        )

        for n, output in zip(n_prompt_tokens, pooling_outputs):
            assert len(output.prompt_token_ids) == n
            assert len(output.outputs.data) == n
            assert output.num_cached_tokens == 0

        # test enable_prefix_caching plus all pooling
        # we need to skip reading cache at this request by
        # request.skip_reading_prefix_cache
        pooling_outputs = vllm_model.llm.encode(
            [TokensPrompt(prompt_token_ids=t) for t in token_prompts],
            pooling_task="token_embed",
        )

        for n, output in zip(n_prompt_tokens, pooling_outputs):
            assert len(output.prompt_token_ids) == n
            assert len(output.outputs.data) == n
            assert output.num_cached_tokens == 0

        # skip_reading_prefix_cache can still write to cache
        # to accelerate following requests
        pooling_outputs = vllm_model.llm.encode(
            [TokensPrompt(prompt_token_ids=t) for t in token_prompts],
            pooling_task="embed",
        )

        for n, output in zip(n_prompt_tokens, pooling_outputs):
            assert len(output.prompt_token_ids) == n
            assert output.num_cached_tokens > 0
