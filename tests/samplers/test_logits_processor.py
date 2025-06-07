# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import SamplingParams

MODELS = ["distilbert/distilgpt2"]


@pytest.fixture(scope="function", autouse=True)
def use_v0_only(monkeypatch):
    """
    This file tests V0 internals, so set VLLM_USE_V1=0.
    """
    monkeypatch.setenv('VLLM_USE_V1', '0')


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_logits_processor_force_generate(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:
    with vllm_runner(model, dtype=dtype) as vllm_model:
        tokenizer = vllm_model.model.get_tokenizer()
        repeat_times = 2
        enforced_answers = " vLLM"
        vllm_token_ids = tokenizer.encode(enforced_answers,
                                          add_special_tokens=False)
        max_tokens = len(vllm_token_ids) * repeat_times

        def pick_vllm(token_ids, logits):
            token_id = vllm_token_ids[len(token_ids) % len(vllm_token_ids)]
            logits[token_id] = torch.finfo(logits.dtype).max
            return logits

        params_with_logprobs = SamplingParams(
            logits_processors=[pick_vllm],
            prompt_logprobs=3,
            max_tokens=max_tokens,
        )

        # test logits_processors when prompt_logprobs is not None
        vllm_model.model._add_request(
            example_prompts[0],
            params=params_with_logprobs,
        )

        # test prompt_logprobs is not None
        vllm_model.model._add_request(
            example_prompts[1],
            params=SamplingParams(
                prompt_logprobs=3,
                max_tokens=max_tokens,
            ),
        )

        # test grouped requests
        vllm_model.model._add_request(
            example_prompts[2],
            params=SamplingParams(max_tokens=max_tokens),
        )

        outputs = vllm_model.model._run_engine(use_tqdm=False)

        assert outputs[0].outputs[0].text == enforced_answers * repeat_times
