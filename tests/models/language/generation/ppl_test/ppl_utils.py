# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://huggingface.co/docs/transformers/perplexity
import math
from typing import Optional, cast

import torch
from datasets import load_dataset

from tests.models.utils import ModelInfo, TokensTextLogprobsPromptLogprobs
from vllm.logprobs import Logprob

PPL_TOL = 1
MAX_LENGTH = 1024


def wikitext_ppl_test(hf_runner,
                      vllm_runner,
                      model_info: ModelInfo,
                      max_length=MAX_LENGTH,
                      vllm_extra_kwargs=None,
                      atol=PPL_TOL):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    vllm_extra_kwargs = vllm_extra_kwargs or {}
    vllm_extra_kwargs["dtype"] = model_info.dtype

    if model_info.hf_overrides is not None:
        vllm_extra_kwargs["hf_overrides"] = model_info.hf_overrides

    with vllm_runner(model_info.name,
                     gpu_memory_utilization=0.7,
                     max_model_len=max_length,
                     max_num_seqs=1,
                     enforce_eager=True,
                     **vllm_extra_kwargs) as vllm_model:
        model_config = vllm_model.llm.llm_engine.model_config

        max_length = min(model_config.max_model_len - 1, max_length)
        stride = max_length

        tokenizer = vllm_model.llm.get_tokenizer()
        tokens = tokenizer.encode("\n\n".join(dataset["text"]))
        n_tokens = len(tokens)

        chunks = []
        for begin_loc in range(0, n_tokens, stride):
            end_loc = min(begin_loc + max_length, n_tokens)
            chunks.append(tokens[begin_loc:end_loc])

        outputs = vllm_model.generate_greedy_logprobs(prompts=chunks,
                                                      max_tokens=1,
                                                      num_logprobs=None,
                                                      num_prompt_logprobs=0)
        nll_sum = 0.0
        n_tokens = 0
        for output in outputs:
            output = cast(TokensTextLogprobsPromptLogprobs, output)
            token_datas = cast(list[Optional[dict[int, Logprob]]], output[3])

            assert token_datas[0] is None
            token_log_probs = []
            for token_data in token_datas[1:]:
                assert token_data is not None
                assert len(token_data) == 1
                token_log_prob = list(token_data.values())[0].logprob
                token_log_probs.append(token_log_prob)

            neg_log_likelihood = -sum(token_log_probs)
            nll_sum += neg_log_likelihood
            n_tokens += len(token_log_probs)
        vllm_ppl = math.exp(nll_sum / n_tokens)
        vllm_dtype = model_config.dtype

    with hf_runner(
            model_info.name,
            dtype=model_info.hf_dtype,
    ) as hf_model:
        nll_sum = 0.0
        n_tokens = 0
        for chunk in chunks:
            with torch.no_grad():
                inputs = hf_model.wrap_device(
                    {"input_ids": torch.tensor([chunk])})
                input_ids = inputs["input_ids"]
                outputs = hf_model.model(input_ids, labels=input_ids)
                neg_log_likelihood = outputs.loss
            num_loss_tokens = len(chunk) - 1
            nll_sum += neg_log_likelihood * num_loss_tokens
            n_tokens += num_loss_tokens

        hf_ppl = math.exp(nll_sum / n_tokens)
        hf_dtype = next(hf_model.model.parameters()).dtype

    print("Model:", model_info.name)
    print("VLLM:", vllm_dtype, vllm_ppl)
    print("Transformers:", hf_dtype, hf_ppl)
    print("Difference:", vllm_ppl - hf_ppl)

    # PPL the smaller, the better
    # We are not concerned that the vllm PPL is less than Transformers,
    # so we only perform one-sided testing.
    assert vllm_ppl - hf_ppl < atol
