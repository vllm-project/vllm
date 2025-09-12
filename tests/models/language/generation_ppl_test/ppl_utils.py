# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://huggingface.co/docs/transformers/perplexity
from typing import Optional, cast

import pytest
import torch
from datasets import load_dataset

import tests.ci_envs as ci_envs
from tests.models.utils import (GenerateModelInfo,
                                TokensTextLogprobsPromptLogprobs)
from vllm.logprobs import Logprob

# See #24485
PPL_TOL = 0.01
MAX_LENGTH = 1024


@torch.inference_mode
def wikitext_ppl_test(hf_runner,
                      vllm_runner,
                      model_info: GenerateModelInfo,
                      max_length=MAX_LENGTH,
                      vllm_extra_kwargs=None,
                      atol=PPL_TOL):

    # A model family has many models with the same architecture,
    # and we don't need to test each one.
    if not ci_envs.VLLM_CI_NO_SKIP and not model_info.enable_test:
        pytest.skip("Skipping test.")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Allow vllm to test using the given dtype, such as float32
    vllm_extra_kwargs = vllm_extra_kwargs or {}
    vllm_extra_kwargs["dtype"] = ci_envs.VLLM_CI_DTYPE or model_info.dtype

    # Allow vllm to test using hf_overrides
    if model_info.hf_overrides is not None:
        vllm_extra_kwargs["hf_overrides"] = model_info.hf_overrides

    # Allow changing the head dtype used by vllm in tests
    if ci_envs.VLLM_CI_HEAD_DTYPE is not None:
        if "hf_overrides" not in vllm_extra_kwargs:
            vllm_extra_kwargs["hf_overrides"] = {}
        vllm_extra_kwargs["hf_overrides"][
            "head_dtype"] = ci_envs.VLLM_CI_HEAD_DTYPE

    with vllm_runner(model_info.name,
                     gpu_memory_utilization=0.7,
                     max_model_len=max_length,
                     max_num_seqs=1,
                     enforce_eager=True,
                     **vllm_extra_kwargs) as vllm_model:
        # Use max_num_seqs=1 to avoid OOM,
        # and avoid batch different requests together.

        model_config = vllm_model.llm.llm_engine.model_config

        # Confirm whether vllm is using the correct architecture
        if model_info.architecture:
            assert (model_info.architecture in model_config.architectures)

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
                                                      num_prompt_logprobs=0,
                                                      use_tqdm=False)
        nll_sum = torch.tensor(0., dtype=torch.float32, device="cpu")
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

            neg_log_likelihood = -torch.tensor(
                token_log_probs, dtype=torch.float32, device="cpu").sum()
            nll_sum += neg_log_likelihood
            n_tokens += len(token_log_probs)
        vllm_ppl = float(torch.exp(nll_sum / n_tokens))
        vllm_dtype = model_config.dtype
        head_dtype = model_config.head_dtype

    # Accelerate ppl test by setting Transformers ppl score to a constant
    if model_info.hf_ppl is None:
        with hf_runner(
                model_info.name,
                dtype=ci_envs.VLLM_CI_HF_DTYPE or model_info.hf_dtype,
        ) as hf_model:
            nll_sum = torch.tensor(0., dtype=torch.float32, device="cpu")
            n_tokens = 0
            for chunk in chunks:
                inputs = hf_model.wrap_device(
                    {"input_ids": torch.tensor([chunk])})
                input_ids = inputs["input_ids"]
                outputs = hf_model.model(input_ids, labels=input_ids)
                neg_log_likelihood = outputs.loss

                neg_log_likelihood = neg_log_likelihood.to(torch.float32).cpu()

                num_loss_tokens = len(chunk) - 1
                nll_sum += neg_log_likelihood * num_loss_tokens
                n_tokens += num_loss_tokens

            hf_ppl = float(torch.exp(nll_sum / n_tokens))
            hf_dtype = next(hf_model.model.parameters()).dtype
    else:
        hf_ppl = model_info.hf_ppl
        hf_dtype = "Constant"

    differ = (vllm_ppl - hf_ppl) / hf_ppl
    print("Model:", model_info.name)
    print("VLLM:", f"dtype:{vllm_dtype}", f"head_dtype:{head_dtype}", vllm_ppl)
    print("Transformers:", hf_dtype, hf_ppl)
    print("Difference (%):", differ * 100)

    # PPL the smaller, the better
    # We are not concerned that the vllm PPL is less than Transformers,
    # so we only perform one-sided testing.
    assert differ < atol
