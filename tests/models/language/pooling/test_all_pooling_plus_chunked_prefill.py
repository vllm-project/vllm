# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from transformers import AutoModel

from tests.models.utils import check_embeddings_close
from vllm import TokensPrompt


@pytest.mark.parametrize(
    "model",
    ["Qwen/Qwen3-Embedding-0.6B"],
)
@torch.inference_mode
def test_embed_models(hf_runner, vllm_runner, model: str):
    chunk_size = 10
    n_prompt_tokens = [55, 56, 57]
    token_prompts = [[1024 + i for i in range(n)] for n in n_prompt_tokens]

    with vllm_runner(
        model,
        runner="pooling",
        max_model_len=128,
        max_num_batched_tokens=chunk_size,
        enforce_eager=True,
        # `enable_chunked_prefill`: Set to `False` instead of `None` in VllmRunner
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
    ) as vllm_model:
        vllm_outputs = vllm_model.token_embed(
            [TokensPrompt(prompt_token_ids=t) for t in token_prompts],
        )

    with hf_runner(
        model,
        auto_cls=AutoModel,
    ) as hf_model:
        hf_outputs = []
        for token_prompt in token_prompts:
            inputs = hf_model.wrap_device({"input_ids": torch.tensor([token_prompt])})
            input_ids = inputs["input_ids"]
            output = hf_model.model(input_ids)
            hf_outputs.append(output.last_hidden_state.cpu().float()[0])

    for hf_output, vllm_output in zip(hf_outputs, vllm_outputs):
        check_embeddings_close(
            embeddings_0_lst=hf_output,
            embeddings_1_lst=vllm_output,
            name_0="hf",
            name_1="vllm",
            tol=1e-2,
        )
