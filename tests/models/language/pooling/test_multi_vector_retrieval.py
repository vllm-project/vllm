# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from transformers import AutoModel

from tests.models.utils import check_embeddings_close


@pytest.mark.parametrize(
    "model",
    ["BAAI/bge-m3"],
)
@pytest.mark.parametrize("dtype", ["half"])
@torch.inference_mode
def test_embed_models(hf_runner, vllm_runner, example_prompts, model: str, dtype: str):
    with vllm_runner(
        model,
        runner="pooling",
        max_model_len=None,
    ) as vllm_model:
        vllm_outputs = vllm_model.token_embed(example_prompts)

    with hf_runner(
        model,
        auto_cls=AutoModel,
    ) as hf_model:
        tokenizer = hf_model.tokenizer
        hf_outputs = []
        for prompt in example_prompts:
            inputs = tokenizer([prompt], return_tensors="pt")
            inputs = hf_model.wrap_device(inputs)
            output = hf_model.model(**inputs)
            embedding = output.last_hidden_state[0].float()
            # normal
            hf_outputs.append(embedding.cpu())

    for hf_output, vllm_output in zip(hf_outputs, vllm_outputs):
        check_embeddings_close(
            embeddings_0_lst=hf_output,
            embeddings_1_lst=vllm_output,
            name_0="hf",
            name_1="vllm",
            tol=1e-2,
        )
