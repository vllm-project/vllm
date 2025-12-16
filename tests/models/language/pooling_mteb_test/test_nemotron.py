# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import pytest
import torch

from tests.conftest import HfRunner
from tests.models.utils import (
    CLSPoolingEmbedModelInfo,
    CLSPoolingRerankModelInfo,
    EmbedModelInfo,
    RerankModelInfo,
)

from .mteb_utils import mteb_test_embed_models, mteb_test_rerank_models

EMBEDDING_MODELS = [
    CLSPoolingEmbedModelInfo(
        "nvidia/llama-nemotron-embed-1b-v2",
        architecture="LlamaBidirectionalModel",
    )
]

RERANK_MODELS = [
    CLSPoolingRerankModelInfo(
        "nvidia/llama-nemotron-rerank-1b-v2",
        architecture="LlamaBidirectionalForSequenceClassification",
    ),
]


class NemotronRerankHfRunner(HfRunner):
    def __init__(
        self, model_name: str, dtype: str = "auto", *args: Any, **kwargs: Any
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        super().__init__(model_name, dtype, auto_cls=AutoModelForCausalLM)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = 8192

    def predict(self, prompts: list[list[str]], *args, **kwargs) -> torch.Tensor:
        def prompt_template(q, p):
            return f"question:{q} \n \n passage:{p}"

        scores = []
        for query, doc, *_ in prompts:
            texts = [prompt_template(query, doc)]
            batch_dict = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            )
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

            logits = self.model(**batch_dict).logits
            scores = logits.view(-1).cpu().tolist()
            scores.append(scores[0].item())

        return torch.Tensor(scores)


@pytest.mark.parametrize("model_info", EMBEDDING_MODELS)
def test_embed_models_mteb(hf_runner, vllm_runner, model_info: EmbedModelInfo) -> None:
    mteb_test_embed_models(hf_runner, vllm_runner, model_info)


@pytest.mark.parametrize("model_info", RERANK_MODELS)
def test_rerank_models_mteb(vllm_runner, model_info: RerankModelInfo) -> None:
    mteb_test_rerank_models(NemotronRerankHfRunner, vllm_runner, model_info)
