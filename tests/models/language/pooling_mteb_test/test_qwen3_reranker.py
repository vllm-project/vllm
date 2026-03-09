# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
from typing import Any

import mteb
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from tests.conftest import HfRunner
from tests.models.utils import RerankModelInfo
from tests.utils import multi_gpu_test

from .mteb_score_utils import MtebCrossEncoderMixin, mteb_test_rerank_models

qwen3_reranker_hf_overrides = {
    "architectures": ["Qwen3ForSequenceClassification"],
    "classifier_from_token": ["no", "yes"],
    "is_original_qwen3_reranker": True,
}

RERANK_MODELS = [
    RerankModelInfo(
        "Qwen/Qwen3-Reranker-0.6B",
        architecture="Qwen3ForSequenceClassification",
        hf_overrides=qwen3_reranker_hf_overrides,
        chat_template_name="qwen3_reranker.jinja",
        seq_pooling_type="LAST",
        attn_type="decoder",
        is_prefix_caching_supported=True,
        is_chunked_prefill_supported=True,
        mteb_score=0.33459,
        enable_test=True,
    ),
    RerankModelInfo(
        "Qwen/Qwen3-Reranker-4B",
        architecture="Qwen3ForSequenceClassification",
        chat_template_name="qwen3_reranker.jinja",
        hf_overrides=qwen3_reranker_hf_overrides,
        enable_test=False,
    ),
]


class Qwen3RerankerHfRunner(MtebCrossEncoderMixin, HfRunner):
    def __init__(
        self, model_name: str, dtype: str = "auto", *args: Any, **kwargs: Any
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        HfRunner.__init__(
            self,
            model_name=model_name,
            auto_cls=AutoModelForCausalLM,
            dtype=dtype,
            **kwargs,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = 40960

    @torch.no_grad
    def predict(
        self,
        inputs1: DataLoader[mteb.types.BatchedInput],
        inputs2: DataLoader[mteb.types.BatchedInput],
        *args,
        **kwargs,
    ) -> np.ndarray:
        queries = [text for batch in inputs1 for text in batch["text"]]
        corpus = [text for batch in inputs2 for text in batch["text"]]

        tokenizer = self.tokenizer
        prompts = []
        for query, document in zip(queries, corpus):
            conversation = [
                {"role": "query", "content": query},
                {"role": "document", "content": document},
            ]

            prompt = tokenizer.apply_chat_template(
                conversation=conversation,
                tools=None,
                chat_template=self.chat_template,
                tokenize=False,
            )
            prompts.append(prompt)

        def compute_logits(inputs):
            batch_scores = self.model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp()
            return scores

        scores = []
        for prompt in prompts:
            inputs = tokenizer([prompt], return_tensors="pt")
            inputs = self.wrap_device(inputs)
            score = compute_logits(inputs)
            scores.append(score[0].item())
        return torch.Tensor(scores)


@pytest.mark.parametrize("model_info", RERANK_MODELS)
def test_rerank_models_mteb(vllm_runner, model_info: RerankModelInfo) -> None:
    mteb_test_rerank_models(vllm_runner, model_info, hf_runner=Qwen3RerankerHfRunner)


@pytest.mark.parametrize("model_info", RERANK_MODELS)
@multi_gpu_test(num_gpus=2)
def test_rerank_models_mteb_tp(vllm_runner, model_info: RerankModelInfo) -> None:
    assert model_info.architecture == "Qwen3ForSequenceClassification"

    vllm_extra_kwargs: dict[str, Any] = {
        "tensor_parallel_size": 2,
    }

    mteb_test_rerank_models(
        vllm_runner,
        model_info,
        vllm_extra_kwargs=vllm_extra_kwargs,
        hf_runner=Qwen3RerankerHfRunner,
    )
