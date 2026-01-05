# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import pytest
import torch

from tests.conftest import HfRunner
from tests.models.utils import RerankModelInfo
from tests.utils import multi_gpu_test

from .mteb_score_utils import mteb_test_rerank_models

qwen3_reranker_hf_overrides = {
    "architectures": ["Qwen3ForSequenceClassification"],
    "classifier_from_token": ["no", "yes"],
    "is_original_qwen3_reranker": True,
}

RERANK_MODELS = [
    RerankModelInfo(
        "Qwen/Qwen3-Reranker-0.6B",
        architecture="Qwen3ForSequenceClassification",
        mteb_score=0.25736,
        hf_overrides=qwen3_reranker_hf_overrides,
        pooling_type="LAST",
        attn_type="decoder",
        is_prefix_caching_supported=True,
        is_chunked_prefill_supported=True,
        enable_test=True,
    ),
    RerankModelInfo(
        "Qwen/Qwen3-Reranker-4B",
        architecture="Qwen3ForSequenceClassification",
        hf_overrides=qwen3_reranker_hf_overrides,
        enable_test=False,
    ),
]


class Qwen3RerankerHfRunner(HfRunner):
    def __init__(
        self, model_name: str, dtype: str = "auto", *args: Any, **kwargs: Any
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        super().__init__(model_name, dtype, auto_cls=AutoModelForCausalLM)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

    def predict(self, prompts: list[list[str]], *args, **kwargs) -> torch.Tensor:
        def process_inputs(pairs):
            inputs = self.tokenizer(
                pairs,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
            )
            for i, ele in enumerate(inputs["input_ids"]):
                inputs["input_ids"][i] = ele
            inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
            for key in inputs:
                inputs[key] = inputs[key].to(self.model.device)
            return inputs

        @torch.no_grad()
        def compute_logits(inputs):
            batch_scores = self.model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp()
            return scores

        scores = []
        for query, doc, *_ in prompts:
            pairs = [(query, doc)]
            inputs = process_inputs(pairs)
            score = compute_logits(inputs)
            scores.append(score[0].item())
        return torch.Tensor(scores)


@pytest.mark.parametrize("model_info", RERANK_MODELS)
def test_rerank_models_mteb(vllm_runner, model_info: RerankModelInfo) -> None:
    mteb_test_rerank_models(Qwen3RerankerHfRunner, vllm_runner, model_info)


@pytest.mark.parametrize("model_info", RERANK_MODELS)
@multi_gpu_test(num_gpus=2)
def test_rerank_models_mteb_tp(vllm_runner, model_info: RerankModelInfo) -> None:
    assert model_info.architecture == "Qwen3ForSequenceClassification"

    vllm_extra_kwargs: dict[str, Any] = {
        "tensor_parallel_size": 2,
    }

    mteb_test_rerank_models(
        Qwen3RerankerHfRunner, vllm_runner, model_info, vllm_extra_kwargs
    )
