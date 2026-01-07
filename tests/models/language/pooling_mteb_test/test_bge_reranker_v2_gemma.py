# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import mteb
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from tests.conftest import HfRunner
from tests.models.utils import RerankModelInfo

from .mteb_score_utils import (
    MtebCrossEncoderMixin,
    mteb_test_rerank_models,
)

RERANK_MODELS = [
    RerankModelInfo(
        "BAAI/bge-reranker-v2-gemma",
        architecture="GemmaForSequenceClassification",
        hf_overrides={
            "architectures": ["GemmaForSequenceClassification"],
            "classifier_from_token": ["Yes"],
            "method": "no_post_processing",
        },
        mteb_score=0.33757,
        pooling_type="LAST",
        attn_type="decoder",
        is_prefix_caching_supported=True,
        is_chunked_prefill_supported=True,
        chat_template_name="bge-reranker-v2-gemma.jinja",
    ),
]

PROMPT = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."  # noqa: E501


class GemmaRerankerHfRunner(MtebCrossEncoderMixin, HfRunner):
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
        self.yes_loc = self.tokenizer.convert_tokens_to_ids("Yes")

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

        def get_inputs(pairs, tokenizer, prompt=None):
            if prompt is None:
                prompt = PROMPT

            sep = "\n"
            prompt_inputs = tokenizer(
                prompt, return_tensors=None, add_special_tokens=False
            )["input_ids"]
            sep_inputs = tokenizer(sep, return_tensors=None, add_special_tokens=False)[
                "input_ids"
            ]
            inputs = []
            for query, passage in pairs:
                query_inputs = tokenizer(
                    f"A: {query}",
                    return_tensors=None,
                    add_special_tokens=False,
                    truncation=True,
                )
                passage_inputs = tokenizer(
                    f"B: {passage}",
                    return_tensors=None,
                    add_special_tokens=False,
                    truncation=True,
                )
                item = tokenizer.prepare_for_model(
                    [tokenizer.bos_token_id] + query_inputs["input_ids"],
                    sep_inputs + passage_inputs["input_ids"],
                    truncation="only_second",
                    padding=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    add_special_tokens=False,
                )
                item["input_ids"] = item["input_ids"] + sep_inputs + prompt_inputs
                item["attention_mask"] = [1] * len(item["input_ids"])
                inputs.append(item)
            return tokenizer.pad(
                inputs,
                padding=True,
                return_tensors="pt",
            )

        scores = []
        for query, document in zip(queries, corpus):
            pairs = [(query, document)]
            inputs = get_inputs(pairs, self.tokenizer)
            inputs = inputs.to(self.model.device)
            _n_tokens = inputs["input_ids"].shape[1]
            logits = self.model(**inputs, return_dict=True).logits
            _scores = (
                logits[:, -1, self.yes_loc]
                .view(
                    -1,
                )
                .float()
                .sigmoid()
            )
            scores.append(_scores[0].item())
        return torch.Tensor(scores)


@pytest.mark.parametrize("model_info", RERANK_MODELS)
def test_rerank_models_mteb(vllm_runner, model_info: RerankModelInfo) -> None:
    mteb_test_rerank_models(
        vllm_runner,
        model_info,
        hf_runner=GemmaRerankerHfRunner,
    )
