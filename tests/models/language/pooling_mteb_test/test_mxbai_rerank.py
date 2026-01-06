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

from .mteb_score_utils import MtebCrossEncoderMixin, mteb_test_rerank_models

mxbai_rerank_hf_overrides = {
    "architectures": ["Qwen2ForSequenceClassification"],
    "classifier_from_token": ["0", "1"],
    "method": "from_2_way_softmax",
}

RERANK_MODELS = [
    RerankModelInfo(
        "mixedbread-ai/mxbai-rerank-base-v2",
        architecture="Qwen2ForSequenceClassification",
        hf_overrides=mxbai_rerank_hf_overrides,
        pooling_type="LAST",
        attn_type="decoder",
        is_prefix_caching_supported=True,
        is_chunked_prefill_supported=True,
        chat_template_name="mxbai_rerank_v2.jinja",
        mteb_score=0.33651,
        enable_test=True,
    ),
    RerankModelInfo(
        "mixedbread-ai/mxbai-rerank-large-v2",
        architecture="Qwen2ForSequenceClassification",
        hf_overrides=mxbai_rerank_hf_overrides,
        chat_template_name="mxbai_rerank_v2.jinja",
        enable_test=False,
    ),
]


class MxbaiRerankerHfRunner(MtebCrossEncoderMixin, HfRunner):
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
        self.yes_loc = self.tokenizer.convert_tokens_to_ids("1")
        self.no_loc = self.tokenizer.convert_tokens_to_ids("0")

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
            logits = self.model(**inputs).logits[:, -1, :]
            yes_logits = logits[:, self.yes_loc]
            no_logits = logits[:, self.no_loc]
            logits = yes_logits - no_logits
            scores = logits.float().sigmoid()
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
    mteb_test_rerank_models(vllm_runner, model_info, hf_runner=MxbaiRerankerHfRunner)
