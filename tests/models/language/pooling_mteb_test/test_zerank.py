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

RERANK_MODELS = [
    RerankModelInfo(
        "polaria-tech/zerank-2-reranker-vllm",
        architecture="Qwen3ForSequenceClassification",
        chat_template_name="zerank2.jinja",
        seq_pooling_type="LAST",
        attn_type="decoder",
        is_prefix_caching_supported=True,
        is_chunked_prefill_supported=True,
        mteb_score=0.34305,
        mteb_tol=1e-2,
        enable_test=True,
    ),
]


class ZerankHfRunner(MtebCrossEncoderMixin, HfRunner):
    # The converted checkpoint has no lm_head, so the reference is the original
    # causal LM: the logit of the "Yes" token at the last position. The converted
    # checkpoint returns sigmoid(logit / 5), which ranks documents identically.
    original_model = "zeroentropy/zerank-2-reranker"

    def __init__(
        self, model_name: str, dtype: str = "auto", *args: Any, **kwargs: Any
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        HfRunner.__init__(
            self,
            model_name=self.original_model,
            auto_cls=AutoModelForCausalLM,
            dtype=dtype,
            **kwargs,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.original_model, padding_side="left"
        )
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("Yes")

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

        scores = []
        for prompt in prompts:
            inputs = tokenizer([prompt], return_tensors="pt")
            inputs = self.wrap_device(inputs)
            logits = self.model(**inputs).logits[:, -1, :]
            scores.append(logits[0, self.token_true_id].item())
        return torch.Tensor(scores)


@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize("model_info", RERANK_MODELS)
def test_rerank_models_mteb(vllm_runner, model_info: RerankModelInfo) -> None:
    mteb_test_rerank_models(vllm_runner, model_info, hf_runner=ZerankHfRunner)
