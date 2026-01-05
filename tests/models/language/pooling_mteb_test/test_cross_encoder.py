# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from tests.models.utils import (
    RerankModelInfo,
)

from .mteb_score_utils import mteb_test_rerank_models

RERANK_MODELS = [
    RerankModelInfo(
        "cross-encoder/ms-marco-TinyBERT-L-2-v2",
        architecture="BertForSequenceClassification",
        pooling_type="CLS",
        attn_type="encoder_only",
        is_prefix_caching_supported=False,
        is_chunked_prefill_supported=False,
        mteb_score=0.32898,
    ),
    RerankModelInfo(
        "tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
        architecture="Qwen3ForSequenceClassification",
        pooling_type="LAST",
        attn_type="decoder",
        is_prefix_caching_supported=True,
        is_chunked_prefill_supported=True,
        chat_template_name="qwen3_reranker.jinja",
        mteb_score=0.33459,
    ),
]


@pytest.mark.parametrize("model_info", RERANK_MODELS)
def test_rerank_models_mteb(vllm_runner, model_info: RerankModelInfo) -> None:
    mteb_test_rerank_models(vllm_runner, model_info)
