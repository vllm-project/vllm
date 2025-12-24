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
        mteb_score=0.32898,
        architecture="BertForSequenceClassification",
        pooling_type="CLS",
        attn_type="encoder_only",
        is_prefix_caching_supported=False,
        is_chunked_prefill_supported=False,
    ),
    RerankModelInfo(
        "tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
        mteb_score=0.25736,
        architecture="Qwen3ForSequenceClassification",
        pooling_type="LAST",
        attn_type="decoder",
        is_prefix_caching_supported=True,
        is_chunked_prefill_supported=True,
    ),
]


@pytest.mark.parametrize("model_info", RERANK_MODELS)
def test_rerank_models_mteb(
    hf_runner, vllm_runner, model_info: RerankModelInfo
) -> None:
    mteb_test_rerank_models(hf_runner, vllm_runner, model_info)
