# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from tests.models.utils import (
    EmbedModelInfo,
    RerankModelInfo,
)

from .mteb_embed_utils import mteb_test_embed_models
from .mteb_score_utils import mteb_test_rerank_models

MODELS = [
    ########## BertModel
    EmbedModelInfo(
        "BAAI/bge-base-en",
        architecture="BertModel",
        mteb_score=0.779336792,
        mteb_tol=2e-3,
        seq_pooling_type="CLS",
        attn_type="encoder_only",
        is_prefix_caching_supported=False,
        is_chunked_prefill_supported=False,
    ),
    ########## XLMRobertaModel
    EmbedModelInfo(
        "BAAI/bge-m3",
        architecture="XLMRobertaModel",
        mteb_score=0.787343078,
        mteb_tol=2e-3,
        seq_pooling_type="CLS",
        attn_type="encoder_only",
        is_prefix_caching_supported=False,
        is_chunked_prefill_supported=False,
    ),
]

RERANK_MODELS = [
    ########## XLMRobertaForSequenceClassification
    RerankModelInfo(
        "BAAI/bge-reranker-base",
        architecture="XLMRobertaForSequenceClassification",
        mteb_score=0.32398,
        mteb_tol=1e-2,
        seq_pooling_type="CLS",
        attn_type="encoder_only",
        is_prefix_caching_supported=False,
        is_chunked_prefill_supported=False,
    ),
]


@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize("model_info", MODELS)
def test_embed_models_mteb(hf_runner, vllm_runner, model_info: EmbedModelInfo) -> None:
    mteb_test_embed_models(hf_runner, vllm_runner, model_info)


@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize("model_info", RERANK_MODELS)
def test_rerank_models_mteb(vllm_runner, model_info: RerankModelInfo) -> None:
    mteb_test_rerank_models(vllm_runner, model_info)
