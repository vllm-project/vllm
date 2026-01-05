# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from tests.models.language.pooling.embed_utils import correctness_test_embed_models
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
        pooling_type="CLS",
        attn_type="encoder_only",
        is_prefix_caching_supported=False,
        is_chunked_prefill_supported=False,
        enable_test=True,
    ),
    EmbedModelInfo("BAAI/bge-base-zh", architecture="BertModel", enable_test=False),
    EmbedModelInfo("BAAI/bge-small-en", architecture="BertModel", enable_test=False),
    EmbedModelInfo("BAAI/bge-small-zh", architecture="BertModel", enable_test=False),
    EmbedModelInfo("BAAI/bge-large-en", architecture="BertModel", enable_test=False),
    EmbedModelInfo("BAAI/bge-large-zh", architecture="BertModel", enable_test=False),
    EmbedModelInfo(
        "BAAI/bge-large-zh-noinstruct", architecture="BertModel", enable_test=False
    ),
    EmbedModelInfo(
        "BAAI/bge-base-en-v1.5", architecture="BertModel", enable_test=False
    ),
    EmbedModelInfo(
        "BAAI/bge-base-zh-v1.5", architecture="BertModel", enable_test=False
    ),
    EmbedModelInfo(
        "BAAI/bge-small-en-v1.5", architecture="BertModel", enable_test=False
    ),
    EmbedModelInfo(
        "BAAI/bge-small-zh-v1.5", architecture="BertModel", enable_test=False
    ),
    EmbedModelInfo(
        "BAAI/bge-large-en-v1.5", architecture="BertModel", enable_test=False
    ),
    EmbedModelInfo(
        "BAAI/bge-large-zh-v1.5", architecture="BertModel", enable_test=False
    ),
    ########## XLMRobertaModel
    EmbedModelInfo(
        "BAAI/bge-m3",
        architecture="XLMRobertaModel",
        mteb_score=0.787343078,
        pooling_type="CLS",
        attn_type="encoder_only",
        is_prefix_caching_supported=False,
        is_chunked_prefill_supported=False,
        enable_test=True,
    ),
    ########## Qwen2Model
    EmbedModelInfo(
        "BAAI/bge-code-v1",
        architecture="Qwen2Model",
        mteb_score=0.75724465,
        dtype="float32",
        pooling_type="LAST",
        attn_type="decoder",
        is_prefix_caching_supported=True,
        is_chunked_prefill_supported=True,
        enable_test=True,
    ),
]

RERANK_MODELS = [
    ########## XLMRobertaForSequenceClassification
    RerankModelInfo(
        "BAAI/bge-reranker-base",
        architecture="XLMRobertaForSequenceClassification",
        mteb_score=0.32398,
        pooling_type="CLS",
        attn_type="encoder_only",
        is_prefix_caching_supported=False,
        is_chunked_prefill_supported=False,
        enable_test=True,
    ),
    RerankModelInfo(
        "BAAI/bge-reranker-large",
        architecture="XLMRobertaForSequenceClassification",
        enable_test=False,
    ),
    RerankModelInfo(
        "BAAI/bge-reranker-v2-m3",
        architecture="XLMRobertaForSequenceClassification",
        enable_test=False,
    ),
]


@pytest.mark.parametrize("model_info", MODELS)
def test_embed_models_mteb(hf_runner, vllm_runner, model_info: EmbedModelInfo) -> None:
    mteb_test_embed_models(hf_runner, vllm_runner, model_info)


@pytest.mark.parametrize("model_info", MODELS)
def test_embed_models_correctness(
    hf_runner, vllm_runner, model_info: EmbedModelInfo, example_prompts
) -> None:
    correctness_test_embed_models(hf_runner, vllm_runner, model_info, example_prompts)


@pytest.mark.parametrize("model_info", RERANK_MODELS)
def test_rerank_models_mteb(vllm_runner, model_info: RerankModelInfo) -> None:
    mteb_test_rerank_models(vllm_runner, model_info)
