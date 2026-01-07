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
        "thenlper/gte-large",
        mteb_score=0.76807651,
        architecture="BertModel",
        pooling_type="MEAN",
        attn_type="encoder_only",
        is_prefix_caching_supported=False,
        is_chunked_prefill_supported=False,
        enable_test=True,
    ),
    EmbedModelInfo("thenlper/gte-base", architecture="BertModel", enable_test=False),
    EmbedModelInfo("thenlper/gte-small", architecture="BertModel", enable_test=False),
    EmbedModelInfo(
        "thenlper/gte-large-zh", architecture="BertModel", enable_test=False
    ),
    EmbedModelInfo("thenlper/gte-base-zh", architecture="BertModel", enable_test=False),
    EmbedModelInfo(
        "thenlper/gte-small-zh", architecture="BertModel", enable_test=False
    ),
    ########### NewModel
    # These three architectures are almost the same, but not exactly the same.
    # For example,
    # - whether to use token_type_embeddings
    # - whether to use context expansion
    # So only test one (the most widely used) model
    EmbedModelInfo(
        "Alibaba-NLP/gte-multilingual-base",
        architecture="GteNewModel",
        mteb_score=0.775074696,
        hf_overrides={"architectures": ["GteNewModel"]},
        pooling_type="CLS",
        attn_type="encoder_only",
        is_prefix_caching_supported=False,
        is_chunked_prefill_supported=False,
        enable_test=True,
    ),
    EmbedModelInfo(
        "Alibaba-NLP/gte-base-en-v1.5",
        architecture="GteNewModel",
        hf_overrides={"architectures": ["GteNewModel"]},
        enable_test=False,
    ),
    EmbedModelInfo(
        "Alibaba-NLP/gte-large-en-v1.5",
        architecture="GteNewModel",
        hf_overrides={"architectures": ["GteNewModel"]},
        enable_test=False,
    ),
    ########### Qwen2ForCausalLM
    EmbedModelInfo(
        "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        mteb_score=0.758473459018872,
        architecture="Qwen2ForCausalLM",
        pooling_type="LAST",
        attn_type="encoder_only",
        is_prefix_caching_supported=False,
        is_chunked_prefill_supported=False,
        enable_test=True,
    ),
    ########## ModernBertModel
    EmbedModelInfo(
        "Alibaba-NLP/gte-modernbert-base",
        mteb_score=0.748193353,
        architecture="ModernBertModel",
        pooling_type="CLS",
        attn_type="encoder_only",
        is_prefix_caching_supported=False,
        is_chunked_prefill_supported=False,
        enable_test=True,
    ),
    ########## Qwen3ForCausalLM
    EmbedModelInfo(
        "Qwen/Qwen3-Embedding-0.6B",
        mteb_score=0.771163695,
        architecture="Qwen3ForCausalLM",
        pooling_type="LAST",
        attn_type="decoder",
        is_prefix_caching_supported=True,
        is_chunked_prefill_supported=True,
        enable_test=True,
    ),
    EmbedModelInfo(
        "Qwen/Qwen3-Embedding-4B",
        architecture="Qwen3ForCausalLM",
        enable_test=False,
    ),
]

RERANK_MODELS = [
    RerankModelInfo(
        # classifier_pooling: mean
        "Alibaba-NLP/gte-reranker-modernbert-base",
        mteb_score=0.33386,
        architecture="ModernBertForSequenceClassification",
        pooling_type="CLS",
        attn_type="encoder_only",
        is_prefix_caching_supported=False,
        is_chunked_prefill_supported=False,
        enable_test=True,
    ),
    RerankModelInfo(
        "Alibaba-NLP/gte-multilingual-reranker-base",
        mteb_score=0.33062,
        architecture="GteNewForSequenceClassification",
        hf_overrides={"architectures": ["GteNewForSequenceClassification"]},
        pooling_type="CLS",
        attn_type="encoder_only",
        is_prefix_caching_supported=False,
        is_chunked_prefill_supported=False,
        enable_test=True,
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
