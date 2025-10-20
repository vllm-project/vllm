# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.language.pooling.embed_utils import correctness_test_embed_models
from tests.models.utils import (
    CLSPoolingEmbedModelInfo,
    CLSPoolingRerankModelInfo,
    EmbedModelInfo,
    LASTPoolingEmbedModelInfo,
    RerankModelInfo,
)

from .mteb_utils import mteb_test_embed_models, mteb_test_rerank_models

MODELS = [
    ########## BertModel
    CLSPoolingEmbedModelInfo(
        "thenlper/gte-large",
        mteb_score=0.76807651,
        architecture="BertModel",
        enable_test=True,
    ),
    CLSPoolingEmbedModelInfo(
        "thenlper/gte-base", architecture="BertModel", enable_test=False
    ),
    CLSPoolingEmbedModelInfo(
        "thenlper/gte-small", architecture="BertModel", enable_test=False
    ),
    CLSPoolingEmbedModelInfo(
        "thenlper/gte-large-zh", architecture="BertModel", enable_test=False
    ),
    CLSPoolingEmbedModelInfo(
        "thenlper/gte-base-zh", architecture="BertModel", enable_test=False
    ),
    CLSPoolingEmbedModelInfo(
        "thenlper/gte-small-zh", architecture="BertModel", enable_test=False
    ),
    ########### NewModel
    # These three architectures are almost the same, but not exactly the same.
    # For example,
    # - whether to use token_type_embeddings
    # - whether to use context expansion
    # So only test one (the most widely used) model
    CLSPoolingEmbedModelInfo(
        "Alibaba-NLP/gte-multilingual-base",
        architecture="GteNewModel",
        mteb_score=0.775074696,
        hf_overrides={"architectures": ["GteNewModel"]},
        enable_test=True,
    ),
    CLSPoolingEmbedModelInfo(
        "Alibaba-NLP/gte-base-en-v1.5",
        architecture="GteNewModel",
        hf_overrides={"architectures": ["GteNewModel"]},
        enable_test=False,
    ),
    CLSPoolingEmbedModelInfo(
        "Alibaba-NLP/gte-large-en-v1.5",
        architecture="GteNewModel",
        hf_overrides={"architectures": ["GteNewModel"]},
        enable_test=False,
    ),
    ########### Qwen2ForCausalLM
    LASTPoolingEmbedModelInfo(
        "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        mteb_score=0.758473459018872,
        architecture="Qwen2ForCausalLM",
        enable_test=True,
    ),
    ########## ModernBertModel
    CLSPoolingEmbedModelInfo(
        "Alibaba-NLP/gte-modernbert-base",
        mteb_score=0.748193353,
        architecture="ModernBertModel",
        enable_test=True,
    ),
    ########## Qwen3ForCausalLM
    LASTPoolingEmbedModelInfo(
        "Qwen/Qwen3-Embedding-0.6B",
        mteb_score=0.771163695,
        architecture="Qwen3ForCausalLM",
        dtype="float32",
        enable_test=True,
    ),
    LASTPoolingEmbedModelInfo(
        "Qwen/Qwen3-Embedding-4B",
        architecture="Qwen3ForCausalLM",
        dtype="float32",
        enable_test=False,
    ),
]

RERANK_MODELS = [
    CLSPoolingRerankModelInfo(
        # classifier_pooling: mean
        "Alibaba-NLP/gte-reranker-modernbert-base",
        mteb_score=0.33386,
        architecture="ModernBertForSequenceClassification",
        enable_test=True,
    ),
    CLSPoolingRerankModelInfo(
        "Alibaba-NLP/gte-multilingual-reranker-base",
        mteb_score=0.33062,
        architecture="GteNewForSequenceClassification",
        hf_overrides={"architectures": ["GteNewForSequenceClassification"]},
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
def test_rerank_models_mteb(
    hf_runner, vllm_runner, model_info: RerankModelInfo
) -> None:
    mteb_test_rerank_models(hf_runner, vllm_runner, model_info)
