# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from ...utils import (CLSPoolingEmbedModelInfo, CLSPoolingRerankModelInfo,
                      EmbedModelInfo, LASTPoolingEmbedModelInfo,
                      RerankModelInfo, check_transformers_version)
from .embed_utils import correctness_test_embed_models
from .mteb_utils import mteb_test_embed_models, mteb_test_rerank_models

MODELS = [
    ########## BertModel
    CLSPoolingEmbedModelInfo("thenlper/gte-large",
                             architecture="BertModel",
                             enable_test=True),
    CLSPoolingEmbedModelInfo("thenlper/gte-base",
                             architecture="BertModel",
                             enable_test=False),
    CLSPoolingEmbedModelInfo("thenlper/gte-small",
                             architecture="BertModel",
                             enable_test=False),
    CLSPoolingEmbedModelInfo("thenlper/gte-large-zh",
                             architecture="BertModel",
                             enable_test=False),
    CLSPoolingEmbedModelInfo("thenlper/gte-base-zh",
                             architecture="BertModel",
                             enable_test=False),
    CLSPoolingEmbedModelInfo("thenlper/gte-small-zh",
                             architecture="BertModel",
                             enable_test=False),
    ########### NewModel
    CLSPoolingEmbedModelInfo("Alibaba-NLP/gte-multilingual-base",
                             architecture="GteNewModel",
                             hf_overrides={"architectures": ["GteNewModel"]},
                             enable_test=True),
    CLSPoolingEmbedModelInfo("Alibaba-NLP/gte-base-en-v1.5",
                             architecture="GteNewModel",
                             hf_overrides={"architectures": ["GteNewModel"]},
                             enable_test=True),
    CLSPoolingEmbedModelInfo("Alibaba-NLP/gte-large-en-v1.5",
                             architecture="GteNewModel",
                             hf_overrides={"architectures": ["GteNewModel"]},
                             enable_test=True),
    ########### Qwen2ForCausalLM
    LASTPoolingEmbedModelInfo("Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                              architecture="Qwen2ForCausalLM",
                              enable_test=True),
    ########## ModernBertModel
    CLSPoolingEmbedModelInfo("Alibaba-NLP/gte-modernbert-base",
                             architecture="ModernBertModel",
                             enable_test=True),
    ########## Qwen3ForCausalLM
    LASTPoolingEmbedModelInfo("Qwen/Qwen3-Embedding-0.6B",
                              architecture="Qwen3ForCausalLM",
                              dtype="float32",
                              enable_test=True),
    LASTPoolingEmbedModelInfo("Qwen/Qwen3-Embedding-4B",
                              architecture="Qwen3ForCausalLM",
                              dtype="float32",
                              enable_test=False),
]

RERANK_MODELS = [
    CLSPoolingRerankModelInfo(
        # classifier_pooling: mean
        "Alibaba-NLP/gte-reranker-modernbert-base",
        architecture="ModernBertForSequenceClassification",
        enable_test=True),
    CLSPoolingRerankModelInfo(
        "Alibaba-NLP/gte-multilingual-reranker-base",
        architecture="GteNewForSequenceClassification",
        hf_overrides={"architectures": ["GteNewForSequenceClassification"]},
        enable_test=True),
]


@pytest.mark.parametrize("model_info", MODELS)
def test_embed_models_mteb(hf_runner, vllm_runner,
                           model_info: EmbedModelInfo) -> None:
    if model_info.name == "Alibaba-NLP/gte-Qwen2-1.5B-instruct":
        check_transformers_version(model_info.name,
                                   max_transformers_version="4.53.2")

    mteb_test_embed_models(hf_runner, vllm_runner, model_info)


@pytest.mark.parametrize("model_info", MODELS)
def test_embed_models_correctness(hf_runner, vllm_runner,
                                  model_info: EmbedModelInfo,
                                  example_prompts) -> None:
    if model_info.name == "Alibaba-NLP/gte-Qwen2-1.5B-instruct":
        check_transformers_version(model_info.name,
                                   max_transformers_version="4.53.2")

    correctness_test_embed_models(hf_runner, vllm_runner, model_info,
                                  example_prompts)


@pytest.mark.parametrize("model_info", RERANK_MODELS)
def test_rerank_models_mteb(hf_runner, vllm_runner,
                            model_info: RerankModelInfo) -> None:
    mteb_test_rerank_models(hf_runner, vllm_runner, model_info)
