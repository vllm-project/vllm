# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from ...utils import (CLSPoolingEmbedModelInfo, CLSPoolingRerankModelInfo,
                      EmbedModelInfo, LASTPoolingEmbedModelInfo,
                      RerankModelInfo)
from .embed_utils import correctness_test_embed_models
from .mteb_utils import mteb_test_embed_models, mteb_test_rerank_models

MODELS = [
    ########## BertModel
    CLSPoolingEmbedModelInfo("BAAI/bge-base-en",
                             architecture="BertModel",
                             enable_test=True),
    CLSPoolingEmbedModelInfo("BAAI/bge-base-zh",
                             architecture="BertModel",
                             enable_test=False),
    CLSPoolingEmbedModelInfo("BAAI/bge-small-en",
                             architecture="BertModel",
                             enable_test=False),
    CLSPoolingEmbedModelInfo("BAAI/bge-small-zh",
                             architecture="BertModel",
                             enable_test=False),
    CLSPoolingEmbedModelInfo("BAAI/bge-large-en",
                             architecture="BertModel",
                             enable_test=False),
    CLSPoolingEmbedModelInfo("BAAI/bge-large-zh",
                             architecture="BertModel",
                             enable_test=False),
    CLSPoolingEmbedModelInfo("BAAI/bge-large-zh-noinstruct",
                             architecture="BertModel",
                             enable_test=False),
    CLSPoolingEmbedModelInfo("BAAI/bge-base-en-v1.5",
                             architecture="BertModel",
                             enable_test=False),
    CLSPoolingEmbedModelInfo("BAAI/bge-base-zh-v1.5",
                             architecture="BertModel",
                             enable_test=False),
    CLSPoolingEmbedModelInfo("BAAI/bge-small-en-v1.5",
                             architecture="BertModel",
                             enable_test=False),
    CLSPoolingEmbedModelInfo("BAAI/bge-small-zh-v1.5",
                             architecture="BertModel",
                             enable_test=False),
    CLSPoolingEmbedModelInfo("BAAI/bge-large-en-v1.5",
                             architecture="BertModel",
                             enable_test=False),
    CLSPoolingEmbedModelInfo("BAAI/bge-large-zh-v1.5",
                             architecture="BertModel",
                             enable_test=False),
    ########## XLMRobertaModel
    CLSPoolingEmbedModelInfo("BAAI/bge-m3",
                             architecture="XLMRobertaModel",
                             enable_test=True),
    ########## Qwen2Model
    LASTPoolingEmbedModelInfo("BAAI/bge-code-v1",
                              architecture="Qwen2Model",
                              dtype="float32",
                              enable_test=True),
]

RERANK_MODELS = [
    ########## XLMRobertaForSequenceClassification
    CLSPoolingRerankModelInfo(
        "BAAI/bge-reranker-base",
        architecture="XLMRobertaForSequenceClassification",
        enable_test=True),
    CLSPoolingRerankModelInfo(
        "BAAI/bge-reranker-large",
        architecture="XLMRobertaForSequenceClassification",
        enable_test=False),
    CLSPoolingRerankModelInfo(
        "BAAI/bge-reranker-v2-m3",
        architecture="XLMRobertaForSequenceClassification",
        enable_test=False)
]


@pytest.mark.parametrize("model_info", MODELS)
def test_embed_models_mteb(hf_runner, vllm_runner,
                           model_info: EmbedModelInfo) -> None:
    mteb_test_embed_models(hf_runner, vllm_runner, model_info)


@pytest.mark.parametrize("model_info", MODELS)
def test_embed_models_correctness(hf_runner, vllm_runner,
                                  model_info: EmbedModelInfo,
                                  example_prompts) -> None:
    correctness_test_embed_models(hf_runner, vllm_runner, model_info,
                                  example_prompts)


@pytest.mark.parametrize("model_info", RERANK_MODELS)
def test_rerank_models_mteb(hf_runner, vllm_runner,
                            model_info: RerankModelInfo) -> None:
    mteb_test_rerank_models(hf_runner, vllm_runner, model_info)
