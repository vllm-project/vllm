# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.utils import (
    EmbedModelInfo,
    LASTPoolingEmbedModelInfo,
    LASTPoolingRerankModelInfo,
    RerankModelInfo,
)

from .mteb_utils import mteb_test_embed_models, mteb_test_rerank_models

EMBEDDING_MODELS = [
    LASTPoolingEmbedModelInfo(
        "nvidia/llama-nemotron-embed-1b-v2",
        architecture="LlamaBidirectionalModel",
        mteb_score=0.689164662128673,
    )
]

RERANK_MODELS = [
    LASTPoolingRerankModelInfo(
        "nvidia/llama-nemotron-rerank-1b-v2",
        architecture="LlamaBidirectionalForSequenceClassification",
        chat_template_name="nemotron-rerank.jinja",
        mteb_score=0.33994,
    ),
]


@pytest.mark.parametrize("model_info", EMBEDDING_MODELS)
def test_embed_models_mteb(hf_runner, vllm_runner, model_info: EmbedModelInfo) -> None:
    mteb_test_embed_models(hf_runner, vllm_runner, model_info)


@pytest.mark.parametrize("model_info", RERANK_MODELS)
def test_rerank_models_mteb(
    hf_runner, vllm_runner, model_info: RerankModelInfo
) -> None:
    mteb_test_rerank_models(hf_runner, vllm_runner, model_info)
