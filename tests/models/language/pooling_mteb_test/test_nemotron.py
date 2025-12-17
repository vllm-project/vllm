# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.utils import (
    CLSPoolingEmbedModelInfo,
    CLSPoolingRerankModelInfo,
    EmbedModelInfo,
    RerankModelInfo,
)

from .mteb_utils import mteb_test_embed_models, mteb_test_rerank_models

EMBEDDING_MODELS = [
    CLSPoolingEmbedModelInfo(
        "nvidia/llama-nemotron-embed-1b-v2",
        architecture="LlamaBidirectionalModel",
        mteb_score=0.6891299463693507,
    )
]

RERANK_MODELS = [
    CLSPoolingRerankModelInfo(
        "nvidia/llama-nemotron-rerank-1b-v2",
        architecture="LlamaBidirectionalForSequenceClassification",
        mteb_score=0.6891299463693507,
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
