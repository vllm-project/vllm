# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.language.pooling.embed_utils import correctness_test_embed_models
from tests.models.utils import CLSPoolingEmbedModelInfo, EmbedModelInfo

from .mteb_utils import mteb_test_embed_models

MODELS = [
    CLSPoolingEmbedModelInfo(
        "nomic-ai/nomic-embed-text-v1",
        architecture="NomicBertModel",
        mteb_score=0.737568559,
        enable_test=True,
    ),
    CLSPoolingEmbedModelInfo(
        "nomic-ai/nomic-embed-text-v1.5",
        architecture="NomicBertModel",
        enable_test=False,
    ),
    CLSPoolingEmbedModelInfo(
        "nomic-ai/CodeRankEmbed", architecture="NomicBertModel", enable_test=False
    ),
    CLSPoolingEmbedModelInfo(
        "nomic-ai/nomic-embed-text-v2-moe",
        architecture="NomicBertModel",
        mteb_score=0.715488912,
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
