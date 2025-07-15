# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from ...utils import CLSEmbedModelInfo
from .embed_utils import correctness_test_embed_models
from .mteb_utils import mteb_test_embed_models

MODELS = [
    CLSEmbedModelInfo("nomic-ai/nomic-embed-text-v1",
                      architecture="NomicBertModel",
                      enable_test=True),
    CLSEmbedModelInfo("nomic-ai/nomic-embed-text-v1.5",
                      architecture="NomicBertModel",
                      enable_test=False),
    CLSEmbedModelInfo("nomic-ai/CodeRankEmbed",
                      architecture="NomicBertModel",
                      enable_test=False),
    CLSEmbedModelInfo("nomic-ai/nomic-embed-text-v2-moe",
                      architecture="NomicBertModel",
                      enable_test=True)
]


@pytest.mark.parametrize("model_info", MODELS)
def test_embed_models_mteb(hf_runner, vllm_runner,
                           model_info: CLSEmbedModelInfo) -> None:
    mteb_test_embed_models(hf_runner, vllm_runner, model_info)


@pytest.mark.parametrize("model_info", MODELS)
def test_embed_models_correctness(hf_runner, vllm_runner,
                                  model_info: CLSEmbedModelInfo,
                                  example_prompts) -> None:
    correctness_test_embed_models(hf_runner, vllm_runner, model_info,
                                  example_prompts)
