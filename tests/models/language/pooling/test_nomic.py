# SPDX-License-Identifier: Apache-2.0

import pytest

from .embed_utils import EmbedModelInfo, correctness_test_embed_models
from .mteb_utils import mteb_test_embed_models

MODELS = [
    EmbedModelInfo("nomic-ai/nomic-embed-text-v1",
                   architecture="NomicBertModel",
                   dtype="float32",
                   enable_test=True),
    EmbedModelInfo("nomic-ai/nomic-embed-text-v1.5",
                   architecture="NomicBertModel",
                   dtype="float32",
                   enable_test=False),
    EmbedModelInfo("nomic-ai/CodeRankEmbed",
                   architecture="NomicBertModel",
                   enable_test=False),
    EmbedModelInfo("nomic-ai/nomic-embed-text-v2-moe",
                   architecture="NomicBertModel",
                   dtype="float32",
                   enable_test=True)
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
