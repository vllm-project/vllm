# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from ...utils import (CLSPoolingEmbedModelInfo, EmbedModelInfo,
                      LASTPoolingEmbedModelInfo)
from .embed_utils import correctness_test_embed_models
from .mteb_utils import mteb_test_embed_models

# ST models with projector (Dense) layers
ST_PROJECTOR_MODELS = [
    CLSPoolingEmbedModelInfo(
        "TencentBAC/Conan-embedding-v1",
        architecture="BertModel",
        mteb_score=0.688611955,
        enable_test=True,
    ),
    LASTPoolingEmbedModelInfo("google/embeddinggemma-300m",
                              architecture="Gemma3TextModel",
                              enable_test=True)
]


@pytest.mark.parametrize("model_info", ST_PROJECTOR_MODELS)
def test_embed_models_mteb(hf_runner, vllm_runner,
                           model_info: EmbedModelInfo) -> None:

    mteb_test_embed_models(hf_runner, vllm_runner, model_info)


@pytest.mark.parametrize("model_info", ST_PROJECTOR_MODELS)
def test_embed_models_correctness(hf_runner, vllm_runner,
                                  model_info: EmbedModelInfo,
                                  example_prompts) -> None:
    # This test is needed to ensure the output embedding_dims are the same
    correctness_test_embed_models(hf_runner,
                                  vllm_runner,
                                  model_info,
                                  example_prompts,
                                  skip=False)
