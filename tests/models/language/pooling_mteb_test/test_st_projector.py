# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from tests.models.utils import (
    EmbedModelInfo,
)

from .mteb_embed_utils import mteb_test_embed_models

# ST models with projector (Dense) layers
ST_PROJECTOR_MODELS = [
    EmbedModelInfo(
        "TencentBAC/Conan-embedding-v1",
        architecture="BertModel",
        mteb_score=0.688611955,
        pooling_type="MEAN",
        attn_type="encoder_only",
        is_prefix_caching_supported=False,
        is_chunked_prefill_supported=False,
        enable_test=True,
    ),
    EmbedModelInfo(
        "google/embeddinggemma-300m",
        architecture="Gemma3TextModel",
        mteb_score=0.7473819294684156,
        pooling_type="MEAN",
        attn_type="encoder_only",
        is_prefix_caching_supported=False,
        is_chunked_prefill_supported=False,
        enable_test=True,
        dtype="float32",
    ),
]


@pytest.mark.parametrize("model_info", ST_PROJECTOR_MODELS)
def test_embed_models_mteb(hf_runner, vllm_runner, model_info: EmbedModelInfo) -> None:
    mteb_test_embed_models(hf_runner, vllm_runner, model_info)
