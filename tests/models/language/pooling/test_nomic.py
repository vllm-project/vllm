# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.models.utils import EmbedModelInfo

MODELS = [
    EmbedModelInfo("nomic-ai/nomic-embed-text-v1",
                   architecture="NomicBertModel",
                   dtype="float32",
                   enable_test=True),
    EmbedModelInfo("nomic-ai/nomic-embed-text-v1.5",
                   architecture="NomicBertModel",
                   dtype="float32",
                   enable_test=True),
    EmbedModelInfo("nomic-ai/nomic-embed-text-v2-moe",
                   architecture="NomicBertModel",
                   dtype="float32",
                   enable_test=True)
]


@pytest.mark.parametrize("model_info", MODELS)
def test_models(
    hf_runner,
    vllm_runner,
    model_info: EmbedModelInfo,
    monkeypatch,
) -> None:
    from .mteb_utils import mteb_test_embed_models
    mteb_test_embed_models(hf_runner, vllm_runner, model_info)
