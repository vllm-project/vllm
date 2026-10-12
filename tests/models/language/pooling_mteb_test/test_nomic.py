# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.utils import EmbedModelInfo

from .mteb_embed_utils import mteb_test_embed_models

MODELS = [
    EmbedModelInfo(
        "nomic-ai/nomic-embed-text-v1",
        architecture="NomicBertModel",
        # Fixme:
        #  Update nomic-embed code to support the latest
        #  HF version and remove revision set.
        revision="720244025c1a7e15661a174c63cce63c8218e52b",
        mteb_score=0.737568559,
        mteb_tol=2e-3,
        seq_pooling_type="MEAN",
        attn_type="encoder_only",
        is_prefix_caching_supported=False,
        is_chunked_prefill_supported=False,
    ),
    EmbedModelInfo(
        "nomic-ai/nomic-embed-text-v2-moe",
        architecture="NomicBertModel",
        mteb_score=0.715488912,
        mteb_tol=2e-3,
        seq_pooling_type="MEAN",
        attn_type="encoder_only",
        is_prefix_caching_supported=False,
        is_chunked_prefill_supported=False,
    ),
]


@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize("model_info", MODELS)
def test_embed_models_mteb(hf_runner, vllm_runner, model_info: EmbedModelInfo) -> None:
    mteb_test_embed_models(hf_runner, vllm_runner, model_info)
