# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.language.pooling.embed_utils import correctness_test_embed_models
from tests.models.utils import EmbedModelInfo

from .mteb_embed_utils import mteb_test_embed_models

MODELS = [
    EmbedModelInfo(
        "voyageai/voyage-4-nano",
        architecture="VoyageQwen3BidirectionalEmbedModel",
        enable_test=True,
        seq_pooling_type="MEAN",
        attn_type="encoder_only",
        is_prefix_caching_supported=False,
        is_chunked_prefill_supported=False,
        hf_overrides={"architectures": ["VoyageQwen3BidirectionalEmbedModel"]},
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
