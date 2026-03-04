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
        hf_overrides={
            "architectures": ["VoyageQwen3BidirectionalEmbedModel"],
            "num_labels": 2048,
        },
        mteb_score=0.7054,
        # === MTEB Results ===
        # STS12: 0.6613
        # STS13: 0.6906
        # STS14: 0.6556
        # STS15: 0.7843
        # STS16: 0.7340
        # STSBenchmark: 0.7063
        # Average score: 0.7054
    ),
]


@pytest.mark.parametrize("model_info", MODELS)
def test_embed_models_mteb(hf_runner, vllm_runner, model_info: EmbedModelInfo) -> None:
    # Encoder-only attention models need enforce_eager=True to avoid
    # CUDA graph capture issues with piecewise compilation
    mteb_test_embed_models(
        hf_runner, vllm_runner, model_info, vllm_extra_kwargs={"enforce_eager": True}
    )


@pytest.mark.parametrize("model_info", MODELS)
def test_embed_models_correctness(
    hf_runner, vllm_runner, model_info: EmbedModelInfo, example_prompts
) -> None:
    correctness_test_embed_models(
        hf_runner,
        vllm_runner,
        model_info,
        example_prompts,
        vllm_extra_kwargs={"enforce_eager": True},
    )
