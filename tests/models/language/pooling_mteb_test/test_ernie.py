# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.language.pooling.embed_utils import correctness_test_embed_models
from tests.models.utils import EmbedModelInfo

from .mteb_embed_utils import mteb_test_embed_models

MODELS = [
    EmbedModelInfo(
        "shibing624/text2vec-base-chinese-sentence",
        architecture="ErnieModel",
        mteb_score=0.536523112,
        seq_pooling_type="MEAN",
        attn_type="encoder_only",
        is_prefix_caching_supported=False,
        is_chunked_prefill_supported=False,
        enable_test=True,
    ),
]


@pytest.mark.parametrize("model_info", MODELS, ids=lambda model_info: model_info.name)
def test_embed_models_mteb(hf_runner, vllm_runner, model_info: EmbedModelInfo) -> None:
    mteb_test_embed_models(
        hf_runner,
        vllm_runner,
        model_info,
        vllm_extra_kwargs={"gpu_memory_utilization": 0.2},
    )


@pytest.mark.parametrize("model_info", MODELS, ids=lambda model_info: model_info.name)
def test_embed_models_correctness(
    hf_runner, vllm_runner, model_info: EmbedModelInfo, example_prompts
) -> None:
    correctness_test_embed_models(
        hf_runner,
        vllm_runner,
        model_info,
        example_prompts,
        vllm_extra_kwargs={"gpu_memory_utilization": 0.2},
    )
