# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from tests.models.language.pooling.embed_utils import correctness_test_embed_models
from tests.models.utils import EmbedModelInfo

from .mteb_embed_utils import mteb_test_embed_models

MODELS = [
    ########## BertModel
    EmbedModelInfo(
        "intfloat/e5-small",
        architecture="BertModel",
        mteb_score=0.742285423,
        pooling_type="MEAN",
        attn_type="encoder_only",
        is_prefix_caching_supported=False,
        is_chunked_prefill_supported=False,
        enable_test=True,
    ),
    EmbedModelInfo("intfloat/e5-base", architecture="BertModel", enable_test=False),
    EmbedModelInfo("intfloat/e5-large", architecture="BertModel", enable_test=False),
    EmbedModelInfo(
        "intfloat/multilingual-e5-small", architecture="BertModel", enable_test=False
    ),
    ########## XLMRobertaModel
    EmbedModelInfo(
        "intfloat/multilingual-e5-base",
        architecture="XLMRobertaModel",
        mteb_score=0.779325955,
        pooling_type="MEAN",
        attn_type="encoder_only",
        is_prefix_caching_supported=False,
        is_chunked_prefill_supported=False,
        enable_test=True,
    ),
    EmbedModelInfo(
        "intfloat/multilingual-e5-large",
        architecture="XLMRobertaModel",
        enable_test=False,
    ),
    EmbedModelInfo(
        "intfloat/multilingual-e5-large-instruct",
        architecture="XLMRobertaModel",
        enable_test=False,
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
