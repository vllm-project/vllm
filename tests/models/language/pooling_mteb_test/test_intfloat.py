# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from tests.models.utils import EmbedModelInfo

from .mteb_embed_utils import mteb_test_embed_models

MODELS = [
    ########## BertModel
    EmbedModelInfo(
        "intfloat/e5-small",
        architecture="BertModel",
        mteb_score=0.742285423,
        mteb_tol=2e-3,
        seq_pooling_type="MEAN",
        attn_type="encoder_only",
        is_prefix_caching_supported=False,
        is_chunked_prefill_supported=False,
    ),
    ########## XLMRobertaModel
    EmbedModelInfo(
        "intfloat/multilingual-e5-base",
        architecture="XLMRobertaModel",
        mteb_score=0.779325955,
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
