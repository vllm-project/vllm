# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from tests.models.utils import (
    RerankModelInfo,
)

from .mteb_score_utils import mteb_test_rerank_models

RERANK_MODELS = [
    RerankModelInfo(
        "noooop9527/llama-nemotron-rerank-1b-v2-STv6",
        architecture="LlamaBidirectionalForSequenceClassification",
        mteb_score=0.33994,
        pooling_type="MEAN",
        attn_type="encoder_only",
        is_prefix_caching_supported=False,
        is_chunked_prefill_supported=False,
    ),
]


@pytest.mark.parametrize("model_info", RERANK_MODELS)
def test_rerank_models_mteb(vllm_runner, model_info: RerankModelInfo) -> None:
    mteb_test_rerank_models(vllm_runner, model_info)
