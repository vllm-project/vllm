# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.language.pooling.embed_utils import correctness_test_embed_models
from tests.models.utils import CLSPoolingEmbedModelInfo, EmbedModelInfo

from .mteb_utils import mteb_test_embed_models

MODELS = [
    CLSPoolingEmbedModelInfo(
        "Snowflake/snowflake-arctic-embed-xs",
        is_matryoshka=False,
        architecture="BertModel",
        mteb_score=0.714927797,
        enable_test=True,
    ),
    CLSPoolingEmbedModelInfo(
        "Snowflake/snowflake-arctic-embed-s",
        is_matryoshka=False,
        architecture="BertModel",
        enable_test=False,
    ),
    CLSPoolingEmbedModelInfo(
        "Snowflake/snowflake-arctic-embed-m",
        is_matryoshka=False,
        architecture="BertModel",
        enable_test=False,
    ),
    CLSPoolingEmbedModelInfo(
        "Snowflake/snowflake-arctic-embed-m-long",
        is_matryoshka=False,
        architecture="NomicBertModel",
        mteb_score=0.681146831,
        enable_test=True,
    ),
    CLSPoolingEmbedModelInfo(
        "Snowflake/snowflake-arctic-embed-l",
        is_matryoshka=False,
        architecture="BertModel",
        enable_test=False,
    ),
    CLSPoolingEmbedModelInfo(
        "Snowflake/snowflake-arctic-embed-m-v1.5",
        is_matryoshka=True,
        architecture="BertModel",
        mteb_score=0.649088363,
        enable_test=True,
    ),
    CLSPoolingEmbedModelInfo(
        "Snowflake/snowflake-arctic-embed-l-v2.0",
        is_matryoshka=True,
        architecture="XLMRobertaModel",
        mteb_score=0.712258299,
        enable_test=True,
    ),
    CLSPoolingEmbedModelInfo(
        "Snowflake/snowflake-arctic-embed-m-v2.0",
        is_matryoshka=True,
        architecture="GteModel",
        mteb_score=0.706622444,
        enable_test=True,
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
