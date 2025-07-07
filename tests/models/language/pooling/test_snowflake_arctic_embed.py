# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from .embed_utils import EmbedModelInfo, correctness_test_embed_models
from .mteb_utils import mteb_test_embed_models

MODELS = [
    EmbedModelInfo("Snowflake/snowflake-arctic-embed-xs",
                   is_matryoshka=False,
                   architecture="BertModel",
                   enable_test=True),
    EmbedModelInfo("Snowflake/snowflake-arctic-embed-s",
                   is_matryoshka=False,
                   architecture="BertModel",
                   enable_test=False),
    EmbedModelInfo("Snowflake/snowflake-arctic-embed-m",
                   is_matryoshka=False,
                   architecture="BertModel",
                   enable_test=False),
    EmbedModelInfo("Snowflake/snowflake-arctic-embed-m-long",
                   is_matryoshka=False,
                   architecture="NomicBertModel",
                   enable_test=True),
    EmbedModelInfo("Snowflake/snowflake-arctic-embed-l",
                   is_matryoshka=False,
                   architecture="BertModel",
                   enable_test=False),
    EmbedModelInfo("Snowflake/snowflake-arctic-embed-m-v1.5",
                   is_matryoshka=True,
                   architecture="BertModel",
                   enable_test=True),
    EmbedModelInfo("Snowflake/snowflake-arctic-embed-l-v2.0",
                   is_matryoshka=True,
                   architecture="XLMRobertaModel",
                   enable_test=True),
    EmbedModelInfo("Snowflake/snowflake-arctic-embed-m-v2.0",
                   is_matryoshka=True,
                   architecture="GteModel",
                   enable_test=True),
]


@pytest.mark.parametrize("model_info", MODELS)
def test_embed_models_mteb(hf_runner, vllm_runner,
                           model_info: EmbedModelInfo) -> None:
    mteb_test_embed_models(hf_runner, vllm_runner, model_info)


@pytest.mark.parametrize("model_info", MODELS)
def test_embed_models_correctness(hf_runner, vllm_runner,
                                  model_info: EmbedModelInfo,
                                  example_prompts) -> None:
    correctness_test_embed_models(hf_runner, vllm_runner, model_info,
                                  example_prompts)
