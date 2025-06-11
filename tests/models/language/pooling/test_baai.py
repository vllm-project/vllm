# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from .embed_utils import EmbedModelInfo, correctness_test_embed_models
from .mteb_utils import mteb_test_embed_models

MODELS = [
    ########## BertModel
    EmbedModelInfo("BAAI/bge-base-en",
                   architecture="BertModel",
                   enable_test=True),
    EmbedModelInfo("BAAI/bge-base-zh",
                   architecture="BertModel",
                   enable_test=False),
    EmbedModelInfo("BAAI/bge-small-en",
                   architecture="BertModel",
                   enable_test=False),
    EmbedModelInfo("BAAI/bge-small-zh",
                   architecture="BertModel",
                   enable_test=False),
    EmbedModelInfo("BAAI/bge-large-en",
                   architecture="BertModel",
                   enable_test=False),
    EmbedModelInfo("BAAI/bge-large-zh",
                   architecture="BertModel",
                   enable_test=False),
    EmbedModelInfo("BAAI/bge-large-zh-noinstruct",
                   architecture="BertModel",
                   enable_test=False),
    EmbedModelInfo("BAAI/bge-base-en-v1.5",
                   architecture="BertModel",
                   enable_test=False),
    EmbedModelInfo("BAAI/bge-base-zh-v1.5",
                   architecture="BertModel",
                   enable_test=False),
    EmbedModelInfo("BAAI/bge-small-en-v1.5",
                   architecture="BertModel",
                   enable_test=False),
    EmbedModelInfo("BAAI/bge-small-zh-v1.5",
                   architecture="BertModel",
                   enable_test=False),
    EmbedModelInfo("BAAI/bge-large-en-v1.5",
                   architecture="BertModel",
                   enable_test=False),
    EmbedModelInfo("BAAI/bge-large-zh-v1.5",
                   architecture="BertModel",
                   enable_test=False),
    ########## XLMRobertaModel
    EmbedModelInfo("BAAI/bge-m3",
                   architecture="XLMRobertaModel",
                   enable_test=True),
    ########## Qwen2Model
    EmbedModelInfo("BAAI/bge-code-v1",
                   architecture="Qwen2Model",
                   dtype="float32",
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
