# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import pytest

from .embed_utils import EmbedModelInfo, correctness_test_embed_models
from .mteb_utils import mteb_test_embed_models

MODELS = [
    ########## BertModel
    EmbedModelInfo("thenlper/gte-large",
                   architecture="BertModel",
                   enable_test=True),
    EmbedModelInfo("thenlper/gte-base",
                   architecture="BertModel",
                   enable_test=False),
    EmbedModelInfo("thenlper/gte-small",
                   architecture="BertModel",
                   enable_test=False),
    EmbedModelInfo("thenlper/gte-large-zh",
                   architecture="BertModel",
                   enable_test=False),
    EmbedModelInfo("thenlper/gte-base-zh",
                   architecture="BertModel",
                   enable_test=False),
    EmbedModelInfo("thenlper/gte-small-zh",
                   architecture="BertModel",
                   enable_test=False),
    ########### NewModel
    EmbedModelInfo("Alibaba-NLP/gte-multilingual-base",
                   architecture="GteNewModel",
                   enable_test=True),
    EmbedModelInfo("Alibaba-NLP/gte-base-en-v1.5",
                   architecture="GteNewModel",
                   enable_test=True),
    EmbedModelInfo("Alibaba-NLP/gte-large-en-v1.5",
                   architecture="GteNewModel",
                   enable_test=True),
    ########### Qwen2ForCausalLM
    EmbedModelInfo("Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                   architecture="Qwen2ForCausalLM",
                   enable_test=True),
    ########## ModernBertModel
    EmbedModelInfo("Alibaba-NLP/gte-modernbert-base",
                   architecture="ModernBertModel",
                   enable_test=True),
    ########## Qwen3ForCausalLM
    EmbedModelInfo("Qwen/Qwen3-Embedding-0.6B",
                   architecture="Qwen3ForCausalLM",
                   dtype="float32",
                   enable_test=True),
    EmbedModelInfo("Qwen/Qwen3-Embedding-4B",
                   architecture="Qwen3ForCausalLM",
                   dtype="float32",
                   enable_test=False),
]


@pytest.mark.parametrize("model_info", MODELS)
def test_embed_models_mteb(hf_runner, vllm_runner,
                           model_info: EmbedModelInfo) -> None:

    vllm_extra_kwargs: dict[str, Any] = {}
    if model_info.architecture == "GteNewModel":
        vllm_extra_kwargs["hf_overrides"] = {"architectures": ["GteNewModel"]}

    mteb_test_embed_models(hf_runner, vllm_runner, model_info,
                           vllm_extra_kwargs)


@pytest.mark.parametrize("model_info", MODELS)
def test_embed_models_correctness(hf_runner, vllm_runner,
                                  model_info: EmbedModelInfo,
                                  example_prompts) -> None:

    vllm_extra_kwargs: dict[str, Any] = {}
    if model_info.architecture == "GteNewModel":
        vllm_extra_kwargs["hf_overrides"] = {"architectures": ["GteNewModel"]}

    correctness_test_embed_models(hf_runner, vllm_runner, model_info,
                                  example_prompts, vllm_extra_kwargs)
