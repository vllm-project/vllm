# SPDX-License-Identifier: Apache-2.0
from typing import Any

import pytest

from ...utils import EmbedModelInfo, run_embedding_correctness_test

MODELS = [
    ########## BertModel
    EmbedModelInfo("thenlper/gte-large",
                   architecture="BertModel",
                   dtype="float32",
                   enable_test=True),
    EmbedModelInfo("thenlper/gte-base",
                   architecture="BertModel",
                   dtype="float32",
                   enable_test=False),
    EmbedModelInfo("thenlper/gte-small",
                   architecture="BertModel",
                   dtype="float32",
                   enable_test=False),
    EmbedModelInfo("thenlper/gte-large-zh",
                   architecture="BertModel",
                   dtype="float32",
                   enable_test=False),
    EmbedModelInfo("thenlper/gte-base-zh",
                   architecture="BertModel",
                   dtype="float32",
                   enable_test=False),
    EmbedModelInfo("thenlper/gte-small-zh",
                   architecture="BertModel",
                   dtype="float32",
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
    EmbedModelInfo("Alibaba-NLP/gte-Qwen2-7B-instruct",
                   architecture="Qwen2ForCausalLM",
                   enable_test=False),
    ########## ModernBertModel
    EmbedModelInfo("Alibaba-NLP/gte-modernbert-base",
                   architecture="ModernBertModel",
                   enable_test=True),
]


@pytest.mark.parametrize("model_info", MODELS)
def test_models_mteb(hf_runner, vllm_runner,
                     model_info: EmbedModelInfo) -> None:
    pytest.skip("Skipping mteb test.")

    from .mteb_utils import mteb_test_embed_models

    vllm_extra_kwargs: dict[str, Any] = {}
    if model_info.name == "Alibaba-NLP/gte-Qwen2-1.5B-instruct":
        vllm_extra_kwargs["hf_overrides"] = {"is_causal": True}

    if model_info.architecture == "GteNewModel":
        vllm_extra_kwargs["hf_overrides"] = {"architectures": ["GteNewModel"]}

    mteb_test_embed_models(hf_runner, vllm_runner, model_info,
                           vllm_extra_kwargs)


@pytest.mark.parametrize("model_info", MODELS)
def test_models_correctness(hf_runner, vllm_runner, model_info: EmbedModelInfo,
                            example_prompts) -> None:
    if not model_info.enable_test:
        pytest.skip("Skipping test.")

    # ST will strip the input texts, see test_embedding.py
    example_prompts = [str(s).strip() for s in example_prompts]

    vllm_extra_kwargs: dict[str, Any] = {}
    if model_info.name == "Alibaba-NLP/gte-Qwen2-1.5B-instruct":
        vllm_extra_kwargs["hf_overrides"] = {"is_causal": True}

    if model_info.architecture == "GteNewModel":
        vllm_extra_kwargs["hf_overrides"] = {"architectures": ["GteNewModel"]}

    with vllm_runner(model_info.name,
                     task="embed",
                     dtype=model_info.dtype,
                     max_model_len=None,
                     **vllm_extra_kwargs) as vllm_model:
        vllm_outputs = vllm_model.encode(example_prompts)

    with hf_runner(
            model_info.name,
            dtype=model_info.dtype,
            is_sentence_transformer=True,
    ) as hf_model:
        run_embedding_correctness_test(hf_model, example_prompts, vllm_outputs)
