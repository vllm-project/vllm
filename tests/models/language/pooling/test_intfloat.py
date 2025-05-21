# SPDX-License-Identifier: Apache-2.0
import pytest

from ...utils import EmbedModelInfo, run_embedding_correctness_test
from .mteb_utils import mteb_test_embed_models


MODELS = [
    ########## BertModel
    EmbedModelInfo("intfloat/e5-small",
                   architecture="BertModel",
                   dtype="hybrid",
                   enable_test=True),
    EmbedModelInfo("intfloat/e5-base",
                   architecture="BertModel",
                   dtype="hybrid",
                   enable_test=False),
    EmbedModelInfo("intfloat/e5-small",
                   architecture="BertModel",
                   #dtype="hybrid",
                   enable_test=False),
    EmbedModelInfo("intfloat/multilingual-e5-small",
                   architecture="BertModel",
                   dtype="hybrid",
                   enable_test=False),
    ########## XLMRobertaModel
    EmbedModelInfo("intfloat/multilingual-e5-base",
                   architecture="XLMRobertaModel",
                   dtype="hybrid",
                   enable_test=True),
    EmbedModelInfo("intfloat/multilingual-e5-large",
                   architecture="XLMRobertaModel",
                   dtype="hybrid",
                   enable_test=False),
    EmbedModelInfo("intfloat/multilingual-e5-large-instruct",
                   architecture="XLMRobertaModel",
                   dtype="hybrid",
                   enable_test=False),
]


@pytest.mark.parametrize("model_info", MODELS)
def test_models_mteb(hf_runner, vllm_runner,
                     model_info: EmbedModelInfo) -> None:
    mteb_test_embed_models(hf_runner, vllm_runner, model_info)


@pytest.mark.parametrize("model_info", MODELS)
def test_models_correctness(hf_runner, vllm_runner, model_info: EmbedModelInfo,
                            example_prompts) -> None:
    if not model_info.enable_test:
        pytest.skip("Skipping test.")

    # ST will strip the input texts, see test_embedding.py
    example_prompts = [str(s).strip() for s in example_prompts]

    with vllm_runner(model_info.name,
                     task="embed",
                     dtype=model_info.dtype,
                     max_model_len=None) as vllm_model:
        vllm_outputs = vllm_model.encode(example_prompts)
        vllm_dtype = vllm_model.model.llm_engine.model_config.dtype
        model_dtype = getattr(
            vllm_model.model.llm_engine.model_config.hf_config, "torch_dtype",
            vllm_dtype)

    with hf_runner(
            model_info.name,
            dtype=model_dtype,
            is_sentence_transformer=True,
    ) as hf_model:
        run_embedding_correctness_test(hf_model, example_prompts, vllm_outputs)
