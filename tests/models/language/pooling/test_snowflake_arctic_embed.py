# SPDX-License-Identifier: Apache-2.0

import pytest

from ...utils import EmbedModelInfo, run_embedding_correctness_test

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
def test_models_mteb(
    hf_runner,
    vllm_runner,
    model_info: EmbedModelInfo,
) -> None:
    pytest.skip("Skipping mteb test.")
    from .mteb_utils import mteb_test_embed_models
    mteb_test_embed_models(hf_runner, vllm_runner, model_info)


@pytest.mark.parametrize("model_info", MODELS)
def test_models_correctness(
    hf_runner,
    vllm_runner,
    model_info: EmbedModelInfo,
    example_prompts,
) -> None:
    if not model_info.enable_test:
        pytest.skip("Skipping test.")

    # ST will strip the input texts, see test_embedding.py
    example_prompts = [str(s).strip() for s in example_prompts]

    with vllm_runner(model_info.name,
                     task="embed",
                     dtype=model_info.dtype,
                     max_model_len=None) as vllm_model:
        vllm_outputs = vllm_model.encode(example_prompts)

    with hf_runner(
            model_info.name,
            dtype=model_info.dtype,
            is_sentence_transformer=True,
    ) as hf_model:
        run_embedding_correctness_test(hf_model, example_prompts, vllm_outputs)
