# SPDX-License-Identifier: Apache-2.0
import pytest

from ...utils import EmbedModelInfo, check_embeddings_close

EMBEDDING_PROMPTS = [
    'what is snowflake?', 'Where can I get the best tacos?', 'The Data Cloud!',
    'Mexico City of Course!'
]

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
@pytest.mark.parametrize("dtype", ["half"])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model_info: EmbedModelInfo,
    dtype: str,
    monkeypatch,
) -> None:
    if not model_info.enable_test:
        # A model family has many models with the same architecture,
        # and we don't need to test each one.
        pytest.skip("Skipping test.")

    example_prompts = example_prompts + EMBEDDING_PROMPTS

    vllm_extra_kwargs = {
        "hf_overrides": {
            "is_matryoshka": model_info.is_matryoshka
        }
    }

    with hf_runner(model_info.name, dtype=dtype,
                   is_sentence_transformer=True) as hf_model:
        hf_outputs = hf_model.encode(example_prompts)

    with vllm_runner(model_info.name,
                     task="embed",
                     dtype=dtype,
                     max_model_len=None,
                     **vllm_extra_kwargs) as vllm_model:

        assert (vllm_model.model.llm_engine.model_config.is_matryoshka ==
                model_info.is_matryoshka)

        if model_info.architecture:
            assert (model_info.architecture
                    in vllm_model.model.llm_engine.model_config.architectures)

        vllm_outputs = vllm_model.encode(example_prompts)

    check_embeddings_close(
        embeddings_0_lst=hf_outputs,
        embeddings_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
        tol=1e-2,
    )
