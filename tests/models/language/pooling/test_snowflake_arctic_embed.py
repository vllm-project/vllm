# SPDX-License-Identifier: Apache-2.0
import math

import pytest

from tests.models.utils import EmbedModelInfo

from .mteb_utils import MTEB_EMBED_TASKS, VllmMtebEncoder, run_mteb_embed_task

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
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model_info: EmbedModelInfo,
    monkeypatch,
) -> None:
    if not model_info.enable_test:
        # A model family has many models with the same architecture,
        # and we don't need to test each one.
        pytest.skip("Skipping test.")

    vllm_extra_kwargs = {
        "hf_overrides": {
            "is_matryoshka": model_info.is_matryoshka
        }
    }

    with hf_runner(model_info.name, is_sentence_transformer=True) as hf_model:
        st_main_score = run_mteb_embed_task(hf_model, MTEB_EMBED_TASKS)

    with vllm_runner(model_info.name,
                     task="embed",
                     max_model_len=None,
                     **vllm_extra_kwargs) as vllm_model:

        assert (vllm_model.model.llm_engine.model_config.is_matryoshka ==
                model_info.is_matryoshka)

        if model_info.architecture:
            assert (model_info.architecture
                    in vllm_model.model.llm_engine.model_config.architectures)

        vllm_main_score = run_mteb_embed_task(VllmMtebEncoder(vllm_model),
                                              MTEB_EMBED_TASKS)

    print("VLLM main score: ", vllm_main_score)
    print("SentenceTransformer main score: ", st_main_score)
    print("Difference: ", st_main_score - vllm_main_score)

    assert math.isclose(st_main_score, vllm_main_score, rel_tol=1e-4)
