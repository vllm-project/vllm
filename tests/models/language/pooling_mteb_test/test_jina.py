# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import partial

import pytest

from tests.models.language.pooling.embed_utils import (
    check_embeddings_close,
    correctness_test_embed_models,
    matryoshka_fy,
)
from tests.models.utils import (
    EmbedModelInfo,
    RerankModelInfo,
)
from vllm import PoolingParams

from .mteb_embed_utils import mteb_test_embed_models
from .mteb_score_utils import mteb_test_rerank_models

EMBEDDING_MODELS = [
    EmbedModelInfo(
        "jinaai/jina-embeddings-v3",
        mteb_score=0.824413164,
        architecture="XLMRobertaModel",
        is_matryoshka=True,
        seq_pooling_type="MEAN",
        attn_type="encoder_only",
        is_prefix_caching_supported=False,
        is_chunked_prefill_supported=False,
    )
]

RERANK_MODELS = [
    RerankModelInfo(
        "jinaai/jina-reranker-v2-base-multilingual",
        mteb_score=0.33643,
        architecture="XLMRobertaForSequenceClassification",
        seq_pooling_type="CLS",
        attn_type="encoder_only",
        is_prefix_caching_supported=False,
        is_chunked_prefill_supported=False,
    )
]


@pytest.mark.parametrize("model_info", EMBEDDING_MODELS)
def test_embed_models_mteb(hf_runner, vllm_runner, model_info: EmbedModelInfo) -> None:
    def hf_model_callback(model):
        model.encode = partial(model.encode, task="text-matching")

    mteb_test_embed_models(
        hf_runner, vllm_runner, model_info, hf_model_callback=hf_model_callback
    )


@pytest.mark.parametrize("model_info", EMBEDDING_MODELS)
def test_embed_models_correctness(
    hf_runner, vllm_runner, model_info: EmbedModelInfo, example_prompts
) -> None:
    def hf_model_callback(model):
        model.encode = partial(model.encode, task="text-matching")

    correctness_test_embed_models(
        hf_runner,
        vllm_runner,
        model_info,
        example_prompts,
        hf_model_callback=hf_model_callback,
    )


@pytest.mark.parametrize("model_info", RERANK_MODELS)
def test_rerank_models_mteb(vllm_runner, model_info: RerankModelInfo) -> None:
    mteb_test_rerank_models(vllm_runner, model_info)


@pytest.mark.parametrize("model_info", EMBEDDING_MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("dimensions", [16, 32])
def test_matryoshka(
    hf_runner,
    vllm_runner,
    model_info,
    dtype: str,
    dimensions: int,
    example_prompts,
    monkeypatch,
) -> None:
    if not model_info.is_matryoshka:
        pytest.skip("Model is not matryoshka")

    # ST will strip the input texts, see test_embedding.py
    example_prompts = [str(s).strip() for s in example_prompts]

    with hf_runner(
        model_info.name,
        dtype=dtype,
        is_sentence_transformer=True,
    ) as hf_model:
        hf_outputs = hf_model.encode(example_prompts, task="text-matching")
        hf_outputs = matryoshka_fy(hf_outputs, dimensions)

    with vllm_runner(
        model_info.name, runner="pooling", dtype=dtype, max_model_len=None
    ) as vllm_model:
        assert vllm_model.llm.llm_engine.model_config.is_matryoshka

        matryoshka_dimensions = (
            vllm_model.llm.llm_engine.model_config.matryoshka_dimensions
        )
        assert matryoshka_dimensions is not None

        if dimensions not in matryoshka_dimensions:
            with pytest.raises(ValueError):
                vllm_model.embed(
                    example_prompts, pooling_params=PoolingParams(dimensions=dimensions)
                )
        else:
            vllm_outputs = vllm_model.embed(
                example_prompts, pooling_params=PoolingParams(dimensions=dimensions)
            )

            check_embeddings_close(
                embeddings_0_lst=hf_outputs,
                embeddings_1_lst=vllm_outputs,
                name_0="hf",
                name_1="vllm",
                tol=1e-2,
            )
