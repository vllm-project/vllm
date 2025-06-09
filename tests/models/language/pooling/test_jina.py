# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import partial

import pytest

from vllm import PoolingParams

from .embed_utils import (EmbedModelInfo, check_embeddings_close,
                          correctness_test_embed_models, matryoshka_fy)
from .mteb_utils import mteb_test_embed_models

SCORING_MODELS = [
    "jinaai/jina-reranker-v2-base-multilingual",  # Roberta
]

TEXTS_1 = ["Organic skincare products for sensitive skin"]

TEXTS_2 = [
    "Organic skincare for sensitive skin with aloe vera and chamomile.",
    "New makeup trends focus on bold colors and innovative techniques",
    "Bio-Hautpflege für empfindliche Haut mit Aloe Vera und Kamille",
    "Neue Make-up-Trends setzen auf kräftige Farben und innovative Techniken",  # noqa: E501
    "Cuidado de la piel orgánico para piel sensible con aloe vera y manzanilla",  # noqa: E501
    "Las nuevas tendencias de maquillaje se centran en colores vivos y técnicas innovadoras",  # noqa: E501
    "针对敏感肌专门设计的天然有机护肤产品",
    "新的化妆趋势注重鲜艳的颜色和创新的技巧",
    "敏感肌のために特別に設計された天然有機スキンケア製品",
    "新しいメイクのトレンドは鮮やかな色と革新的な技術に焦点を当てています",
]

EMBEDDING_MODELS = [
    EmbedModelInfo("jinaai/jina-embeddings-v3",
                   architecture="XLMRobertaModel",
                   is_matryoshka=True)
]


@pytest.fixture(scope="module", params=SCORING_MODELS)
def model_name(request):
    yield request.param


@pytest.mark.parametrize("dtype", ["half"])
def test_llm_1_to_1(vllm_runner, hf_runner, model_name, dtype: str):

    text_pair = [TEXTS_1[0], TEXTS_2[0]]

    with hf_runner(model_name, dtype=dtype, is_cross_encoder=True) as hf_model:
        hf_outputs = hf_model.predict([text_pair]).tolist()

    with vllm_runner(model_name, task="score", dtype=dtype,
                     max_model_len=None) as vllm_model:
        vllm_outputs = vllm_model.score(text_pair[0], text_pair[1])

    assert len(vllm_outputs) == 1
    assert len(hf_outputs) == 1

    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.01)


@pytest.mark.parametrize("dtype", ["half"])
def test_llm_1_to_N(vllm_runner, hf_runner, model_name, dtype: str):

    text_pairs = [[TEXTS_1[0], text] for text in TEXTS_2]

    with hf_runner(model_name, dtype=dtype, is_cross_encoder=True) as hf_model:
        hf_outputs = hf_model.predict(text_pairs).tolist()

    with vllm_runner(model_name, task="score", dtype=dtype,
                     max_model_len=None) as vllm_model:
        vllm_outputs = vllm_model.score(TEXTS_1[0], TEXTS_2)

    assert len(vllm_outputs) == 10
    assert len(hf_outputs) == 10

    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.01)
    assert hf_outputs[1] == pytest.approx(vllm_outputs[1], rel=0.01)


@pytest.mark.parametrize("model_info", EMBEDDING_MODELS)
def test_embed_models_mteb(hf_runner, vllm_runner,
                           model_info: EmbedModelInfo) -> None:

    def hf_model_callback(model):
        model.encode = partial(model.encode, task="text-matching")

    mteb_test_embed_models(hf_runner,
                           vllm_runner,
                           model_info,
                           hf_model_callback=hf_model_callback)


@pytest.mark.parametrize("model_info", EMBEDDING_MODELS)
def test_embed_models_correctness(hf_runner, vllm_runner,
                                  model_info: EmbedModelInfo,
                                  example_prompts) -> None:

    def hf_model_callback(model):
        model.encode = partial(model.encode, task="text-matching")

    correctness_test_embed_models(hf_runner,
                                  vllm_runner,
                                  model_info,
                                  example_prompts,
                                  hf_model_callback=hf_model_callback)


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

    with vllm_runner(model_info.name,
                     task="embed",
                     dtype=dtype,
                     max_model_len=None) as vllm_model:
        assert vllm_model.model.llm_engine.model_config.is_matryoshka

        matryoshka_dimensions = (
            vllm_model.model.llm_engine.model_config.matryoshka_dimensions)
        assert matryoshka_dimensions is not None

        if dimensions not in matryoshka_dimensions:
            with pytest.raises(ValueError):
                vllm_model.encode(
                    example_prompts,
                    pooling_params=PoolingParams(dimensions=dimensions))
        else:
            vllm_outputs = vllm_model.encode(
                example_prompts,
                pooling_params=PoolingParams(dimensions=dimensions))

            check_embeddings_close(
                embeddings_0_lst=hf_outputs,
                embeddings_1_lst=vllm_outputs,
                name_0="hf",
                name_1="vllm",
                tol=1e-2,
            )
