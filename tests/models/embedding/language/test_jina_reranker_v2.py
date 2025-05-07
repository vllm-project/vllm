# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
"""Compare the scoring outputs of HF and vLLM models.

Run `pytest tests/models/embedding/language/test_jina_reranker_v2.py`.
"""
import math

import pytest

MODELS = [
    "jinaai/jina-reranker-v2-base-multilingual",  # Roberta
]

TEXTS_1 = ["Organic skincare products for sensitive skin"]

TEXTS_2 = [
    "Organic skincare for sensitive skin with aloe vera and chamomile.",
    "New makeup trends focus on bold colors and innovative techniques",
    "Bio-Hautpflege für empfindliche Haut mit Aloe Vera und Kamille",
    "Neue Make-up-Trends setzen auf kräftige Farben und innovative Techniken",
    "Cuidado de la piel orgánico para piel sensible con aloe vera y manzanilla",
    "Las nuevas tendencias de maquillaje se centran en colores vivos y técnicas innovadoras",
    "针对敏感肌专门设计的天然有机护肤产品",
    "新的化妆趋势注重鲜艳的颜色和创新的技巧",
    "敏感肌のために特別に設計された天然有機スキンケア製品",
    "新しいメイクのトレンドは鮮やかな色と革新的な技術に焦点を当てています",
]


@pytest.fixture(scope="module", params=MODELS)
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

    assert math.isclose(hf_outputs[0], vllm_outputs[0], rel_tol=0.01)


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

    assert math.isclose(hf_outputs[0], vllm_outputs[0], rel_tol=0.01)
    assert math.isclose(hf_outputs[1], vllm_outputs[1], rel_tol=0.01)
