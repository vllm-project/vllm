# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from transformers import AutoModelForSequenceClassification

from tests.models.language.pooling.mteb_utils import mteb_test_rerank_models
from tests.models.utils import RerankModelInfo


@pytest.mark.parametrize(
    "model",
    [
        "cirimus/modernbert-base-go-emotions",
    ],
)
@pytest.mark.parametrize("dtype", ["half"])
def test_classify_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:
    example_prompts = ["This is a good movie."]

    with vllm_runner(model, max_model_len=512, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.classify(example_prompts, )

    with hf_runner(model,
                   dtype=dtype,
                   auto_cls=AutoModelForSequenceClassification) as hf_model:
        hf_outputs = hf_model.classify(example_prompts)

    for hf_output, vllm_output in zip(hf_outputs, vllm_outputs):
        hf_output = torch.tensor(hf_output)
        vllm_output = torch.tensor(vllm_output)

        assert torch.allclose(hf_output, vllm_output,
                              1e-3 if dtype == "float" else 1e-2)


RERANK_MODELS = [
    # classifier_pooling: mean
    RerankModelInfo("Alibaba-NLP/gte-reranker-modernbert-base",
                    architecture="ModernBertForSequenceClassification",
                    enable_test=True),
    # classifier_pooling: cls
    RerankModelInfo("cl-nagoya/ruri-v3-reranker-310m",
                    architecture="ModernBertForSequenceClassification",
                    enable_test=True),
]


@pytest.mark.parametrize(
    "model",
    [m.name for m in RERANK_MODELS],
)
@pytest.mark.parametrize("dtype", ["half"])
def test_reranker_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:
    TEXTS_1 = [
        "What is the capital of France?",
        "What is the capital of Germany?",
    ]

    TEXTS_2 = [
        "The capital of France is Paris.",
        "The capital of Germany is Berlin.",
    ]

    text_pairs = [
        [TEXTS_1[0], TEXTS_2[0]],
        [TEXTS_1[1], TEXTS_2[1]],
    ]

    with hf_runner(model, dtype=dtype, is_cross_encoder=True) as hf_model:
        hf_outputs = hf_model.predict(text_pairs).tolist()

    with vllm_runner(model, runner="pooling", dtype=dtype,
                     max_model_len=None) as vllm_model:
        vllm_outputs = vllm_model.score(TEXTS_1, TEXTS_2)

    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.01)
    assert hf_outputs[1] == pytest.approx(vllm_outputs[1], rel=0.01)


@pytest.mark.parametrize("model_info", RERANK_MODELS)
def test_rerank_models_mteb(hf_runner, vllm_runner,
                            model_info: RerankModelInfo) -> None:
    mteb_test_rerank_models(hf_runner, vllm_runner, model_info)
