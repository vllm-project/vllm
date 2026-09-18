# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import AutoModelForSequenceClassification

from vllm.config import ModelConfig, PoolerConfig
from vllm.model_executor.models.bert import BertPooler
from vllm.model_executor.models.modernbert import ModernBertPooler
from vllm.platforms import current_platform
from vllm.pooling_params import PoolingParams
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates


def _pooling_metadata(*use_activation: bool) -> PoolingMetadata:
    num_prompts = len(use_activation)
    return PoolingMetadata(
        prompt_lens=torch.ones(num_prompts, dtype=torch.long),
        prompt_token_ids=None,
        prompt_token_ids_cpu=None,
        pooling_params=[
            PoolingParams(task="classify", use_activation=flag)
            for flag in use_activation
        ],
        pooling_states=[PoolingStates() for _ in range(num_prompts)],
    )


@pytest.mark.parametrize("architecture", ["bert", "modernbert"])
def test_classifier_architecture_pooler_is_unconditional(architecture: str) -> None:
    if architecture == "bert":
        hf_config = SimpleNamespace(hidden_size=4)
        pooler_cls = BertPooler
    else:
        hf_config = SimpleNamespace(
            hidden_size=4,
            classifier_bias=True,
            classifier_pooling="cls",
            norm_eps=1e-5,
            norm_bias=True,
        )
        pooler_cls = ModernBertPooler

    model_config = Mock(spec=ModelConfig)
    model_config.pooler_config = PoolerConfig(seq_pooling_type="CLS")
    model_config.hf_config = hf_config
    model_config.head_dtype = torch.float32
    with torch.random.fork_rng():
        torch.manual_seed(0)
        pooler = pooler_cls(model_config)
    hidden_states = torch.tensor([[0.25, -0.5, 1.0, -1.5]])

    with_activation = pooler.head(hidden_states, _pooling_metadata(True))
    without_activation = pooler.head(hidden_states, _pooling_metadata(False))
    mixed_activation = pooler.head(
        hidden_states.repeat(2, 1), _pooling_metadata(True, False)
    )
    if isinstance(mixed_activation, list):
        mixed_activation = torch.stack(mixed_activation)

    if isinstance(pooler, BertPooler):
        expected = pooler.act_fn(pooler.dense(hidden_states))
        expected_mixed = pooler.act_fn(pooler.dense(hidden_states.repeat(2, 1)))
    else:
        expected = pooler.norm(pooler.act(pooler.dense(hidden_states)))
        expected_mixed = pooler.norm(
            pooler.act(pooler.dense(hidden_states.repeat(2, 1)))
        )

    assert torch.equal(without_activation, expected)
    assert torch.equal(without_activation, with_activation)
    assert torch.equal(mixed_activation, expected_mixed)


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(
            "jason9693/Qwen2.5-1.5B-apeach",
            marks=[
                pytest.mark.core_model,
                pytest.mark.cpu_model,
                pytest.mark.slow_test,
            ],
        ),
    ],
)
@pytest.mark.parametrize("dtype", ["half"] if current_platform.is_rocm() else ["float"])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:
    with vllm_runner(model, max_model_len=512, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.classify(example_prompts)

    with hf_runner(
        model, dtype=dtype, auto_cls=AutoModelForSequenceClassification
    ) as hf_model:
        hf_outputs = hf_model.classify(example_prompts)

    # check logits difference
    for hf_output, vllm_output in zip(hf_outputs, vllm_outputs):
        hf_output = torch.tensor(hf_output)
        vllm_output = torch.tensor(vllm_output)

        # the tolerance value of 1e-2 is selected based on the
        # half datatype tests in
        # tests/models/language/pooling/test_embedding.py
        assert torch.allclose(
            hf_output,
            vllm_output,
            rtol=2e-3 if dtype == "float" else 1e-2,
        )


@pytest.mark.core_model
def test_bert_model_runner_v2(hf_runner, vllm_runner, monkeypatch) -> None:
    model = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    score_inputs = (
        "What is the capital of France?",
        [
            "Paris.",
            "Paris is the capital and largest city of France.",
            "William Shakespeare wrote Hamlet in the early seventeenth century.",
        ],
    )
    prompt_batches = [
        ["short input"],
        [
            "short input",
            "a longer input that exercises mixed sequence lengths",
        ],
    ]

    with hf_runner(
        model, dtype="half", auto_cls=AutoModelForSequenceClassification
    ) as hf_model:
        # HfRunner uses problem_type to preserve the model's
        # sbert_ce_default_activation_function=Identity raw logits.
        hf_model.config.problem_type = "regression"
        hf_outputs = [hf_model.classify(prompts) for prompts in prompt_batches]

    text_1, text_2 = score_inputs
    text_pairs = [[text_1, document] for document in text_2]
    with hf_runner(model, dtype="half", is_cross_encoder=True) as hf_model:
        hf_scores = hf_model.predict(text_pairs).tolist()

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    with vllm_runner(
        model,
        runner="pooling",
        dtype="half",
        max_model_len=64,
    ) as vllm_model:
        assert vllm_model.llm.llm_engine.vllm_config.use_v2_model_runner
        vllm_outputs = [vllm_model.classify(prompts) for prompts in prompt_batches]
        vllm_scores = vllm_model.score(*score_inputs)

    for hf_batch, vllm_batch in zip(hf_outputs, vllm_outputs):
        hf_tensor = torch.tensor(hf_batch)
        vllm_tensor = torch.tensor(vllm_batch)
        assert vllm_tensor.shape == hf_tensor.shape
        assert torch.allclose(vllm_tensor, hf_tensor, rtol=1e-2, atol=1e-4)

    assert torch.allclose(
        torch.tensor(vllm_scores),
        torch.tensor(hf_scores),
        rtol=1e-2,
        atol=1e-4,
    )
