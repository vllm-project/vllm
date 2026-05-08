# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref

import pytest
import torch
from transformers import AutoModelForSequenceClassification

from vllm import LLM, PoolingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.platforms import current_platform

# DeBERTa-v3 NLI model: 3 labels (contradiction, neutral, entailment).
# num_labels=3 means score() is blocked; this file tests the classify() API.
MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
NUM_LABELS = 3  # contradiction, neutral, entailment

TEXTS_1 = [
    "What is the capital of France?",
    "What is the capital of Germany?",
]

TEXTS_2 = [
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
]


@pytest.fixture(scope="module")
def llm():
    # ROCm: Use FLEX_ATTENTION backend as it's the only attention backend
    # that supports encoder-only models on ROCm.
    attention_config = None
    if current_platform.is_rocm():
        attention_config = {"backend": "FLEX_ATTENTION"}

    # pytest caches the fixture so we use weakref.proxy to
    # enable garbage collection
    llm = LLM(
        model=MODEL_NAME,
        max_num_batched_tokens=32768,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
        enforce_eager=True,
        seed=0,
        attention_config=attention_config,
    )

    yield weakref.proxy(llm)

    del llm

    cleanup_dist_env_and_memory()


@pytest.fixture(scope="module")
def hf_model(hf_runner):
    return hf_runner(
        MODEL_NAME,
        auto_cls=AutoModelForSequenceClassification,
    )


@pytest.mark.skip_global_cleanup
def test_1_to_1(llm, hf_model):
    prompts = [f"{TEXTS_1[0]} [SEP] {TEXTS_2[0]}"]

    vllm_outputs = llm.classify(prompts, use_tqdm=False)
    hf_outputs = hf_model.classify(prompts)

    vllm_logits = torch.tensor(vllm_outputs[0].outputs.probs)
    hf_logits = torch.tensor(hf_outputs[0])

    assert len(vllm_logits) == NUM_LABELS
    assert torch.allclose(hf_logits, vllm_logits, atol=1e-2)


@pytest.mark.skip_global_cleanup
def test_1_to_n(llm, hf_model):
    prompts = [
        f"{TEXTS_1[0]} [SEP] {TEXTS_2[0]}",
        f"{TEXTS_1[0]} [SEP] {TEXTS_2[1]}",
    ]

    vllm_outputs = llm.classify(prompts, use_tqdm=False)
    hf_outputs = hf_model.classify(prompts)

    assert len(vllm_outputs) == 2
    for vllm_out, hf_out in zip(vllm_outputs, hf_outputs):
        vllm_logits = torch.tensor(vllm_out.outputs.probs)
        hf_logits = torch.tensor(hf_out)
        assert torch.allclose(hf_logits, vllm_logits, atol=1e-2)


@pytest.mark.skip_global_cleanup
def test_n_to_n(llm, hf_model):
    prompts = [
        f"{TEXTS_1[0]} [SEP] {TEXTS_2[0]}",
        f"{TEXTS_1[1]} [SEP] {TEXTS_2[1]}",
    ]

    vllm_outputs = llm.classify(prompts, use_tqdm=False)
    hf_outputs = hf_model.classify(prompts)

    assert len(vllm_outputs) == 2
    for vllm_out, hf_out in zip(vllm_outputs, hf_outputs):
        vllm_logits = torch.tensor(vllm_out.outputs.probs)
        hf_logits = torch.tensor(hf_out)
        assert torch.allclose(hf_logits, vllm_logits, atol=1e-2)


@pytest.mark.skip_global_cleanup
def test_classify(llm):
    # classify returns raw logits over all NUM_LABELS labels
    outputs = llm.encode(TEXTS_1[0], pooling_task="classify", use_tqdm=False)
    assert len(outputs) == 1
    assert len(outputs[0].outputs.data) == NUM_LABELS


@pytest.mark.skip_global_cleanup
def test_pooling_params(llm: LLM):
    """Verify that use_activation controls softmax application on logits."""
    prompt = f"{TEXTS_1[0]} [SEP] {TEXTS_2[0]}"

    def get_outputs(use_activation):
        outputs = llm.classify(
            [prompt],
            pooling_params=PoolingParams(use_activation=use_activation),
            use_tqdm=False,
        )
        return torch.tensor(outputs[0].outputs.probs)

    default = get_outputs(use_activation=None)
    w_activation = get_outputs(use_activation=True)
    wo_activation = get_outputs(use_activation=False)

    # Default and w_activation should be identical (softmax applied)
    assert torch.allclose(default, w_activation, atol=1e-4), (
        "Default should apply activation (softmax)."
    )
    # softmax(raw_logits) == activated scores
    assert torch.allclose(
        torch.softmax(wo_activation, dim=0),
        w_activation,
        atol=1e-3,
    ), "softmax(wo_activation) should match w_activation."
