# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import weakref

import pytest
import torch

from tests.models.utils import softmax
from vllm import LLM, PoolingParams

MODEL_NAME = "vllm-project/Qwen3-Reranker-0.6B-seq-cls"
PROMPT = "The chef prepared a delicious meal."
TEXTS_1 = [
    "What is the capital of France?",
    "What is the capital of Germany?",
]

TEXTS_2 = [
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
]


@pytest.fixture(scope="module")
def llm(vllm_runner):
    with vllm_runner(
        MODEL_NAME,
        max_model_len=None,
        max_num_batched_tokens=32768,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
        enforce_eager=True,
        seed=0,
        enable_chunked_prefill=None,
    ) as runner:
        # pytest caches yielded fixtures until after teardown, so use a proxy to
        # avoid retaining the LLM while VllmRunner.__exit__ releases ROCm memory.
        yield weakref.proxy(runner.llm)


@pytest.fixture(scope="module")
def hf_model(hf_runner):
    return hf_runner(MODEL_NAME, is_cross_encoder=True)


@pytest.mark.skip_global_cleanup
def test_1_to_1(llm, hf_model):
    text_pair = [TEXTS_1[0], TEXTS_2[0]]

    hf_outputs = hf_model.predict([text_pair]).tolist()
    vllm_outputs = [
        output.outputs.score for output in llm.score(text_pair[0], text_pair[1])
    ]

    assert len(vllm_outputs) == 1
    assert len(hf_outputs) == 1

    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.01)


@pytest.mark.skip_global_cleanup
def test_1_to_n(llm, hf_model):
    text_pairs = [
        [TEXTS_1[0], TEXTS_2[0]],
        [TEXTS_1[0], TEXTS_2[1]],
    ]

    vllm_outputs = [output.outputs.score for output in llm.score(TEXTS_1[0], TEXTS_2)]
    hf_outputs = hf_model.predict(text_pairs).tolist()

    assert len(vllm_outputs) == 2
    assert len(hf_outputs) == 2

    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.01)
    assert hf_outputs[1] == pytest.approx(vllm_outputs[1], rel=0.01)


@pytest.mark.skip_global_cleanup
def test_n_to_n(llm, hf_model):
    text_pairs = [
        [TEXTS_1[0], TEXTS_2[0]],
        [TEXTS_1[1], TEXTS_2[1]],
    ]

    vllm_outputs = [output.outputs.score for output in llm.score(TEXTS_1, TEXTS_2)]
    hf_outputs = hf_model.predict(text_pairs).tolist()

    assert len(vllm_outputs) == 2
    assert len(hf_outputs) == 2

    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.01)
    assert hf_outputs[1] == pytest.approx(vllm_outputs[1], rel=0.01)


@pytest.mark.skip_global_cleanup
def test_classify(llm):
    outputs = llm.encode(PROMPT, pooling_task="classify", use_tqdm=False)
    assert len(outputs) == 1
    assert len(outputs[0].outputs.data) == 1


@pytest.mark.parametrize("content_format", [None, "string", "openai"])
@pytest.mark.skip_global_cleanup
def test_score_content_format_override_matches_classify(llm: LLM, content_format):
    # Legacy character loops keep strings unless structured content is requested.
    query, document = TEXTS_1[0], TEXTS_2[0]
    expected_prompt = json.dumps(
        [
            {"role": "query", "content": [{"type": "text", "text": query}]},
            {"role": "document", "content": [{"type": "text", "text": document}]},
        ]
    )
    chat_template = "{{ messages | tojson }}"
    if content_format != "openai":
        chat_template = (
            "{% for message in messages %}"
            "{% for char in message.content %}{{ char }}{% endfor %}"
            "{% endfor %}"
        )
        expected_prompt = query + document
    # Compare full-prefill executions; a prefix-cache hit changes the compute path.
    llm.reset_prefix_cache()
    format_kwargs = (
        {}
        if content_format is None
        else {"score_template_content_format": content_format}
    )
    (scored,) = llm.score(
        query,
        document,
        chat_template=chat_template,
        **format_kwargs,
        tokenization_kwargs={"add_special_tokens": False},
        use_tqdm=False,
    )
    llm.reset_prefix_cache()
    (classified,) = llm.classify(
        expected_prompt,
        tokenization_kwargs={"add_special_tokens": False},
        use_tqdm=False,
    )
    assert scored.prompt_token_ids == classified.prompt_token_ids
    assert scored.outputs.score == pytest.approx(classified.outputs.probs[0])


@pytest.mark.skip_global_cleanup
def test_score_rejects_automatic_content_format(llm: LLM):
    with pytest.raises(ValueError, match="score_template_content_format"):
        llm.score(
            "query",
            "document",
            score_template_content_format="auto",  # type: ignore[arg-type]
        )


@pytest.mark.skip_global_cleanup
def test_max_tokens_per_doc(llm: LLM):
    """Test max_tokens_per_doc via PoolingParams.extra_kwargs (offline)."""
    long_doc = "The capital of France is Paris. " * 20

    # Without truncation
    outputs_no_limit = llm.score(
        TEXTS_1[0],
        long_doc,
        use_tqdm=False,
    )

    # With truncation via extra_kwargs
    outputs_with_limit = llm.score(
        TEXTS_1[0],
        long_doc,
        pooling_params=PoolingParams(extra_kwargs={"max_tokens_per_doc": 10}),
        use_tqdm=False,
    )

    assert len(outputs_no_limit) == 1
    assert len(outputs_with_limit) == 1

    # Truncated version should have fewer prompt tokens
    no_limit_tokens = len(outputs_no_limit[0].prompt_token_ids)
    with_limit_tokens = len(outputs_with_limit[0].prompt_token_ids)
    assert with_limit_tokens < no_limit_tokens


def test_pooling_params(llm: LLM):
    def get_outputs(use_activation):
        outputs = llm.score(
            TEXTS_1[0],
            TEXTS_2[0],
            pooling_params=PoolingParams(use_activation=use_activation),
            use_tqdm=False,
        )
        return torch.tensor([x.outputs.score for x in outputs])

    default = get_outputs(use_activation=None)
    w_activation = get_outputs(use_activation=True)
    wo_activation = get_outputs(use_activation=False)

    assert torch.allclose(default, w_activation, atol=1e-2), (
        "Default should use activation."
    )
    assert not torch.allclose(w_activation, wo_activation, atol=1e-2), (
        "wo_activation should not use activation."
    )
    assert torch.allclose(softmax(wo_activation), w_activation, atol=1e-2), (
        "w_activation should be close to activation(wo_activation)."
    )
