# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small CPU protocol tests; no trained checkpoint or serving process."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
import torch

from vllm.entrypoints.openai.decisions.adapters import make_answer, make_read_question
from vllm.entrypoints.openai.decisions.protocol import (
    ChoiceQuestion,
    DecisionRequest,
    PredicateQuestion,
    ScoreQuestion,
)
from vllm.entrypoints.openai.decisions.serving import OpenAIServingDecisions
from vllm.entrypoints.openai.decisions.strategies import (
    D3Strategy,
    StartLuxStrategy,
    StructuredDecisionError,
    select_read_strategy,
)


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(
            tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return json.dumps(messages) + StartLuxStrategy.THINK_OFF_SUFFIX

    def encode(self, text, **kwargs):
        return list(map(ord, text))


def question(kind, names, descriptions=None, instructions="Choose."):
    return SimpleNamespace(
        id="q",
        type=SimpleNamespace(name=kind),
        instructions=instructions,
        options=tuple(
            SimpleNamespace(name=name, description=description)
            for name, description in zip(names, descriptions or [None] * len(names))
        ),
    )


def strategy(temperatures=None):
    config = SimpleNamespace(
        model="/local/checkpoint", revision=None, max_model_len=8192
    )
    context = SimpleNamespace(
        engine_client=SimpleNamespace(model_config=config),
        online_renderer=SimpleNamespace(
            renderer=SimpleNamespace(get_tokenizer=Tokenizer)
        ),
    )
    decision = {
        "letter_token_ids": list(range(ord("A"), ord("Z") + 1)),
        "temperature_by_type": temperatures or {"choice": 2, "noul": 1, "score": 3},
    }
    with patch(
        "vllm.entrypoints.openai.decisions.strategies.get_hf_file_to_dict",
        return_value=decision,
    ):
        return StartLuxStrategy(context)


@pytest.mark.parametrize(
    "architecture",
    [
        "StartLuxDecisionForSequenceClassification",
        "StartLuxDecisionMoeForSequenceClassification",
    ],
)
def test_native_strategy_selection(architecture):
    config = SimpleNamespace(architecture=architecture)
    assert select_read_strategy(config, ["classify"]) is StartLuxStrategy
    with pytest.raises(ValueError, match="classify"):
        select_read_strategy(config, ["generate"])


@pytest.mark.parametrize(
    "kind,names,descriptions,lines",
    [
        ("choice", ["A", "B"], ["Paris", "London"], ["A) Paris", "B) London"]),
        (
            "choice",
            ["billing", "tech"],
            [None, "repair"],
            ["A) billing", "B) tech: repair"],
        ),
        ("noul", ["yes", "no"], [None, None], ["A) yes", "B) no"]),
        ("score", ["low", "high"], [None, None], ["A) 0: low", "B) 1: high"]),
    ],
)
def test_trained_prompt_and_order(kind, names, descriptions, lines):
    actual = strategy()._prompt("state", question(kind, names, descriptions), [])
    messages = json.loads(actual.removesuffix(StartLuxStrategy.THINK_OFF_SUFFIX))
    assert messages[0]["content"] == (
        "Apply the criterion to the evidence. Choose exactly one listed option. "
        "Answer with its letter only."
    )
    assert messages[1]["content"] == (
        "Evidence:\nstate\n\nQuestion: Choose.\nOptions:\n" + "\n".join(lines)
    )


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_invalid_temperatures_rejected(temperature):
    with pytest.raises(ValueError, match="finite and positive"):
        strategy({"choice": temperature, "noul": 1, "score": 1})


def test_candidate_bounds_and_media_rejected():
    reader = strategy()
    for n in (0, 1, 27):
        with pytest.raises(StructuredDecisionError):
            reader._options(question("choice", [str(i) for i in range(n)]))
    assert (
        len(reader._options(question("choice", [str(i) for i in range(26)]))[0]) == 26
    )
    with pytest.raises(StructuredDecisionError):
        reader._options(question("score", [str(i) for i in range(11)]))
    with pytest.raises(StructuredDecisionError, match="text-only"):
        reader._prompt("state", question("choice", ["x", "y"]), ["image"])


def test_native_read_restricts_candidates_then_calibrates():
    reader = strategy()

    async def encode(prompt, params, request_id, **kwargs):
        assert params.task == "classify" and params.use_activation is False
        # Excluded labels dominate the full vocabulary: normalization must only
        # include the current question's two candidates.
        yield SimpleNamespace(
            error=None,
            outputs=SimpleNamespace(data=torch.tensor([0.0, 2.0] + [100.0] * 24)),
        )

    reader.context.engine_client.encode = encode
    questions = [question("choice", ["x", "y"]), question("noul", ["yes", "no"])]
    result = asyncio.run(
        reader.read(
            questions,
            None,
            "state",
            request_id="cpu",
            chat_template_kwargs=None,
            lora_request=None,
            priority=0,
        )
    )
    for read, temperature in zip(result, [2.0, 1.0]):
        expected = torch.softmax(
            torch.tensor([0.0, 2.0], dtype=torch.float64) / temperature, 0
        )
        assert read.probs == pytest.approx(expected.tolist())
        assert read.output_tokens == 0
    d3 = D3Strategy.__new__(D3Strategy)
    d3.temperature = 1.75
    assert d3._temperature(questions[0]) == 1.75


def test_api_conversion_preserves_types_and_probability_order():
    reader = strategy()
    predicate = PredicateQuestion(type="predicate", instructions="Is it safe?")
    q = reader.make_question(0, predicate)
    assert q.type.name == "noul"
    texts, order = reader._options(q)
    assert texts == ["yes", "no"] and order == [1, 0]
    # The model reads yes/no, while the public API expects false/true.
    probs = [0.8, 0.2]
    result = make_answer(predicate, [probs[i] for i in order], 1.0)
    assert result.probability == pytest.approx(0.8)
    assert reader._temperature(q) == 1.0

    choice = ChoiceQuestion(
        type="choice",
        instructions="Choose.",
        choices=[
            {"value": "A", "description": "Paris"},
            {"value": "B", "description": "London"},
        ],
    )
    q = reader.make_question(1, choice)
    assert reader._options(q)[0] == ["Paris", "London"]
    assert reader._temperature(q) == 2.0
    score = ScoreQuestion(
        type="score",
        instructions="Impact?",
        levels=[{"label": "low"}, {"label": "high", "description": "urgent"}],
    )
    q = reader.make_question(2, score)
    assert q.type.name == "score"
    assert reader._options(q)[0] == ["0: low", "1: high: urgent"]
    assert reader._temperature(q) == 3.0

    # Unmodified strategies retain their previous generic conversion.
    d3 = D3Strategy.__new__(D3Strategy)
    d3.codes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    actual = d3.make_question(0, predicate)
    expected = make_read_question(0, predicate, 26)
    assert actual.type.name == expected.type.name == "choice"
    assert (actual.id, actual.instructions, actual.options, actual.labels) == (
        expected.id,
        expected.instructions,
        expected.options,
        expected.labels,
    )


def test_complete_prompt_and_tokenizer_metadata():
    reader = strategy()
    reader.max_model_len = 1
    with pytest.raises(StructuredDecisionError, match="nothing is truncated"):
        asyncio.run(
            reader.read(
                [question("choice", ["x", "y"])],
                None,
                "state",
                request_id="cpu",
                chat_template_kwargs=None,
                lora_request=None,
                priority=0,
            )
        )
    with (
        patch.object(Tokenizer, "encode", return_value=[0]),
        pytest.raises(ValueError, match="reply tokens|letter IDs"),
    ):
        strategy()


def test_native_handler_preserves_question_types_and_answers():
    reader = strategy()

    async def encode(prompt, params, request_id, **kwargs):
        yield SimpleNamespace(
            error=None,
            outputs=SimpleNamespace(data=torch.tensor([0.0, 2.0] + [100.0] * 24)),
        )

    reader.context.engine_client.encode = encode
    reader.context.engine_client.errored = False
    reader.context.engine_client.check_admission = Mock()
    handler = OpenAIServingDecisions.__new__(OpenAIServingDecisions)
    handler.strategy = reader
    handler.limits = reader.limits()
    handler.models = SimpleNamespace(model_name=lambda _: "test")
    handler._check_model = AsyncMock(return_value=None)
    handler._base_request_id = lambda _: "cpu"
    handler._maybe_get_adapters = lambda _: None
    handler._log_inputs = Mock()
    request = DecisionRequest(
        model="test",
        input="state",
        questions=[
            {
                "type": "choice",
                "instructions": "Pick.",
                "choices": [{"value": "Paris"}, {"value": "London"}],
            },
            {"type": "predicate", "instructions": "Safe?"},
            {
                "type": "score",
                "instructions": "Rate.",
                "levels": [{"label": "low"}, {"label": "high"}],
            },
        ],
    )
    response = asyncio.run(handler.create_decisions(request))
    assert response.answers[0].choice == "London"
    assert response.answers[0].confidence == pytest.approx(
        torch.softmax(torch.tensor([0.0, 1.0]), 0)[1].item()
    )
    # Candidate A is yes but the API's true probability is at position 1.
    assert response.answers[1].probability == pytest.approx(
        torch.softmax(torch.tensor([0.0, 2.0]), 0)[0].item()
    )
    assert response.answers[2].score == pytest.approx(
        torch.softmax(torch.tensor([0.0, 2.0 / 3.0]), 0)[1].item()
    )
    assert response.usage.output_tokens == 0
    reader.context.engine_client.check_admission.assert_called_once_with(3)


@pytest.mark.parametrize(
    "kind,task,expected",
    [
        ("startlux", "classify", True),
        ("default", "generate", True),
        ("default", "classify", False),
        ("default", "token_classify", False),
    ],
)
def test_native_state_initializes_only_supported_openai_strategies(
    kind, task, expected
):
    """Handler-level success is insufficient if startup never registers it."""
    from vllm.entrypoints.openai.decisions.state import init_decisions_state

    reader = strategy() if kind == "startlux" else object.__new__(D3Strategy)
    reader.task = task
    state = SimpleNamespace(online_renderer=object(), openai_serving_models=object())
    args = SimpleNamespace(chat_template_content_format="auto")
    engine = SimpleNamespace(model_config=object())
    with (
        patch(
            "vllm.entrypoints.openai.decisions.state.select_read_strategy",
            return_value=lambda context: reader,
        ),
        patch(
            "vllm.entrypoints.openai.decisions.state.OpenAIServingDecisions"
        ) as handler,
    ):
        returned = init_decisions_state(engine, state, args, None, None, {}, (task,))
    assert returned is reader
    assert handler.called is expected
    assert (state.openai_serving_decisions is not None) is expected
