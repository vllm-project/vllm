# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
from pydantic import ValidationError

from vllm.entrypoints.openai.decisions.adapters import (
    input_state,
    input_text,
    make_answer,
    make_read_question,
)
from vllm.entrypoints.openai.decisions.protocol import DecisionRequest
from vllm.entrypoints.openai.decisions.question_types import (
    StructuredDecisionError,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def request(**kwargs):
    return DecisionRequest.model_validate(
        {
            "model": "test-model",
            "input": "The package is broken.",
            "questions": [{"type": "predicate", "instructions": "Is it damaged?"}],
            **kwargs,
        }
    )


def test_predicate_returns_probability_of_true_and_null_name():
    question = request().questions[0]
    read = make_read_question(0, question)
    assert [option.name for option in read.options] == ["false", "true"]
    assert make_answer(question, [0.1, 0.9], 0.5).model_dump() == {
        "type": "predicate",
        "name": None,
        "probability": 0.9,
    }


def test_choice_preserves_boolean_and_string_values():
    question = request(
        questions=[
            {
                "type": "choice",
                "name": "decision",
                "instructions": "Select a value.",
                "choices": [{"value": True}, {"value": "true"}, {"value": False}],
            }
        ]
    ).questions[0]
    read = make_read_question(0, question)
    assert [option.name for option in read.options] == ["true", '"true"', "false"]
    answer = make_answer(question, [0.7, 0.2, 0.1], 0.5).model_dump()
    assert answer["name"] == "decision"
    assert answer["choice"] is True
    assert answer["probabilities"] == [
        {"value": True, "probability": 0.7},
        {"value": "true", "probability": 0.2},
        {"value": False, "probability": 0.1},
    ]
    assert answer["confidence"] == pytest.approx(0.35)


def test_score_is_weighted_average_of_zero_based_level_indices():
    question = request(
        questions=[
            {
                "type": "score",
                "instructions": "Rate severity.",
                "levels": [{"label": label} for label in ("low", "medium", "high")],
            }
        ]
    ).questions[0]
    answer = make_answer(question, [0.1, 0.7, 0.2], 0.8).model_dump()
    assert answer["score"] == pytest.approx(1.1)
    assert answer["probabilities"] == [
        {"value": 0, "label": "low", "probability": 0.1},
        {"value": 1, "label": "medium", "probability": 0.7},
        {"value": 2, "label": "high", "probability": 0.2},
    ]
    assert answer["confidence"] == pytest.approx(0.56)


def test_text_messages_preserve_evidence_order():
    parsed = request(
        input=[
            {"role": "user", "content": "first"},
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "second"},
                    {"type": "input_text", "text": "third"},
                ],
            },
        ]
    )
    assert input_text(parsed.input) == "first\n\nsecond\nthird"
    assert input_text([]) == ""


@pytest.mark.parametrize("evidence", ["", [], [{"role": "user", "content": []}]])
def test_empty_input_is_valid_under_the_openai_schema(evidence):
    assert input_text(request(input=evidence).input) == ""


def test_score_levels_with_the_same_label_keep_their_positions():
    question = request(
        questions=[
            {
                "type": "score",
                "instructions": "Rate severity.",
                "levels": [
                    {"label": "damage", "description": "cosmetic"},
                    {"label": "damage", "description": "unusable"},
                ],
            }
        ]
    ).questions[0]
    read = make_read_question(0, question)
    assert read.labels == ("A", "B")
    answer = make_answer(question, [0.1, 0.9], 0.8).model_dump()
    assert answer["score"] == pytest.approx(0.9)
    assert [p["value"] for p in answer["probabilities"]] == [0, 1]


@pytest.mark.parametrize(
    "override,match",
    [
        ({"input": [{"role": "assistant", "content": "x"}]}, "user"),
        (
            {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": "file:///etc/passwd",
                            }
                        ],
                    }
                ]
            },
            "pattern",
        ),
        ({"input": {"text": "x"}}, "Input should"),
        ({"questions": []}, "at least 1"),
        (
            {"questions": [{"type": "predicate", "instructions": "x", "name": None}]},
            "name must be a string",
        ),
        ({"questions": [{"type": "predicate", "instructions": 1}]}, "valid string"),
        (
            {"questions": [{"type": "predicate", "instructions": "x", "criteria": {}}]},
            "Extra inputs",
        ),
        ({"safety_identifier": "x" * 129}, "at most 128"),
        ({"stream": True}, "Extra inputs"),
    ],
)
def test_rejects_invalid_or_unsupported_inputs(override, match):
    with pytest.raises(ValidationError, match=match):
        request(**override)


@pytest.mark.parametrize("values", [[True, True], ["a", "a"], [1, "a"], [None, "a"]])
def test_choice_rejects_duplicates_and_non_string_non_boolean_values(values):
    with pytest.raises(ValidationError):
        request(
            questions=[
                {
                    "type": "choice",
                    "instructions": "x",
                    "choices": [{"value": v} for v in values],
                }
            ]
        )


@pytest.mark.parametrize(
    "count,valid", [(1, False), (2, True), (255, True), (256, False)]
)
def test_choice_limits(count, valid):
    data = {
        "questions": [
            {
                "type": "choice",
                "instructions": "x",
                "choices": [{"value": str(i)} for i in range(count)],
            }
        ]
    }
    if valid:
        assert len(request(**data).questions[0].choices) == count
    else:
        with pytest.raises(ValidationError):
            request(**data)


@pytest.mark.parametrize(
    "count,valid", [(1, False), (2, True), (10, True), (11, False)]
)
def test_score_limits(count, valid):
    data = {
        "questions": [
            {
                "type": "score",
                "instructions": "x",
                "levels": [{"label": str(i)} for i in range(count)],
            }
        ]
    }
    if valid:
        request(**data)
    else:
        with pytest.raises(ValidationError):
            request(**data)


def test_question_limit_and_optional_safety_identifier():
    questions = [{"type": "predicate", "instructions": "x"}] * 200
    assert len(request(questions=questions, safety_identifier=None).questions) == 200
    with pytest.raises(ValidationError, match="at most 200"):
        request(questions=[*questions, questions[0]])


def test_backend_rejects_more_choices_than_single_token_labels():
    question = request(
        questions=[
            {
                "type": "choice",
                "instructions": "x",
                "choices": [{"value": str(i)} for i in range(27)],
            }
        ]
    ).questions[0]
    with pytest.raises(StructuredDecisionError, match="at most 26 choices"):
        make_read_question(0, question)


def test_image_content_preserves_order_without_changing_text_only_input():
    parts = [
        {"type": "input_text", "text": "Before"},
        {"type": "input_image", "image_url": "https://example.com/one.png"},
        {"type": "input_text", "text": "Between"},
        {
            "type": "input_image",
            "image_url": "data:image/png;base64,AA==",
            "detail": "low",
        },
    ]
    parsed = request(input=[{"role": "user", "content": parts}])
    assert input_state(parsed.input) == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Before"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/one.png",
                        "detail": "auto",
                    },
                },
                {"type": "text", "text": "Between"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AA==", "detail": "low"},
                },
            ],
        }
    ]
    assert input_state(request(input="unchanged").input) == "unchanged"
