# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import pytest

from vllm.entrypoints.generate.structured_decisions.question_types import (
    QUESTION_TYPES,
    Option,
    Question,
    QuestionType,
    StructuredDecisionError,
    build_question,
    register_question_type,
)


def choice(qid="bucket", criteria=None, instructions="Which team?", max_options=128):
    return build_question(
        qid,
        "choice",
        instructions,
        criteria or {"billing": "money", "outage": None, "other": None},
        max_options,
    )


def test_choice_labels_follow_option_order():
    q = choice()
    assert q.labels == ("A", "B", "C")
    assert [a.name for a in q.options] == ["billing", "outage", "other"]
    assert q.options[0].description == "money"
    wide = choice(criteria={str(i): None for i in range(26)})
    assert wide.labels[-1] == "Z"


def test_choice_prompt():
    q = choice()
    assert q.type.prompt(q) == (
        "Question: Which team?\n"
        "A: billing - money\n"
        "B: outage\n"
        "C: other\n"
        "Answer with the letter of one option only."
    )


def test_choice_answer_shape():
    q = choice()
    answer = q.type.answer(q, [0.2, 0.7, 0.1], 0.5)
    assert answer == {
        "type": "choice",
        "choice": "outage",
        "probabilities": {"billing": 0.2, "outage": 0.7, "other": 0.1},
        "confidence": 0.35,
    }


@pytest.mark.parametrize(
    "qid,type_name,criteria,match",
    [
        ("", "choice", {"x": None, "y": None}, "non-empty"),
        ("q", "nope", {"x": None, "y": None}, "unknown question type"),
        ("q", "choice", ["x", "y"], "must map option names"),
    ],
)
def test_build_question_rejects(qid, type_name, criteria, match):
    with pytest.raises(StructuredDecisionError, match=match):
        build_question(qid, type_name, "", criteria, 128)


def test_one_option_choice():
    q = choice(criteria={"only": None})
    assert q.labels == ("A",)
    assert q.type.answer(q, [1.0], 0.4)["confidence"] == 0.4


def test_option_limit():
    with pytest.raises(StructuredDecisionError, match="at most 2 options"):
        choice(criteria={"x": None, "y": None, "z": None}, max_options=2)
    with pytest.raises(StructuredDecisionError, match="at most 26 options"):
        choice(criteria={str(i): None for i in range(27)})


def test_registered_type_plugs_in():
    class BinaryQuestion(QuestionType):
        name = "test_binary"

        def parse_options(self, qid: str, criteria: Any) -> list[Option]:
            return [Option("yes"), Option("no")]

        def answer(
            self, question: Question, probs: list[float], label_mass: float
        ) -> dict[str, Any]:
            return {"type": self.name, "yes": probs[0]}

    register_question_type(BinaryQuestion)
    try:
        q = build_question("ok", "test_binary", "Is it fine?", None, 128)
        assert q.labels == ("A", "B")
        assert q.type.answer(q, [0.9, 0.1], 1.0) == {"type": "test_binary", "yes": 0.9}
        with pytest.raises(ValueError, match="already registered"):
            register_question_type(BinaryQuestion)
    finally:
        del QUESTION_TYPES["test_binary"]
