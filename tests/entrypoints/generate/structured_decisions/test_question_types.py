# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import string
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

LETTERS = tuple(string.ascii_uppercase)


def choice(qid="bucket", criteria=None, instructions="Which team?"):
    return build_question(
        qid,
        "choice",
        instructions,
        criteria or {"billing": "money", "outage": None, "other": None},
        LETTERS,
        128,
    )


def test_choice_labels_and_options():
    q = choice()
    assert len(set(q.labels)) == 3 and set(q.labels) <= set(LETTERS)
    assert choice().labels == q.labels
    assert choice(instructions="Which queue?").labels != q.labels
    wide = build_question(
        "q",
        "choice",
        "",
        {str(i): None for i in range(30)},
        LETTERS + ("AA", "AB", "AC", "AD"),
        128,
    )
    assert set(wide.labels[:26]) == set(LETTERS)
    assert [a.name for a in q.options] == ["billing", "outage", "other"]
    assert q.options[0].description == "money"


def test_choice_answer_shape():
    q = choice()
    answer = q.type.answer(q, [0.2, 0.7, 0.1])
    assert answer == {
        "type": "choice",
        "choice": "outage",
        "probabilities": {"billing": 0.2, "outage": 0.7, "other": 0.1},
        "confidence": 0.7,
    }


@pytest.mark.parametrize(
    "qid,type_name,criteria,match",
    [
        ("a:b", "choice", {"x": None, "y": None}, "without ':'"),
        ("", "choice", {"x": None, "y": None}, "non-empty"),
        ("q", "nope", {"x": None, "y": None}, "unknown question type"),
        ("q", "choice", ["x", "y"], "must map option names"),
        ("q", "choice", {"x": None}, "at least 2"),
        ("q", "choice", {str(i): None for i in range(27)}, "at most 26"),
    ],
)
def test_build_question_rejects(qid, type_name, criteria, match):
    with pytest.raises(StructuredDecisionError, match=match):
        build_question(qid, type_name, "", criteria, LETTERS, 128)


def test_option_limit_is_the_smaller_cap():
    with pytest.raises(StructuredDecisionError, match="at most 2 options"):
        build_question("q", "choice", "", {"x": None, "y": None, "z": None}, LETTERS, 2)


def test_registered_type_plugs_in():
    class BinaryQuestion(QuestionType):
        name = "test_binary"

        def parse_options(self, qid: str, criteria: Any) -> list[Option]:
            return [Option("yes"), Option("no")]

        def labels(self, options: list[Option], alphabet: list[str]) -> list[str]:
            return ["yes", "no"]

        def answer(self, question: Question, probs: list[float]) -> dict[str, Any]:
            return {"type": self.name, "yes": probs[0]}

    register_question_type(BinaryQuestion)
    try:
        q = build_question("ok", "test_binary", "Is it fine?", None, LETTERS, 128)
        assert q.labels == ("yes", "no")
        assert q.type.answer(q, [0.9, 0.1]) == {"type": "test_binary", "yes": 0.9}
        with pytest.raises(ValueError, match="already registered"):
            register_question_type(BinaryQuestion)
    finally:
        del QUESTION_TYPES["test_binary"]
