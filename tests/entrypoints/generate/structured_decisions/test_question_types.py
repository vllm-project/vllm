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


def noul(qid="urgent", criteria=None, instructions="Reply within the hour?"):
    return build_question(qid, "noul", instructions, criteria, 128)


def score(qid="tone", criteria=None, instructions="How angry?", max_options=128):
    return build_question(
        qid,
        "score",
        instructions,
        criteria or ["calm", "annoyed", "furious"],
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
        ("q", "noul", ["yes", "no"], "must be an object"),
        ("q", "noul", {"maybe": "same day"}, "must be an object"),
        ("q", "score", {"0": "calm", "1": "furious"}, "ordered list of levels"),
        ("q", "score", ["calm"], "ordered list of levels"),
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


def test_noul_labels_and_prompt():
    q = noul()
    assert q.labels == ("yes", "no")
    assert [a.name for a in q.options] == ["yes", "no"]
    assert q.type.prompt(q) == (
        "Question: Reply within the hour?\nAnswer with yes or no only."
    )
    described = noul(criteria={"true": "same day", "false": "later"})
    assert described.type.prompt(described) == (
        "Question: Reply within the hour?\n"
        "yes: same day\n"
        "no: later\n"
        "Answer with yes or no only."
    )


def test_noul_answer_shape():
    q = noul()
    answer = q.type.answer(q, [0.8, 0.2], 0.5)
    assert answer == {
        "type": "noul",
        "noul": 0.8,
        "probabilities": {"yes": 0.8, "no": 0.2},
        "confidence": 0.4,
    }


def test_score_labels_are_levels():
    assert score().labels == ("0", "1", "2")
    with pytest.raises(StructuredDecisionError, match="at most 10"):
        score(criteria=[str(i) for i in range(11)])


def test_score_prompt():
    q = score()
    assert q.type.prompt(q) == (
        "Question: How angry?\n"
        "0: calm\n"
        "1: annoyed\n"
        "2: furious\n"
        "Answer with the number of one level only."
    )


def test_score_answer_shape():
    q = score()
    answer = q.type.answer(q, [0.5, 0.25, 0.25], 0.4)
    assert answer == {
        "type": "score",
        "score": 0.75,
        "legend": {"0": "calm", "1": "annoyed", "2": "furious"},
        "probabilities": {"0": 0.5, "1": 0.25, "2": 0.25},
        "confidence": 0.2,
    }


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
