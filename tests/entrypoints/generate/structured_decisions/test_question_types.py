# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from typing import Any

import pytest

from vllm.entrypoints.generate.structured_decisions.prompts import (
    answer_prefix,
    label_token_ids,
    system_text,
)
from vllm.entrypoints.generate.structured_decisions.question_types import (
    QUESTION_TYPES,
    Alternative,
    Question,
    QuestionType,
    StructuredDecisionError,
    build_question,
    label_softmax,
    register_question_type,
)
from vllm.tokenizers import get_tokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"


def choice(qid="bucket", criteria=None, instructions="Which team?"):
    return build_question(
        qid,
        "choice",
        instructions,
        criteria or {"billing": "money", "outage": None, "other": None},
    )


def test_choice_labels_and_alternatives():
    q = choice()
    assert q.labels == ("A", "B", "C")
    assert [a.name for a in q.alternatives] == ["billing", "outage", "other"]
    assert q.alternatives[0].description == "money"


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
        ("q", "choice", {"x": None}, "2 to 26"),
        ("q", "choice", {str(i): None for i in range(27)}, "2 to 26"),
    ],
)
def test_build_question_rejects(qid, type_name, criteria, match):
    with pytest.raises(StructuredDecisionError, match=match):
        build_question(qid, type_name, "", criteria)


def test_registered_type_plugs_in():
    class BinaryQuestion(QuestionType):
        name = "test_binary"

        def parse_alternatives(self, qid: str, criteria: Any) -> list[Alternative]:
            return [Alternative("yes"), Alternative("no")]

        def labels(self, alternatives: list[Alternative]) -> list[str]:
            return ["yes", "no"]

        def answer(self, question: Question, probs: list[float]) -> dict[str, Any]:
            return {"type": self.name, "yes": probs[0]}

    register_question_type(BinaryQuestion)
    try:
        q = build_question("ok", "test_binary", "Is it fine?", None)
        assert q.labels == ("yes", "no")
        assert q.type.answer(q, [0.9, 0.1]) == {"type": "test_binary", "yes": 0.9}
        with pytest.raises(ValueError, match="already registered"):
            register_question_type(BinaryQuestion)
    finally:
        del QUESTION_TYPES["test_binary"]


def test_label_softmax_normalizes_over_labels():
    probs = label_softmax([math.log(0.3), math.log(0.1)])
    assert probs == pytest.approx([0.75, 0.25])


def test_system_text_lists_every_alternative():
    q = choice()
    text = system_text("Support inbox.", [q])
    assert "Support inbox." in text
    assert "Question bucket: Which team?" in text
    assert "  A: billing (money)\n" in text
    assert "  B: outage\n" in text
    assert '"id: label"' in text


def test_label_token_ids_are_single_distinct_tokens():
    tokenizer = get_tokenizer(MODEL_NAME)
    q = choice(criteria={chr(ord("a") + i): None for i in range(26)})
    ids = label_token_ids(tokenizer, q)
    assert len(ids) == 26 and len(set(ids)) == 26
    prefix = tokenizer.encode(answer_prefix(q), add_special_tokens=False)
    for label, token in zip(q.labels, ids):
        full = tokenizer.encode(f"{answer_prefix(q)} {label}", add_special_tokens=False)
        assert full == prefix + [token]


def test_label_token_ids_rejects_multi_token_labels():
    class WordQuestion(QuestionType):
        name = "test_words"

        def parse_alternatives(self, qid: str, criteria: Any) -> list[Alternative]:
            return [Alternative(n) for n in criteria]

        def labels(self, alternatives: list[Alternative]) -> list[str]:
            return [a.name for a in alternatives]

        def answer(self, question: Question, probs: list[float]) -> dict[str, Any]:
            return {}

    register_question_type(WordQuestion)
    try:
        q = build_question(
            "q", "test_words", "", ["antidisestablishmentarianism", "no"]
        )
        with pytest.raises(StructuredDecisionError, match="is not one token"):
            label_token_ids(get_tokenizer(MODEL_NAME), q)
    finally:
        del QUESTION_TYPES["test_words"]
