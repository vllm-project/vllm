# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Question types for structured decisions.

A question type turns a request's criteria into alternatives, names the label
the model answers with for each, and shapes the answer from the label
probabilities. Everything else (prompts, slot resolution, reads) is shared, so
a new type is one registered subclass.
"""

import math
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

MAX_LETTER_LABELS = 26


class StructuredDecisionError(ValueError):
    """A request the decision API cannot serve. Reported as a 400."""


@dataclass(frozen=True)
class Alternative:
    name: str
    description: str | None = None


@dataclass(frozen=True)
class Question:
    id: str
    type: "QuestionType"
    instructions: str
    alternatives: tuple[Alternative, ...]
    labels: tuple[str, ...]


class QuestionType(ABC):
    name: ClassVar[str]

    @abstractmethod
    def parse_alternatives(self, qid: str, criteria: Any) -> list[Alternative]:
        """The alternatives described by the request's ``criteria``."""

    @abstractmethod
    def answer(self, question: Question, probs: list[float]) -> dict[str, Any]:
        """The answer for ``question``. ``probs`` follows ``question.labels``
        and sums to 1."""

    def labels(self, alternatives: list[Alternative]) -> list[str]:
        """The label the model answers with for each alternative. The default
        is capital letters: one token after a space in every tokenizer tried,
        with no meaning of their own to bias the read."""
        if len(alternatives) > MAX_LETTER_LABELS:
            raise StructuredDecisionError(
                f"{self.name}: at most {MAX_LETTER_LABELS} alternatives with "
                "letter labels"
            )
        return list(string.ascii_uppercase[: len(alternatives)])


QUESTION_TYPES: dict[str, QuestionType] = {}


def register_question_type(cls: type[QuestionType]) -> type[QuestionType]:
    if cls.name in QUESTION_TYPES:
        raise ValueError(f"question type {cls.name!r} is already registered")
    QUESTION_TYPES[cls.name] = cls()
    return cls


def get_question_type(name: str) -> QuestionType:
    try:
        return QUESTION_TYPES[name]
    except KeyError:
        raise StructuredDecisionError(
            f"unknown question type {name!r}; supported: {sorted(QUESTION_TYPES)}"
        ) from None


def build_question(
    qid: str, type_name: str, instructions: Any, criteria: Any
) -> Question:
    if not qid or ":" in qid or "\n" in qid:
        raise StructuredDecisionError(
            f"question id {qid!r} must be non-empty, without ':' or a newline"
        )
    qtype = get_question_type(type_name)
    alternatives = qtype.parse_alternatives(qid, criteria)
    if len(alternatives) < 2:
        raise StructuredDecisionError(
            f"question {qid!r}: needs at least 2 alternatives, got {len(alternatives)}"
        )
    names = [a.name for a in alternatives]
    if len(set(names)) != len(names):
        raise StructuredDecisionError(f"question {qid!r}: duplicate alternative names")
    try:
        labels = qtype.labels(alternatives)
    except StructuredDecisionError as e:
        raise StructuredDecisionError(f"question {qid!r}: {e}") from None
    if not isinstance(instructions, str):
        instructions = "" if instructions is None else str(instructions)
    return Question(
        id=qid,
        type=qtype,
        instructions=instructions,
        alternatives=tuple(alternatives),
        labels=tuple(labels),
    )


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def label_softmax(logprobs: list[float]) -> list[float]:
    """Probabilities over the labels alone from their full-vocabulary logprobs."""
    top = max(logprobs)
    weights = [math.exp(lp - top) for lp in logprobs]
    total = sum(weights)
    return [w / total for w in weights]


@register_question_type
class ChoiceQuestion(QuestionType):
    """Pick one option. ``criteria`` maps each option name to a description,
    or to null."""

    name = "choice"

    def parse_alternatives(self, qid: str, criteria: Any) -> list[Alternative]:
        if not isinstance(criteria, dict) or not criteria:
            raise StructuredDecisionError(
                f"question {qid!r}: choice criteria must map option names to "
                "descriptions"
            )
        return [
            Alternative(str(name), None if desc is None else str(desc))
            for name, desc in criteria.items()
        ]

    def answer(self, question: Question, probs: list[float]) -> dict[str, Any]:
        top = argmax(probs)
        return {
            "type": self.name,
            "choice": question.alternatives[top].name,
            "probabilities": {a.name: p for a, p in zip(question.alternatives, probs)},
            "confidence": probs[top],
        }
