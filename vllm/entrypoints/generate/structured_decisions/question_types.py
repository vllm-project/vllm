# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Question types for structured decisions.

A question type turns a request's criteria into options, names the label the
model answers with for each, and shapes the answer from the label
probabilities.
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
class Option:
    name: str
    description: str | None = None


@dataclass(frozen=True)
class Question:
    id: str
    type: "QuestionType"
    instructions: str
    options: tuple[Option, ...]
    labels: tuple[str, ...]


class QuestionType(ABC):
    name: ClassVar[str]

    @abstractmethod
    def parse_options(self, qid: str, criteria: Any) -> list[Option]: ...

    @abstractmethod
    def answer(self, question: Question, probs: list[float]) -> dict[str, Any]:
        """The answer for ``question``. ``probs`` follows ``question.labels``
        and sums to 1."""

    def labels(self, options: list[Option]) -> list[str]:
        """The label the model answers with for each option. The default is
        capital letters, which carry no meaning of their own to bias the read."""
        if len(options) > MAX_LETTER_LABELS:
            raise StructuredDecisionError(
                f"{self.name}: at most {MAX_LETTER_LABELS} options with letter labels"
            )
        return list(string.ascii_uppercase[: len(options)])


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
    options = qtype.parse_options(qid, criteria)
    if len(options) < 2:
        raise StructuredDecisionError(
            f"question {qid!r}: needs at least 2 options, got {len(options)}"
        )
    names = [o.name for o in options]
    if len(set(names)) != len(names):
        raise StructuredDecisionError(f"question {qid!r}: duplicate option names")
    try:
        labels = qtype.labels(options)
    except StructuredDecisionError as e:
        raise StructuredDecisionError(f"question {qid!r}: {e}") from None
    if not isinstance(instructions, str):
        instructions = "" if instructions is None else str(instructions)
    return Question(
        id=qid,
        type=qtype,
        instructions=instructions,
        options=tuple(options),
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

    def parse_options(self, qid: str, criteria: Any) -> list[Option]:
        if not isinstance(criteria, dict) or not criteria:
            raise StructuredDecisionError(
                f"question {qid!r}: choice criteria must map option names to "
                "descriptions"
            )
        return [
            Option(str(name), None if desc is None else str(desc))
            for name, desc in criteria.items()
        ]

    def answer(self, question: Question, probs: list[float]) -> dict[str, Any]:
        top = argmax(probs)
        return {
            "type": self.name,
            "choice": question.options[top].name,
            "probabilities": {a.name: p for a, p in zip(question.options, probs)},
            "confidence": probs[top],
        }
