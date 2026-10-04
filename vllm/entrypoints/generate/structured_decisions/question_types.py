# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Question types for structured decisions.

A question type turns a request's criteria into options, writes the question
for the model with a label per option, and builds the answer from the label
probabilities.
"""

import math
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar


class StructuredDecisionError(ValueError):
    """An invalid request. Returned as a 400."""


#: Options are labeled A to Z in order. A single capital letter is one token
#: at the start of a reply for the tokenizers tested; every read checks it.
LABELS = tuple(string.ascii_uppercase)


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
    def answer(
        self, question: Question, probs: list[float], label_mass: float
    ) -> dict[str, Any]:
        """The answer for ``question``. ``probs[i]`` is the probability of
        ``question.labels[i]`` among the labels, and the list sums to 1.
        ``label_mass`` is the labels' total probability over the vocabulary."""

    def prompt(self, question: Question) -> str:
        """The question as the model reads it, after the state."""
        lines = [f"Question: {question.instructions}"] if question.instructions else []
        for label, o in zip(question.labels, question.options):
            lines.append(
                f"{label}: {o.name} - {o.description}"
                if o.description
                else f"{label}: {o.name}"
            )
        lines.append("Answer with the letter of one option only.")
        return "\n".join(lines)


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
    qid: str,
    type_name: str,
    instructions: Any,
    criteria: Any,
    max_options: int,
) -> Question:
    if not qid:
        raise StructuredDecisionError("question ids must be non-empty")
    qtype = get_question_type(type_name)
    options = qtype.parse_options(qid, criteria)
    if not options:
        raise StructuredDecisionError(f"question {qid!r}: needs at least one option")
    names = [o.name for o in options]
    if len(set(names)) != len(names):
        raise StructuredDecisionError(f"question {qid!r}: duplicate option names")
    limit = min(max_options, len(LABELS))
    if len(options) > limit:
        raise StructuredDecisionError(
            f"question {qid!r}: at most {limit} options for this model"
        )
    if not isinstance(instructions, str):
        instructions = "" if instructions is None else str(instructions)
    return Question(
        id=qid,
        type=qtype,
        instructions=instructions,
        options=tuple(options),
        labels=LABELS[: len(options)],
    )


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def label_softmax(logprobs: list[float]) -> list[float]:
    """Softmax of the label logprobs."""
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
                "a description or null"
            )
        return [
            Option(str(name), None if desc is None else str(desc))
            for name, desc in criteria.items()
        ]

    def answer(
        self, question: Question, probs: list[float], label_mass: float
    ) -> dict[str, Any]:
        top = argmax(probs)
        return {
            "type": self.name,
            "choice": question.options[top].name,
            "probabilities": {a.name: p for a, p in zip(question.options, probs)},
            # The chosen label's probability over the whole vocabulary, so a
            # read that mostly wanted a non-label token reports low confidence.
            "confidence": probs[top] * label_mass,
        }
