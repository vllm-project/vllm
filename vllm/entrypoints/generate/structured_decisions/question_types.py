# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Question types for structured decisions.

A question type turns a request's criteria into options, picks the label the
model answers with for each option, and builds the answer from the label
probabilities.
"""

import hashlib
import json
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar


class StructuredDecisionError(ValueError):
    """An invalid request. Returned as a 400."""


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
        """The answer for ``question``. ``probs[i]`` is the probability of
        ``question.labels[i]``, and the list sums to 1."""

    def labels(self, options: list[Option], alphabet: list[str]) -> list[str]:
        """The label the model answers with for each option. ``alphabet`` holds
        at least one label per option, shuffled for this question."""
        return alphabet[: len(options)]


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
    alphabet: tuple[str, ...],
    max_options: int,
    seed: int | None = None,
) -> Question:
    """Labels come from ``alphabet``, shuffled with a seed from ``seed`` and
    the question's content, so a repeated question gets the same labels and
    prompt. Single letters are used first, and two-letter labels only for
    options past the 26th."""
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
    limit = min(len(alphabet), max_options)
    if len(options) > limit:
        raise StructuredDecisionError(
            f"question {qid!r}: at most {limit} options for this model"
        )
    if not isinstance(instructions, str):
        instructions = "" if instructions is None else str(instructions)
    content = [
        seed,
        qid,
        type_name,
        instructions,
        [(o.name, o.description) for o in options],
    ]
    digest = hashlib.sha256(json.dumps(content).encode()).digest()
    rng = random.Random(digest)
    shuffled = []
    for size in sorted({len(label) for label in alphabet}):
        tier = [label for label in alphabet if len(label) == size]
        rng.shuffle(tier)
        shuffled += tier
    labels = qtype.labels(options, shuffled)
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

    def answer(self, question: Question, probs: list[float]) -> dict[str, Any]:
        top = argmax(probs)
        return {
            "type": self.name,
            "choice": question.options[top].name,
            "probabilities": {a.name: p for a, p in zip(question.options, probs)},
            "confidence": probs[top],
        }
