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


#: A choice labels its options A to Z in order. A single capital letter is one
#: token at the start of a reply for the tokenizers tested. Every read checks it.
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
    #: The labels of a question's options, in option order.
    label_set: ClassVar[tuple[str, ...]] = LABELS
    reply_instruction: ClassVar[str] = "Answer with the letter of one option only."

    @abstractmethod
    def parse_options(self, qid: str, criteria: Any) -> list[Option]: ...

    @abstractmethod
    def answer(
        self, question: Question, probs: list[float], label_mass: float
    ) -> dict[str, Any]:
        """The answer for ``question``. ``probs[i]`` is the probability of
        ``question.labels[i]`` among the labels, and the list sums to 1.
        ``label_mass`` is the labels' total probability over the vocabulary."""

    def reply_line(self, label: str, option: Option) -> str:
        return (
            f"{label}: {option.name} - {option.description}"
            if option.description
            else f"{label}: {option.name}"
        )

    def prompt(self, question: Question) -> str:
        """The question as the model reads it, after the state."""
        lines = [f"Question: {question.instructions}"] if question.instructions else []
        for label, o in zip(question.labels, question.options):
            lines.append(self.reply_line(label, o))
        lines.append(self.reply_instruction)
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
    limit = min(max_options, len(qtype.label_set))
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
        labels=qtype.label_set[: len(options)],
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


@register_question_type
class NoulQuestion(QuestionType):
    """Yes or no. ``criteria`` may describe what true and false mean."""

    name = "noul"
    label_set = ("yes", "no")
    reply_instruction = "Answer with yes or no only."

    def parse_options(self, qid: str, criteria: Any) -> list[Option]:
        if criteria is not None and (
            not isinstance(criteria, dict) or not criteria.keys() <= {"true", "false"}
        ):
            raise StructuredDecisionError(
                f"question {qid!r}: noul criteria must be an object with true and false"
            )
        criteria = criteria or {}
        true = criteria.get("true")
        false = criteria.get("false")
        return [
            Option("yes", None if true is None else str(true)),
            Option("no", None if false is None else str(false)),
        ]

    def reply_line(self, label: str, option: Option) -> str:
        # A side without a description still shows its label. Without the
        # label in the prompt, models answer "Yes" or "No" instead.
        return f"{label}: {option.description}" if option.description else label

    def answer(
        self, question: Question, probs: list[float], label_mass: float
    ) -> dict[str, Any]:
        return {
            "type": self.name,
            "noul": probs[0],
            "probabilities": {"yes": probs[0], "no": probs[1]},
            # The chosen label's probability over the whole vocabulary, like a
            # choice's confidence.
            "confidence": max(probs) * label_mass,
        }


@register_question_type
class ScoreQuestion(QuestionType):
    """Rate the state on an ordered scale. ``criteria`` is the ordered list of
    level names, from the first to the last level of the scale."""

    name = "score"
    # 0-indexed like the answer, so a level's label is its score.
    label_set = tuple(string.digits)
    reply_instruction = "Answer with the number of one level only."

    def parse_options(self, qid: str, criteria: Any) -> list[Option]:
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise StructuredDecisionError(
                f"question {qid!r}: score criteria must be an ordered list of levels"
            )
        return [Option(str(level)) for level in criteria]

    def answer(
        self, question: Question, probs: list[float], label_mass: float
    ) -> dict[str, Any]:
        top = argmax(probs)
        return {
            "type": self.name,
            # The expected level, 0-indexed, so a level's index in the request's
            # criteria is its value.
            "score": sum(i * p for i, p in enumerate(probs)),
            "legend": {str(i): o.name for i, o in enumerate(question.options)},
            "probabilities": {str(i): p for i, p in enumerate(probs)},
            "confidence": probs[top] * label_mass,
        }
