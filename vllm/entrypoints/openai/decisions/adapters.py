# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Translate Decisions questions and answers to and from label reads."""

import json

from .protocol import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionAnswer,
    DecisionInputMessage,
    DecisionQuestion,
    PredicateAnswer,
    PredicateQuestion,
    ScoreAnswer,
)
from .question_types import (
    LABELS,
    Option,
    Question,
    StructuredDecisionError,
    argmax,
)
from .question_types import (
    ChoiceQuestion as LabelQuestion,
)


def make_read_question(
    index: int, question: DecisionQuestion, max_options: int = len(LABELS)
) -> Question:
    if isinstance(question, PredicateQuestion):
        options = [Option("false"), Option("true")]
    elif isinstance(question, ChoiceQuestion):
        options = [
            Option(json.dumps(c.value, ensure_ascii=False), c.description)
            for c in question.choices
        ]
    else:
        options = [Option(level.label, level.description) for level in question.levels]
    limit = min(max_options, len(LABELS))
    if len(options) > limit:
        raise StructuredDecisionError(
            f"This model supports at most {limit} choices per question"
        )
    winnow_options: tuple[str, ...]
    if isinstance(question, PredicateQuestion):
        winnow_options = ("false", "true")
    elif isinstance(question, ChoiceQuestion):
        winnow_options = tuple(
            (
                c.value
                if isinstance(c.value, str)
                else json.dumps(c.value, ensure_ascii=False)
            )
            + (": " + c.description if c.description else "")
            for c in question.choices
        )
    else:
        winnow_options = tuple(level.label for level in question.levels)
    return Question(
        id=str(index),
        type=LabelQuestion(),
        instructions=question.instructions,
        options=tuple(options),
        labels=LABELS[: len(options)],
        winnow_options=winnow_options,
    )


def make_answer(
    question: DecisionQuestion, probs: list[float], label_mass: float
) -> DecisionAnswer:
    if isinstance(question, PredicateQuestion):
        return PredicateAnswer(name=question.name, probability=probs[1])
    top = argmax(probs)
    confidence = probs[top] * label_mass
    if isinstance(question, ChoiceQuestion):
        return ChoiceAnswer(
            name=question.name,
            choice=question.choices[top].value,
            probabilities=[
                {"value": choice.value, "probability": probability}
                for choice, probability in zip(question.choices, probs)
            ],
            confidence=confidence,
        )
    return ScoreAnswer(
        name=question.name,
        score=sum(i * probability for i, probability in enumerate(probs)),
        probabilities=[
            {"value": i, "label": level.label, "probability": probability}
            for i, (level, probability) in enumerate(zip(question.levels, probs))
        ],
        confidence=confidence,
    )


def input_text(input: str | list[DecisionInputMessage]) -> str:
    if isinstance(input, str):
        return input
    return "\n\n".join(
        message.content
        if isinstance(message.content, str)
        else "\n".join(part.text for part in message.content)
        for message in input
    )
