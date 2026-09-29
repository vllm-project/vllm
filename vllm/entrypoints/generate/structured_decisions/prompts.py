# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The prompt text and answer slots every question type shares.

The model is asked to reply with one "id: label" line per question. A read
prefills the reply up to a question's "id:" and scores the label tokens that
come next.
"""

import json
from typing import Any

from vllm.tokenizers import TokenizerLike

from .question_types import Question, StructuredDecisionError


def system_text(instructions: str | None, questions: list[Question]) -> str:
    text = (
        "Answer a fixed set of questions about the state the user provides. "
        "Each question lists its allowed answers; reply with exactly one label "
        "per question.\n"
    )
    if instructions:
        text += "\n" + instructions.strip() + "\n"
    for q in questions:
        text += f"\nQuestion {q.id}: {q.instructions.strip()}\n"
        for label, alternative in zip(q.labels, q.alternatives):
            text += "  " + q.type.describe(label, alternative) + "\n"
    text += (
        '\nReply with one line per question, in this order, formatted as "id: label".'
    )
    return text


def state_text(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state)


def answer_prefix(question: Question) -> str:
    # No trailing space: tokenizers attach the space to the label that follows,
    # and chat templates may strip trailing whitespace from a message.
    return f"{question.id}:"


def label_token_ids(tokenizer: TokenizerLike, question: Question) -> list[int]:
    """The token each label adds after the answer prefix. Every label must be
    exactly one token there, and no two labels may share one."""
    prefix_text = answer_prefix(question)
    prefix = tokenizer.encode(prefix_text, add_special_tokens=False)
    ids = []
    for label in question.labels:
        full = tokenizer.encode(f"{prefix_text} {label}", add_special_tokens=False)
        if len(full) != len(prefix) + 1 or full[: len(prefix)] != prefix:
            raise StructuredDecisionError(
                f"question {question.id!r}: label {label!r} is not one token "
                f"after {prefix_text!r} for this tokenizer"
            )
        ids.append(full[-1])
    if len(set(ids)) != len(ids):
        raise StructuredDecisionError(
            f"question {question.id!r}: two labels share a token"
        )
    return ids
