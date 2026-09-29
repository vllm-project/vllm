# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decision templates: Jinja that renders a decision's system prompt and says
how an answer is written.

The template receives ``instructions`` (a string or None) and ``questions``,
each with ``id``, ``type``, ``instructions`` and ``options`` (``label``,
``name``, ``description``).

It may define a macro ``answer(question, label)`` that returns one question's
answer as the model should write it, ``"<id>: <label>"`` by default. The
server renders the answer once per label and compares the tokens to find
where the label goes. The system prompt can call the same macro to show the
model the exact reply format.
"""

import string
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import jinja2
import jinja2.ext
import jinja2.sandbox

from vllm.tokenizers import TokenizerLike

from .question_types import (
    Option,
    Question,
    StructuredDecisionError,
    get_question_type,
)

LABEL_CANDIDATES = tuple(string.ascii_uppercase) + tuple(
    a + b for a in string.ascii_uppercase for b in string.ascii_uppercase
)

DEFAULT_DECISION_TEMPLATE = """\
{%- macro answer(question, label) -%}{{ question.id }}: {{ label }}{%- endmacro -%}
Answer a fixed set of questions about the state the user provides. Each \
question lists its allowed answers; reply with exactly one label per question. \
Labels are chosen randomly.
{% if instructions %}

{{ instructions | trim }}
{% endif %}
{% for question in questions %}

Question {{ question.id }}: {{ question.instructions | trim }}
{% for option in question.options %}
  {{ option.label }}: {{ option.name }}\
{% if option.description %} ({{ option.description | trim }}){% endif %}

{% endfor %}
{% endfor %}

Reply with one line per question, in this order, formatted as \
"{{ answer({"id": "id"}, "label") }}".\
"""


def question_vars(question: Question) -> dict[str, Any]:
    return {
        "id": question.id,
        "type": question.type.name,
        "instructions": question.instructions,
        "options": [
            {"label": label, "name": o.name, "description": o.description}
            for label, o in zip(question.labels, question.options)
        ],
    }


@contextmanager
def template_errors() -> Iterator[None]:
    """Report a template that fails to compile or render as a 400."""
    try:
        yield
    except jinja2.TemplateError as e:
        raise StructuredDecisionError(f"decision template: {e}") from e


@dataclass(frozen=True)
class AnswerSlot:
    """Where a question's label goes in its rendered answer."""

    prefix_ids: list[int]
    label_ids: list[int]  # in the order of question.labels


class DecisionTemplate:
    def __init__(self, source: str):
        env = jinja2.sandbox.ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=[jinja2.ext.loopcontrols],
        )
        with template_errors():
            self._template = env.from_string(source)
        self._alphabets: dict[int, tuple[str, ...]] = {}

    def label_alphabet(self, tokenizer: TokenizerLike) -> tuple[str, ...]:
        """The labels this template can use with ``tokenizer``.

        Each candidate must sit inside one token of its answer. Candidates are
        grouped by the tokens around the label and by what the label token
        holds besides the label, such as a fused space or colon. The largest
        group is kept, so every label is tokenized the same way."""
        key = id(tokenizer)
        if key not in self._alphabets:
            self._alphabets[key] = self._sweep_labels(tokenizer)
        return self._alphabets[key]

    def _sweep_labels(self, tokenizer: TokenizerLike) -> tuple[str, ...]:
        probe = Question(
            id="q",
            type=get_question_type("choice"),
            instructions="",
            options=(Option("a"), Option("b")),
            labels=("A", "B"),
        )
        rendered = self.render(None, [probe])
        groups: dict[tuple, dict[int, str]] = {}
        for label in LABEL_CANDIDATES:
            text = rendered.answer(probe, label)
            if text.count(label) != 1:
                continue
            start = text.index(label)
            end = start + len(label)
            ids = tokenizer.encode(text, add_special_tokens=False)
            # The token bounds below come from prefix decodes. The bounds only
            # line up when the text round-trips.
            if tokenizer.decode(ids) != text:
                continue
            bounds = [0] + [
                len(tokenizer.decode(ids[: i + 1])) for i in range(len(ids))
            ]
            hits = [
                i
                for i in range(len(ids))
                if bounds[i] <= start and end <= bounds[i + 1]
            ]
            if not hits:
                continue
            (i,) = hits
            fused = text[bounds[i] : start] + "{}" + text[end : bounds[i + 1]]
            group = groups.setdefault((tuple(ids[:i]), tuple(ids[i + 1 :]), fused), {})
            group.setdefault(ids[i], label)
        if not groups:
            return ()
        return tuple(max(groups.values(), key=len).values())

    def render(
        self, instructions: str | None, questions: list[Question]
    ) -> "RenderedDecision":
        with template_errors():
            # The module's text is the rendered template, and its macros see
            # the same variables as the template body.
            module = self._template.make_module(
                vars={
                    "instructions": instructions,
                    "questions": [question_vars(q) for q in questions],
                }
            )
        return RenderedDecision(
            system_text=str(module), answer_macro=getattr(module, "answer", None)
        )


@dataclass(frozen=True)
class RenderedDecision:
    system_text: str
    answer_macro: Any

    def answer(self, question: Question, label: str) -> str:
        if self.answer_macro is None:
            return f"{question.id}: {label}"
        with template_errors():
            return str(self.answer_macro(question_vars(question), label))

    def slot(self, tokenizer: TokenizerLike, question: Question) -> AnswerSlot:
        variants = [
            tokenizer.encode(self.answer(question, label), add_special_tokens=False)
            for label in question.labels
        ]
        pos, label_ids = label_position(question, variants)
        if pos == 0:
            raise StructuredDecisionError(
                f"question {question.id!r}: the answer must have text before the "
                "label, such as the question id"
            )
        return AnswerSlot(prefix_ids=variants[0][:pos], label_ids=label_ids)


def label_position(
    question: Question, variants: list[list[int]]
) -> tuple[int, list[int]]:
    """``variants`` holds one tokenization per label of the same text. Returns
    the one position where they differ and each label's token there. Every
    label must be one token, the rest of the text must not change with the
    label, and no two labels may share a token."""
    if any(len(ids) != len(variants[0]) for ids in variants):
        raise StructuredDecisionError(
            f"question {question.id!r}: its labels are not all one token in the answer"
        )
    diffs = {
        i
        for ids in variants[1:]
        for i, (a, b) in enumerate(zip(variants[0], ids))
        if a != b
    }
    if len(diffs) != 1:
        raise StructuredDecisionError(
            f"question {question.id!r}: the labels change {len(diffs)} tokens of "
            "the answer, and must change exactly one"
        )
    (pos,) = diffs
    label_ids = [ids[pos] for ids in variants]
    if len(set(label_ids)) != len(label_ids):
        raise StructuredDecisionError(
            f"question {question.id!r}: two labels share a token"
        )
    return pos, label_ids


@lru_cache(maxsize=16)
def compile_template(source: str) -> DecisionTemplate:
    return DecisionTemplate(source)


def select_template(
    server_source: str | None,
    request_source: str | None,
    trust_request_template: bool,
) -> DecisionTemplate:
    """The template for one request: its own when the server trusts request
    templates, otherwise the server's, otherwise the default."""
    if request_source is not None:
        if not trust_request_template:
            raise StructuredDecisionError(
                "decision_template in a request needs the server to run with "
                "--trust-request-chat-template"
            )
        return compile_template(request_source)
    return compile_template(server_source or DEFAULT_DECISION_TEMPLATE)
