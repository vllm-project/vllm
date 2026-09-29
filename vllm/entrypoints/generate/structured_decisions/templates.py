# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decision templates: Jinja that renders the system prompt of a decision.

A template works like a chat template. It is rendered in the same sandboxed
environment, loaded from a file or given inline with --decision-template, and
a request may carry its own when the server trusts request templates.

The template receives ``instructions`` (a string or None) and ``questions``,
each with ``id``, ``type``, ``instructions`` and ``options`` (``label``,
``name``, ``description``). It may define a macro ``answer_prefix(question)``:
the reply text that comes right before a question's label. A read prefills the
reply up to that text, so a template that asks for a different reply format
defines the matching macro. Without the macro the prefix is ``"<id>:"``.
"""

import json
from functools import lru_cache
from typing import Any

import jinja2
import jinja2.ext
import jinja2.sandbox

from vllm.tokenizers import TokenizerLike

from .question_types import Question, StructuredDecisionError

DEFAULT_DECISION_TEMPLATE = """\
{%- macro answer_prefix(question) -%}{{ question.id }}:{%- endmacro -%}
Answer a fixed set of questions about the state the user provides. Each \
question lists its allowed answers; reply with exactly one label per question.
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

Reply with one line per question, in this order, formatted as "id: label".\
"""


def question_vars(question: Question) -> dict[str, Any]:
    return {
        "id": question.id,
        "type": question.type.name,
        "instructions": question.instructions,
        "options": [
            {"label": label, "name": a.name, "description": a.description}
            for label, a in zip(question.labels, question.alternatives)
        ],
    }


class DecisionTemplate:
    def __init__(self, source: str):
        env = jinja2.sandbox.ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=[jinja2.ext.loopcontrols],
        )
        try:
            self._template = env.from_string(source)
            self._answer_prefix = getattr(self._template.module, "answer_prefix", None)
        except jinja2.TemplateError as e:
            raise StructuredDecisionError(f"decision template: {e}") from e

    def system_text(self, instructions: str | None, questions: list[Question]) -> str:
        try:
            return self._template.render(
                instructions=instructions,
                questions=[question_vars(q) for q in questions],
            )
        except jinja2.TemplateError as e:
            raise StructuredDecisionError(f"decision template: {e}") from e

    def answer_prefix(self, question: Question) -> str:
        if self._answer_prefix is None:
            return f"{question.id}:"
        try:
            return str(self._answer_prefix(question_vars(question)))
        except jinja2.TemplateError as e:
            raise StructuredDecisionError(f"decision template: {e}") from e


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


def state_text(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state)


def label_token_ids(
    tokenizer: TokenizerLike, question: Question, prefix_text: str
) -> list[int]:
    """The token each label adds after ``prefix_text`` and a space. Every label
    must be exactly one token there, and no two labels may share one."""
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
