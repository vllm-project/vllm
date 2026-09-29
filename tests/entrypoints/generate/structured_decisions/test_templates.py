# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses

import pytest

from vllm.entrypoints.generate.structured_decisions.question_types import (
    StructuredDecisionError,
    build_question,
)
from vllm.entrypoints.generate.structured_decisions.templates import (
    DEFAULT_DECISION_TEMPLATE,
    DecisionTemplate,
    select_template,
)
from vllm.tokenizers import get_tokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"


def questions():
    return [
        build_question(
            "bucket", "choice", "Which team?", {"billing": "money", "outage": None}
        ),
        build_question("lang", "choice", " Which language? ", {"en": None, "fr": None}),
    ]


def test_default_template_text():
    text = (
        DecisionTemplate(DEFAULT_DECISION_TEMPLATE)
        .render("Support inbox.", questions())
        .system_text
    )
    assert text == (
        "Answer a fixed set of questions about the state the user provides. Each "
        "question lists its allowed answers; reply with exactly one label per "
        "question.\n"
        "\nSupport inbox.\n"
        "\nQuestion bucket: Which team?\n"
        "  A: billing (money)\n"
        "  B: outage\n"
        "\nQuestion lang: Which language?\n"
        "  A: en\n"
        "  B: fr\n"
        '\nReply with one line per question, in this order, formatted as "id: label".'
    )


def test_custom_template_and_answer():
    template = DecisionTemplate(
        "{% macro answer(question, label) %}[{{ question.id }}] ({{ label }})"
        "{% endmacro %}"
        "{% for q in questions %}{{ q.id }}={{ q.type }}"
        "{% for o in q.options %} {{ o.label }}/{{ o.name }}{% endfor %};{% endfor %}"
        'Write "{{ answer(questions[0], "A") }}".'
    )
    rendered = template.render(None, questions())
    assert rendered.system_text == (
        'bucket=choice A/billing B/outage;lang=choice A/en B/fr;Write "[bucket] (A)".'
    )
    assert rendered.answer(questions()[1], "B") == "[lang] (B)"


def test_bad_template_is_a_request_error():
    with pytest.raises(StructuredDecisionError, match="decision template"):
        DecisionTemplate("{% for q in questions %}")


def test_select_template_trust_gate():
    qs = questions()
    assert select_template(None, None, False).render(None, qs).answer(qs[0], "A") == (
        "bucket: A"
    )
    with pytest.raises(StructuredDecisionError, match="trust-request-chat-template"):
        select_template(None, "custom", False)
    assert select_template(None, "custom", True).render(None, []).system_text == (
        "custom"
    )
    assert select_template("server", None, False).render(None, []).system_text == (
        "server"
    )


def test_slot_on_default_answer():
    tokenizer = get_tokenizer(MODEL_NAME)
    q = build_question("q", "choice", "", {chr(ord("a") + i): None for i in range(26)})
    slot = (
        DecisionTemplate(DEFAULT_DECISION_TEMPLATE).render(None, [q]).slot(tokenizer, q)
    )
    assert len(slot.label_ids) == 26 and len(set(slot.label_ids)) == 26
    for label, token in zip(q.labels, slot.label_ids):
        assert tokenizer.encode(f"q: {label}", add_special_tokens=False) == (
            slot.prefix_ids + [token]
        )


def test_slot_with_text_after_the_label():
    tokenizer = get_tokenizer(MODEL_NAME)
    template = DecisionTemplate(
        "{% macro answer(question, label) %}{{ question.id }} ({{ label }})"
        "{% endmacro %}"
    )
    q = build_question("team", "choice", "", {"a": None, "b": None, "c": None})
    slot = template.render(None, [q]).slot(tokenizer, q)
    for label, token in zip(q.labels, slot.label_ids):
        full = tokenizer.encode(f"team ({label})", add_special_tokens=False)
        assert full[: len(slot.prefix_ids)] == slot.prefix_ids
        assert full[len(slot.prefix_ids)] == token


def test_slot_rejects_multi_token_labels():
    q = build_question("q", "choice", "", {"a": None, "b": None})
    q = dataclasses.replace(q, labels=("antidisestablishmentarianism", "B"))
    with pytest.raises(StructuredDecisionError, match="not all one token"):
        DecisionTemplate(DEFAULT_DECISION_TEMPLATE).render(None, [q]).slot(
            get_tokenizer(MODEL_NAME), q
        )


def test_slot_needs_text_before_the_label():
    template = DecisionTemplate(
        "{% macro answer(question, label) %}{{ label }}{% endmacro %}"
    )
    q = build_question("q", "choice", "", {"a": None, "b": None})
    with pytest.raises(StructuredDecisionError, match="text before the label"):
        template.render(None, [q]).slot(get_tokenizer(MODEL_NAME), q)
