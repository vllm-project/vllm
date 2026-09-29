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
    label_token_ids,
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
    text = DecisionTemplate(DEFAULT_DECISION_TEMPLATE).system_text(
        "Support inbox.", questions()
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


def test_default_template_without_instructions():
    text = DecisionTemplate(DEFAULT_DECISION_TEMPLATE).system_text(None, questions())
    assert "\n\nQuestion bucket: Which team?\n" in text
    assert "None" not in text


def test_default_answer_prefix():
    q = questions()[0]
    assert DecisionTemplate(DEFAULT_DECISION_TEMPLATE).answer_prefix(q) == "bucket:"
    assert DecisionTemplate("Just the questions.").answer_prefix(q) == "bucket:"


def test_custom_template_and_prefix():
    template = DecisionTemplate(
        "{% macro answer_prefix(question) %}[{{ question.id }}] ->{% endmacro %}"
        "{% for q in questions %}{{ q.id }}={{ q.type }}"
        "{% for o in q.options %} {{ o.label }}/{{ o.name }}{% endfor %};{% endfor %}"
    )
    assert template.system_text(None, questions()) == (
        "bucket=choice A/billing B/outage;lang=choice A/en B/fr;"
    )
    assert template.answer_prefix(questions()[1]) == "[lang] ->"


def test_bad_template_is_a_request_error():
    with pytest.raises(StructuredDecisionError, match="decision template"):
        DecisionTemplate("{% for q in questions %}")


def test_select_template_trust_gate():
    assert select_template(None, None, False).answer_prefix(questions()[0]) == "bucket:"
    with pytest.raises(StructuredDecisionError, match="trust-request-chat-template"):
        select_template(None, "custom", False)
    assert select_template(None, "custom", True).system_text(None, []) == "custom"
    assert select_template("server", None, False).system_text(None, []) == "server"


def test_label_token_ids_are_single_distinct_tokens():
    tokenizer = get_tokenizer(MODEL_NAME)
    q = build_question("q", "choice", "", {chr(ord("a") + i): None for i in range(26)})
    ids = label_token_ids(tokenizer, q, "q:")
    assert len(ids) == 26 and len(set(ids)) == 26
    prefix = tokenizer.encode("q:", add_special_tokens=False)
    for label, token in zip(q.labels, ids):
        assert tokenizer.encode(f"q: {label}", add_special_tokens=False) == (
            prefix + [token]
        )


def test_label_token_ids_rejects_multi_token_labels():
    q = build_question("q", "choice", "", {"a": None, "b": None})
    q = dataclasses.replace(q, labels=("antidisestablishmentarianism", "B"))
    with pytest.raises(StructuredDecisionError, match="is not one token"):
        label_token_ids(get_tokenizer(MODEL_NAME), q, "q:")
