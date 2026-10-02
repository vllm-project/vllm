# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.entrypoints.generate.structured_decisions.question_types import (
    Option,
    Question,
    StructuredDecisionError,
    get_question_type,
)
from vllm.entrypoints.generate.structured_decisions.templates import (
    DEFAULT_DECISION_TEMPLATE,
    LABEL_CANDIDATES,
    DecisionTemplate,
)
from vllm.tokenizers import get_tokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"
LETTERS = LABEL_CANDIDATES[:26]


def choice(qid, instructions, options, labels=None):
    return Question(
        id=qid,
        type=get_question_type("choice"),
        instructions=instructions,
        options=tuple(Option(name, desc) for name, desc in options.items()),
        labels=labels or LETTERS[: len(options)],
    )


def questions():
    return [
        choice("bucket", "Which team?", {"billing": "money", "outage": None}),
        choice("lang", " Which language? ", {"en": None, "fr": None}),
    ]


def test_default_template_text():
    text = (
        DecisionTemplate(DEFAULT_DECISION_TEMPLATE)
        .render("Support inbox.", questions())
        .text
    )
    assert text == (
        "Answer a fixed set of questions about the state the user provides. Each "
        "question lists its allowed answers; reply with exactly one label per "
        "question. Labels are chosen randomly.\n"
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
    assert rendered.text == (
        'bucket=choice A/billing B/outage;lang=choice A/en B/fr;Write "[bucket] (A)".'
    )
    assert rendered.answer(questions()[1], "B") == "[lang] (B)"


def test_bad_template_is_a_request_error():
    with pytest.raises(StructuredDecisionError, match="decision template"):
        DecisionTemplate("{% for q in questions %}")


def test_slot_on_default_answer():
    tokenizer = get_tokenizer(MODEL_NAME)
    q = choice("q", "", {chr(ord("a") + i): None for i in range(26)})
    slot = (
        DecisionTemplate(DEFAULT_DECISION_TEMPLATE).render(None, [q]).slot(tokenizer, q)
    )
    assert len(slot.label_ids) == 26 and len(set(slot.label_ids)) == 26
    for label, token in zip(q.labels, slot.label_ids):
        assert tokenizer.encode(f"q: {label}", add_special_tokens=False) == (
            slot.prefix_ids + [token]
        )


def test_slot_for_one_option():
    tokenizer = get_tokenizer(MODEL_NAME)
    q = choice("q", "", {"only": None}, ("K",))
    slot = (
        DecisionTemplate(DEFAULT_DECISION_TEMPLATE).render(None, [q]).slot(tokenizer, q)
    )
    assert tokenizer.encode("q: K", add_special_tokens=False) == (
        slot.prefix_ids + slot.label_ids
    )


def test_slot_with_text_after_the_label():
    tokenizer = get_tokenizer(MODEL_NAME)
    template = DecisionTemplate(
        "{% macro answer(question, label) %}{{ question.id }} ({{ label }})"
        "{% endmacro %}"
    )
    q = choice("team", "", {"a": None, "b": None, "c": None})
    slot = template.render(None, [q]).slot(tokenizer, q)
    for label, token in zip(q.labels, slot.label_ids):
        full = tokenizer.encode(f"team ({label})", add_special_tokens=False)
        assert full[: len(slot.prefix_ids)] == slot.prefix_ids
        assert full[len(slot.prefix_ids)] == token


def test_slot_rejects_multi_token_labels():
    q = choice("q", "", {"a": None, "b": None}, ("antidisestablishmentarianism", "B"))
    with pytest.raises(StructuredDecisionError, match="not all one token"):
        DecisionTemplate(DEFAULT_DECISION_TEMPLATE).render(None, [q]).slot(
            get_tokenizer(MODEL_NAME), q
        )


def test_slot_needs_text_before_the_label():
    template = DecisionTemplate(
        "{% macro answer(question, label) %}{{ label }}{% endmacro %}"
    )
    q = choice("q", "", {"a": None, "b": None})
    with pytest.raises(StructuredDecisionError, match="text before the label"):
        template.render(None, [q]).slot(get_tokenizer(MODEL_NAME), q)


def test_label_alphabet_on_default_template():
    tokenizer = get_tokenizer(MODEL_NAME)
    alphabet = DecisionTemplate(DEFAULT_DECISION_TEMPLATE).label_alphabet(tokenizer)
    assert set(LETTERS) < set(alphabet)
    for label in alphabet:
        ids = tokenizer.encode(f"q: {label}", add_special_tokens=False)
        assert tokenizer.decode(ids[-1:]) == f" {label}"


def test_label_alphabet_keeps_one_fusion_pattern():
    # Qwen3 fuses the colon into some labels written right after it.
    tokenizer = get_tokenizer(MODEL_NAME)
    template = DecisionTemplate(
        "{% macro answer(question, label) %}{{ question.id }}:{{ label }}{% endmacro %}"
    )
    alphabet = template.label_alphabet(tokenizer)
    assert 2 <= len(alphabet) < len(LABEL_CANDIDATES)
    patterns = set()
    for label in alphabet:
        ids = tokenizer.encode(f"q:{label}", add_special_tokens=False)
        patterns.add(tokenizer.decode(ids[-1:]).replace(label, "{}"))
    assert len(patterns) == 1
