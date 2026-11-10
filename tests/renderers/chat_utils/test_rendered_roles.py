# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from unittest.mock import patch

import jinja2.sandbox
import pytest
import regex as re

from vllm.entrypoints.serve.exception_handling.error_response import (
    create_error_response,
)
from vllm.exceptions import VLLMValidationError
from vllm.renderers.chat_utils import (
    assert_chat_template_rendered_messages,
    message_has_nonempty_payload,
    roles_rendered_by_chat_template,
)
from vllm.renderers.chat_utils.rendered_roles import _STANDARD_CHAT_ROLES
from vllm.transformers_utils.chat_templates import get_chat_template_fallback_path

# Qwen-style dispatch: unknown roles fall through and emit nothing.
_FILTERING_TEMPLATE = """
{%- if messages[0]['role'] == 'system' -%}
system:{{ messages[0]['content'] }}
{%- endif -%}
{%- for message in messages -%}
{%- if message.role == "user" or message.role == "assistant"
    or message.role == "tool" -%}
{{ message.role }}:{{ message.content }}
{%- elif message.role == "system" -%}
system:{{ message.content }}
{%- endif -%}
{%- endfor -%}
"""

_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] + '\\n' + message['content'] }}"
    "{% endfor %}"
    "{% if messages[-1]['role'] != 'assistant' %}assistant{% endif %}"
)

_CUSTOM_ROLE_TEMPLATE = """
{% for message in messages %}
{% if message['role'] == 'user' %}
user:{{ message['content'] }}
{% else %}
{{ message['role'] }}:{{ message['content'] }}
{% endif %}
{% endfor %}
"""


def _filtering_render(messages):
    """Render only system/user/assistant/tool, matching ``_FILTERING_TEMPLATE``."""
    handled = {"system", "user", "assistant", "tool"}
    parts = []
    for message in messages:
        role = message.get("role")
        if role not in handled:
            continue
        content = message.get("content")
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, str):
                    texts.append(part)
                elif isinstance(part, dict) and isinstance(part.get("text"), str):
                    texts.append(part["text"])
                elif isinstance(part, dict) and part.get("type") == "image":
                    texts.append("<image>")
            content = "".join(texts)
        parts.append(f"{role}:{content}")
        if message.get("tool_calls"):
            parts.append(f"{role}:tool_calls")
    return "\n".join(parts)


def test_role_filter_extracts_qwen_style_roles():
    assert roles_rendered_by_chat_template(_FILTERING_TEMPLATE) == frozenset(
        {"system", "user", "assistant", "tool"}
    )


def test_chatml_inequality_guard_is_not_an_allow_list():
    """ChatML's ``!= assistant`` guard must not skip the drop check.

    The template still prints every role. The guard is a possible filter
    with no literal allow-list, not a reason to trust ``assistant`` only.
    """
    assert roles_rendered_by_chat_template(_CHATML_TEMPLATE) == frozenset()


def test_negated_role_comparisons_are_filters_without_an_allow_list():
    """``not in``, ``!=``, and ``not (role == ...)`` do not name kept roles."""
    not_in = "{% if message.role not in ['user', 'assistant'] %}skip{% endif %}"
    unequal = (
        "{% if message.role != 'user' and message.role != 'assistant' %}skip{% endif %}"
    )
    negated_eq = (
        "{% if not (message.role == 'user' or message.role == 'assistant') %}"
        "skip{% endif %}"
    )
    for template in (not_in, unequal, negated_eq):
        assert roles_rendered_by_chat_template(template) == frozenset()


def test_dynamic_role_allow_list_is_treated_as_a_filter():
    template = "{% if message.role in allowed %}{{ message.content }}{% endif %}"
    assert roles_rendered_by_chat_template(template) == frozenset()


def test_dynamic_role_equality_is_treated_as_a_filter():
    """``message.role == some_var`` is a filter even though the name is hidden."""
    template = "{% if message.role == allowed_role %}{{ message.content }}{% endif %}"
    assert roles_rendered_by_chat_template(template) == frozenset()


def test_selectattr_role_filter_extracts_literals():
    template = (
        "{% for message in messages | selectattr('role', 'equalto', 'user') %}"
        "{{ message.content }}{% endfor %}"
    )
    assert roles_rendered_by_chat_template(template) == frozenset({"user"})
    eq_template = "{{ messages | selectattr('role', 'eq', 'assistant') }}"
    assert roles_rendered_by_chat_template(eq_template) == frozenset({"assistant"})


def test_selectattr_on_a_non_role_attribute_is_not_a_role_filter():
    template = "{{ messages | selectattr('type', 'equalto', 'text') }}"
    assert roles_rendered_by_chat_template(template) is None


def test_rejectattr_role_filter_does_not_allow_the_rejected_role():
    """``rejectattr`` drops the named role, so that name is not an allow-list."""
    template = (
        "{% for message in messages | rejectattr('role', 'equalto', 'tool') %}"
        "{{ message.content }}{% endfor %}"
    )
    assert roles_rendered_by_chat_template(template) == frozenset()


def test_selectattr_non_literal_role_is_a_filter():
    template = (
        "{% for message in messages | selectattr('role', 'equalto', allowed) %}"
        "{{ message.content }}{% endfor %}"
    )
    assert roles_rendered_by_chat_template(template) == frozenset()


@pytest.mark.parametrize(
    "message,expected",
    [
        ({"role": "user", "content": "hello"}, True),
        ({"role": "user", "content": ""}, False),
        ({"role": "user", "content": "   \n"}, False),
        ({"role": "user", "content": None}, False),
        ({"role": "user"}, False),
        ({"role": "tool", "content": "", "tool_call_id": "call_1"}, False),
        (
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            True,
        ),
        (
            {
                "role": "user",
                "content": [{"type": "image", "image_url": "http://example.test/a"}],
            },
            True,
        ),
        (
            {"role": "user", "content": [{"type": "text", "text": "  "}]},
            False,
        ),
    ],
)
def test_message_payload_ignores_empty_text(message, expected):
    """Empty and whitespace-only turns are not content that can be dropped."""
    assert message_has_nonempty_payload(message) is expected


def test_known_roles_are_not_probed():
    calls = 0

    def render_text(messages):
        nonlocal calls
        calls += 1
        return _filtering_render(messages)

    assert_chat_template_rendered_messages(
        [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "tool", "content": "result", "tool_call_id": "call_1"},
        ],
        _FILTERING_TEMPLATE,
        render_text,
    )
    assert calls == 0


def test_chatml_custom_role_is_not_rejected():
    """A template that emits every role must keep custom roles."""

    def render_text(messages):
        return "\n".join(
            f"{message['role']}:{message.get('content')}" for message in messages
        )

    assert_chat_template_rendered_messages(
        [{"role": "wizard", "content": "hello"}],
        _CHATML_TEMPLATE,
        render_text,
    )


@pytest.mark.parametrize("role", ["usr", "assistan", "USER", "wizard", "", " "])
def test_unhandled_role_with_text_is_rejected(role):
    with pytest.raises(VLLMValidationError, match=re.escape(repr(role))) as exc_info:
        assert_chat_template_rendered_messages(
            [{"role": role, "content": "hello"}],
            _FILTERING_TEMPLATE,
            _filtering_render,
        )
    assert exc_info.value.parameter == "messages"
    assert exc_info.value.value == role


def test_middle_turn_typo_names_that_role():
    with pytest.raises(VLLMValidationError, match="'usr'"):
        assert_chat_template_rendered_messages(
            [
                {"role": "system", "content": "You are terse."},
                {"role": "usr", "content": "The launch code is TANGERINE-42."},
                {"role": "user", "content": "What is the launch code?"},
            ],
            _FILTERING_TEMPLATE,
            _filtering_render,
        )


def test_several_dropped_roles_are_all_named():
    with pytest.raises(VLLMValidationError, match="'usr'") as exc_info:
        assert_chat_template_rendered_messages(
            [
                {"role": "usr", "content": "one"},
                {"role": "user", "content": "kept"},
                {"role": "wizard", "content": "two"},
            ],
            _FILTERING_TEMPLATE,
            _filtering_render,
        )
    assert exc_info.value.value == ["usr", "wizard"]
    assert "wizard" in str(exc_info.value)


def test_custom_role_that_template_renders_is_kept():
    def render_text(messages):
        return "\n".join(
            f"{message.get('role')}:{message.get('content')}" for message in messages
        )

    assert_chat_template_rendered_messages(
        [{"role": "wizard", "content": "hello from a custom role"}],
        _CUSTOM_ROLE_TEMPLATE,
        render_text,
    )


def test_role_header_without_content_is_rejected():
    """Emitting the role token and dropping the text is still data loss."""
    template = """
    {% for message in messages %}
    {% if message['role'] == 'user' %}
    {{ message['content'] }}
    {% endif %}
    {% endfor %}
    """

    def render_text(messages):
        return "\n".join(str(message.get("role")) for message in messages)

    with pytest.raises(VLLMValidationError, match="'wizard'"):
        assert_chat_template_rendered_messages(
            [{"role": "wizard", "content": "secret"}],
            template,
            render_text,
        )


def test_else_branch_custom_role_is_probed_and_kept():
    """A custom role printed from ``else`` is probed, and kept when rendered."""
    calls = 0

    def render_text(messages):
        nonlocal calls
        calls += 1
        parts = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role == "user":
                parts.append(f"user:{content}")
            else:
                parts.append(f"{role}:{content}")
        return "\n".join(parts)

    assert_chat_template_rendered_messages(
        [{"role": "wizard", "content": "secret"}],
        _CUSTOM_ROLE_TEMPLATE,
        render_text,
    )
    assert 0 < calls <= 8


@pytest.mark.parametrize("content", ["", "   ", None])
def test_empty_content_unknown_role_is_allowed(content):
    """No text to lose: an unknown role with empty content is not a 400."""

    def render_text(messages):
        raise AssertionError("empty messages must not be probed")

    message = {"role": "wizard"}
    if content is not None:
        message["content"] = content
    assert_chat_template_rendered_messages(
        [message],
        _FILTERING_TEMPLATE,
        render_text,
    )


def test_whitespace_only_tool_id_is_allowed():
    def render_text(messages):
        raise AssertionError("empty tool results must not be probed")

    assert_chat_template_rendered_messages(
        [{"role": "tol", "content": " \n", "tool_call_id": "call_1"}],
        _FILTERING_TEMPLATE,
        render_text,
    )


def test_unhandled_role_with_only_tool_calls_is_rejected():
    with pytest.raises(VLLMValidationError, match="'assistan'"):
        assert_chat_template_rendered_messages(
            [
                {
                    "role": "assistan",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                }
            ],
            _FILTERING_TEMPLATE,
            _filtering_render,
        )


def test_unhandled_multimodal_message_is_rejected():
    with pytest.raises(VLLMValidationError, match="'wizard'"):
        assert_chat_template_rendered_messages(
            [
                {
                    "role": "wizard",
                    "content": [
                        {
                            "type": "image",
                            "image_url": "http://example.test/a",
                        }
                    ],
                }
            ],
            _FILTERING_TEMPLATE,
            _filtering_render,
        )


def test_known_role_multimodal_message_is_kept():
    def render_text(messages):
        raise AssertionError("handled roles must not be probed")

    assert_chat_template_rendered_messages(
        [
            {
                "role": "user",
                "content": [{"type": "image", "image_url": "http://example.test/a"}],
            }
        ],
        _FILTERING_TEMPLATE,
        render_text,
    )


def test_structured_text_unknown_role_is_rejected():
    with pytest.raises(VLLMValidationError, match="'usr'"):
        assert_chat_template_rendered_messages(
            [
                {
                    "role": "usr",
                    "content": [{"type": "text", "text": "hello"}],
                }
            ],
            _FILTERING_TEMPLATE,
            _filtering_render,
        )


def test_probe_tokens_do_not_hide_an_earlier_dropped_message():
    """Index 1's probe must not match inside index 10's probe."""
    template = """
    {% for message in messages %}
    {% if message.role == 'user' %}
    user:{{ message.content }}
    {% elif message.role != 'usr' %}
    {{ message.role }}:{{ message.content }}
    {% endif %}
    {% endfor %}
    """
    messages = []
    for index in range(11):
        if index == 1:
            messages.append({"role": "usr", "content": "secret"})
        elif index == 10:
            messages.append({"role": "wizard", "content": "visible"})
        else:
            messages.append({"role": "user", "content": f"turn {index}"})

    def render_text(rendered_messages):
        parts = []
        for message in rendered_messages:
            role = message.get("role")
            content = message.get("content")
            if role == "usr" or not isinstance(content, str):
                continue
            parts.append(f"{role}:{content}")
        return "\n".join(parts)

    with pytest.raises(VLLMValidationError, match="'usr'") as exc_info:
        assert_chat_template_rendered_messages(messages, template, render_text)
    assert exc_info.value.value == "usr"


def test_many_omitted_structural_messages_share_one_render():
    """Dropping many image turns must not re-render once per message."""
    messages = [
        {
            "role": f"wizard{index}",
            "content": [{"type": "image", "image_url": "http://example.test/a"}],
        }
        for index in range(24)
    ]
    calls = 0

    def render_text(rendered_messages):
        nonlocal calls
        calls += 1
        return _filtering_render(rendered_messages)

    with pytest.raises(VLLMValidationError, match="'wizard0'") as exc_info:
        assert_chat_template_rendered_messages(
            messages, _FILTERING_TEMPLATE, render_text
        )
    assert calls == 2
    assert exc_info.value.value[0] == "wizard0"
    assert exc_info.value.value[-1] == "wizard23"


def test_many_rendered_tool_calls_are_checked_together():
    """A template that prints every tool name is not probed once per call."""
    messages = [
        {
            "role": "wizard",
            "content": "",
            "tool_calls": [
                {
                    "id": f"call_{index}",
                    "type": "function",
                    "function": {"name": f"lookup{index}", "arguments": "{}"},
                }
            ],
        }
        for index in range(24)
    ]
    calls = 0

    def render_text(rendered_messages):
        nonlocal calls
        calls += 1
        names = []
        for message in rendered_messages:
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                name = function.get("name")
                if isinstance(name, str):
                    names.append(name)
        return "\n".join(names)

    assert_chat_template_rendered_messages(messages, _FILTERING_TEMPLATE, render_text)
    assert calls <= 8


def test_batched_omission_keeps_a_rendered_tool_call():
    """Many dropped images plus one printed tool call stay a handful of renders."""
    messages = [
        {
            "role": f"drop{index}",
            "content": [{"type": "image", "image_url": "http://example.test/a"}],
        }
        for index in range(12)
    ]
    messages.append(
        {
            "role": "keeper",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        }
    )
    calls = 0

    def render_text(rendered_messages):
        nonlocal calls
        calls += 1
        names = []
        for message in rendered_messages:
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                name = function.get("name")
                if isinstance(name, str):
                    names.append(name)
        return "\n".join(names)

    with pytest.raises(VLLMValidationError, match="'drop0'") as exc_info:
        assert_chat_template_rendered_messages(
            messages, _FILTERING_TEMPLATE, render_text
        )
    assert calls <= 8
    assert "keeper" not in str(exc_info.value)
    # The last dropped image sits in the same leftover group as the tool
    # call when the render budget runs out, so it is not named. Earlier
    # dropped images are.
    assert "drop0" in exc_info.value.value


def test_selectattr_template_rejects_an_unselected_role():
    """A selectattr allow-list must not skip the drop check."""
    template = (
        "{% for message in messages | selectattr('role', 'equalto', 'user') %}"
        "user:{{ message.content }}{% endfor %}"
    )

    def render_text(rendered_messages):
        return "\n".join(
            f"user:{message.get('content')}"
            for message in rendered_messages
            if message.get("role") == "user"
        )

    with pytest.raises(VLLMValidationError, match="'usr'") as exc_info:
        assert_chat_template_rendered_messages(
            [{"role": "usr", "content": "hello"}],
            template,
            render_text,
        )
    assert exc_info.value.parameter == "messages"


def test_rejectattr_keeps_a_role_it_does_not_name():
    template = (
        "{% for message in messages | rejectattr('role', 'equalto', 'tool') %}"
        "{{ message.role }}:{{ message.content }}{% endfor %}"
    )

    def render_text(rendered_messages):
        return "\n".join(
            f"{message.get('role')}:{message.get('content')}"
            for message in rendered_messages
            if message.get("role") != "tool"
        )

    assert_chat_template_rendered_messages(
        [{"role": "usr", "content": "hello"}],
        template,
        render_text,
    )
    with patch("vllm.renderers.chat_utils.rendered_roles.logger.warning_once") as warn:
        assert_chat_template_rendered_messages(
            [{"role": "tool", "content": "result", "tool_call_id": "call_1"}],
            template,
            render_text,
        )
    warn.assert_called_once()
    assert "tool" in str(warn.call_args)


def _render_loop_template(template: str, messages) -> str:
    """Render with ``{% continue %}`` enabled, matching HF's Jinja env."""
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=["jinja2.ext.loopcontrols"],
    )
    return env.from_string(template).render(
        messages=[dict(message) for message in messages],
        add_generation_prompt=False,
    )


@pytest.mark.parametrize(
    "template",
    [
        pytest.param(
            """
            {% for message in messages %}
            {% if message.role not in ['user', 'assistant'] %}{% continue %}{% endif %}
            {{ message.role }}:{{ message.content }}
            {% endfor %}
            """,
            id="not-in",
        ),
        pytest.param(
            """
            {% for message in messages %}
            {% if message.role != 'user' and message.role != 'assistant' %}
            {% continue %}{% endif %}
            {{ message.role }}:{{ message.content }}
            {% endfor %}
            """,
            id="ne",
        ),
        pytest.param(
            """
            {% for message in messages %}
            {% if not (message.role == 'user' or message.role == 'assistant') %}
            {% continue %}{% endif %}
            {{ message.role }}:{{ message.content }}
            {% endfor %}
            """,
            id="not-eq",
        ),
    ],
)
def test_exclusion_template_rejects_a_dropped_custom_role(template):
    """A ``continue`` on a negated role check must not accept ``usr``."""
    dropped = [{"role": "usr", "content": "secret launch code"}]
    rendered = _render_loop_template(template, dropped)
    assert "secret launch code" not in rendered

    def render_text(messages):
        return _render_loop_template(template, messages)

    with pytest.raises(VLLMValidationError, match="'usr'") as exc_info:
        assert_chat_template_rendered_messages(dropped, template, render_text)
    assert exc_info.value.parameter == "messages"
    assert exc_info.value.value == "usr"

    kept = [{"role": "user", "content": "hello"}]
    assert "hello" in _render_loop_template(template, kept)
    calls = 0

    def render_kept(messages):
        nonlocal calls
        calls += 1
        return _render_loop_template(template, messages)

    assert_chat_template_rendered_messages(kept, template, render_kept)
    assert calls == 0


def test_negated_equality_rejects_the_named_role_when_content_is_dropped():
    """``not (role == 'wizard')`` must not treat wizard as rendered."""
    template = """
    {% for message in messages %}
    {% if not (message.role == 'wizard') %}
    {{ message.content }}
    {% endif %}
    {% endfor %}
    """
    assert roles_rendered_by_chat_template(template) == frozenset()

    def render_text(messages):
        return _render_chat_template(template, messages)

    rendered = render_text([{"role": "wizard", "content": "secret"}])
    assert "secret" not in rendered
    with pytest.raises(VLLMValidationError, match="'wizard'"):
        assert_chat_template_rendered_messages(
            [{"role": "wizard", "content": "secret"}],
            template,
            render_text,
        )


def test_selectattr_ne_rejects_a_role_also_named_by_equality():
    """``selectattr('role', 'ne', ...)`` drops that role even if ``==`` names it."""
    template = """
    {% for message in messages %}
    {% if message.role == 'wizard' %}{% endif %}
    {% endfor %}
    {% for message in messages | selectattr('role', 'ne', 'wizard') %}
    {{ message.content }}
    {% endfor %}
    """
    # The ``==`` still records wizard. The ``ne`` filter has to win, or the
    # allow-list would skip the probe and the dropped turn would pass.
    assert roles_rendered_by_chat_template(template) == frozenset({"wizard"})

    def render_text(messages):
        return _render_chat_template(template, messages)

    with pytest.raises(VLLMValidationError, match="'wizard'"):
        assert_chat_template_rendered_messages(
            [{"role": "wizard", "content": "secret"}],
            template,
            render_text,
        )


def test_rejectattr_ne_rejects_every_other_role():
    """``rejectattr('role', '!=', 'user')`` keeps user and drops ``usr``."""
    template = """
    {% for message in messages | rejectattr('role', '!=', 'user') %}
    {{ message.content }}
    {% endfor %}
    """
    assert roles_rendered_by_chat_template(template) == frozenset()

    def render_text(messages):
        return _render_chat_template(template, messages)

    assert "hello" in render_text([{"role": "user", "content": "hello"}])
    rendered = render_text([{"role": "usr", "content": "secret"}])
    assert "secret" not in rendered
    with pytest.raises(VLLMValidationError, match="'usr'"):
        assert_chat_template_rendered_messages(
            [{"role": "usr", "content": "secret"}],
            template,
            render_text,
        )


def test_rejectattr_still_rejects_a_nonstandard_role():
    template = (
        "{% for message in messages | rejectattr('role', 'equalto', 'wizard') %}"
        "{{ message.role }}:{{ message.content }}{% endfor %}"
    )

    def render_text(rendered_messages):
        return "\n".join(
            f"{message.get('role')}:{message.get('content')}"
            for message in rendered_messages
            if message.get("role") != "wizard"
        )

    with pytest.raises(VLLMValidationError, match="'wizard'"):
        assert_chat_template_rendered_messages(
            [{"role": "wizard", "content": "secret"}],
            template,
            render_text,
        )


def test_echoed_text_does_not_hide_a_dropped_tool_call():
    """Text and tool calls on one unhandled turn are checked separately."""

    def render_text(rendered_messages):
        parts = []
        for message in rendered_messages:
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                parts.append(content)
        return "\n".join(parts)

    message = {
        "role": "wizard",
        "content": "hello",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }
        ],
    }
    with pytest.raises(VLLMValidationError, match="'wizard'"):
        assert_chat_template_rendered_messages(
            [message],
            _FILTERING_TEMPLATE,
            render_text,
        )


def test_mixed_text_and_rendered_tool_call_is_kept():
    def render_text(rendered_messages):
        parts = []
        for message in rendered_messages:
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                parts.append(content)
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                name = function.get("name")
                if isinstance(name, str):
                    parts.append(name)
        return "\n".join(parts)

    assert_chat_template_rendered_messages(
        [
            {
                "role": "wizard",
                "content": "hello",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            }
        ],
        _FILTERING_TEMPLATE,
        render_text,
    )


def test_many_mixed_messages_do_not_render_per_suspect():
    """Echoed text plus a dropped tool call stays a constant number of renders."""
    messages = [
        {
            "role": f"wizard{index}",
            "content": "hello",
            "tool_calls": [
                {
                    "id": f"call_{index}",
                    "type": "function",
                    "function": {"name": f"lookup{index}", "arguments": "{}"},
                }
            ],
        }
        for index in range(20)
    ]
    calls = 0

    def render_text(rendered_messages):
        nonlocal calls
        calls += 1
        parts = []
        for message in rendered_messages:
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                parts.append(content)
        return "\n".join(parts)

    with pytest.raises(VLLMValidationError, match="'wizard0'"):
        assert_chat_template_rendered_messages(
            messages, _FILTERING_TEMPLATE, render_text
        )
    assert calls == 3
    assert calls < len(messages)


def test_structural_drop_ignores_index_shift():
    """A dropped tool call is still a 400 when later turns are indexed by position."""

    def render_text(rendered_messages):
        parts = [f"n={len(rendered_messages)}"]
        for index, message in enumerate(rendered_messages):
            role = message.get("role")
            if role in {"user", "assistant"}:
                parts.append(f"{index}:{role}:{message.get('content')}")
        return "\n".join(parts)

    with pytest.raises(VLLMValidationError, match="'wizard'"):
        assert_chat_template_rendered_messages(
            [
                {"role": "user", "content": "a"},
                {
                    "role": "wizard",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "assistant", "content": "b"},
            ],
            _FILTERING_TEMPLATE,
            render_text,
        )


def test_variable_role_equality_still_rejects_a_dropped_turn():
    template = "{% if message.role == required %}{{ message.content }}{% endif %}"

    def render_text(rendered_messages):
        return "".join(
            message["content"]
            for message in rendered_messages
            if message.get("role") == "user" and isinstance(message.get("content"), str)
        )

    with pytest.raises(VLLMValidationError, match="'usr'"):
        assert_chat_template_rendered_messages(
            [{"role": "usr", "content": "hello"}],
            template,
            render_text,
        )


def test_dropped_role_error_is_http_400():
    with pytest.raises(VLLMValidationError) as exc_info:
        assert_chat_template_rendered_messages(
            [{"role": "usr", "content": "hello"}],
            _FILTERING_TEMPLATE,
            _filtering_render,
        )
    response = create_error_response(exc_info.value)
    assert response.error.code == 400
    assert response.error.type == "BadRequestError"
    assert response.error.param == "messages"
    assert "usr" in response.error.message


def _read_repo_template(*parts: str) -> str:
    root = Path(__file__).resolve().parents[3]
    return root.joinpath(*parts).read_text()


def _read_user_assistant_template(template_path: str | tuple[str, ...]) -> str:
    """Load a repo template, or a built-in fallback by model type.

    The CPU test image installs the wheel and does not keep the source tree,
    so built-in templates are read from the same registry the renderer uses.
    """
    if isinstance(template_path, str):
        path = get_chat_template_fallback_path(template_path, "")
        if path is None:
            raise FileNotFoundError(template_path)
        return path.read_text()
    return _read_repo_template(*template_path)


def _render_chat_template(template: str, messages) -> str:
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.from_string(template).render(
        messages=[dict(message) for message in messages],
        add_generation_prompt=False,
        bos_token="",
    )


def _tool_call(name: str = "get_weather") -> dict:
    return {
        "id": "call_1",
        "type": "function",
        "function": {"name": name, "arguments": '{"city": "Paris"}'},
    }


def test_llama31_tool_call_with_text_is_not_rejected():
    """Assistant text plus a tool call still contributes, so it is not a 400."""
    template = _read_repo_template("examples", "tool_chat_template_llama3.1_json.jinja")
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is the weather?"},
        {
            "role": "assistant",
            "content": "Let me check",
            "tool_calls": [_tool_call()],
        },
    ]
    rendered = _render_chat_template(template, messages)
    assert "get_weather" in rendered
    assert "Let me check" not in rendered
    calls = 0

    def render_text(rendered_messages):
        nonlocal calls
        calls += 1
        return _render_chat_template(template, rendered_messages)

    with patch("vllm.renderers.chat_utils.rendered_roles.logger.warning_once") as warn:
        assert_chat_template_rendered_messages(messages, template, render_text)
    warn.assert_not_called()
    assert 0 < calls <= 8


def test_llama31_reasoning_with_tool_call_is_not_rejected():
    """Empty content plus reasoning must not be probed as dropped text."""
    template = _read_repo_template("examples", "tool_chat_template_llama3.1_json.jinja")
    messages = [
        {"role": "user", "content": "What is the weather?"},
        {
            "role": "assistant",
            "content": None,
            "reasoning": "I should call a tool",
            "tool_calls": [_tool_call()],
        },
    ]
    rendered = _render_chat_template(template, messages)
    assert "get_weather" in rendered
    assert "I should call a tool" not in rendered
    calls = 0

    def render_text(rendered_messages):
        nonlocal calls
        calls += 1
        return _render_chat_template(template, rendered_messages)

    with patch("vllm.renderers.chat_utils.rendered_roles.logger.warning_once") as warn:
        assert_chat_template_rendered_messages(messages, template, render_text)
    warn.assert_not_called()
    assert 0 < calls <= 8


def test_reasoning_only_custom_role_is_not_rejected():
    """A probe token belongs in content, never in reasoning."""

    def render_text(messages):
        raise AssertionError("reasoning-only turns must not be probed")

    assert_chat_template_rendered_messages(
        [{"role": "wizard", "content": "", "reasoning": "chain of thought"}],
        _FILTERING_TEMPLATE,
        render_text,
    )


def test_rendered_tool_call_keeps_a_turn_whose_text_was_omitted():
    """A turn that still prints its tool call is not a silent drop."""
    template = """
    {% for message in messages %}
    {% if message['role'] == 'user' or message['role'] == 'assistant' %}
    {{ message['content'] }}
    {% elif message.tool_calls %}
    {{ message.tool_calls[0].function.name }}
    {% endif %}
    {% endfor %}
    """
    message = {
        "role": "wizard",
        "content": "Let me check",
        "tool_calls": [_tool_call()],
    }

    def render_text(rendered_messages):
        return _render_chat_template(template, rendered_messages)

    rendered = render_text([message])
    assert "get_weather" in rendered
    assert "Let me check" not in rendered
    assert_chat_template_rendered_messages([message], template, render_text)


_USER_ASSISTANT_TEMPLATES: list[str | tuple[str, ...]] = [
    ("examples", "template_falcon.jinja"),
    ("examples", "template_chatglm2.jinja"),
    # Registered fallback. Resolved from the installed package, not the source tree.
    "blip-2",
]


@pytest.mark.parametrize("template_path", _USER_ASSISTANT_TEMPLATES)
def test_ignored_system_prompt_warns_and_is_not_a_400(template_path):
    template = _read_user_assistant_template(template_path)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "hello"},
    ]
    rendered = _render_chat_template(template, messages)
    assert "hello" in rendered
    assert "You are a helpful assistant." not in rendered
    calls = 0

    def render_text(rendered_messages):
        nonlocal calls
        calls += 1
        return _render_chat_template(template, rendered_messages)

    with patch("vllm.renderers.chat_utils.rendered_roles.logger.warning_once") as warn:
        assert_chat_template_rendered_messages(messages, template, render_text)
    assert 0 < calls <= 8
    warn.assert_called_once()
    assert "system" in str(warn.call_args)


def test_ignored_developer_role_warns_and_is_not_a_400():
    template = _read_repo_template("examples", "template_falcon.jinja")
    messages = [
        {"role": "developer", "content": "Be concise."},
        {"role": "user", "content": "hello"},
    ]
    calls = 0

    def render_text(rendered_messages):
        nonlocal calls
        calls += 1
        return _render_chat_template(template, rendered_messages)

    with patch("vllm.renderers.chat_utils.rendered_roles.logger.warning_once") as warn:
        assert_chat_template_rendered_messages(messages, template, render_text)
    assert 0 < calls <= 8
    warn.assert_called_once()
    assert "developer" in str(warn.call_args)


def test_standard_roles_follow_the_openai_message_schema():
    """Legacy ``function`` is a schema role, so it is not treated as a typo."""
    assert (
        frozenset({"system", "user", "assistant", "tool", "developer", "function"})
        == _STANDARD_CHAT_ROLES
    )


def test_ignored_function_role_warns_and_is_not_a_400():
    """A template that skips legacy function messages warns and still returns."""
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "function", "name": "lookup", "content": "result"},
    ]
    calls = 0

    def render_text(rendered_messages):
        nonlocal calls
        calls += 1
        return _filtering_render(rendered_messages)

    with patch("vllm.renderers.chat_utils.rendered_roles.logger.warning_once") as warn:
        assert_chat_template_rendered_messages(
            messages, _FILTERING_TEMPLATE, render_text
        )
    assert 0 < calls <= 8
    warn.assert_called_once()
    assert "function" in str(warn.call_args)


def test_formatting_only_assistant_check_does_not_warn():
    """Checking assistant for a tag is not proof that user content was dropped."""
    template = """
    {% for message in messages %}
    {% if message.role == 'assistant' %}
    <assistant>{{ message.content }}</assistant>
    {% else %}
    {{ message.content }}
    {% endif %}
    {% endfor %}
    """
    messages = [
        {"role": "user", "content": "hello from the user"},
        {"role": "assistant", "content": "hi"},
    ]
    rendered = _render_chat_template(template, messages)
    assert "hello from the user" in rendered
    assert "<assistant>hi</assistant>" in rendered
    calls = 0

    def render_text(rendered_messages):
        nonlocal calls
        calls += 1
        return _render_chat_template(template, rendered_messages)

    with patch("vllm.renderers.chat_utils.rendered_roles.logger.warning_once") as warn:
        assert_chat_template_rendered_messages(messages, template, render_text)
    warn.assert_not_called()
    assert 0 < calls <= 8


def test_loop_controls_and_generation_still_parse():
    """HF's Jinja env accepts ``{% break %}`` and ``{% generation %}``."""
    template = """
    {% for message in messages %}
    {% if message['role'] == 'user' %}
    {% generation %}{{ message['content'] }}{% endgeneration %}
    {% endif %}
    {% if loop.first %}{% break %}{% endif %}
    {% endfor %}
    """
    assert roles_rendered_by_chat_template(template) == frozenset({"user"})


def test_role_header_without_content_path_rejects_typo():
    """A role header for every turn does not keep a typo whose text is dropped."""
    template = """
    {% for message in messages %}
    <|{{ message['role'] }}|>
    {% if message['role'] == 'user' %}
    {% generation %}{{ message['content'] }}{% endgeneration %}
    {% elif message['role'] == 'assistant' %}
    {{ message['content'] }}
    {% endif %}
    {% endfor %}
    """

    def render_text(rendered_messages):
        parts = []
        for message in rendered_messages:
            role = message.get("role")
            parts.append(f"<|{role}|>")
            content = message.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                parts.append(content)
        return "".join(parts)

    with pytest.raises(VLLMValidationError, match="'usr'"):
        assert_chat_template_rendered_messages(
            [{"role": "usr", "content": "secret"}],
            template,
            render_text,
        )
    calls = 0

    def render_named(rendered_messages):
        nonlocal calls
        calls += 1
        return render_text(rendered_messages)

    assert_chat_template_rendered_messages(
        [{"role": "user", "content": "hello"}],
        template,
        render_named,
    )
    assert calls == 0


@pytest.mark.parametrize("template_path", _USER_ASSISTANT_TEMPLATES)
@pytest.mark.parametrize("role", ["usr", "assistan", "USER"])
def test_typo_role_on_user_assistant_template_is_rejected(template_path, role):
    template = _read_user_assistant_template(template_path)
    with pytest.raises(VLLMValidationError, match=re.escape(repr(role))):
        assert_chat_template_rendered_messages(
            [{"role": role, "content": "hello"}],
            template,
            lambda rendered_messages: _render_chat_template(
                template, rendered_messages
            ),
        )


# HuggingFaceH4/zephyr-7b-beta. No branch for any other role.
_ZEPHYR_TEMPLATE = """
{% for message in messages %}
{% if message['role'] == 'user' %}
{{ '<|user|>\\n' + message['content'] }}
{% elif message['role'] == 'system' %}
{{ '<|system|>\\n' + message['content'] }}
{% elif message['role'] == 'assistant' %}
{{ '<|assistant|>\\n' + message['content'] }}
{% endif %}
{% endfor %}
"""


@pytest.mark.parametrize(
    "content",
    ["what is 1+1?", [{"type": "text", "text": "what is 1+1?"}]],
)
def test_zephyr_template_rejects_custom_role(content):
    """Zephyr drops my-custom-role for both string and list content."""

    def render_text(messages):
        return _render_chat_template(_ZEPHYR_TEMPLATE, messages)

    rendered = render_text([{"role": "my-custom-role", "content": content}])
    assert "what is 1+1?" not in rendered
    with pytest.raises(VLLMValidationError, match="my-custom-role"):
        assert_chat_template_rendered_messages(
            [{"role": "my-custom-role", "content": content}],
            _ZEPHYR_TEMPLATE,
            render_text,
        )


# Llama Guard prints `content['text']` from text parts and ignores a string.
_TEXT_PART_TEMPLATE = """
{% for message in messages %}
{% if message['role'] == 'user' %}
{% set role = 'User' %}
{% elif message['role'] == 'assistant' %}
{% set role = 'Agent' %}
{% endif %}
{% for part in message['content'] | selectattr('type', 'equalto', 'text') %}
{{ role }}: {{ part['text'] }}
{% endfor %}
{% endfor %}
"""


def test_text_part_template_keeps_user_text():
    """A user turn is present once content is a text part, and a printed typo stays."""

    def render_text(messages):
        return _render_chat_template(_TEXT_PART_TEMPLATE, messages)

    plain = render_text([{"role": "user", "content": "hello from the user"}])
    assert "hello from the user" not in plain

    parts = [{"type": "text", "text": "hello from the user"}]
    rendered = render_text([{"role": "user", "content": parts}])
    assert "User: hello from the user" in rendered
    assert_chat_template_rendered_messages(
        [{"role": "user", "content": parts}],
        _TEXT_PART_TEMPLATE,
        render_text,
    )

    typo = [{"type": "text", "text": "typo payload zebra"}]
    assert "typo payload zebra" in render_text([{"role": "usr", "content": typo}])
    assert_chat_template_rendered_messages(
        [{"role": "usr", "content": typo}],
        _TEXT_PART_TEMPLATE,
        render_text,
    )


def _image_message(role: str, index: int) -> dict:
    return {
        "role": role,
        "content": [
            {"type": "image", "image_url": f"http://example.test/img-{index:04d}"}
        ],
    }


def _render_kept_images(messages, *, prefix_role: bool, echo_url: bool) -> str:
    parts: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image":
                continue
            if echo_url:
                url = part.get("image_url")
                if isinstance(url, str):
                    parts.append(url)
                continue
            token = "<image>"
            if prefix_role:
                token = f"{message.get('role')}:{token}"
            parts.append(token)
    return "".join(parts)


@pytest.mark.parametrize("count", [8, 32, 128])
@pytest.mark.parametrize(
    ("prefix_role", "echo_url"),
    [(False, False), (True, False), (False, True)],
)
def test_rendered_image_turns_stay_a_bounded_number_of_renders(
    count, prefix_role, echo_url
):
    """Successfully rendered images must not be re-rendered once per message."""
    messages = [
        _image_message(f"wizard{index}" if prefix_role else "wizard", index)
        for index in range(count)
    ]
    calls = 0

    def render_text(rendered_messages):
        nonlocal calls
        calls += 1
        return _render_kept_images(
            rendered_messages, prefix_role=prefix_role, echo_url=echo_url
        )

    assert_chat_template_rendered_messages(messages, _FILTERING_TEMPLATE, render_text)
    assert calls <= 8


def test_two_messages_one_dropped_image_is_rejected():
    """A dropped image next to one rendered image still fits in the budget."""
    messages = [_image_message("drop", 0), _image_message("wizard", 1)]

    def render_text(rendered_messages):
        parts: list[str] = []
        for message in rendered_messages:
            if message.get("role") == "drop":
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    parts.append("<image>")
        return "".join(parts)

    with pytest.raises(VLLMValidationError, match="'drop'") as exc_info:
        assert_chat_template_rendered_messages(
            messages, _FILTERING_TEMPLATE, render_text
        )
    assert exc_info.value.value == "drop"


def test_lone_dropped_image_among_many_rendered_is_kept():
    """The render budget stops before a single drop among many images.

    Stripping any large group still changes the prompt, because the other
    images render. After eight probes the unresolved turn is accepted.
    """
    messages = [
        _image_message("drop" if index == 17 else f"wizard{index}", index)
        for index in range(32)
    ]
    calls = 0

    def render_text(rendered_messages):
        nonlocal calls
        calls += 1
        parts: list[str] = []
        for message in rendered_messages:
            if message.get("role") == "drop":
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    parts.append("<image>")
        return "".join(parts)

    assert_chat_template_rendered_messages(messages, _FILTERING_TEMPLATE, render_text)
    assert calls <= 8


# `content[:5]` keeps "hello" and clips an appended probe marker.
_SLICE_TEMPLATE = """
{% for message in messages %}
{% if message['role'] == 'user' %}
{{ message['content'] }}
{% else %}
{{ message['content'][:5] }}
{% endif %}
{% endfor %}
"""


def test_sliced_content_is_not_rejected_when_the_probe_marker_is_clipped():
    messages = [{"role": "wizard", "content": "hello"}]
    rendered = _render_chat_template(_SLICE_TEMPLATE, messages)
    assert "hello" in rendered
    assert "vllmprobe" not in rendered
    calls = 0

    def render_text(rendered_messages):
        nonlocal calls
        calls += 1
        return _render_chat_template(_SLICE_TEMPLATE, rendered_messages)

    assert_chat_template_rendered_messages(messages, _SLICE_TEMPLATE, render_text)
    assert calls > 0


@pytest.mark.parametrize("count", [64, 256, 1024])
def test_distinct_truncated_messages_stay_within_render_budget(count):
    """``content[:5]`` clips every marker when each prefix is unique.

    The omission split cannot prove which turn contributed, so it stops
    after the shared render budget and keeps the request.
    """
    messages = [
        {"role": f"wizard{index}", "content": f"{index:05d}-payload"}
        for index in range(count)
    ]
    calls = 0

    def render_text(rendered_messages):
        nonlocal calls
        calls += 1
        parts: list[str] = []
        for message in rendered_messages:
            content = message.get("content")
            if isinstance(content, str):
                parts.append(content[:5])
        return "".join(parts)

    assert_chat_template_rendered_messages(messages, _SLICE_TEMPLATE, render_text)
    assert calls <= 8


def test_sliced_template_still_rejects_a_role_it_omits():
    template = """
    {% for message in messages %}
    {% if message['role'] == 'user' %}
    {{ message['content'][:5] }}
    {% endif %}
    {% endfor %}
    """

    def render_text(rendered_messages):
        return _render_chat_template(template, rendered_messages)

    with pytest.raises(VLLMValidationError, match="'wizard'"):
        assert_chat_template_rendered_messages(
            [{"role": "wizard", "content": "hello"}],
            template,
            render_text,
        )
