# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import lru_cache
from typing import Any, Literal

import jinja2.ext
import jinja2.nodes
import jinja2.parser
import jinja2.sandbox

from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.utils import random_uuid

logger = init_logger(__name__)


class _ChatTemplateParseExtension(jinja2.ext.Extension):
    """Accept HF ``{% generation %}`` blocks while parsing a chat template."""

    tags = {"generation"}

    def parse(self, parser: jinja2.parser.Parser) -> jinja2.nodes.Node:
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return jinja2.nodes.Scope(body).set_lineno(lineno)


# Same sandbox + loop/generation extensions used when a template is rendered.
_CHAT_TEMPLATE_JINJA_ENV = jinja2.sandbox.ImmutableSandboxedEnvironment(
    trim_blocks=True,
    lstrip_blocks=True,
    extensions=[_ChatTemplateParseExtension, jinja2.ext.loopcontrols],
)
_TEXT_PART_TYPES = frozenset(
    {
        "text",
        "input_text",
        "output_text",
        "refusal",
        "thinking",
    }
)


def _is_role_access(node: jinja2.nodes.Node) -> bool:
    """Whether ``node`` reads ``.role`` or ``['role']``."""
    if isinstance(node, jinja2.nodes.Getattr):
        return node.attr == "role"
    if isinstance(node, jinja2.nodes.Getitem):
        return isinstance(node.arg, jinja2.nodes.Const) and node.arg.value == "role"
    return False


def _literal_role_names(node: jinja2.nodes.Node) -> list[str] | None:
    """String literal, or a list/tuple of them. ``None`` if not pure literals."""
    if isinstance(node, jinja2.nodes.Const) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (jinja2.nodes.List, jinja2.nodes.Tuple)):
        names: list[str] = []
        for item in node.items:
            if not isinstance(item, jinja2.nodes.Const) or not isinstance(
                item.value, str
            ):
                return None
            names.append(item.value)
        return names
    return None


def _roles_from_compare(
    node: jinja2.nodes.Compare,
) -> tuple[Literal["ignore", "literal", "dynamic"], set[str]]:
    """Classify one comparison that might select rendered roles.

    Only ``==`` and ``in`` count. ``!=`` and ``not in`` are generation-prompt
    guards (ChatML checks ``messages[-1]['role'] != 'assistant'``) and do not
    select which roles are rendered. A non-literal ``in`` target such as
    ``message.role in allowed_roles`` is dynamic.

    Returns:
        ``("ignore", empty)`` when the node is not a role inclusion test,
        ``("literal", names)`` when the accepted roles are string literals
        (an empty list is a literal filter that names nothing), or
        ``("dynamic", empty)`` when the allow-list is not a literal.

    """
    if not node.ops:
        return "ignore", set()
    # Chained comparisons compare later operands with each other, not with
    # the role expression.
    operand = node.ops[0]
    op = operand.op
    if op not in {"eq", "in"}:
        return "ignore", set()

    left_is_role = _is_role_access(node.expr)
    right_is_role = _is_role_access(operand.expr)
    if op == "eq":
        if not left_is_role and not right_is_role:
            return "ignore", set()
        literal_node = operand.expr if left_is_role else node.expr
        names = _literal_role_names(literal_node)
        if names is None:
            return "ignore", set()
        return "literal", set(names)

    if not left_is_role:
        return "ignore", set()
    names = _literal_role_names(operand.expr)
    if names is None:
        return "dynamic", set()
    return "literal", set(names)


@lru_cache(maxsize=32)
def roles_rendered_by_chat_template(chat_template: str) -> frozenset[str] | None:
    """Return roles a template explicitly dispatches on.

    The Jinja AST is walked for comparisons that read ``.role`` or
    ``['role']``. ``None`` means the template does not filter by role, so
    every role is emitted (ChatML-style ``{{ message['role'] }}``). A
    frozenset, possibly empty, means unknown roles can be dropped and must
    be checked after render.

    Args:
        chat_template: Resolved Jinja source.

    Returns:
        Roles named by ``==`` / ``in`` tests. An empty set means the
        template filters on role, but the allowed names are not literals
        (``message.role in allowed_roles``). ``None`` means there is no
        role filter.

    """
    try:
        parsed = _CHAT_TEMPLATE_JINJA_ENV.parse(chat_template)
    except Exception:
        logger.warning(
            "Could not parse the chat template while checking message roles.",
            exc_info=True,
        )
        return None

    roles: set[str] = set()
    saw_role_filter = False
    for compare in parsed.find_all(jinja2.nodes.Compare):
        kind, found = _roles_from_compare(compare)
        if kind == "ignore":
            continue
        saw_role_filter = True
        if kind == "literal":
            roles.update(found)
    if not saw_role_filter:
        return None
    return frozenset(roles)


def _has_nonempty_text(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return _has_nonempty_text(value.get("text"))
    if isinstance(value, list):
        return any(_has_nonempty_text(part) for part in value)
    return False


def message_has_nonempty_payload(message: Mapping[str, Any]) -> bool:
    """Whether a message carries content the template must not drop.

    Empty and whitespace-only text is not a payload. Templates may omit
    those turns, and there is no user-visible content to lose. A tool-call
    id with no arguments or result text is treated the same way. Multimodal
    parts and tool calls are payloads even when they have no text.
    """
    if _has_nonempty_text(message.get("content")):
        return True
    if _has_nonempty_text(message.get("reasoning")):
        return True
    if _has_nonempty_text(message.get("reasoning_content")):
        return True
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and len(tool_calls) > 0:
        return True
    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                part_type = part.get("type")
                if part_type not in _TEXT_PART_TYPES and part_type is not None:
                    return True
    return False


def _inject_probe_token(message: dict[str, Any], token: str) -> bool:
    """Append ``token`` to every text field so a render can prove inclusion."""
    injected = False
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        message["content"] = content + token
        injected = True
    elif isinstance(content, list):
        parts: list[Any] = []
        for part in content:
            if isinstance(part, str) and part.strip():
                parts.append(part + token)
                injected = True
            elif isinstance(part, dict):
                updated = dict(part)
                text = updated.get("text")
                if isinstance(text, str) and text.strip():
                    updated["text"] = text + token
                    injected = True
                parts.append(updated)
            else:
                parts.append(part)
        message["content"] = parts
    for key in ("reasoning", "reasoning_content"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            message[key] = value + token
            injected = True
    return injected


def _unrendered_role_error(roles: list[str]) -> VLLMValidationError:
    shown = ", ".join(repr(role) for role in roles)
    noun = "role" if len(roles) == 1 else "roles"
    return VLLMValidationError(
        "Chat template produced no prompt content for non-empty "
        f"message {noun} {shown}. The template did not emit that "
        "content, so the turn would be dropped. Custom roles are "
        "accepted when the template renders their content; check for "
        "a typo in `role`.",
        parameter="messages",
        value=roles[0] if len(roles) == 1 else roles,
    )


def _unique_roles(roles: Iterable[str]) -> list[str]:
    seen: list[str] = []
    for role in roles:
        if role not in seen:
            seen.append(role)
    return seen


def _roles_dropped_by_omission(
    conversation: Sequence[Mapping[str, Any]],
    suspects: Sequence[tuple[int, str]],
    render_text: Callable[[Sequence[Mapping[str, Any]]], str],
) -> list[str]:
    """Roles whose removal leaves the rendered prompt unchanged."""
    try:
        baseline = render_text(conversation)
    except Exception:
        logger.warning(
            "Could not re-render the conversation to check dropped roles.",
            exc_info=True,
        )
        return []
    if not isinstance(baseline, str):
        logger.warning_once(
            "Chat template probe did not return text; skipping the dropped-role check."
        )
        return []
    dropped: list[str] = []
    for index, role in suspects:
        reduced = [message for i, message in enumerate(conversation) if i != index]
        try:
            alternate = render_text(reduced)
        except Exception:
            # Removing the message changed rendering enough to fail.
            # That is not a silent drop.
            continue
        if isinstance(alternate, str) and alternate == baseline:
            dropped.append(role)
    return dropped


def assert_chat_template_rendered_messages(
    conversation: Sequence[Mapping[str, Any]],
    chat_template: str,
    render_text: Callable[[Sequence[Mapping[str, Any]]], str],
) -> None:
    """Raise if a non-empty message contributed no prompt content.

    This is a post-condition on chat-template rendering. Roles the template
    names (``user``, ``assistant``, ``tool``, and template-specific roles)
    are trusted. Any other role is probed: a unique token is appended to
    its text and the conversation is rendered again. If the token is absent,
    the message was dropped and the request is rejected. Templates that
    pass every role through are not probed.

    Empty and whitespace-only messages are not an error. There is no
    content to drop, including a tool message that only carries
    ``tool_call_id``.

    Args:
        conversation: Messages passed to the template, after rewrites such
            as developer → system.
        chat_template: Resolved Jinja source.
        render_text: Renders a conversation to prompt text with the same
            template options as the request.

    Raises:
        VLLMValidationError: A non-empty message produced no prompt
            content. The error names the role and is returned as HTTP 400.

    """
    handled = roles_rendered_by_chat_template(chat_template)
    if handled is None:
        return

    suspects = [
        (index, role)
        for index, message in enumerate(conversation)
        if isinstance((role := message.get("role")), str)
        and role not in handled
        and message_has_nonempty_payload(message)
    ]
    if not suspects:
        return

    text_suspects = [
        (index, role)
        for index, role in suspects
        if _has_nonempty_text(conversation[index].get("content"))
        or _has_nonempty_text(conversation[index].get("reasoning"))
        or _has_nonempty_text(conversation[index].get("reasoning_content"))
    ]
    text_indexes = {index for index, _ in text_suspects}
    structural_suspects = [
        (index, role) for index, role in suspects if index not in text_indexes
    ]

    dropped: list[str] = []
    if text_suspects:
        dropped.extend(
            _roles_dropped_by_text_probe(conversation, text_suspects, render_text)
        )
    if structural_suspects:
        dropped.extend(
            _roles_dropped_by_omission(conversation, structural_suspects, render_text)
        )
    dropped = _unique_roles(dropped)
    if dropped:
        raise _unrendered_role_error(dropped)


def _roles_dropped_by_text_probe(
    conversation: Sequence[Mapping[str, Any]],
    suspects: Sequence[tuple[int, str]],
    render_text: Callable[[Sequence[Mapping[str, Any]]], str],
) -> list[str]:
    try:
        probed = copy.deepcopy(list(conversation))
    except Exception:
        logger.warning(
            "Could not copy the conversation to check dropped roles.",
            exc_info=True,
        )
        return _roles_dropped_by_omission(conversation, suspects, render_text)

    prefix = "vllmprobe" + random_uuid().replace("-", "")[:12]
    tokens: list[tuple[str, str]] = []
    for index, role in suspects:
        token = f"{prefix}{index}"
        message = probed[index]
        if isinstance(message, dict) and _inject_probe_token(message, token):
            tokens.append((token, role))
    if not tokens:
        return _roles_dropped_by_omission(conversation, suspects, render_text)

    try:
        rendered = render_text(probed)
    except Exception:
        logger.warning(
            "Chat template probe failed while checking role(s) %s.",
            [role for _, role in suspects],
            exc_info=True,
        )
        return _roles_dropped_by_omission(conversation, suspects, render_text)
    if not isinstance(rendered, str):
        return _roles_dropped_by_omission(conversation, suspects, render_text)

    folded = rendered.lower()
    missing = [role for token, role in tokens if token.lower() not in folded]
    return missing
