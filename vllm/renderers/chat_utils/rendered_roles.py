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


# Inclusion tests passed to selectattr/rejectattr. Exclusion tests (``ne``)
# and rejectattr do not name roles the template keeps.
_ROLE_INCLUDE_TESTS = frozenset({"equalto", "eq", "==", "sameas", "in"})


def _roles_from_attr_filter(
    node: jinja2.nodes.Filter,
) -> tuple[Literal["ignore", "literal", "dynamic"], set[str]]:
    """Classify a ``selectattr`` / ``rejectattr`` call on ``role``.

    ``selectattr('role', 'equalto', 'user')`` keeps that literal role.
    ``rejectattr``, a non-literal target, or a test that is not an
    inclusion (``ne``) still means the template filters by role, but the
    kept names are not a literal allow-list.

    Returns:
        ``("ignore", empty)`` when the filter does not test ``role``,
        ``("literal", names)`` when ``selectattr`` names kept roles, or
        ``("dynamic", empty)`` when the filter is real but the kept
        roles are not literals.

    """
    if node.name not in {"selectattr", "rejectattr"} or not node.args:
        return "ignore", set()
    attr = node.args[0]
    if not isinstance(attr, jinja2.nodes.Const) or attr.value != "role":
        return "ignore", set()
    if node.name != "selectattr" or len(node.args) < 3:
        return "dynamic", set()
    test = node.args[1]
    if (
        not isinstance(test, jinja2.nodes.Const)
        or test.value not in _ROLE_INCLUDE_TESTS
    ):
        return "dynamic", set()
    names = _literal_role_names(node.args[2])
    if names is None:
        return "dynamic", set()
    return "literal", set(names)


def _roles_from_compare(
    node: jinja2.nodes.Compare,
) -> tuple[Literal["ignore", "literal", "dynamic"], set[str]]:
    """Classify one comparison that might select rendered roles.

    Only ``==`` and ``in`` count. ``!=`` and ``not in`` are generation-prompt
    guards (ChatML checks ``messages[-1]['role'] != 'assistant'``) and do not
    select which roles are rendered. A non-literal target, whether
    ``message.role in allowed_roles`` or ``message.role == some_var``, is
    dynamic.

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
            # ``message.role == some_var`` names no literal. Same as
            # ``role in allowed_roles``: a filter may exist, so probe
            # instead of skipping the check.
            return "dynamic", set()
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
    ``['role']``, and for ``selectattr`` / ``rejectattr`` on ``role``.
    ``None`` means the template does not filter by role, so every role is
    emitted (ChatML-style ``{{ message['role'] }}``). A frozenset, possibly
    empty, means unknown roles can be dropped and must be checked after
    render.

    Args:
        chat_template: Resolved Jinja source.

    Returns:
        Roles named by ``==`` / ``in`` tests or by ``selectattr`` on
        ``role``. An empty set means the template filters on role, but the
        allowed names are not literals (``message.role in allowed_roles``,
        ``message.role == some_var``, or ``rejectattr``). ``None`` means
        there is no role filter.

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
    findings = [
        _roles_from_compare(node) for node in parsed.find_all(jinja2.nodes.Compare)
    ]
    findings.extend(
        _roles_from_attr_filter(node) for node in parsed.find_all(jinja2.nodes.Filter)
    )
    for kind, found in findings:
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
    return _has_structural_payload(message)


def _has_structural_payload(message: Mapping[str, Any]) -> bool:
    """Whether a message carries tool calls or non-text multimodal parts."""
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


def _strip_structural_payload(message: Mapping[str, Any]) -> dict[str, Any]:
    """Copy ``message`` without tool calls or multimodal parts.

    Text stays, and the message keeps its index. Removing the whole turn
    would hide a dropped tool call behind echoed text, and would also shift
    later messages in templates that read neighbors by position.
    """
    updated = dict(message)
    tool_calls = updated.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        updated["tool_calls"] = []
    content = updated.get("content")
    if isinstance(content, list):
        kept: list[Any] = []
        for part in content:
            if isinstance(part, dict):
                part_type = part.get("type")
                if part_type not in _TEXT_PART_TYPES and part_type is not None:
                    continue
            kept.append(part)
        updated["content"] = kept
    return updated


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


def _role_probe_token(prefix: str, index: int) -> str:
    """Build a probe token that is not a substring of another index's token."""
    return f"{prefix}_{index:06d}_"


def _render_probe_text(
    render_text: Callable[[Sequence[Mapping[str, Any]]], str],
    messages: Sequence[Mapping[str, Any]],
) -> str | None:
    try:
        rendered = render_text(messages)
    except Exception:
        return None
    if not isinstance(rendered, str):
        return None
    return rendered


def _messages_without(
    conversation: Sequence[Mapping[str, Any]],
    indexes: set[int],
) -> list[Mapping[str, Any]]:
    return [
        message for index, message in enumerate(conversation) if index not in indexes
    ]


def _messages_without_structural_payload(
    conversation: Sequence[Mapping[str, Any]],
    indexes: set[int],
) -> list[Mapping[str, Any]]:
    return [
        _strip_structural_payload(message) if index in indexes else message
        for index, message in enumerate(conversation)
    ]


def _stamp_tool_call_probe(message: dict[str, Any], token: str) -> bool:
    """Append ``token`` to tool-call names this message already has.

    A new text part is not added. A template can echo that part while still
    dropping the tool call or image that made the message a suspect.
    """
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return False
    stamped: list[Any] = []
    wrote = False
    for call in tool_calls:
        if not isinstance(call, dict):
            stamped.append(call)
            continue
        call = copy.deepcopy(call)
        function = call.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            function["name"] = function["name"] + token
            wrote = True
        stamped.append(call)
    if wrote:
        message["tool_calls"] = stamped
    return wrote


def _tool_call_indexes_echoed(
    conversation: Sequence[Mapping[str, Any]],
    suspects: Sequence[tuple[int, str]],
    render_text: Callable[[Sequence[Mapping[str, Any]]], str],
) -> set[int] | None:
    """Suspects whose tool-call name is still present after one probe render.

    ``None`` means the probe did not return text. An empty set means no
    stamped name showed up; that is not proof the message was dropped.
    """
    try:
        probed = copy.deepcopy(list(conversation))
    except Exception:
        logger.warning(
            "Could not copy the conversation to check dropped roles.",
            exc_info=True,
        )
        return None

    prefix = "vllmprobe" + random_uuid().replace("-", "")[:12]
    tokens: list[tuple[int, str]] = []
    for index, _role in suspects:
        message = probed[index]
        if not isinstance(message, dict):
            continue
        token = _role_probe_token(prefix, index)
        if _stamp_tool_call_probe(message, token):
            tokens.append((index, token))
    if not tokens:
        return set()

    rendered = _render_probe_text(render_text, probed)
    if rendered is None:
        return None
    folded = rendered.lower()
    return {index for index, token in tokens if token.lower() in folded}


def _roles_dropped_by_omission(
    conversation: Sequence[Mapping[str, Any]],
    suspects: Sequence[tuple[int, str]],
    render_text: Callable[[Sequence[Mapping[str, Any]]], str],
    *,
    strip_payload: bool = False,
) -> list[str]:
    """Roles whose structural payload leaves the rendered prompt unchanged.

    Every suspect is covered by one render. An unchanged prompt means each
    of them was omitted. If the prompt changes, tool-call names are stamped
    and checked in a second render so messages the template actually prints
    are cleared together. Only a mixed remainder is split, and a fully
    omitted subgroup is still decided by one render.

    When ``strip_payload`` is true, tool calls and multimodal parts are
    removed but the message stays at the same index. That catches a tool
    call dropped beside echoed text, and keeps neighbor lookups stable.
    When false, the whole message is removed. The text-probe fallback uses
    that to see whether the turn itself was omitted.
    """
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

    group = list(suspects)
    if not group:
        return []

    def without(indexes: set[int]) -> list[Mapping[str, Any]]:
        if strip_payload:
            return _messages_without_structural_payload(conversation, indexes)
        return _messages_without(conversation, indexes)

    def dropped_in(
        subgroup: Sequence[tuple[int, str]],
        known_alternate: str | None = None,
        *,
        have_known: bool = False,
    ) -> list[str]:
        if not subgroup:
            return []
        if have_known:
            alternate = known_alternate
        else:
            alternate = _render_probe_text(
                render_text,
                without({index for index, _role in subgroup}),
            )
        if isinstance(alternate, str) and alternate == baseline:
            return [role for _index, role in subgroup]
        # One comparison cannot tell which member of a changed group was
        # the one that contributed. A single message that changes the
        # prompt, or whose removal fails to render, is not a silent drop.
        if len(subgroup) == 1:
            return []
        mid = len(subgroup) // 2
        return dropped_in(subgroup[:mid]) + dropped_in(subgroup[mid:])

    alternate = _render_probe_text(
        render_text,
        without({index for index, _role in group}),
    )
    if isinstance(alternate, str) and alternate == baseline:
        return [role for _index, role in group]

    has_tool_calls = False
    for index, _role in group:
        message = conversation[index]
        tool_calls = message.get("tool_calls") if isinstance(message, Mapping) else None
        if isinstance(tool_calls, list) and tool_calls:
            has_tool_calls = True
            break
    if has_tool_calls:
        echoed = _tool_call_indexes_echoed(conversation, group, render_text)
        if echoed:
            group = [(index, role) for index, role in group if index not in echoed]
            if not group:
                return []
            # ``alternate`` also removed the echoed messages, so it does
            # not describe this smaller group.
            return dropped_in(group)

    return dropped_in(group, alternate, have_known=True)


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
    the message was dropped and the request is rejected. Tool calls and
    multimodal parts on that same message are checked too, so echoed text
    cannot hide a dropped tool call or image. Templates that pass every
    role through are not probed.

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
    structural_suspects = [
        (index, role)
        for index, role in suspects
        if _has_structural_payload(conversation[index])
    ]

    dropped: list[str] = []
    if text_suspects:
        dropped.extend(
            _roles_dropped_by_text_probe(conversation, text_suspects, render_text)
        )
    if structural_suspects:
        dropped.extend(
            _roles_dropped_by_omission(
                conversation,
                structural_suspects,
                render_text,
                strip_payload=True,
            )
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
        token = _role_probe_token(prefix, index)
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
