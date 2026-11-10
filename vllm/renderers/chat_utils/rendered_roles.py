# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import lru_cache
from types import UnionType
from typing import Any, Literal, NamedTuple, Union, get_args, get_origin

import jinja2.nodes
import transformers.utils.chat_template_utils as hf_chat_utils
from typing_extensions import get_type_hints

from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.renderers.chat_utils.types import OpenAIChatCompletionMessageParam
from vllm.utils import random_uuid

logger = init_logger(__name__)

_TEXT_PART_TYPES = frozenset(
    {
        "text",
        "input_text",
        "output_text",
        "refusal",
        "thinking",
    }
)


def _parse_chat_template(chat_template: str) -> jinja2.nodes.Template:
    """Parse with the Jinja environment Hugging Face renders templates with.

    `transformers.utils.chat_template_utils._compile_jinja_template` is
    private, but it is the only place that environment is built (the
    `generation` tag, loop controls, and the `tojson` /
    `raise_exception` / `strftime_now` helpers). The renderer already
    parses through it for content-format detection, and the supported
    transformers range (`>=5.16.1,<5.20`) keeps the helper.
    """
    compiled = hf_chat_utils._compile_jinja_template(chat_template)
    return compiled.environment.parse(chat_template)


def _literal_strings(annotation: Any) -> set[str]:
    """String members of a `Literal`, including wrapped annotations."""
    if get_origin(annotation) is Literal:
        return {value for value in get_args(annotation) if isinstance(value, str)}
    found: set[str] = set()
    for arg in get_args(annotation):
        found |= _literal_strings(arg)
    return found


def _roles_from_message_params(annotation: Any) -> frozenset[str]:
    """Literal `role` values on a chat-message union.

    `CustomChatCompletionMessageParam.role` is a free-form string, so
    those stay out of this set and can still be a hard error. Only the
    fixed OpenAI roles are collected. Legacy `function` is one of them.
    """
    roles: set[str] = set()
    pending: list[Any] = [annotation]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        origin = get_origin(current)
        if origin is Union or origin is UnionType:
            pending.extend(get_args(current))
            continue
        if not isinstance(current, type):
            continue
        try:
            hints = get_type_hints(current)
        except Exception:
            logger.warning(
                "Could not read message role literals from %s.",
                getattr(current, "__name__", current),
                exc_info=True,
            )
            continue
        roles |= _literal_strings(hints.get("role"))
    return frozenset(roles)


# Fixed roles on OpenAI's chat message union, which vLLM's request schema
# accepts. A template may ignore these on purpose. A typo or a custom role
# is outside this set and can still be a hard error.
_STANDARD_CHAT_ROLES = _roles_from_message_params(OpenAIChatCompletionMessageParam)


def _is_role_access(node: jinja2.nodes.Node) -> bool:
    """Whether `node` reads `.role` or `['role']`."""
    if isinstance(node, jinja2.nodes.Getattr):
        return node.attr == "role"
    if isinstance(node, jinja2.nodes.Getitem):
        return isinstance(node.arg, jinja2.nodes.Const) and node.arg.value == "role"
    return False


def _literal_role_names(node: jinja2.nodes.Node) -> list[str] | None:
    """String literal, or a list/tuple of them. `None` if not pure literals."""
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


# Inclusion tests passed to selectattr/rejectattr. A negated test (``ne``)
# does not name roles the template keeps.
_ROLE_INCLUDE_TESTS = frozenset({"equalto", "eq", "==", "sameas", "in"})
_ROLE_NEGATED_TESTS = frozenset({"ne", "!=", "notin", "not in"})
_ROLE_COMPARE_OPS = frozenset({"eq", "ne", "in", "notin"})


# Extra conversation renders spent proving a drop. Past this, unresolved
# turns are kept: a clipped marker or a rendered image must not walk the
# whole conversation again.
_MAX_EXTRA_RENDERS = 8

_RoleFinding = Literal["ignore", "literal", "dynamic", "exclude"]


class _TemplateRoles(NamedTuple):
    """Literal roles named by one parsed template.

    `handled is None` means the template does not filter on role.
    `excluded` is roles an inclusion `rejectattr`, or a negated
    `selectattr`, drops by name. Neither set proves the content was
    omitted. A standard role in `excluded`, or outside a non-empty
    `handled` set, is probed and warned only when that probe shows the
    content is gone.
    """

    handled: frozenset[str] | None
    excluded: frozenset[str]


def _roles_from_attr_filter(
    node: jinja2.nodes.Filter,
) -> tuple[_RoleFinding, set[str]]:
    """Classify a `selectattr` / `rejectattr` call on `role`.

    `selectattr('role', 'equalto', 'user')` keeps that literal role.
    `rejectattr('role', 'equalto', 'tool')` names a role the template
    drops. `selectattr('role', 'ne', 'wizard')` drops the named role, so
    those names are exclusions and must not stay on an allow-list.
    `rejectattr('role', 'ne', 'user')` keeps the named role and drops
    every other one; that name is not an allow-list either. A non-literal
    target still means the template filters by role.

    Returns:
        `("ignore", empty)` when the filter does not test `role`,
        `("literal", names)` when `selectattr` names kept roles,
        `("exclude", names)` when the filter names dropped roles,
        or `("dynamic", empty)` when the filter is real but the kept
        names are not a literal allow-list.

    """
    if node.name not in {"selectattr", "rejectattr"} or not node.args:
        return "ignore", set()
    attr = node.args[0]
    if not isinstance(attr, jinja2.nodes.Const) or attr.value != "role":
        return "ignore", set()
    if len(node.args) < 3:
        return "dynamic", set()
    test = node.args[1]
    if not isinstance(test, jinja2.nodes.Const):
        return "dynamic", set()
    negative = test.value in _ROLE_NEGATED_TESTS
    if not negative and test.value not in _ROLE_INCLUDE_TESTS:
        return "dynamic", set()
    names = _literal_role_names(node.args[2])
    if names is None or (negative and node.name == "rejectattr"):
        return "dynamic", set()
    if node.name == "rejectattr" or negative:
        return "exclude", set(names)
    return "literal", set(names)


def _negated_compare_ids(parsed: jinja2.nodes.Node) -> set[int]:
    """Compares sitting under an odd number of `not` expressions."""
    negated: set[int] = set()

    def walk(node: jinja2.nodes.Node, depth: int) -> None:
        if isinstance(node, jinja2.nodes.Not):
            depth += 1
        elif isinstance(node, jinja2.nodes.Compare) and depth % 2 == 1:
            negated.add(id(node))
        for child in node.iter_child_nodes():
            walk(child, depth)

    walk(parsed, 0)
    return negated


def _roles_from_compare(
    node: jinja2.nodes.Compare,
    *,
    negated: bool = False,
) -> tuple[_RoleFinding, set[str]]:
    """Classify one comparison that might select rendered roles.

    Positive `==` / `in` with string literals name roles the template
    may keep. `!=`, `not in`, and `not (role == ...)` are possible
    filters, but the literals are not an allow-list: a generation-prompt
    guard (`messages[-1]['role'] != 'assistant'`) and a `continue` that
    skips every other role look the same here. Those roles stay unresolved
    until a render probe. A non-literal target, whether `message.role in
    allowed_roles` or `message.role == some_var`, is dynamic too.

    Returns:
        `("ignore", empty)` when the node does not test a role,
        `("literal", names)` when the accepted roles are string literals
        (an empty list is a literal filter that names nothing), or
        `("dynamic", empty)` when a filter exists but the kept names are
        not literals.

    """
    if not node.ops:
        return "ignore", set()
    # Chained comparisons compare later operands with each other, not with
    # the role expression.
    operand = node.ops[0]
    op = operand.op
    if op not in _ROLE_COMPARE_OPS:
        return "ignore", set()

    left_is_role = _is_role_access(node.expr)
    right_is_role = _is_role_access(operand.expr)
    if op in {"eq", "ne"}:
        if not left_is_role and not right_is_role:
            return "ignore", set()
        if negated or op == "ne":
            return "dynamic", set()
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
    if negated or op == "notin":
        return "dynamic", set()
    names = _literal_role_names(operand.expr)
    if names is None:
        return "dynamic", set()
    return "literal", set(names)


@lru_cache(maxsize=32)
def _template_roles(chat_template: str) -> _TemplateRoles:
    """Parse `chat_template` once for literal role names and exclusions."""
    try:
        parsed = _parse_chat_template(chat_template)
    except Exception:
        logger.warning(
            "Could not parse the chat template while checking message roles.",
            exc_info=True,
        )
        return _TemplateRoles(None, frozenset())

    included: set[str] = set()
    excluded: set[str] = set()
    saw_role_filter = False
    negated = _negated_compare_ids(parsed)
    findings = [
        _roles_from_compare(node, negated=id(node) in negated)
        for node in parsed.find_all(jinja2.nodes.Compare)
    ]
    findings.extend(
        _roles_from_attr_filter(node) for node in parsed.find_all(jinja2.nodes.Filter)
    )
    for kind, found in findings:
        if kind == "ignore":
            continue
        saw_role_filter = True
        if kind == "literal":
            included.update(found)
        elif kind == "exclude":
            excluded.update(found)
    if not saw_role_filter:
        return _TemplateRoles(None, frozenset())
    return _TemplateRoles(frozenset(included), frozenset(excluded))


def roles_rendered_by_chat_template(chat_template: str) -> frozenset[str] | None:
    """Return roles a template explicitly dispatches on.

    The Jinja AST is walked once for comparisons that read `.role` or
    `['role']`, and for `selectattr` / `rejectattr` on `role`.
    `None` means the template does not filter by role, so every role is
    emitted. A frozenset, possibly empty, means unknown roles can be
    dropped and must be checked after render. A ChatML generation guard
    (`messages[-1]['role'] != 'assistant'`) is one of those filters: it
    does not name an allow-list, but it is not ignored.

    Args:
        chat_template: Resolved Jinja source.

    Returns:
        Roles named by positive `==` / `in` tests or by `selectattr`
        on `role`. An empty set means the template filters on role, but
        the kept names are not a literal allow-list (`not in`, `!=`,
        `not (role == ...)`, `message.role in allowed_roles`,
        `message.role == some_var`, or `rejectattr`). `None` means
        there is no role filter.

    """
    return _template_roles(chat_template).handled


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
    """Copy `message` without tool calls or multimodal parts.

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


def _is_text_part(part: Mapping[str, Any]) -> bool:
    part_type = part.get("type")
    return part_type in _TEXT_PART_TYPES or (
        part_type is None and isinstance(part.get("text"), str)
    )


def _blank_text_payload(message: Mapping[str, Any]) -> dict[str, Any]:
    """Copy `message` with content text cleared and everything else kept."""
    updated = dict(message)
    content = updated.get("content")
    if isinstance(content, str):
        updated["content"] = ""
        return updated
    if isinstance(content, list):
        parts: list[Any] = []
        for part in content:
            if isinstance(part, str):
                parts.append("")
                continue
            if isinstance(part, dict) and _is_text_part(part):
                blank = dict(part)
                if isinstance(blank.get("text"), str):
                    blank["text"] = ""
                parts.append(blank)
                continue
            parts.append(part)
        updated["content"] = parts
    return updated


def _inject_probe_token(message: dict[str, Any], token: str) -> bool:
    """Append `token` to content text so a render can prove inclusion.

    Reasoning is not probed. Templates often keep it out of the prompt, and
    a missing reasoning string is not a dropped turn.
    """
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
    return injected


def _warn_ignored_standard_roles(roles: Sequence[str]) -> None:
    """Log once when a template ignores a standard chat role."""
    shown = tuple(_unique_roles(roles))
    if not shown:
        return
    logger.warning_once(
        "Chat template produced no prompt content for non-empty "
        "standard message role(s) %s. The request is still accepted; "
        "this template does not render that role.",
        shown,
    )


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


class _RenderBudget:
    """At most `_MAX_EXTRA_RENDERS` calls. Further probes are unresolved."""

    def __init__(self) -> None:
        self.remaining = _MAX_EXTRA_RENDERS

    def render(
        self,
        render_text: Callable[[Sequence[Mapping[str, Any]]], str],
        messages: Sequence[Mapping[str, Any]],
    ) -> str | None:
        if self.remaining <= 0:
            return None
        self.remaining -= 1
        try:
            rendered = render_text(messages)
        except Exception:
            return None
        if not isinstance(rendered, str):
            return None
        return rendered


def _replace_messages(
    conversation: Sequence[Mapping[str, Any]],
    indexes: set[int],
    transform: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [
        transform(message) if index in indexes else message
        for index, message in enumerate(conversation)
    ]


def _dropped_by_omission(
    conversation: Sequence[Mapping[str, Any]],
    suspects: Sequence[tuple[int, str]],
    baseline: str,
    render_text: Callable[[Sequence[Mapping[str, Any]]], str],
    budget: _RenderBudget,
    transform: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> set[int]:
    """Indexes whose transformed copy leaves `baseline` unchanged.

    A group that still changes the prompt is split only while `budget`
    has renders left. When the budget is spent, the rest are kept.
    """

    def dropped_in(subgroup: Sequence[tuple[int, str]]) -> set[int]:
        if not subgroup:
            return set()
        indexes = {index for index, _role in subgroup}
        alternate = budget.render(
            render_text,
            _replace_messages(conversation, indexes, transform),
        )
        if alternate is None:
            return set()
        if alternate == baseline:
            return indexes
        if len(subgroup) == 1:
            return set()
        mid = len(subgroup) // 2
        return dropped_in(subgroup[:mid]) | dropped_in(subgroup[mid:])

    return dropped_in(list(suspects))


def _indexes_missing_text_marker(
    conversation: Sequence[Mapping[str, Any]],
    suspects: Sequence[tuple[int, str]],
    baseline: str,
    render_text: Callable[[Sequence[Mapping[str, Any]]], str],
    budget: _RenderBudget,
) -> set[int]:
    """Suspects whose text is absent, confirmed by clearing that text.

    One render appends a marker to every suspect. A marker in the prompt
    means that content was rendered. A missing marker is not enough: the
    template may keep the original text and clip the suffix. Those indexes
    are dropped only when clearing the original text leaves the prompt
    unchanged.
    """
    try:
        probed = copy.deepcopy(list(conversation))
    except Exception:
        logger.warning(
            "Could not copy the conversation to check dropped roles.",
            exc_info=True,
        )
        return _dropped_by_omission(
            conversation,
            suspects,
            baseline,
            render_text,
            budget,
            _blank_text_payload,
        )

    prefix = "vllmprobe" + random_uuid().replace("-", "")[:12]
    tokens: list[tuple[int, str]] = []
    for index, _role in suspects:
        message = probed[index]
        if not isinstance(message, dict):
            continue
        token = _role_probe_token(prefix, index)
        if _inject_probe_token(message, token):
            tokens.append((index, token))
    if not tokens:
        return _dropped_by_omission(
            conversation,
            suspects,
            baseline,
            render_text,
            budget,
            _blank_text_payload,
        )

    rendered = budget.render(render_text, probed)
    if rendered is None:
        return set()
    folded = rendered.lower()
    missing_ids = {index for index, token in tokens if token.lower() not in folded}
    if not missing_ids:
        return set()
    missing = [(index, role) for index, role in suspects if index in missing_ids]
    return _dropped_by_omission(
        conversation,
        missing,
        baseline,
        render_text,
        budget,
        _blank_text_payload,
    )


def _standard_role_needs_probe(info: _TemplateRoles, role: str) -> bool:
    """Whether the AST leaves this standard role unresolved.

    Absence from a literal allow-list, or a name on an exclusion filter,
    does not mean the content was dropped. A formatting check on
    `assistant` still prints every other role. The caller probes, and
    warns only when the rendered prompt loses the content. An empty
    allow-list is a dynamic filter and does not put every standard role
    on that list.
    """
    handled = info.handled
    if role in info.excluded:
        return True
    if not handled:
        return False
    return role not in handled


def _named_by_template(info: _TemplateRoles, role: str) -> bool:
    handled = info.handled
    if not handled or role in info.excluded:
        return False
    return role in handled


def assert_chat_template_rendered_messages(
    conversation: Sequence[Mapping[str, Any]],
    chat_template: str,
    render_text: Callable[[Sequence[Mapping[str, Any]]], str],
    rendered: str | None = None,
) -> None:
    """Raise if a non-empty message contributed no prompt content.

    Roles named by the template are trusted. Any other non-standard role
    with content is probed: one render appends a marker to each suspect.
    A missing marker is checked by clearing the original text. Tool calls
    and multimodal parts are checked by stripping them in place, so a
    neighbor lookup by index stays put. Each check stops after
    `_MAX_EXTRA_RENDERS` extra renders and keeps whatever is still
    unresolved.

    Standard roles from the OpenAI message schema (`system`, `user`,
    `assistant`, `tool`, `developer`, and legacy `function`) are
    not rejected. When a non-empty literal set omits one, or an exclusion
    filter names it, the turn is probed with the same budget as a custom
    role. A warning is logged only when that probe shows the content is
    absent. Dynamic filters do not probe every standard role. Empty and
    whitespace-only messages are not an error.

    Args:
        conversation: Messages passed to the template, after rewrites such
            as developer → system.
        chat_template: Resolved Jinja source.
        render_text: Renders a conversation to prompt text with the same
            template options as the request.
        rendered: Prompt text already produced for `conversation`, when
            the caller has it. Tokenized calls leave this empty and one
            text baseline is rendered only if a probe is needed.

    Raises:
        VLLMValidationError: A non-empty non-standard message produced no
            prompt content. The error names the role and is returned as
            HTTP 400.

    """
    info = _template_roles(chat_template)
    if info.handled is None:
        return

    hard: list[tuple[int, str]] = []
    soft: list[tuple[int, str]] = []
    for index, message in enumerate(conversation):
        role = message.get("role")
        if not isinstance(role, str) or _named_by_template(info, role):
            continue
        if not message_has_nonempty_payload(message):
            continue
        if role in _STANDARD_CHAT_ROLES:
            if _standard_role_needs_probe(info, role):
                soft.append((index, role))
        else:
            hard.append((index, role))
    if not hard and not soft:
        return

    budget = _RenderBudget()
    baseline = (
        rendered
        if isinstance(rendered, str)
        else budget.render(render_text, conversation)
    )
    if baseline is None:
        return

    # One probe pass so a standard role and a custom role share the budget.
    dropped = _roles_with_no_prompt_contribution(
        conversation, hard + soft, baseline, render_text, budget
    )
    ignored = [role for role in dropped if role in _STANDARD_CHAT_ROLES]
    rejected = [role for role in dropped if role not in _STANDARD_CHAT_ROLES]
    if ignored:
        _warn_ignored_standard_roles(ignored)
    if rejected:
        raise _unrendered_role_error(rejected)


def _roles_with_no_prompt_contribution(
    conversation: Sequence[Mapping[str, Any]],
    suspects: Sequence[tuple[int, str]],
    baseline: str,
    render_text: Callable[[Sequence[Mapping[str, Any]]], str],
    budget: _RenderBudget,
) -> list[str]:
    """Roles that added neither content text nor a structural payload.

    A missing content token is not a drop when the same turn's tool call
    or image still shows up in the prompt. Turns the budget cannot settle
    are kept.
    """
    text_suspects = [
        (index, role)
        for index, role in suspects
        if _has_nonempty_text(conversation[index].get("content"))
    ]
    structural_suspects = [
        (index, role)
        for index, role in suspects
        if _has_structural_payload(conversation[index])
    ]
    if not text_suspects and not structural_suspects:
        return []

    text_missing = (
        _indexes_missing_text_marker(
            conversation, text_suspects, baseline, render_text, budget
        )
        if text_suspects
        else set()
    )
    struct_missing = (
        _dropped_by_omission(
            conversation,
            structural_suspects,
            baseline,
            render_text,
            budget,
            _strip_structural_payload,
        )
        if structural_suspects
        else set()
    )
    struct_ids = {index for index, _role in structural_suspects}
    dropped: list[str] = []
    for index, role in suspects:
        if index in struct_ids and index in struct_missing:
            dropped.append(role)
            continue
        contributed = index in struct_ids and index not in struct_missing
        if index in text_missing and not contributed:
            dropped.append(role)
    return _unique_roles(dropped)
