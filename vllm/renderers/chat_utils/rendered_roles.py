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

    ``transformers.utils.chat_template_utils._compile_jinja_template`` is
    private, but it is the only place that environment is built (the
    ``generation`` tag, loop controls, and the ``tojson`` /
    ``raise_exception`` / ``strftime_now`` helpers). The renderer already
    parses through it for content-format detection, and the supported
    transformers range (``>=5.16.1,<5.20``) keeps the helper.
    """
    compiled = hf_chat_utils._compile_jinja_template(chat_template)
    return compiled.environment.parse(chat_template)


def _literal_strings(annotation: Any) -> set[str]:
    """String members of a ``Literal``, including wrapped annotations."""
    if get_origin(annotation) is Literal:
        return {value for value in get_args(annotation) if isinstance(value, str)}
    found: set[str] = set()
    for arg in get_args(annotation):
        found |= _literal_strings(arg)
    return found


def _roles_from_message_params(annotation: Any) -> frozenset[str]:
    """Literal ``role`` values on a chat-message union.

    ``CustomChatCompletionMessageParam.role`` is a free-form string, so
    those stay out of this set and can still be a hard error. Only the
    fixed OpenAI roles are collected. Legacy ``function`` is one of them.
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
        parsed = _parse_chat_template(chat_template)
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


class _RoleConstraint(NamedTuple):
    """How one template path limits ``message.role``.

    ``roles is None`` means the path does not pin the role to literals.
    ``excludes`` means the role must fall outside ``roles``.
    """

    roles: frozenset[str] | None
    excludes: bool


def _any_role() -> _RoleConstraint:
    return _RoleConstraint(None, False)


def _fixed_roles(names: Iterable[str]) -> _RoleConstraint:
    return _RoleConstraint(frozenset(names), False)


def _excluded_roles(names: Iterable[str]) -> _RoleConstraint:
    return _RoleConstraint(frozenset(names), True)


def _and_role_constraints(
    left: _RoleConstraint, right: _RoleConstraint
) -> _RoleConstraint:
    if left.roles is None and not left.excludes:
        return right
    if right.roles is None and not right.excludes:
        return left
    left_roles = left.roles or frozenset()
    right_roles = right.roles or frozenset()
    if not left.excludes and not right.excludes:
        return _fixed_roles(left_roles & right_roles)
    if left.excludes and right.excludes:
        return _excluded_roles(left_roles | right_roles)
    fixed = left_roles if not left.excludes else right_roles
    excluded = right_roles if not left.excludes else left_roles
    return _fixed_roles(fixed - excluded)


def _or_role_constraints(
    left: _RoleConstraint, right: _RoleConstraint
) -> _RoleConstraint:
    if (left.roles is None and not left.excludes) or (
        right.roles is None and not right.excludes
    ):
        return _any_role()
    left_roles = left.roles or frozenset()
    right_roles = right.roles or frozenset()
    if not left.excludes and not right.excludes:
        return _fixed_roles(left_roles | right_roles)
    if left.excludes and right.excludes:
        return _excluded_roles(left_roles & right_roles)
    fixed = left_roles if not left.excludes else right_roles
    excluded = right_roles if not left.excludes else left_roles
    return _excluded_roles(excluded - fixed)


def _not_role_constraint(constraint: _RoleConstraint) -> _RoleConstraint:
    if constraint.roles is None:
        return _any_role()
    if not constraint.excludes:
        return _excluded_roles(constraint.roles)
    return _fixed_roles(constraint.roles)


def _is_field_access(node: jinja2.nodes.Node, field: str) -> bool:
    if isinstance(node, jinja2.nodes.Getattr):
        return node.attr == field
    if isinstance(node, jinja2.nodes.Getitem):
        return isinstance(node.arg, jinja2.nodes.Const) and node.arg.value == field
    return False


def _field_base_name(node: jinja2.nodes.Node) -> str | None:
    base = node.node  # type: ignore[attr-defined]
    if isinstance(base, jinja2.nodes.Name):
        return base.name
    return None


def _role_vars_in(expr: jinja2.nodes.Node) -> set[str]:
    found: set[str] = set()

    def walk(node: jinja2.nodes.Node) -> None:
        if _is_field_access(node, "role"):
            base = _field_base_name(node)
            if base:
                found.add(base)
        for child in node.iter_child_nodes():
            walk(child)

    walk(expr)
    return found


def _compare_role_constraint(node: jinja2.nodes.Compare, var: str) -> _RoleConstraint:
    if not node.ops:
        return _any_role()
    operand = node.ops[0]

    def role_var(expr: jinja2.nodes.Node) -> str | None:
        if _is_field_access(expr, "role"):
            return _field_base_name(expr)
        return None

    left = role_var(node.expr)
    right = role_var(operand.expr)
    op = operand.op
    if op in {"eq", "ne"}:
        if left != var and right != var:
            return _any_role()
        literal = operand.expr if left == var else node.expr
        names = _literal_role_names(literal)
        if names is None:
            return _any_role()
        if op == "eq":
            return _fixed_roles(names)
        return _excluded_roles(names)
    if op in {"in", "notin"} and left == var:
        names = _literal_role_names(operand.expr)
        if names is None:
            return _any_role()
        if op == "in":
            return _fixed_roles(names)
        return _excluded_roles(names)
    return _any_role()


def _constraint_on_var(expr: jinja2.nodes.Node, var: str) -> _RoleConstraint:
    if isinstance(expr, jinja2.nodes.And):
        return _and_role_constraints(
            _constraint_on_var(expr.left, var),
            _constraint_on_var(expr.right, var),
        )
    if isinstance(expr, jinja2.nodes.Or):
        return _or_role_constraints(
            _constraint_on_var(expr.left, var),
            _constraint_on_var(expr.right, var),
        )
    if isinstance(expr, jinja2.nodes.Not):
        return _not_role_constraint(_constraint_on_var(expr.node, var))
    if isinstance(expr, jinja2.nodes.Compare):
        return _compare_role_constraint(expr, var)
    return _any_role()


def _filter_role_constraint(node: jinja2.nodes.Filter) -> _RoleConstraint:
    if node.name not in {"selectattr", "rejectattr"} or not node.args:
        return _any_role()
    attr = node.args[0]
    if not isinstance(attr, jinja2.nodes.Const) or attr.value != "role":
        return _any_role()
    if len(node.args) < 3:
        return _any_role()
    test = node.args[1]
    if (
        not isinstance(test, jinja2.nodes.Const)
        or test.value not in _ROLE_INCLUDE_TESTS
    ):
        return _any_role()
    names = _literal_role_names(node.args[2])
    if names is None:
        return _any_role()
    if node.name == "rejectattr":
        return _excluded_roles(names)
    return _fixed_roles(names)


def _iterator_role_constraint(node: jinja2.nodes.Node) -> _RoleConstraint:
    if isinstance(node, jinja2.nodes.Filter):
        return _and_role_constraints(
            _iterator_role_constraint(node.node),
            _filter_role_constraint(node),
        )
    return _any_role()


def _merged_role_constraint(
    var: str,
    bindings: Mapping[str, _RoleConstraint],
    guards: Sequence[tuple[str, _RoleConstraint]],
) -> _RoleConstraint:
    constraint = bindings.get(var, _any_role())
    for guard_var, guard in guards:
        if guard_var == var:
            constraint = _and_role_constraints(constraint, guard)
    return constraint


def _collect_output_fields(
    expr: jinja2.nodes.Node,
    bindings: Mapping[str, _RoleConstraint],
    guards: Sequence[tuple[str, _RoleConstraint]],
    role_paths: list[_RoleConstraint],
    content_paths: list[_RoleConstraint],
) -> None:
    if isinstance(expr, jinja2.nodes.CondExpr):
        mentioned = _role_vars_in(expr.test)
        true_guards = [
            *guards,
            *[(var, _constraint_on_var(expr.test, var)) for var in mentioned],
        ]
        false_guards = [
            *guards,
            *[
                (var, _not_role_constraint(_constraint_on_var(expr.test, var)))
                for var in mentioned
            ],
        ]
        _collect_output_fields(
            expr.expr1, bindings, true_guards, role_paths, content_paths
        )
        if expr.expr2 is not None:
            _collect_output_fields(
                expr.expr2, bindings, false_guards, role_paths, content_paths
            )
        return
    if _is_field_access(expr, "role"):
        base = _field_base_name(expr)
        if base:
            role_paths.append(_merged_role_constraint(base, bindings, guards))
    if _is_field_access(expr, "content"):
        base = _field_base_name(expr)
        if base:
            content_paths.append(_merged_role_constraint(base, bindings, guards))
    for child in expr.iter_child_nodes():
        _collect_output_fields(child, bindings, guards, role_paths, content_paths)


def _walk_guarded_stmts(
    nodes: Iterable[jinja2.nodes.Node],
    bindings: Mapping[str, _RoleConstraint],
    guards: Sequence[tuple[str, _RoleConstraint]],
    role_paths: list[_RoleConstraint],
    content_paths: list[_RoleConstraint],
) -> None:
    for node in nodes:
        _walk_template_node(node, bindings, role_paths, content_paths, guards)


def _walk_template_node(
    node: jinja2.nodes.Node,
    bindings: Mapping[str, _RoleConstraint],
    role_paths: list[_RoleConstraint],
    content_paths: list[_RoleConstraint],
    guards: Sequence[tuple[str, _RoleConstraint]] = (),
) -> None:
    if isinstance(node, jinja2.nodes.If):
        _walk_if_chain(node, bindings, role_paths, content_paths, guards)
        return
    if isinstance(node, jinja2.nodes.For):
        new_bindings = bindings
        if isinstance(node.target, jinja2.nodes.Name):
            var = node.target.name
            constraint = _iterator_role_constraint(node.iter)
            if node.test is not None:
                constraint = _and_role_constraints(
                    constraint, _constraint_on_var(node.test, var)
                )
            constraint = _and_role_constraints(
                bindings.get(var, _any_role()), constraint
            )
            new_bindings = {**bindings, var: constraint}
        _walk_guarded_stmts(node.body, new_bindings, guards, role_paths, content_paths)
        _walk_guarded_stmts(node.else_, bindings, guards, role_paths, content_paths)
        return
    if isinstance(node, jinja2.nodes.Output):
        for expr in node.nodes:
            _collect_output_fields(expr, bindings, guards, role_paths, content_paths)
        return
    for child in node.iter_child_nodes():
        if isinstance(child, jinja2.nodes.Stmt):
            _walk_template_node(child, bindings, role_paths, content_paths, guards)


def _walk_if_chain(
    node: jinja2.nodes.If,
    bindings: Mapping[str, _RoleConstraint],
    role_paths: list[_RoleConstraint],
    content_paths: list[_RoleConstraint],
    outer_guards: Sequence[tuple[str, _RoleConstraint]] = (),
) -> None:
    """Walk an if/elif/else chain, keeping earlier failed tests."""
    pending: list[tuple[str, _RoleConstraint]] = list(outer_guards)
    for branch in (node, *node.elif_):
        mentioned = _role_vars_in(branch.test)
        passed = pending + [
            (var, _constraint_on_var(branch.test, var)) for var in mentioned
        ]
        _walk_guarded_stmts(branch.body, bindings, passed, role_paths, content_paths)
        pending = pending + [
            (var, _not_role_constraint(_constraint_on_var(branch.test, var)))
            for var in mentioned
        ]
    _walk_guarded_stmts(node.else_, bindings, pending, role_paths, content_paths)


def _constraint_passes_unlisted(constraint: _RoleConstraint) -> bool:
    return constraint.roles is None or constraint.excludes


def _constraint_allows_role(constraint: _RoleConstraint, role: str) -> bool:
    if constraint.roles is None:
        return True
    if constraint.excludes:
        return role not in constraint.roles
    return role in constraint.roles


def _role_output_is_trusted(
    passes_unlisted: bool,
    excluded: frozenset[str],
    content_paths: Sequence[_RoleConstraint],
    role: str,
) -> bool:
    """Whether a dynamically printed role does not need a probe.

    The role header has to be reachable, and so does the message content.
    A template can print ``message['role']`` for every turn and still emit
    content only inside named-role branches. That header alone is not
    enough. When the walker saw no content output, there is no content
    constraint to apply, so the header path stays trusted.
    """
    if not (passes_unlisted and role not in excluded):
        return False
    if not content_paths:
        return True
    return any(_constraint_allows_role(path, role) for path in content_paths)


@lru_cache(maxsize=32)
def _template_role_emission(
    chat_template: str,
) -> tuple[bool, frozenset[str], tuple[_RoleConstraint, ...]]:
    """How a template prints roles, and which content paths exist.

    Returns:
        ``passes_unlisted`` is true when some output prints
        ``message['role']`` on a path that is not limited to the literal
        roles the template compares. ``excluded`` is the roles that cannot
        reach any such output. ``content_paths`` are the role constraints
        on outputs that read message content.

    """
    try:
        parsed = _parse_chat_template(chat_template)
    except Exception:
        logger.warning(
            "Could not parse the chat template while checking message roles.",
            exc_info=True,
        )
        return False, frozenset(), ()

    role_paths: list[_RoleConstraint] = []
    content_paths: list[_RoleConstraint] = []
    _walk_guarded_stmts(parsed.body, {}, (), role_paths, content_paths)

    exclusions: list[frozenset[str]] = []
    for constraint in role_paths:
        if not _constraint_passes_unlisted(constraint):
            continue
        if constraint.excludes and constraint.roles is not None:
            exclusions.append(constraint.roles)
        else:
            exclusions.append(frozenset())
    if not exclusions:
        return False, frozenset(), tuple(content_paths)
    excluded = exclusions[0]
    for item in exclusions[1:]:
        excluded &= item
    return True, excluded, tuple(content_paths)


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
    """Append ``token`` to content text so a render can prove inclusion.

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


def _indexes_dropped_by_omission(
    conversation: Sequence[Mapping[str, Any]],
    suspects: Sequence[tuple[int, str]],
    render_text: Callable[[Sequence[Mapping[str, Any]]], str],
    *,
    strip_payload: bool = False,
) -> set[int]:
    """Indexes whose structural payload leaves the rendered prompt unchanged.

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
        return set()
    if not isinstance(baseline, str):
        logger.warning_once(
            "Chat template probe did not return text; skipping the dropped-role check."
        )
        return set()

    group = list(suspects)
    if not group:
        return set()

    def without(indexes: set[int]) -> list[Mapping[str, Any]]:
        if strip_payload:
            return _messages_without_structural_payload(conversation, indexes)
        return _messages_without(conversation, indexes)

    def dropped_in(
        subgroup: Sequence[tuple[int, str]],
        known_alternate: str | None = None,
        *,
        have_known: bool = False,
    ) -> set[int]:
        if not subgroup:
            return set()
        if have_known:
            alternate = known_alternate
        else:
            alternate = _render_probe_text(
                render_text,
                without({index for index, _role in subgroup}),
            )
        if isinstance(alternate, str) and alternate == baseline:
            return {index for index, _role in subgroup}
        # One comparison cannot tell which member of a changed group was
        # the one that contributed. A single message that changes the
        # prompt, or whose removal fails to render, is not a silent drop.
        if len(subgroup) == 1:
            return set()
        mid = len(subgroup) // 2
        return dropped_in(subgroup[:mid]) | dropped_in(subgroup[mid:])

    alternate = _render_probe_text(
        render_text,
        without({index for index, _role in group}),
    )
    if isinstance(alternate, str) and alternate == baseline:
        return {index for index, _role in group}

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
                return set()
            # ``alternate`` also removed the echoed messages, so it does
            # not describe this smaller group.
            return dropped_in(group)

    return dropped_in(group, alternate, have_known=True)


def _ignored_standard_roles(
    roles: Sequence[str],
    content_paths: Sequence[_RoleConstraint],
    handled: frozenset[str],
) -> list[str]:
    """Standard roles the template can be shown to skip, without re-rendering."""
    ignored: list[str] = []
    for role in _unique_roles(roles):
        if content_paths:
            if any(_constraint_allows_role(path, role) for path in content_paths):
                continue
            ignored.append(role)
        elif handled and role not in handled:
            ignored.append(role)
    return ignored


def assert_chat_template_rendered_messages(
    conversation: Sequence[Mapping[str, Any]],
    chat_template: str,
    render_text: Callable[[Sequence[Mapping[str, Any]]], str],
) -> None:
    """Raise if a non-empty message contributed no prompt content.

    This is a post-condition on chat-template rendering. Roles the template
    names are trusted. So is a role printed with ``message['role']`` when
    that role can also reach an output of message content. A header with
    no such content path is still probed. Any other non-standard role is
    probed: a unique token is appended to its content and the conversation
    is rendered again. The turn is rejected only when it adds nothing to
    the prompt. A rendered tool call or image keeps the turn, and echoed
    text does not hide a dropped tool call or image.

    Standard roles from the OpenAI message schema (``system``, ``user``,
    ``assistant``, ``tool``, ``developer``, and legacy ``function``) are
    not rejected. When the template ignores one, a warning is logged and
    the request continues. That path does not copy the conversation or
    render it again. Empty and whitespace-only messages are not an error.

    Args:
        conversation: Messages passed to the template, after rewrites such
            as developer → system.
        chat_template: Resolved Jinja source.
        render_text: Renders a conversation to prompt text with the same
            template options as the request.

    Raises:
        VLLMValidationError: A non-empty non-standard message produced no
            prompt content. The error names the role and is returned as
            HTTP 400.

    """
    handled = roles_rendered_by_chat_template(chat_template)
    if handled is None:
        return

    passes_unlisted, excluded, content_paths = _template_role_emission(chat_template)
    suspects = [
        (index, role)
        for index, message in enumerate(conversation)
        if isinstance((role := message.get("role")), str)
        and role not in handled
        and not _role_output_is_trusted(passes_unlisted, excluded, content_paths, role)
        and message_has_nonempty_payload(message)
    ]
    if not suspects:
        return

    # A standard role cannot be a 400. Skip the copy and the extra render
    # unless some other role still has to be checked.
    hard = [
        (index, role) for index, role in suspects if role not in _STANDARD_CHAT_ROLES
    ]
    soft = [role for _index, role in suspects if role in _STANDARD_CHAT_ROLES]
    ignored = _ignored_standard_roles(soft, content_paths, handled)
    if ignored:
        _warn_ignored_standard_roles(ignored)
    if not hard:
        return

    dropped = _roles_with_no_prompt_contribution(conversation, hard, render_text)
    if dropped:
        raise _unrendered_role_error(dropped)


def _roles_with_no_prompt_contribution(
    conversation: Sequence[Mapping[str, Any]],
    suspects: Sequence[tuple[int, str]],
    render_text: Callable[[Sequence[Mapping[str, Any]]], str],
) -> list[str]:
    """Roles that added neither content text nor a structural payload.

    A missing content token is not a drop when the same turn's tool call
    or multimodal part still shows up in the prompt.
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
        _indexes_dropped_by_text_probe(conversation, text_suspects, render_text)
        if text_suspects
        else set()
    )
    struct_missing = (
        _indexes_dropped_by_omission(
            conversation,
            structural_suspects,
            render_text,
            strip_payload=True,
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
        # Text that the template omitted is still a drop, unless this turn
        # contributed a tool call or image.
        contributed = index in struct_ids and index not in struct_missing
        if index in text_missing and not contributed:
            dropped.append(role)
    return _unique_roles(dropped)


def _indexes_dropped_by_text_probe(
    conversation: Sequence[Mapping[str, Any]],
    suspects: Sequence[tuple[int, str]],
    render_text: Callable[[Sequence[Mapping[str, Any]]], str],
) -> set[int]:
    try:
        probed = copy.deepcopy(list(conversation))
    except Exception:
        logger.warning(
            "Could not copy the conversation to check dropped roles.",
            exc_info=True,
        )
        return _indexes_dropped_by_omission(conversation, suspects, render_text)

    prefix = "vllmprobe" + random_uuid().replace("-", "")[:12]
    tokens: list[tuple[int, str]] = []
    for index, _role in suspects:
        token = _role_probe_token(prefix, index)
        message = probed[index]
        if isinstance(message, dict) and _inject_probe_token(message, token):
            tokens.append((index, token))
    if not tokens:
        return _indexes_dropped_by_omission(conversation, suspects, render_text)

    try:
        rendered = render_text(probed)
    except Exception:
        logger.warning(
            "Chat template probe failed while checking role(s) %s.",
            [role for _, role in suspects],
            exc_info=True,
        )
        return _indexes_dropped_by_omission(conversation, suspects, render_text)
    if not isinstance(rendered, str):
        return _indexes_dropped_by_omission(conversation, suspects, render_text)

    folded = rendered.lower()
    return {index for index, token in tokens if token.lower() not in folded}
