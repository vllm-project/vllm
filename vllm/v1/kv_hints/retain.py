# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ``kv.retain`` KV-hint action: per-request retention directives for
priority-based KV-cache eviction.

Payload: ``{"scope": str | null, "directives": [directive, ...]}`` where a
directive is ``{"start", "end", "priority", "duration"}`` or
``{"covers_output": true, "priority", "duration"}``. Priority 1-100 protects
the token range, 0 releases it; ranges no directive covers are left as they
are. Priorities must be non-increasing along the token axis, because a
prefix cache can only ever serve a block whose predecessors survived.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from vllm.v1.kv_hints.protocol import KvHintAction, KvHintsEnvelope

RETAIN_ACTION_TYPE = "kv.retain"
RETAIN_ACTION_VERSION = "1.0"
MAX_PRIORITY = 100
MAX_DURATION_S = 30 * 24 * 3600.0
MAX_ACTIONS = 16
MAX_DIRECTIVES = 128

_RETAIN_ACTION_MAJOR = 1
_DIRECTIVE_FIELDS = frozenset({"start", "end", "priority", "duration", "covers_output"})


@dataclass(frozen=True, slots=True)
class RetainDirective:
    """One token range of a request and how strongly to keep its blocks.

    ``end`` None means the end of the sequence, generated tokens included.
    ``duration`` is seconds until the protection lapses, None for never. A
    ``covers_output`` directive has no position of its own: the engine
    resolves it to the generated tail once the prompt length is known.
    """

    priority: int
    start: int = 0
    end: int | None = None
    duration: float | None = None
    covers_output: bool = False


@dataclass(frozen=True, slots=True)
class RetainDirectives:
    scope: str | None
    directives: tuple[RetainDirective, ...]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _directive_from_payload(item: Any) -> RetainDirective:
    if not isinstance(item, dict):
        raise ValueError("each directive must be an object")
    unknown = set(item) - _DIRECTIVE_FIELDS
    if unknown:
        raise ValueError(f"unknown directive fields: {sorted(unknown)}")
    priority_val = item.get("priority")
    if not _is_int(priority_val):
        raise ValueError(f"priority must be an integer in [0, {MAX_PRIORITY}]")
    priority = cast(int, priority_val)
    if not 0 <= priority <= MAX_PRIORITY:
        raise ValueError(f"priority must be an integer in [0, {MAX_PRIORITY}]")
    start_val = item.get("start", 0)
    if not _is_int(start_val):
        raise ValueError("start must be a non-negative integer")
    start = cast(int, start_val)
    if start < 0:
        raise ValueError("start must be a non-negative integer")
    end_val = item.get("end")
    if end_val is not None:
        if not _is_int(end_val):
            raise ValueError("end must be null or an integer greater than start")
        end = cast(int, end_val)
        if end <= start:
            raise ValueError("end must be null or an integer greater than start")
    else:
        end = None
    duration_val = item.get("duration")
    duration: float | None = None
    if duration_val is not None:
        message = f"duration must be null or a number in [0, {MAX_DURATION_S:g}]"
        if isinstance(duration_val, bool) or not isinstance(duration_val, (int, float)):
            raise ValueError(message)
        try:
            duration = float(duration_val)
        except (TypeError, OverflowError):
            raise ValueError(message) from None
        if not (math.isfinite(duration) and 0 <= duration <= MAX_DURATION_S):
            raise ValueError(message)
    covers_output = item.get("covers_output", False)
    if not isinstance(covers_output, bool):
        raise ValueError("covers_output must be a boolean")
    if covers_output and ("start" in item or "end" in item):
        raise ValueError("covers_output cannot be combined with start or end")
    return RetainDirective(
        priority=priority,
        start=start,
        end=end,
        duration=duration,
        covers_output=covers_output,
    )


def _check_monotonic(directives: Sequence[RetainDirective]) -> None:
    ordered = sorted(
        enumerate(directives),
        key=lambda pair: (
            float("inf") if pair[1].covers_output else pair[1].start,
            pair[0],
        ),
    )
    prev: RetainDirective | None = None
    for _, d in ordered:
        if prev is not None and d.priority > prev.priority:
            raise ValueError(
                "priorities must be non-increasing across token positions "
                f"(prefix-cache constraint): directive at start={d.start} has "
                f"priority={d.priority} > earlier directive at start={prev.start} "
                f"with priority={prev.priority}"
            )
        prev = d


def _major_version(version: str) -> int | None:
    head = version.split(".", 1)[0]
    return int(head) if head.isdigit() else None


def _parse_action(
    action: KvHintAction, max_directives: int, strict: bool
) -> RetainDirectives:
    if _major_version(action.action_version) != _RETAIN_ACTION_MAJOR:
        raise ValueError(
            f"unsupported {RETAIN_ACTION_TYPE} action_version {action.action_version!r}"
        )
    payload = action.payload
    scope = payload.get("scope")
    if scope is not None and not isinstance(scope, str):
        raise ValueError("scope must be a string or null")
    raw = payload.get("directives")
    if not isinstance(raw, list):
        raise ValueError("directives must be a list")
    if len(raw) > max_directives:
        if strict:
            raise ValueError(f"at most {MAX_DIRECTIVES} directives per envelope")
        raw = raw[:max_directives]
    directives = tuple(_directive_from_payload(item) for item in raw)
    _check_monotonic(directives)
    return RetainDirectives(scope=scope, directives=directives)


def parse_retain_actions(
    envelope: KvHintsEnvelope | None, *, strict: bool
) -> RetainDirectives | None:
    """Collect the ``kv.retain`` actions of an envelope into one claim.

    Args:
        envelope: The request's KV hints, or None.
        strict: Raise ``ValueError`` on the first invalid action (frontend
            validation). When False, an invalid action is skipped and the
            valid ones still apply, as the KV-hints RFC asks of executors;
            actions beyond ``MAX_ACTIONS`` and directives beyond
            ``MAX_DIRECTIVES`` are dropped instead of rejected.

    Returns:
        The scope and the concatenated directives, or None when the envelope
        carries no usable ``kv.retain`` action.

    Raises:
        ValueError: If ``strict`` and the ``kv.retain`` actions are invalid or
            exceed the caps.

    """
    if envelope is None:
        return None
    scope: str | None = None
    found = False
    num_actions = 0
    directives: list[RetainDirective] = []
    for action in envelope.actions:
        if action.action_type != RETAIN_ACTION_TYPE:
            continue
        num_actions += 1
        if num_actions > MAX_ACTIONS:
            if strict:
                raise ValueError(
                    f"at most {MAX_ACTIONS} {RETAIN_ACTION_TYPE} actions per envelope"
                )
            break
        try:
            parsed = _parse_action(action, MAX_DIRECTIVES - len(directives), strict)
            if found and parsed.scope != scope:
                raise ValueError(
                    f"{RETAIN_ACTION_TYPE} actions in one envelope must share a scope"
                )
        except (ValueError, TypeError, OverflowError) as exc:
            if strict:
                raise ValueError(f"action {action.action_id!r}: {exc}") from exc
            continue
        if not found:
            scope = parsed.scope
            found = True
        directives.extend(parsed.directives)
    if not found:
        return None
    if strict:
        _check_monotonic(directives)
    return RetainDirectives(scope=scope, directives=tuple(directives))
