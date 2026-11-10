# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from .deepseek_v41_encoding import REASONING_EFFORT_MAPPINGS


def _parse_controls(
    kwargs: dict[str, Any], effort: Any = None
) -> tuple[bool | None, Any]:
    toggle = None
    for key in ("thinking", "enable_thinking"):
        if key in kwargs:
            toggle = kwargs[key]
            if type(toggle) is not bool:
                raise ValueError(f"template kwarg `{key}` must be a boolean")
            break
    if toggle is False:
        return False, None
    if effort is None:
        effort = kwargs.get("reasoning_effort")
    if effort is not None and (
        isinstance(effort, bool) or not isinstance(effort, (str, int, float))
    ):
        raise ValueError("reasoning_effort must be a string or number")
    if effort == "none":
        return toggle is True, None
    return (True if effort is not None else toggle), effort


def resolve_reasoning_kwargs(
    kwargs: dict[str, Any],
    defaults: dict[str, Any] | None = None,
    *,
    effort: Any = None,
) -> dict[str, Any]:
    """Resolve DeepSeek V4.1 controls in request, deployment, model order.

    The explicit toggle wins within each source. Only enabled sources inherit
    effort from enabled defaults; a disabled request never inherits effort.
    """
    thinking, selected = _parse_controls(kwargs, effort)
    default_thinking, default_effort = _parse_controls(defaults or {})
    if thinking is None:
        thinking, selected = default_thinking, default_effort
    elif thinking and selected is None and default_thinking:
        selected = default_effort
    if thinking is None:
        thinking = True
    if not thinking:
        return {
            "thinking": False,
            "enable_thinking": False,
            "reasoning_effort": "none",
        }

    if selected is None:
        selected = "high"
    if isinstance(selected, str):
        selected = REASONING_EFFORT_MAPPINGS.get(selected)
    if type(selected) is not int or not 1 <= selected <= 100:
        raise ValueError(
            "DeepSeek V4.1 reasoning_effort must be low, high, xhigh, max, "
            "or an integer within [1, 100] in chat_template_kwargs"
        )
    return {
        "thinking": True,
        "enable_thinking": True,
        "reasoning_effort": selected,
    }
