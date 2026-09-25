# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal, cast

from vllm.logger import init_logger

logger = init_logger(__name__)

ReasoningEffortRounding = Literal["down", "up"]

ReasoningEffortValue = Literal[
    "none", "minimal", "low", "medium", "high", "xhigh", "max"
]

REASONING_EFFORT_LADDER = ("minimal", "low", "medium", "high", "xhigh", "max")
"""Reasoning effort levels ordered from least to most effort.

The special value "none" disables reasoning entirely, so it is not part of
the ladder and is never remapped.
"""


def validate_supported_reasoning_efforts(supported: list[str] | None) -> None:
    """Validate the `--supported-reasoning-efforts` values.

    Raises:
        ValueError: If the list is empty or contains values outside
            `REASONING_EFFORT_LADDER`.
    """
    if supported is None:
        return
    if not supported:
        raise ValueError("--supported-reasoning-efforts must not be empty")
    invalid = [effort for effort in supported if effort not in REASONING_EFFORT_LADDER]
    if invalid:
        raise ValueError(
            f"Invalid --supported-reasoning-efforts values: {invalid}. "
            f"Choose from {list(REASONING_EFFORT_LADDER)} ('none' always "
            "passes through and cannot be listed)."
        )


def normalize_reasoning_effort(
    effort: ReasoningEffortValue | None,
    supported: list[str] | None,
    rounding: ReasoningEffortRounding = "down",
) -> ReasoningEffortValue | None:
    """Map an unsupported reasoning effort to the nearest supported level.

    Levels are compared on `REASONING_EFFORT_LADDER`. With `rounding="down"`,
    the highest supported level at or below the requested one is chosen,
    falling back to the lowest supported level when nothing is at or below.
    With `rounding="up"`, the lowest supported level at or above the requested
    one is chosen, falling back to the highest supported level.

    `None` and `"none"` always pass through unchanged, as do values already
    in `supported` and values outside the ladder. No-op when `supported` is
    `None` (the default server configuration).
    """
    if not supported or effort is None or effort == "none" or effort in supported:
        return effort
    if effort not in REASONING_EFFORT_LADDER:
        return effort
    requested_idx = REASONING_EFFORT_LADDER.index(effort)
    ordered = sorted(set(supported), key=REASONING_EFFORT_LADDER.index)
    if rounding == "up":
        candidates = [
            e for e in ordered if REASONING_EFFORT_LADDER.index(e) >= requested_idx
        ]
        effective = candidates[0] if candidates else ordered[-1]
    else:
        candidates = [
            e for e in ordered if REASONING_EFFORT_LADDER.index(e) <= requested_idx
        ]
        effective = candidates[-1] if candidates else ordered[0]
    logger.warning_once(
        "reasoning_effort=%r is not supported by this server "
        "(--supported-reasoning-efforts=%s); using %r (rounding=%s).",
        effort,
        ",".join(ordered),
        effective,
        rounding,
    )
    return cast(ReasoningEffortValue, effective)
