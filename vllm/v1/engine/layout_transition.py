# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collective preparation for drained, external-launcher layout changes."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import torch.distributed as dist

from vllm.distributed.parallel_state import get_world_group


@dataclass(frozen=True)
class LayoutTransitionRequest:
    # Validation happens inside the collective, never in __post_init__.
    transition_id: str
    tensor_parallel_size: int
    weight_version: str


class LayoutTransitionRejected(ValueError):
    """Collective preflight rejection, retaining the reservation and phase."""


class LayoutTransitionFailed(RuntimeError):
    """A transition stage failed. Admission must remain closed on every rank."""


class LayoutTransitionPhase(Enum):
    PREPARED = "prepared"
    INSTALLING = "installing"
    REFITTING = "refitting"
    WEIGHTS_READY = "weights_ready"
    FAILED = "failed"


def agree_layout_transition(
    request: LayoutTransitionRequest,
    reasons: list[str],
    *,
    phase: str,
    current_layout: tuple[int, int] | None = None,
    details: object = None,
) -> None:
    """Exchange votes on the persistent physical world, including DP peers.

    All physical ranks must call in the same order. A transport failure is not
    a rejection: callers must keep admission closed when consensus is unknown.
    """
    world = get_world_group()
    votes: list = [None] * world.world_size
    dist.all_gather_object(
        votes, (phase, request, current_layout, details, reasons), group=world.cpu_group
    )
    failures = [
        f"rank {rank}: {reason}"
        for rank, (_, _, _, _, errors) in enumerate(votes)
        for reason in errors
    ]
    if any(vote[:4] != votes[0][:4] for vote in votes[1:]):
        failures.insert(
            0, "physical ranks disagree on transition phase, target or current layout"
        )
    if failures:
        raise LayoutTransitionRejected("; ".join(failures))


def run_layout_stage(
    request: LayoutTransitionRequest, phase: str, operation: Callable[[], None]
) -> None:
    """Report local stage errors over the retained physical CPU world.

    Operations with internal collectives still require healthy participating
    ranks and the configured distributed timeouts. This is not rank recovery.
    """
    reasons = []
    try:
        operation()
    except Exception as error:
        reasons.append(f"{type(error).__name__}: {error}")
    try:
        agree_layout_transition(request, reasons, phase=phase)
    except LayoutTransitionRejected as error:
        raise LayoutTransitionFailed(
            f"Layout transition {phase} failed; engine remains reserved: {error}"
        ) from error
