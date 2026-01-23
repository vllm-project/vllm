"""Request Journey Event tracing for lifecycle observability.

This module provides sparse lifecycle event tracking for requests as they
move through the vLLM scheduler. Events are emitted at key state transitions
with full progress snapshots, enabling detailed request journey analysis.
"""

import enum
from typing import TYPE_CHECKING, Literal

import msgspec

if TYPE_CHECKING:
    from vllm.v1.request import RequestStatus


class RequestJourneyEventType(enum.IntEnum):
    """Request lifecycle event types.

    Events are emitted at key state transitions during a request's journey
    through the scheduler. Each event captures a full progress snapshot.
    """
    QUEUED = 1  # Added to scheduler waiting queue
    SCHEDULED = 2  # Moved to RUNNING (first time or resume after preemption)
    FIRST_TOKEN = 3  # First decode token produced
    PREEMPTED = 4  # Moved to PREEMPTED status
    FINISHED = 5  # Request completed (terminal state)
    DEPARTED = 6  # Response left system (stretch goal, not yet implemented)


class ScheduleKind(enum.IntEnum):
    """Type of scheduling transition for SCHEDULED events."""
    FIRST = 1  # First time scheduled (WAITING → RUNNING)
    RESUME = 2  # Resumed after preemption (PREEMPTED → RUNNING)


class RequestJourneyEvent(msgspec.Struct, frozen=True):
    """A single request lifecycle event with full progress snapshot.

    Each event captures the complete state of a request at a specific
    lifecycle transition, including accurate progress counters that
    survive preemption.

    Progress Tracking:
    - prefill_done_tokens: High-water mark of prompt tokens processed
      (survives preemption, tracked via scheduler-side dict)
    - decode_done_tokens: Output tokens generated (from num_output_tokens)
    - phase: Current processing phase (PREFILL or DECODE)

    Scheduler Context:
    - scheduler_step: Monotonic counter from Scheduler.scheduler_step_counter
      (None only for QUEUED events before first schedule)
    """
    # Identity
    request_id: str
    event_type: RequestJourneyEventType
    ts_monotonic: float  # time.monotonic() supplied by scheduler

    # Scheduler context (None only for QUEUED before first schedule)
    scheduler_step: int | None

    # Progress snapshot (accurate even after preemption)
    prefill_done_tokens: int  # Prompt tokens processed (from _journey_prefill_hiwater)
    prefill_total_tokens: int  # Total prompt tokens (len(prompt_token_ids))
    decode_done_tokens: int  # Output tokens generated (num_output_tokens)
    decode_max_tokens: int  # Max generation tokens (max_tokens)
    phase: Literal["PREFILL", "DECODE"]  # "DECODE" if num_output_tokens > 0

    # Lifecycle tracking
    num_preemptions_so_far: int  # Number of preemptions for this request

    # Event-specific fields (None when not applicable)
    schedule_kind: ScheduleKind | None  # SCHEDULED only: FIRST or RESUME
    finish_status: Literal["stopped", "length", "aborted", "ignored",
                           "error"] | None  # FINISHED only


def _map_finish_status(
    status: "RequestStatus",
) -> Literal["stopped", "length", "aborted", "ignored", "error"]:
    """Map RequestStatus terminal state to journey event finish_status string.

    Args:
        status: Terminal RequestStatus (FINISHED_*)

    Returns:
        Human-readable finish status string

    Raises:
        ValueError: If status is not a terminal FINISHED_* status
    """
    # Import here to avoid circular dependency
    from vllm.v1.request import RequestStatus

    mapping = {
        RequestStatus.FINISHED_STOPPED: "stopped",
        RequestStatus.FINISHED_LENGTH_CAPPED: "length",
        RequestStatus.FINISHED_ABORTED: "aborted",
        RequestStatus.FINISHED_IGNORED: "ignored",
        RequestStatus.FINISHED_ERROR: "error",
    }

    if status not in mapping:
        raise ValueError(f"Invalid terminal status: {status}")

    return mapping[status]
