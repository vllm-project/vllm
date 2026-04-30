# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exception types used by the fault tolerance framework.

These exist so callers that detect a fault on a hot path (e.g.,
`dp_utils._run_ar`) can raise a typed exception that the supervisor's
`on_step_error` hook can recognize without resorting to string parsing.

This is intentionally small. Most cross-process signals travel through
typed fields on the existing output objects (e.g.,
`ModelRunnerOutput.fault_signal`), not exceptions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.fault_tolerance.types import FaultSignal


class FaultToleranceError(Exception):
    """Base class for fault-tolerance exceptions.

    The optional `signal` attribute carries a typed `FaultSignal` so the
    supervisor's `on_step_error` hook can read structured detail (kind +
    payload) without parsing the exception message.
    """

    def __init__(
        self,
        message: str = "",
        signal: FaultSignal | None = None,
    ) -> None:
        super().__init__(message)
        self.signal = signal


class DpAllReduceFaultError(FaultToleranceError):
    """Raised when the DP synchronization all-reduce in
    `vllm.v1.worker.dp_utils._run_ar` returns a failure result.

    The supervisor's `on_step_error` hook can read `self.signal` to surface
    the failure as a `FaultInfo` on the bus.
    """

    def __init__(self, signal: FaultSignal | None = None) -> None:
        detail = signal.detail if signal is not None else ""
        super().__init__(f"DP all-reduce failed: {detail}", signal=signal)


class EngineLoopPausedError(FaultToleranceError):
    """Raised by the busy loop when a paused state has propagated across DP
    ranks (e.g., one rank failed; healthy ranks must pause coordinated).

    Compatibility shim: existing PR #38534 raises an exception with this
    name and pattern-matches its prefix in the executor. The redesign uses
    typed exceptions instead — callers read `self.signal` rather than the
    message text.
    """
