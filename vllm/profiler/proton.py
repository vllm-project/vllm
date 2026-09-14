# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from tempfile import NamedTemporaryFile
from types import ModuleType

from vllm.config.profiler import ProtonOutputFormat


@dataclass
class _PendingPhase:
    phase: int
    output_path: str | None
    future: Future[None] | None = None
    exported: bool = False


class ProtonPhaseManager:
    """Own activity phases within a retained Proton session.

    Construct once for a new session and reuse across captures and intervals.
    Management methods must be called serially by the worker thread, which owns
    phase advancement and the pending queue. In periodic mode,
    a single background exporter reads and clears completed phases. The caller
    owns session activation and finalization. Native periodic flushing must be
    disabled: it would also export and clear these phases.
    """

    def __init__(
        self,
        proton: ModuleType,
        session_id: int,
        output_format: ProtonOutputFormat | None,
        *,
        asynchronous: bool = False,
    ) -> None:
        self._proton = proton
        self._session_id = session_id
        self._output_format = output_format or "hatchet"
        self._phase = 0
        self._pending: deque[_PendingPhase] = deque()
        self._flushed_through = -1
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="proton-export")
            if asynchronous
            else None
        )

    def discard(self) -> None:
        """Finish the current phase without exporting its activity."""
        self._finish(output_path=None)

    def export(self, output_path: str) -> None:
        """Finish the current phase and export it under the given prefix."""
        self._finish(output_path=output_path)

    def rotate(self, output_path: str) -> None:
        """Seal the current phase without interrupting GPU collection."""
        self._advance(output_path)

    def _advance(self, output_path: str | None) -> None:
        phase = self._phase
        # A failed advance leaves the current phase queued with its original path.
        if not self._pending or self._pending[-1].phase != phase:
            self._pending.append(_PendingPhase(phase, output_path))
        self._phase = self._proton.data.advance_phase(self._session_id)

    def poll(self, *, wait: bool = False) -> None:
        """Export complete phases, retaining failed exports for retry.

        ``wait=True`` waits for submitted exports, not incomplete GPU phases.
        Use ``drain()`` to synchronously flush and export all pending activity.
        """
        for pending in self._pending:
            if pending.phase >= self._phase:
                break
            if pending.future is not None:
                continue
            if pending.phase > self._flushed_through and not (
                self._proton.data.is_phase_complete(self._session_id, pending.phase)
            ):
                break
            if self._executor is None:
                # Synchronous managers are drained only after a successful flush.
                self._export_phase(pending)
                pending.future = Future()
                pending.future.set_result(None)
            else:
                pending.future = self._executor.submit(self._export_phase, pending)

        while self._pending:
            pending = self._pending[0]
            future = pending.future
            if future is None or (not wait and not future.done()):
                break
            try:
                future.result()
            except Exception:
                pending.future = None
                raise
            self._pending.popleft()

    def _export_phase(self, pending: _PendingPhase) -> None:
        if pending.output_path is not None and not pending.exported:
            self._write(pending.phase, pending.output_path)
            pending.exported = True
        self._proton.data.clear(self._session_id, pending.phase)

    def _finish(self, output_path: str | None) -> None:
        try:
            self._advance(output_path)
        finally:
            self._proton.deactivate(session=self._session_id, flushing=True)
        self._flushed_through = self._phase - 1
        self.poll(wait=True)

    def drain(self) -> None:
        """Synchronously flush and export all retained phases, waiting for writes.

        Call before starting another run or finalizing the session.
        """
        if self._pending:
            # A failed deactivate may already have marked the session inactive.
            self._proton.activate(session=self._session_id)
            try:
                if self._pending[-1].phase == self._phase:
                    self._advance(output_path=None)
            finally:
                self._proton.deactivate(session=self._session_id, flushing=True)
            self._flushed_through = self._phase - 1
            self.poll(wait=True)

    def close(self) -> None:
        """Drain exports and stop the exporter before session finalization."""
        self.drain()
        if self._executor is not None:
            self._executor.shutdown(wait=True)

    def _write(self, phase: int, output_path: str) -> None:
        output_path = f"{output_path}.{self._output_format}"
        with NamedTemporaryFile(
            dir=os.path.dirname(output_path),
            prefix=".proton_",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = temporary.name
        try:
            self._write_data(phase, temporary_path)
            os.replace(temporary_path, output_path)
        finally:
            with suppress(FileNotFoundError):
                os.remove(temporary_path)

    def _write_data(self, phase: int, output_path: str) -> None:
        if self._output_format == "hatchet_msgpack":
            with open(output_path, "wb") as output_file:
                output_file.write(
                    self._proton.data.get_msgpack(self._session_id, phase)
                )
        else:
            with open(output_path, "w", encoding="utf-8") as output_file:
                json.dump(self._proton.data.get(self._session_id, phase), output_file)
