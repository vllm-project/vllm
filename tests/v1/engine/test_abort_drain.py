# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for abort-and-drain shutdown behavior in EngineCoreProc.

These tests validate that:
1. In-flight requests are aborted during graceful shutdown.
2. Abort outputs are sent to clients before the engine core exits.
3. The output thread is drained (joined) before shutdown completes.
"""

import queue
import threading
from unittest.mock import MagicMock

import pytest

from vllm.v1.engine import EngineCoreOutputs, FinishReason
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.request import RequestStatus


@pytest.fixture
def mock_engine_core():
    """Create a minimal mock of EngineCoreProc for testing shutdown."""
    engine = MagicMock(spec=EngineCoreProc)
    engine.output_queue = queue.Queue()

    # Use real methods for the ones we're testing.
    engine._abort_and_drain_outputs = EngineCoreProc._abort_and_drain_outputs.__get__(
        engine
    )
    engine._send_abort_outputs = EngineCoreProc._send_abort_outputs.__get__(engine)

    return engine


class TestAbortAndDrainOutputs:
    def test_aborts_all_in_flight_requests(self, mock_engine_core):
        """Verify all in-flight requests are aborted via scheduler."""
        mock_engine_core.scheduler.finish_requests.return_value = [
            ("req-1", 0),
            ("req-2", 0),
        ]
        # No output thread.
        del mock_engine_core.output_thread

        mock_engine_core._abort_and_drain_outputs()

        mock_engine_core.scheduler.finish_requests.assert_called_once_with(
            None, RequestStatus.FINISHED_ABORTED
        )

    def test_sends_abort_outputs_to_clients(self, mock_engine_core):
        """Verify abort outputs are placed in the output queue."""
        mock_engine_core.scheduler.finish_requests.return_value = [
            ("req-1", 0),
            ("req-2", 1),
        ]
        del mock_engine_core.output_thread

        mock_engine_core._abort_and_drain_outputs()

        # Should have 2 output entries (one per client_index).
        items = []
        while not mock_engine_core.output_queue.empty():
            items.append(mock_engine_core.output_queue.get_nowait())

        # Filter out ENGINE_CORE_DEAD sentinel if present.
        output_items = [i for i in items if not isinstance(i, bytes)]
        assert len(output_items) == 2

        # Check that abort outputs have the right finish reason.
        for client_index, eco in output_items:
            assert isinstance(eco, EngineCoreOutputs)
            for output in eco.outputs:
                assert output.finish_reason == FinishReason.ABORT

    def test_drains_output_thread(self, mock_engine_core):
        """Verify output thread is signaled and joined."""
        mock_engine_core.scheduler.finish_requests.return_value = []
        mock_thread = MagicMock(spec=threading.Thread)
        mock_thread.is_alive.return_value = False
        mock_engine_core.output_thread = mock_thread

        mock_engine_core._abort_and_drain_outputs()

        # ENGINE_CORE_DEAD sentinel should be in the queue.
        sentinel = mock_engine_core.output_queue.get_nowait()
        assert sentinel == EngineCoreProc.ENGINE_CORE_DEAD

        # Output thread should be joined.
        mock_thread.join.assert_called_once_with(timeout=5.0)

    def test_no_output_thread_still_works(self, mock_engine_core):
        """Verify graceful handling when output_thread doesn't exist."""
        mock_engine_core.scheduler.finish_requests.return_value = [
            ("req-1", 0),
        ]
        del mock_engine_core.output_thread

        # Should not raise.
        mock_engine_core._abort_and_drain_outputs()

    def test_no_requests_to_abort(self, mock_engine_core):
        """Verify clean shutdown when no requests are in flight."""
        mock_engine_core.scheduler.finish_requests.return_value = []
        mock_thread = MagicMock(spec=threading.Thread)
        mock_thread.is_alive.return_value = False
        mock_engine_core.output_thread = mock_thread

        mock_engine_core._abort_and_drain_outputs()

        # Should still signal the output thread to drain.
        sentinel = mock_engine_core.output_queue.get_nowait()
        assert sentinel == EngineCoreProc.ENGINE_CORE_DEAD
        mock_thread.join.assert_called_once_with(timeout=5.0)
