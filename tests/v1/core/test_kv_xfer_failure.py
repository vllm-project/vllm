import logging
from unittest.mock import MagicMock

from vllm.v1.core.sched.scheduler import Scheduler


def test_kv_xfer_finished_unknown_request_no_crash(caplog):
    """
    Test that late KV transfer completions for already-deleted/failed
    requests do not crash the scheduler.
    """
    # Initialize a mock scheduler instance.
    scheduler = MagicMock(spec=Scheduler)

    # Simulate an environment where the requests have already failed
    # and been removed from the scheduler's tracked requests.
    scheduler.requests = {}
    scheduler.connector = None

    # Create a mock KVConnectorOutput with dummy request IDs.
    class MockKVConnectorOutput:
        def __init__(self):
            self.finished_recving = ["req_dead_1"]
            self.finished_sending = ["req_dead_2"]

    kv_output = MockKVConnectorOutput()

    # Capture logs at the WARNING level to verify the new error handling.
    with caplog.at_level(logging.WARNING):
        # Call the target method directly.
        Scheduler._update_from_kv_xfer_finished(scheduler, kv_output)

    # Verify that the engine gracefully handled the missing requests
    # and generated the expected warnings instead of raising AssertionError.
    log_messages = [record.message for record in caplog.records]

    expected_recv_warning = (
        "Late KV xfer-finished (recv) for unknown request req_dead_1"
    )
    expected_send_warning = (
        "Late KV xfer-finished (send) for unknown request req_dead_2"
    )

    assert any(expected_recv_warning in msg for msg in log_messages)
    assert any(expected_send_warning in msg for msg in log_messages)

    # Ensure that no attempt was made to free blocks for non-existent requests.
    scheduler._free_blocks.assert_not_called()
