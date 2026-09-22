# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Mooncake EC consumer reservations and push completion.

Covers the concurrent-duplicate-push path: two producers reserve the same
mm_hash (one writer, one follower), and completion rejections must not abort
an entire producer push batch.
"""

import logging
from concurrent.futures import Future

import torch

from vllm.distributed.ec_transfer.ec_connector.mooncake.memory import (
    ConsumerMemoryPool,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.metadata import (
    ECMooncakePushSpec,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.producer import (
    ProducerPushRecord,
    ProducerPushState,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.reservation import (
    ConsumerReservationManager,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.worker import (
    ECMooncakeWorker,
)


class _FakeTransfer:
    def register_memory(self, tensor: torch.Tensor) -> int:
        return 0

    def unregister_memory(self, tensor: torch.Tensor) -> bool:
        return True

    def local_session(self) -> str:
        return "fake-session"


def _manager() -> ConsumerReservationManager:
    pool = ConsumerMemoryPool(1 << 20, _FakeTransfer())
    pool.prepare(torch.device("cpu"))
    return ConsumerReservationManager(pool, lease_ttl=60.0, tombstone_limit=16)


def _reserve(manager: ConsumerReservationManager, transfer_id: str):
    return manager.reserve(
        transfer_id, "hash-A", 8, (4,), "torch.float16", torch.float16
    )


def test_follower_complete_is_idempotent_noop():
    """A follower completion must not be rejected: the writer publishes it."""
    manager = _manager()
    writer, should_write = _reserve(manager, "t-writer")
    assert should_write and writer is not None and not writer.writer_id
    follower, should_write = _reserve(manager, "t-follower")
    assert not should_write and follower is not None
    assert follower.writer_id == "t-writer"

    # Before the fix this returned (False, False), which producers reported
    # as "Unknown EC reservation" and failed the whole push batch.
    assert manager.complete("t-follower", follower.reservation_id) == (True, False)

    # The writer completing publishes the follower.
    assert manager.complete("t-writer", writer.reservation_id) == (True, True)
    assert "t-follower" in manager.drain_ready()

    # Re-completing either id stays idempotent once both are ready.
    assert manager.complete("t-follower", follower.reservation_id) == (True, False)
    assert manager.complete("t-writer", writer.reservation_id) == (True, False)


def test_complete_rejects_unknown_transfer():
    manager = _manager()
    assert manager.complete("no-such-transfer", "no-such-reservation") == (
        False,
        False,
    )


def _push(transfer_id: str) -> ProducerPushRecord:
    spec = ECMooncakePushSpec(
        mm_hash=f"hash-{transfer_id}",
        nbytes=8,
        shape=(4,),
        dtype="torch.float16",
        consumer_zmq="tcp://consumer:1",
        transfer_id=transfer_id,
    )
    return ProducerPushRecord(spec, ProducerPushState.NOTIFYING, Future())


class _FakeControlClient:
    def __init__(self, completions: list[bool]):
        self.completions = completions

    def request(self, addr: str, payload: dict):
        assert payload["op"] == "complete_batch"
        return {
            "items": [
                {"completed": completed, "became_ready": completed}
                for completed in self.completions
            ]
        }


def test_notify_completions_tolerates_rejected_reservation(caplog):
    """One stale reservation must not fail the whole push batch."""
    worker = ECMooncakeWorker.__new__(ECMooncakeWorker)
    worker._control_client = _FakeControlClient([True, False])
    pushes = [_push("t-ok"), _push("t-stale")]
    reservations = [
        {"addr": "tcp://consumer:1", "reservation_id": "r-ok"},
        {"addr": "tcp://consumer:1", "reservation_id": "r-stale"},
    ]
    with caplog.at_level(logging.WARNING):
        worker._notify_completions(list(zip(pushes, reservations)))
    assert "rejected" in caplog.text
    assert "hash-t-stale" in caplog.text
