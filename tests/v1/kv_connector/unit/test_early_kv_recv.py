# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
import time
from unittest.mock import MagicMock

import pytest
import zmq

from vllm.distributed.kv_transfer.kv_connector.v1.early_recv import (
    EarlyKVOutputAggregator,
    EarlyKVRecvNotifier,
    EarlyKVRecvReceiver,
)
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput


@pytest.mark.cpu_test
def test_notifier_receiver_round_trip_and_thread_safety():
    receiver = EarlyKVRecvReceiver()
    aggregator = EarlyKVOutputAggregator(2, receiver=receiver)
    notifier = EarlyKVRecvNotifier(aggregator.endpoint)
    try:
        errors: list[Exception] = []

        def send_batch(prefix: str) -> None:
            try:
                for i in range(10):
                    assert notifier.try_notify({f"{prefix}-{i}"})
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=send_batch, args=("req",)),
            threading.Thread(target=send_batch, args=("req",)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert not errors

        finished: set[str] = set()
        deadline = time.monotonic() + 5.0
        while len(finished) < 10 and time.monotonic() < deadline:
            batch = aggregator.take_early_finished_recving()
            if batch:
                finished |= batch
            else:
                time.sleep(0.001)
        assert finished == {f"req-{i}" for i in range(10)}
    finally:
        notifier.close()
        notifier.close()  # idempotent
        aggregator.close()
        aggregator.close()  # idempotent


@pytest.mark.cpu_test
def test_notifier_falls_back_when_closed_or_would_block():
    receiver = EarlyKVRecvReceiver()
    notifier = EarlyKVRecvNotifier(receiver.path)
    try:
        assert notifier.try_notify(set()) is True
        notifier._socket = MagicMock()
        notifier._socket.send.side_effect = zmq.Again()
        assert notifier.try_notify({"req-1"}) is False
    finally:
        notifier.close()
        receiver.close()
    assert notifier.try_notify({"req-2"}) is False


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "ordinary_rank,failed,invalid",
    [(None, False, False)]
    + [
        (rank, failed, invalid)
        for rank in (0, 1)
        for failed, invalid in ((False, False), (False, True), (True, False))
    ],
)
def test_early_receives_share_normal_quorum(ordinary_rank, failed, invalid):
    aggregator = EarlyKVOutputAggregator(2)

    def aggregate(kv_output=None):
        output = ModelRunnerOutput([], {}, kv_connector_output=kv_output)
        aggregator.aggregate([output])
        assert output.kv_connector_output is not None
        return output.kv_connector_output

    early = set()
    normal = KVConnectorOutput()
    for rank in range(2):
        if rank == ordinary_rank:
            # A step-output vote: a failure, invalid blocks, or a success that
            # was not reported early.
            normal = aggregate(
                KVConnectorOutput(
                    finished_recving={"request"},
                    failed_recving={"request"} if failed else set(),
                    invalid_block_ids={7} if invalid else set(),
                ),
            )
        else:
            early |= aggregator.add_finished_recving({"request"})
        if rank == 0:
            assert not early and not normal.finished_recving
    # A step without output must keep a deferred vote for the next one.
    assert aggregator.aggregate([None]) is None
    if not normal.finished_recving:
        normal = aggregate()
    # A step output with a vote may carry invalid blocks or a failure that the
    # scheduler has not processed yet, so a receive finishes early only when
    # every worker reported it early.
    assert early == ({"request"} if ordinary_rank is None else set())
    assert (normal.finished_recving or set()) == {"request"} - early
    assert normal.failed_recving == ({"request"} if failed else set())
    assert aggregate().is_empty()
