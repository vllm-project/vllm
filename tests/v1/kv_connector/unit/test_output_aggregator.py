# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.kv_events import BlockStored, KVEventAggregator
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorWorkerMetadata,
)
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

pytestmark = pytest.mark.cpu_test


class DummyWorkerMeta(KVConnectorWorkerMetadata):
    def __init__(self, tags: set[str]):
        self.tags = set(tags)

    def aggregate(self, other: "DummyWorkerMeta") -> "DummyWorkerMeta":
        return DummyWorkerMeta(self.tags | other.tags)


def make_events(*token_ids: int) -> KVEventAggregator:
    events = KVEventAggregator(num_workers=1)
    events.add_events(
        [
            BlockStored(
                block_hashes=[b"\xab" * 32],
                parent_block_hash=None,
                token_ids=list(token_ids),
                block_size=4,
                lora_id=None,
                medium="GPU",
                lora_name=None,
            )
        ]
    )
    return events


class DummyModelRunnerOutput(ModelRunnerOutput):
    def __init__(
        self,
        finished_sending: set[str] | None = None,
        finished_recving: set[str] | None = None,
        invalid_block_ids: set[int] | None = None,
        failed_recving: set[str] | None = None,
        expected_finished_count: int = 0,
        kv_connector_worker_meta: KVConnectorWorkerMetadata | None = None,
        kv_cache_events: KVEventAggregator | None = None,
    ):
        self.kv_connector_output = KVConnectorOutput(
            finished_sending=finished_sending,
            finished_recving=finished_recving,
            invalid_block_ids=invalid_block_ids or set(),
            failed_recving=failed_recving or set(),
            expected_finished_count=expected_finished_count,
            kv_connector_worker_meta=kv_connector_worker_meta,
            kv_cache_events=kv_cache_events,
        )

    def __repr__(self):
        return (
            f"DummyModelRunnerOutput("
            f"finished_sending={self.kv_connector_output.finished_sending},"
            f"finished_recving={self.kv_connector_output.finished_recving})"
            f"invalid_block_ids={self.kv_connector_output.invalid_block_ids})"
        )


def test_aggregate_workers_output():
    aggregator = KVOutputAggregator(expected_finished_count=2)

    output1 = DummyModelRunnerOutput()
    output2 = DummyModelRunnerOutput()

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving is None
    assert not aggregated.invalid_block_ids

    output1 = DummyModelRunnerOutput(
        finished_sending={"req1"}, finished_recving={"req2"}
    )
    output2 = DummyModelRunnerOutput(invalid_block_ids={1})

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving is None
    assert aggregated.invalid_block_ids == {1}

    output1 = DummyModelRunnerOutput(invalid_block_ids={2})
    output2 = DummyModelRunnerOutput(finished_sending={"req1"})

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending == {"req1"}
    assert aggregated.finished_recving is None
    assert aggregated.invalid_block_ids == {2}

    output1 = DummyModelRunnerOutput(invalid_block_ids={3, 4})
    output2 = DummyModelRunnerOutput(
        finished_recving={"req2"},
        invalid_block_ids={4, 5},
        failed_recving={"req3"},
    )

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving == {"req2"}
    assert aggregated.invalid_block_ids == {3, 4, 5}
    assert not aggregated.failed_recving

    output1 = DummyModelRunnerOutput(finished_recving={"req3"})
    output2 = DummyModelRunnerOutput(finished_recving={"req3"})
    aggregated = aggregator.aggregate([output1, output2])
    assert aggregated.kv_connector_output.failed_recving == {"req3"}


def test_aggregate_workers_output_with_expected_finished_count():
    # We create the aggregator expecting to collect from 4 workers
    aggregator = KVOutputAggregator(expected_finished_count=4)
    assert aggregator._expected_finished_count == 4
    # Some request with default expected finished requests
    output1 = DummyModelRunnerOutput(finished_sending={"req1"})
    aggregated = aggregator.aggregate([output1])
    # still expecting to collect from 4 workers
    assert aggregator._send_remaining_count["req1"] == 3
    assert not aggregated.kv_connector_output.finished_sending
    assert not aggregated.kv_connector_output.finished_recving

    # Workers discover and find that in this setup they only need to
    # collect from 2
    output1 = DummyModelRunnerOutput(
        finished_sending={"req1"}, expected_finished_count=2
    )
    output2 = DummyModelRunnerOutput(
        finished_recving={"req2"}, expected_finished_count=2
    )
    output3 = DummyModelRunnerOutput(finished_recving={"req2"})
    # Req2 only needs 2 acks
    aggregated = aggregator.aggregate([output1, output2, output3])
    assert aggregated.kv_connector_output.expected_finished_count == 2

    assert not aggregated.kv_connector_output.finished_sending

    # Req2 is finished
    assert "req2" not in aggregator._recv_remaining_count
    assert aggregated.kv_connector_output.finished_recving == {"req2"}

    # Req1 is still waiting for 2 more acks (expected_finished_count has no effect)
    # NOTE: This is to showcase dynamic update. Workers are responsible for
    # ensuring "req1" termination in this case
    assert aggregator._send_remaining_count["req1"] == 2


def test_merge_preserves_get_and_clear_fields():
    """merge must preserve get-and-clear fields and surface them later."""
    aggregator = KVOutputAggregator(expected_finished_count=4)

    # Failed step: worker 0's output is salvaged, worker 1 is lost.
    salvaged = KVConnectorOutput(
        finished_sending={"req1"},
        finished_recving={"req2"},
        failed_recving={"req3"},
        invalid_block_ids={7},
        kv_connector_worker_meta=DummyWorkerMeta({"salvaged"}),
        kv_cache_events=make_events(1, 2, 3, 4),
        expected_finished_count=2,
    )
    aggregator.merge_kv_connector_output([salvaged, None])
    assert aggregator._expected_finished_count == 2

    # Recovered step: both workers succeed.
    output0 = DummyModelRunnerOutput(
        finished_recving={"req3"},
        kv_connector_worker_meta=DummyWorkerMeta({"recovered"}),
        kv_cache_events=make_events(9, 10, 11, 12),
    )
    output1 = DummyModelRunnerOutput(
        finished_sending={"req1"},
        finished_recving={"req2", "req3"},
    )
    aggregated = aggregator.aggregate([output0, output1])
    kv = aggregated.kv_connector_output

    assert kv.finished_sending == {"req1"}
    assert kv.finished_recving == {"req2", "req3"}
    assert kv.failed_recving == {"req3"}
    assert kv.invalid_block_ids == {7}
    assert kv.kv_connector_worker_meta.tags == {"salvaged", "recovered"}
    assert kv.kv_cache_events is not None
    all_token_ids = sorted(
        tuple(e.token_ids) for e in kv.kv_cache_events.get_all_events()
    )
    assert all_token_ids == [(1, 2, 3, 4), (9, 10, 11, 12)]


def test_merge_seeds_pending_finished_recving():
    """Pending finished_recving must be seeded before the failed_recving
    intersection, or the request waits on WAITING_FOR_REMOTE_KVS forever."""
    aggregator = KVOutputAggregator(expected_finished_count=2)

    # Failed step: req1's recv votes completed (2/2), flagged as failed.
    aggregator.merge_kv_connector_output(
        [
            KVConnectorOutput(finished_recving={"req1"}),
            KVConnectorOutput(finished_recving={"req1"}, failed_recving={"req1"}),
        ]
    )
    assert aggregator._pending_finished_recving == {"req1"}

    # Recovered step: nobody reports req1 anymore.
    aggregated = aggregator.aggregate(
        [DummyModelRunnerOutput(), DummyModelRunnerOutput()]
    )
    kv = aggregated.kv_connector_output
    assert kv.finished_recving == {"req1"}
    assert kv.failed_recving == {"req1"}


def test_merge_kv_cache_events_do_not_raise_quorum():
    """Pending events are extra votes only; quorum stays at current workers."""
    aggregator = KVOutputAggregator(expected_finished_count=2)

    # Failed step: only worker 0 reported event E0.
    aggregator.merge_kv_connector_output(
        [KVConnectorOutput(kv_cache_events=make_events(1, 2, 3, 4)), None]
    )

    # Recovered step: both workers report event E1.
    output0 = DummyModelRunnerOutput(kv_cache_events=make_events(5, 6, 7, 8))
    output1 = DummyModelRunnerOutput(kv_cache_events=make_events(5, 6, 7, 8))
    aggregated = aggregator.aggregate([output0, output1])

    events = aggregated.kv_connector_output.kv_cache_events
    assert events is not None
    assert events.get_number_of_workers() == 2
    common = events.get_common_events()
    assert len(common) == 1
    assert common[0].token_ids == [5, 6, 7, 8]
