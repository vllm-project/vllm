# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.kv_events import (
    BlockStored,
    KVCacheEvent,
    KVConnectorKVEvents,
)
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
    LMCacheKVEvents,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector import (
    MooncakeStoreKVEvents,
)
from vllm.v1.mrope import MRoPEPositionState
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

pytestmark = pytest.mark.cpu_test


class DummyWorkerMeta(KVConnectorWorkerMetadata):
    def __init__(self, tags: set[str]):
        self.tags = set(tags)

    def aggregate(self, other: "DummyWorkerMeta") -> "DummyWorkerMeta":
        return DummyWorkerMeta(self.tags | other.tags)


def _block_stored(*token_ids: int) -> BlockStored:
    return BlockStored(
        block_hashes=[b"\xab" * 32],
        parent_block_hash=None,
        token_ids=list(token_ids),
        block_size=4,
        lora_id=None,
        medium="CPU",
        lora_name=None,
    )


def make_lmcache_events(*token_ids: int) -> LMCacheKVEvents:
    events = LMCacheKVEvents(num_workers=1)
    events.add_events([_block_stored(*token_ids)])
    return events


def make_mooncake_events(*token_ids: int) -> MooncakeStoreKVEvents:
    events = MooncakeStoreKVEvents(num_workers=1)
    events.add_events([_block_stored(*token_ids)])
    return events


def lmcache_published(events: LMCacheKVEvents) -> list[KVCacheEvent]:
    """Mirror LMCacheConnectorV1.take_events: aggregate, then collect."""
    events.aggregate()
    return events.get_all_events()


class DummyModelRunnerOutput(ModelRunnerOutput):
    kv_connector_output: KVConnectorOutput

    def __init__(
        self,
        finished_sending: set[str] | None = None,
        finished_recving: set[str] | None = None,
        invalid_block_ids: set[int] | None = None,
        failed_recving: set[str] | None = None,
        expected_finished_count: int = 0,
        kv_connector_worker_meta: KVConnectorWorkerMetadata | None = None,
        kv_cache_events: KVConnectorKVEvents | None = None,
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

    aggregated_output = aggregator.aggregate([output1, output2])

    assert aggregated_output is output1
    aggregated_kv_output = output1.kv_connector_output
    assert aggregated_kv_output.finished_sending is None
    assert aggregated_kv_output.finished_recving is None
    assert not aggregated_kv_output.invalid_block_ids

    output1 = DummyModelRunnerOutput(
        finished_sending={"req1"}, finished_recving={"req2"}
    )
    output2 = DummyModelRunnerOutput(invalid_block_ids={1})

    aggregated_output = aggregator.aggregate([output1, output2])

    assert aggregated_output is output1
    aggregated_kv_output = output1.kv_connector_output
    assert aggregated_kv_output.finished_sending is None
    assert aggregated_kv_output.finished_recving is None
    assert aggregated_kv_output.invalid_block_ids == {1}

    output1 = DummyModelRunnerOutput(invalid_block_ids={2})
    output2 = DummyModelRunnerOutput(finished_sending={"req1"})

    aggregated_output = aggregator.aggregate([output1, output2])

    assert aggregated_output is output1
    aggregated_kv_output = output1.kv_connector_output
    assert aggregated_kv_output.finished_sending == {"req1"}
    assert aggregated_kv_output.finished_recving is None
    assert aggregated_kv_output.invalid_block_ids == {2}

    output1 = DummyModelRunnerOutput(invalid_block_ids={3, 4})
    output2 = DummyModelRunnerOutput(
        finished_recving={"req2"},
        invalid_block_ids={4, 5},
        failed_recving={"req3"},
    )

    aggregated_output = aggregator.aggregate([output1, output2])

    assert aggregated_output is output1
    aggregated_kv_output = output1.kv_connector_output
    assert aggregated_kv_output.finished_sending is None
    assert aggregated_kv_output.finished_recving == {"req2"}
    assert aggregated_kv_output.invalid_block_ids == {3, 4, 5}
    assert not aggregated_kv_output.failed_recving

    output1 = DummyModelRunnerOutput(finished_recving={"req3"})
    output2 = DummyModelRunnerOutput(finished_recving={"req3"})
    aggregated_output = aggregator.aggregate([output1, output2])
    assert aggregated_output is output1
    aggregated_kv_output = output1.kv_connector_output
    assert aggregated_kv_output.failed_recving == {"req3"}


def test_aggregate_workers_output_with_expected_finished_count():
    # We create the aggregator expecting to collect from 4 workers
    aggregator = KVOutputAggregator(expected_finished_count=4)
    assert aggregator._expected_finished_count == 4
    # Some request with default expected finished requests
    output1 = DummyModelRunnerOutput(finished_sending={"req1"})
    aggregated_output = aggregator.aggregate([output1])
    assert aggregated_output is output1
    aggregated_kv_output = output1.kv_connector_output
    # still expecting to collect from 4 workers
    assert aggregator._send_remaining_count["req1"] == 3
    assert not aggregated_kv_output.finished_sending
    assert not aggregated_kv_output.finished_recving

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
    aggregated_output = aggregator.aggregate([output1, output2, output3])
    assert aggregated_output is output1
    aggregated_kv_output = output1.kv_connector_output
    assert aggregated_kv_output.expected_finished_count == 2

    assert not aggregated_kv_output.finished_sending

    # Req2 is finished
    assert "req2" not in aggregator._recv_remaining_count
    assert aggregated_kv_output.finished_recving == {"req2"}

    # Req1 is still waiting for 2 more acks (expected_finished_count has no effect)
    # NOTE: This is to showcase dynamic update. Workers are responsible for
    # ensuring "req1" termination in this case
    assert aggregator._send_remaining_count["req1"] == 2


@pytest.mark.parametrize("replay", [False, True])
def test_aggregate_mrope_position_state_from_first_pipeline_stage(replay):
    state = MRoPEPositionState(positions=[[0, 1], [0, 2], [0, 3]], delta=-7)
    first_stage = DummyModelRunnerOutput()
    first_stage.kv_connector_output.mrope_position_states = {"req": state}
    last_stage = DummyModelRunnerOutput()
    aggregator = KVOutputAggregator(expected_finished_count=2)
    if replay:
        aggregator.ft_defer_kv_connector_outputs(
            [first_stage.kv_connector_output, None]
        )
        first_stage = DummyModelRunnerOutput()

    output = aggregator.aggregate([first_stage, last_stage], output_rank=1)

    assert output is last_stage
    assert output.kv_connector_output.mrope_position_states == {"req": state}


def test_replay_preserves_get_and_clear_fields():
    """Salvaged outputs must be replayed and their fields surfaced later."""
    aggregator = KVOutputAggregator(expected_finished_count=4)

    # Failed step: worker 0's output is salvaged, worker 1 is lost.
    salvaged = KVConnectorOutput(
        finished_sending={"req1"},
        finished_recving={"req2"},
        failed_recving={"req3"},
        invalid_block_ids={7},
        kv_connector_worker_meta=DummyWorkerMeta({"salvaged"}),
        kv_cache_events=make_lmcache_events(1, 2, 3, 4),
        expected_finished_count=2,
    )
    aggregator.ft_defer_kv_connector_outputs([salvaged, None])

    # Recovered step: both workers succeed.
    output0 = DummyModelRunnerOutput(
        finished_recving={"req3"},
        kv_connector_worker_meta=DummyWorkerMeta({"recovered"}),
        kv_cache_events=make_lmcache_events(9, 10, 11, 12),
    )
    output1 = DummyModelRunnerOutput(
        finished_sending={"req1"},
        finished_recving={"req2", "req3"},
    )
    aggregated = aggregator.aggregate([output0, output1])
    kv = aggregated.kv_connector_output

    assert kv.expected_finished_count == 2
    assert kv.finished_sending == {"req1"}
    assert kv.finished_recving == {"req2", "req3"}
    assert kv.failed_recving == {"req3"}
    assert kv.invalid_block_ids == {7}
    assert kv.kv_connector_worker_meta.tags == {"salvaged", "recovered"}
    assert kv.kv_cache_events is not None
    all_token_ids = sorted(
        tuple(e.token_ids) for e in kv.kv_cache_events.get_all_events()
    )
    assert all_token_ids == [(9, 10, 11, 12)]


def test_replay_completes_finished_recving_votes():
    """Salvaged recv votes must complete via replay, or the request waits
    on WAITING_FOR_REMOTE_KVS forever."""
    aggregator = KVOutputAggregator(expected_finished_count=2)

    # Failed step: req1's recv votes completed (2/2), flagged as failed.
    aggregator.ft_defer_kv_connector_outputs(
        [
            KVConnectorOutput(finished_recving={"req1"}),
            KVConnectorOutput(finished_recving={"req1"}, failed_recving={"req1"}),
        ]
    )

    # Recovered step: nobody reports req1 anymore.
    aggregated = aggregator.aggregate(
        [DummyModelRunnerOutput(), DummyModelRunnerOutput()]
    )
    kv = aggregated.kv_connector_output
    assert kv.finished_recving == {"req1"}
    assert kv.failed_recving == {"req1"}


def test_salvaged_kv_cache_events_do_not_raise_quorum():
    """Salvaged events are extra votes only; quorum stays at live workers."""
    aggregator = KVOutputAggregator(expected_finished_count=2)

    # Failed step: only worker 0 reported event E0.
    aggregator.ft_defer_kv_connector_outputs(
        [KVConnectorOutput(kv_cache_events=make_lmcache_events(1, 2, 3, 4)), None]
    )

    # Recovered step: both workers report event E1.
    output0 = DummyModelRunnerOutput(kv_cache_events=make_lmcache_events(5, 6, 7, 8))
    output1 = DummyModelRunnerOutput(kv_cache_events=make_lmcache_events(5, 6, 7, 8))
    aggregated = aggregator.aggregate([output0, output1])

    events = aggregated.kv_connector_output.kv_cache_events
    assert events is not None
    assert events.get_number_of_workers() == 2
    # Mirror the LMCache consumer's take_events.
    assert [e.token_ids for e in lmcache_published(events)] == [[5, 6, 7, 8]]


def test_salvaged_votes_dropped_live_votes_unaffected_lmcache_exact_quorum():
    """A dropped salvaged vote must not take a live event down with it
    (LMCache exact-match quorum)."""
    aggregator = KVOutputAggregator(expected_finished_count=2)

    # Failed step: worker 0's vote for block E is salvaged.
    aggregator.ft_defer_kv_connector_outputs(
        [KVConnectorOutput(kv_cache_events=make_lmcache_events(1, 2, 3, 4)), None]
    )

    # Recovered step: the surviving worker reports the same block E while
    # the other live worker stays quiet.
    output0 = DummyModelRunnerOutput()
    output1 = DummyModelRunnerOutput(kv_cache_events=make_lmcache_events(1, 2, 3, 4))
    aggregated = aggregator.aggregate([output0, output1])

    events = aggregated.kv_connector_output.kv_cache_events
    assert events is not None
    assert len(events.get_all_events()) == 1
    assert [e.token_ids for e in lmcache_published(events)] == [[1, 2, 3, 4]]


def test_salvaged_votes_dropped_live_votes_unaffected_mooncake_quorum():
    """Same as the LMCache test, under MooncakeStore's >= quorum."""
    aggregator = KVOutputAggregator(expected_finished_count=2)

    # Failed step: worker 0's vote for block E is salvaged.
    aggregator.ft_defer_kv_connector_outputs(
        [KVConnectorOutput(kv_cache_events=make_mooncake_events(1, 2, 3, 4)), None]
    )

    # Recovered step: the surviving worker reports the same block E.
    output0 = DummyModelRunnerOutput()
    output1 = DummyModelRunnerOutput(kv_cache_events=make_mooncake_events(1, 2, 3, 4))
    aggregated = aggregator.aggregate([output0, output1])

    events = aggregated.kv_connector_output.kv_cache_events
    assert events is not None
    # Mirror the MooncakeStore consumer (take_events -> pop_common_events).
    published = events.pop_common_events()
    assert [e.token_ids for e in published] == [[1, 2, 3, 4]]


def test_salvaged_event_votes_are_one_shot():
    """Salvaged votes must not leak into a later step's quorum."""
    aggregator = KVOutputAggregator(expected_finished_count=2)

    # Failed step: worker 0's vote for block E0 is salvaged.
    aggregator.ft_defer_kv_connector_outputs(
        [KVConnectorOutput(kv_cache_events=make_lmcache_events(1, 2, 3, 4)), None]
    )

    # Recovered step is quiet: no live events, salvaged votes discarded.
    aggregated = aggregator.aggregate(
        [DummyModelRunnerOutput(), DummyModelRunnerOutput()]
    )
    assert aggregated.kv_connector_output.kv_cache_events is None

    # A later step reports a different event: the salvaged votes must not
    # leak into this step's quorum.
    output0 = DummyModelRunnerOutput(kv_cache_events=make_lmcache_events(5, 6, 7, 8))
    output1 = DummyModelRunnerOutput(kv_cache_events=make_lmcache_events(5, 6, 7, 8))
    aggregated = aggregator.aggregate([output0, output1])

    events = aggregated.kv_connector_output.kv_cache_events
    assert events is not None
    assert [e.token_ids for e in lmcache_published(events)] == [[5, 6, 7, 8]]
