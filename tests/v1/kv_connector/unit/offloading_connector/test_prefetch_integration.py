# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefetch hints driven through a real Scheduler + OffloadingConnector pair.

`test_prefetch.py` covers the arithmetic of a hint (which hashes to pin, which
offload key owns each chunk). What is covered here is the wiring: a hint that
arrives on a request turns into a transfer that runs while no request needs it,
its blocks are published to the prefix cache on success and handed back to the
pool either way, and the request that follows then finds them resident instead
of loading them itself.

The CPU tier is mocked, so the transfers move no bytes; every assertion is about
the scheduler-side bookkeeping around them.
"""

import pytest

from tests.v1.kv_connector.unit.offloading_connector.utils import (  # noqa: F401
    GPUBlock,
    MockLoadStoreSpec,
    generate_store_output,
    request_runner,
)
from tests.v1.kv_connector.unit.utils import EOS_TOKEN_ID
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.prefetch import (
    PREFETCH_ACTION_TYPE,
    PREFETCH_ACTION_VERSION,
    PREFETCH_REQ_ID,
)
from vllm.v1.kv_hints import KvHintsEnvelope
from vllm.v1.kv_offload.base import (
    LookupResult,
    get_offload_block_hash,
    make_offload_key,
)

try:
    from vllm.v1.kv_hints import KvHintAction
except ImportError:  # pragma: no cover - the export location moved once already
    from vllm.v1.kv_hints.protocol import KvHintAction

BLOCK_SIZE = 4
"""Tokens per block. One chunk is one block, so hashes and blocks line up."""

NUM_BLOCKS = 24

NUM_WARM_BLOCKS = 3
"""Blocks the warmed request fills, i.e. how many chunks a hint can name."""

WARM_TOKENS = [7] * BLOCK_SIZE * NUM_WARM_BLOCKS

CARRIER_TOKENS = [9] * BLOCK_SIZE
"""The request that carries a hint shares no prefix with the warmed request, so
any load it triggers can only be the prefetch."""


def make_runner(request_runner):
    """A runner whose pool may reserve blocks ahead of their request."""
    runner = request_runner(
        block_size=BLOCK_SIZE, num_gpu_blocks=NUM_BLOCKS, async_scheduling=False
    )
    # The harness always enables KV cache events in order to drive the offloading
    # tracker, and pinning stands down while they are on: a block filled from the
    # tier must not be announced as stored before the bytes are there. Turning
    # the flag off here is what a deployment without an event subscriber looks
    # like; `test_hint_is_ignored_while_kv_events_are_published` covers the other
    # side.
    runner.scheduler.kv_cache_manager.block_pool.enable_kv_cache_events = False
    return runner


def prefetch_envelope(hashes, version=PREFETCH_ACTION_VERSION):
    return KvHintsEnvelope(
        protocol_version="0.1",
        message_id="hint-1",
        actions=[
            KvHintAction(
                action_id="prefetch-1",
                action_type=PREFETCH_ACTION_TYPE,
                action_version=version,
                payload={"block_hashes": [h.hex() for h in hashes]},
            )
        ],
    )


def record_loads(runner):
    """Record (req_id, key count) of every load the scheduler prepares."""
    calls = []

    def prepare_load(keys, req_context):
        keys = list(keys)
        calls.append((req_context.req_id, len(keys)))
        return MockLoadStoreSpec(keys)

    runner.manager.prepare_load = prepare_load
    return calls


def tier_with(runner, hashes):
    """Answer a lookup as a tier that holds exactly these blocks."""
    keys = {make_offload_key(h, 0) for h in hashes}
    runner.manager.lookup.side_effect = lambda key, req_context: (
        LookupResult.HIT if key in keys else LookupResult.MISS
    )


def free_blocks(runner):
    return runner.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()


def track_prefetch_transfers(runner):
    """Map the GPU blocks a prefetch reserves, so the harness can translate them.

    A prefetch fills blocks that belong to no request, and the harness only maps
    blocks of scheduled requests; without this the completed transfer names
    blocks it has never seen. The handover point is the connector metadata, which
    is exactly what the worker submits from.
    """
    worker = runner.worker_connector
    bind = worker.bind_connector_metadata

    def track(metadata):
        for entry in metadata.load_jobs.values():
            if entry.req_id != PREFETCH_REQ_ID:
                continue
            keys = list(entry.src_spec.offload_keys)
            block_ids = [int(block_id) for block_id in entry.dst_spec.block_ids]
            offset = 0
            for group_idx, group_size in enumerate(entry.dst_spec.group_sizes):
                for position in range(group_size):
                    runner.gpu_blocks[block_ids[offset + position]] = GPUBlock(
                        group_idx, position
                    )
                    runner.offloaded[(keys[offset + position], 0)] = GPUBlock(
                        group_idx, position
                    )
                offset += group_size
        return bind(metadata)

    worker.bind_connector_metadata = track


def warm_and_offload(runner, calls):
    """Fill blocks, store them to the mocked tier, drop the GPU copies."""
    stored = []

    def prepare_store(keys, req_context, trigger):
        stored.extend(keys)
        return generate_store_output(keys)

    runner.manager.prepare_store.side_effect = prepare_store
    runner.new_request(token_ids=WARM_TOKENS, req_id="warm")
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=tuple(range(NUM_WARM_BLOCKS)),
    )
    assert len(stored) == NUM_WARM_BLOCKS
    # The GPU copies go away, so only the tier still has these blocks: that is
    # the state a prefetch exists for.
    runner.scheduler.reset_prefix_cache()
    assert calls == []
    return [get_offload_block_hash(key) for key in stored]


def run_carrier(runner, hashes, version=PREFETCH_ACTION_VERSION):
    track_prefetch_transfers(runner)
    runner.new_request(
        token_ids=CARRIER_TOKENS,
        req_id="carrier",
        kv_hints=prefetch_envelope(hashes, version=version),
    )
    runner._run(decoded_tokens=[EOS_TOKEN_ID], complete_transfers=True)


def test_hint_loads_blocks_and_the_next_request_finds_them(request_runner):
    runner = make_runner(request_runner)
    calls = record_loads(runner)
    hashes = warm_and_offload(runner, calls)
    before = free_blocks(runner)
    tier_with(runner, hashes)

    run_carrier(runner, hashes)

    assert calls == [(PREFETCH_REQ_ID, NUM_WARM_BLOCKS)], (
        "the hint should load exactly the chunks it named, and the carrier has no "
        "prefix in the tier so it loads nothing itself"
    )
    assert runner.connector_scheduler._prefetch_jobs == {}
    assert free_blocks(runner) == before, "a finished prefetch returns its blocks"
    assert len(runner.completed_loads) == 1
    loaded = {
        block for transfer in runner.completed_loads for block in transfer.gpu_blocks
    }
    assert len(loaded) == NUM_WARM_BLOCKS

    runner.new_request(token_ids=WARM_TOKENS, req_id="probe")
    runner._run(decoded_tokens=[], complete_transfers=True)
    # The prompt's last block is always recomputed - its logits are the point, so
    # the cache-hit length stops one token short of a full hit. A probe that finds
    # the prefetched prefix therefore still asks the tier for that one tail block,
    # the same block the core would otherwise recompute; without the hint it asks
    # for all three.
    assert [call for call in calls if call[0] == "probe"] == [("probe", 1)], (
        "the prefetched prefix is resident: only the tail block is fetched"
    )
    assert len(runner.completed_loads) == 2


def test_without_a_hint_the_request_loads_the_tier_itself(request_runner):
    """The counterweight: same setup, no envelope.

    Without a hint the probe prepares its own load, which is exactly what the
    hint saves. If it ever stops doing so, the test above proves nothing.
    """
    runner = make_runner(request_runner)
    calls = record_loads(runner)
    hashes = warm_and_offload(runner, calls)
    tier_with(runner, hashes)

    runner.new_request(token_ids=WARM_TOKENS, req_id="probe")
    runner._run(decoded_tokens=[], complete_transfers=True)

    assert [call for call in calls if call[0] == "probe"] == [
        ("probe", NUM_WARM_BLOCKS)
    ], "an unhinted prefix is loaded by the request that needs it"


def test_hint_for_a_prefix_the_tier_lacks_is_dropped(request_runner):
    runner = make_runner(request_runner)
    calls = record_loads(runner)
    hashes = warm_and_offload(runner, calls)
    before = free_blocks(runner)
    runner.manager.lookup.return_value = LookupResult.MISS

    run_carrier(runner, hashes)

    assert calls == []
    assert runner.connector_scheduler._prefetch_jobs == {}
    assert runner.completed_loads == []
    assert free_blocks(runner) == before, "a dropped hint reserves nothing"


def test_only_the_served_prefix_is_prefetched(request_runner):
    """A load has to be a contiguous prefix of the tier's answer, so a gap ends it."""
    runner = make_runner(request_runner)
    calls = record_loads(runner)
    hashes = warm_and_offload(runner, calls)
    before = free_blocks(runner)
    tier_with(runner, hashes[:1])

    run_carrier(runner, hashes)

    assert calls == [(PREFETCH_REQ_ID, 1)]
    assert free_blocks(runner) == before


def test_failed_prefetch_returns_blocks_without_publishing(request_runner):
    runner = make_runner(request_runner)
    calls = record_loads(runner)
    hashes = warm_and_offload(runner, calls)
    before = free_blocks(runner)
    tier_with(runner, hashes)

    runner.offloading_spec.handler.fail_loads = True
    run_carrier(runner, hashes)

    assert free_blocks(runner) == before, "a failed prefetch must not leak blocks"

    calls.clear()
    runner.offloading_spec.handler.fail_loads = False
    runner.new_request(token_ids=WARM_TOKENS, req_id="probe")
    runner._run(decoded_tokens=[], complete_transfers=True)
    assert [call for call in calls if call[0] == "probe"] == [
        ("probe", NUM_WARM_BLOCKS)
    ], "nothing was published, so the probe still loads from the tier"


def test_hint_with_an_unknown_action_version_is_ignored(request_runner):
    runner = make_runner(request_runner)
    calls = record_loads(runner)
    hashes = warm_and_offload(runner, calls)
    before = free_blocks(runner)
    tier_with(runner, hashes)

    run_carrier(runner, hashes, version="9.9")

    assert calls == []
    assert free_blocks(runner) == before


def test_hint_is_ignored_while_kv_events_are_published(request_runner):
    """Pinning stands down when block events have subscribers.

    A reserved block holds bytes that are still in flight. Publishing it as
    stored would tell event consumers about KV that has not arrived, so the pool
    declines the reservation instead and the hint costs nothing but a lookup.
    """
    runner = request_runner(
        block_size=BLOCK_SIZE, num_gpu_blocks=NUM_BLOCKS, async_scheduling=False
    )
    calls = record_loads(runner)
    hashes = warm_and_offload(runner, calls)
    before = free_blocks(runner)
    tier_with(runner, hashes)

    run_carrier(runner, hashes)

    assert calls == []
    assert runner.completed_loads == []
    assert free_blocks(runner) == before
