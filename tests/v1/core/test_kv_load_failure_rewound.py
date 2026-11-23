# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV load-failure rollback must flag the request as count-rewound.

RED-first regression test for the Mission-6 Workstream-A crash
(Qwen3-32B TP=2 32K load-failure fallback, NPU vector-core 507035,
deterministic on the m5 and m6 stacks; see
mission6/A-fallback-crash/A1-REPRO-GEOMETRY.md and A2-ROOT-CAUSE.md).

Contract under test (introduced by core commit a3863c1e5d "Gate the
GPU-side computed-token staging on the scheduler"): every scheduler-side
move of request.num_computed_tokens OTHER THAN the ordinary step advance
must flag the request through _mark_num_computed_tokens_rewound, because
the worker stages the authoritative count over its GPU-side
req_states.num_computed_tokens tensor ONLY for requests the next
SchedulerOutput names in rewound_req_ids. The worker builds positions
and seq_lens from that GPU-side tensor (the
_prepare_pos_seq_lens_kernel: pos = num_computed + block), so an
unflagged rollback leaves the worker at the stale optimistic count and
the rescheduled recompute runs with shifted positions / inflated
seq_lens.

The flat invalid-blocks channel (KVConnectorOutput.invalid_block_ids --
the block-level load-failure report the AscendStoreConnector emits on a
mid-transfer GET failure) truncates request.num_computed_tokens at the
first failed block in _update_requests_with_invalid_blocks, and on the
unfixed tree that move is NOT flagged. Hardware evidence (m6 stack, dies
12/13, logs m6a_rawcrash_tp2_p32768_v1.log / m6a_guard_tp2_p32768_v1.log):
sync claim 32768 + 64 in-flight suffix -> truncation 32832 -> 16896 with
marked_rewound=False; the worker keeps gpu_nct=32832 while echoing
16896; the fallback chunk (8192 tokens) then builds positions
32832..41023 -- positions cross the rope-cache bound
max_position_embeddings=40960 and the rope gather (GatherV3<half,long>
over the [40960, 128] fp16 cos/sin cache) asserts "Index 40960 out of
range[0 40960)" -> vector-core 507035, both ranks, engine death.
Below 32K the same stale count silently shifts the fallback positions
(no device fault) -- the divergence is wrong at every length, it only
faults when the shifted positions cross the rope-cache bound.
"""

import pytest

from tests.v1.core.test_scheduler_repair_budget import (
    admit_async_external_hit,
    admit_sync_external_hit,
    fail_and_truncate,
    make_scheduler,
)
from tests.v1.kv_connector.unit.utils import (
    create_model_runner_output,
    create_request,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

BS = 16  # fixture block size


def test_sync_load_failure_rollback_is_rewound_flagged():
    """SYNC arm (the crash geometry): a block-level load failure
    truncates num_computed_tokens at the first failed block; the move
    must be flagged rewound so the next SchedulerOutput tells the worker
    to stage the truncated count over its stale GPU-side count (or the
    fallback chunk builds positions/seq_lens from the stale count -- the
    32K crash)."""
    scheduler = make_scheduler()  # kv_load_failure_policy=recompute
    r1 = create_request(num_tokens=3072)  # 192 blocks of BS
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 2048)  # claim: 128 blocks

    out1 = scheduler.schedule()
    # The step ran (the request is RUNNING with the claim committed and
    # the suffix scheduled); one mid-claim block failed to load.
    (block_ids,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)
    output = create_model_runner_output(
        reqs=[r1],
        invalid_block_ids={block_ids[96]},
        use_eos=False,
    )
    scheduler.update_from_output(out1, output)

    # The truncation itself (the walk rewinds to the first failed block,
    # excluding the in-flight scheduled tokens).
    assert r1.num_computed_tokens == 96 * BS
    # RED on the unfixed tree: the rollback moved the count other than
    # by the ordinary step advance but was never flagged.
    assert (
        r1.request_id in scheduler._pending_rewound_req_ids
    ), "the invalid-blocks rollback must flag the request as rewound"
    # The full contract: the fallback step's SchedulerOutput names the
    # request so the worker stages the truncated count.
    out2 = scheduler.schedule()
    assert r1.request_id in out2.num_scheduled_tokens
    assert r1.request_id in out2.rewound_req_ids


def test_async_load_failure_rollback_promotes_with_truncated_count():
    """ASYNC arm: the same truncation site (evict_blocks=False) flags the
    request, and the promotion after the failed transfer still re-admits
    it carrying the truncated count (the flag is harmless there: the
    promotion re-adds the request as new and the worker stages its init
    from the echoed count)."""
    scheduler = make_scheduler()  # kv_load_failure_policy=recompute
    r1 = create_request(num_tokens=3072)
    scheduler.add_request(r1)
    admit_async_external_hit(scheduler, r1, 2048)

    out1 = scheduler.schedule()
    fail_and_truncate(scheduler, r1, out1, failed_block_idx=96)
    assert r1.num_computed_tokens == 96 * BS
    # RED on the unfixed tree: the async arm truncates at the same site
    # without the flag.
    assert (
        r1.request_id in scheduler._pending_rewound_req_ids
    ), "the invalid-blocks rollback must flag the request as rewound"

    out2 = scheduler.schedule()  # promotion re-admits the request
    assert r1.request_id in out2.num_scheduled_tokens
    # The promoted prefill resumes exactly at the truncated boundary.
    assert out2.num_scheduled_tokens[r1.request_id] == 3072 - 96 * BS
    new_reqs = {d.req_id: d for d in out2.scheduled_new_reqs}
    assert new_reqs[r1.request_id].num_computed_tokens == 96 * BS
    assert r1.request_id in out2.rewound_req_ids
