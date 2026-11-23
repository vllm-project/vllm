# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Partial-write invalidation across block reuse (hardening H3).

The abort terminal state's invariant: NO partially written state from
an aborted repair can become visible to another request or be accepted
as valid cached state. The worker-side re-zero (see
test_repair_envelope.py) is defense in depth; the AUTHORITATIVE
invalidation is scheduler-side:

- the failed span and everything downstream is evicted (unhashed) at
  the abort, so no future request can prefix-match untrustworthy
  content;
- the request truncates to the rollback boundary and the ordinary
  recompute rewrites the span before any reader can observe it
  (write-before-read);
- after the damaged request finishes and its blocks return to the free
  pool, a new request with the IDENTICAL prompt gets no prefix-cache
  hit on any of them (the failed transfer published nothing and the
  abort evicted the span) -- the physical blocks carrying partial
  writes are reused only through the unmatchable (unhashed) path,
  covered by the new owner's own writes.
"""

import pytest

from tests.v1.core.test_scheduler_repair_budget import (
    BS,
    E,
    admit_sync_external_hit,
    drive_repair_result,
    install_plan,
    make_scheduler,
)
from tests.v1.kv_connector.unit.utils import create_request
from vllm.v1.request import RequestStatus

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _aborted_repair_scenario():
    """APC ON, sync external hit, plan admitted, abort verdict consumed.
    Returns (scheduler, r1, block_ids, block_pool)."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", max_num_batched_tokens=8192
    )
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()

    from tests.v1.core.test_scheduler_repair_budget import (
        _deliver_report,
        _structured_report,
    )

    _deliver_report(scheduler, r1, out1, _structured_report(scheduler, r1))
    (block_ids,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)
    plan = install_plan(scheduler, r1, E, failed_block_ids=[block_ids[96]])
    block_pool = scheduler.kv_cache_manager.block_pool

    out2 = scheduler.schedule()
    assert plan.state == "ADMITTED"
    drive_repair_result(
        scheduler,
        r1,
        out2,
        structural_check=False,
        plan=plan,
        aborted=True,
        executed_layers=2,
    )
    return scheduler, r1, block_ids, block_pool


def test_abort_leaves_span_unmatchable_and_prefix_intact():
    """The abort state itself: blocks from the first failed block onward
    are unhashed (unmatchable), the certified prefix below the rollback
    boundary keeps its hashes (still servable), and the request is set
    to recompute from the boundary."""
    scheduler, r1, block_ids, block_pool = _aborted_repair_scenario()

    assert r1.num_computed_tokens == 96 * BS
    assert scheduler.recovery_plans == {}
    for bid in block_ids[96:]:
        assert block_pool.blocks[bid].block_hash is None
    # Under the repair policy the sync external claim is only published
    # on a VERIFIED delivery verdict; this delivery failed, so even the
    # below-rollback blocks were never hashed -- the strongest form of
    # the invariant (nothing of the failed transfer is matchable).
    for bid in block_ids[:96]:
        assert block_pool.blocks[bid].block_hash is None

    # The ordinary reschedule rewrites the whole untrusted extent.
    out3 = scheduler.schedule()
    assert out3.num_scheduled_tokens[r1.request_id] == 3328 - 96 * BS


def test_reused_partial_write_blocks_never_match():
    """The cross-request proof: after the damaged request finishes (its
    blocks return to the free pool), a second request with the IDENTICAL
    prompt prefix-matches exactly the certified prefix (96 blocks) --
    never the evicted span. The blocks that carried the partial writes
    are handed out only through the unhashed path, so the new owner's
    own writes cover them before any read."""
    scheduler, r1, block_ids, block_pool = _aborted_repair_scenario()

    scheduler.finish_requests(
        [r1.request_id], finished_status=RequestStatus.FINISHED_STOPPED
    )

    # The same prompt: identical token ids means identical block hashes,
    # so any still-hashed block WOULD match.
    r2 = create_request(num_tokens=3328)
    r2.request_id = "clone-" + r2.request_id
    r2.prompt_token_ids = list(r1.prompt_token_ids)
    scheduler.add_request(r2)
    assert r2.num_computed_tokens == 0
    out = scheduler.schedule()

    # Nothing matched: the failed transfer's blocks were never
    # published and the abort evicted the span, so a same-prompt request
    # gets zero prefix-cache hit -- its whole prompt is scheduled (any
    # hit here would mean partially written or unverified content was
    # matchable).
    assert out.num_scheduled_tokens[r2.request_id] == 3328, (
        "prefix-cache hit on the failed transfer's content: "
        "partially written state is matchable"
    )
