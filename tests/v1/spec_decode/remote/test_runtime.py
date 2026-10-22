# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone own-KV runtime: draft KV pool, step planning, and the
target-conditioned adapter driven end to end through the server."""

import threading
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.v1.spec_decode.remote.capabilities import (
    SpeculatorPlacementCapabilities,
    TargetFeatureKind,
)
from vllm.v1.spec_decode.remote.protocol import (
    CloseSequence,
    FeatureSlot,
    OpenSequence,
    RemoteServerLimits,
    SequenceKey,
    SpeculatorStatusCode,
    TargetFeatureSchema,
)
from vllm.v1.spec_decode.remote.runtime.adapter import (
    DraftStepExecutor,
    TargetConditionedDraftAdapter,
)
from vllm.v1.spec_decode.remote.runtime.kv import (
    NULL_BLOCK_ID,
    DraftBlockPool,
    DraftKVExhausted,
)
from vllm.v1.spec_decode.remote.runtime.step import (
    BlockTableUpdate,
    DraftStepPlan,
    DraftStepPlanner,
    DraftStepRow,
)
from vllm.v1.spec_decode.remote.server import (
    RemoteDraftServer,
    SequenceCleanupFailed,
    _ServerSession,
)
from vllm.v1.spec_decode.remote.session import RemoteDraftSession, VerifierIdentity
from vllm.v1.spec_decode.remote.state import (
    FeatureBatch,
    RemoteProposalBatch,
    SequenceState,
)

OK = SpeculatorStatusCode.OK
K = 3
VOCAB = 1000
BLOCK_SIZE = 2
STRIDE = 1 << 20


def key(sequence_id: int) -> SequenceKey:
    return SequenceKey("verifier", sequence_id, 0)


def make_planner(num_blocks=64, max_num_reqs=4, max_model_len=64):
    return DraftStepPlanner(
        DraftBlockPool(num_blocks, BLOCK_SIZE),
        max_num_reqs=max_num_reqs,
        max_model_len=max_model_len,
        num_speculative_tokens=K,
    )


# ----------------------------------------------------------------------
# Block pool
# ----------------------------------------------------------------------


def test_pool_never_hands_out_the_null_block():
    pool = DraftBlockPool(num_blocks=4, block_size=2)
    assert pool.ensure(0, 6) == [1, 2, 3]
    assert NULL_BLOCK_ID not in pool.blocks(0)
    assert pool.num_free_blocks == 0


def test_pool_growth_is_all_or_nothing():
    pool = DraftBlockPool(num_blocks=4, block_size=2)
    assert pool.ensure(0, 3) == [1, 2]
    with pytest.raises(DraftKVExhausted):
        pool.ensure(1, 4)
    assert pool.blocks(1) == ()
    assert pool.ensure(0, 4) == []
    assert pool.ensure(0, 5) == [3]


class _FailNextInsert(dict):
    def __setitem__(self, key, value):
        if not getattr(self, "failed", False):
            self.failed = True
            raise MemoryError("injected ownership allocation failure")
        super().__setitem__(key, value)


@pytest.mark.parametrize("existing_positions", [0, 2])
def test_pool_metadata_oom_preserves_capacity(existing_positions):
    pool = DraftBlockPool(num_blocks=4, block_size=2)
    if existing_positions:
        pool.ensure(0, existing_positions)
    previous = pool.blocks(0)
    free = pool.num_free_blocks
    pool._owned = _FailNextInsert(pool._owned)
    with pytest.raises(MemoryError, match="ownership allocation"):
        pool.ensure(0, 6)
    assert pool.blocks(0) == previous
    assert pool.num_free_blocks == free
    pool.release(0)
    # Request cleanup must recover all capacity, including a failed growth.
    assert pool.ensure(1, 6) == [1, 2, 3]


def test_pool_release_returns_blocks_and_slots_cross_blocks():
    pool = DraftBlockPool(num_blocks=8, block_size=2)
    pool.ensure(0, 2)
    pool.ensure(1, 4)
    assert pool.slot_mapping(1, np.arange(4)).tolist() == [4, 5, 6, 7]
    pool.release(1)
    assert pool.num_free_blocks == 6
    assert pool.ensure(2, 2) == [2]


class _NoGrowthList(list):
    def append(self, value):
        raise MemoryError("injected free-list growth failure")

    def extend(self, values):
        raise MemoryError("injected free-list growth failure")


class _FailReleaseWrite(list):
    def __init__(self, values, fail_after):
        super().__init__(values)
        self.remaining = fail_after

    def __setitem__(self, index, value):
        if self.remaining == 0:
            self.remaining = -1
            raise MemoryError("injected free-list write failure")
        self.remaining -= 1
        super().__setitem__(index, value)


def test_pool_release_does_not_grow_free_storage():
    pool = DraftBlockPool(num_blocks=4, block_size=2)
    pool.ensure(0, 6)
    pool._free = _NoGrowthList(pool._free)
    pool.release(0)
    pool.release(0)
    assert pool.num_free_blocks == 3
    assert pool.ensure(1, 6) == [1, 2, 3]


@pytest.mark.parametrize("fail_after", [0, 1])
def test_failed_pool_release_preserves_ownership_until_retry(fail_after):
    pool = DraftBlockPool(num_blocks=4, block_size=2)
    pool.ensure(0, 4)
    pool._free = _FailReleaseWrite(pool._free, fail_after)
    with pytest.raises(MemoryError, match="free-list write"):
        pool.release(0)
    assert pool.blocks(0) == (1, 2)
    assert pool.num_free_blocks == 1
    assert pool.ensure(1, 2) == [3]
    pool.release(0)
    pool.release(1)
    assert pool.num_free_blocks == 3
    assert set(pool.ensure(2, 6)) == {1, 2, 3}


# ----------------------------------------------------------------------
# Step planning
# ----------------------------------------------------------------------


def test_plan_mirrors_a_mixed_runner_step():
    planner = make_planner()
    for i in (1, 2):
        assert planner.open(key(i))
    plan, failures = planner.plan(
        [
            DraftStepRow(key(1), 3, num_sampled=0, next_prefill_tokens=(9,)),
            DraftStepRow(key(2), 2, num_sampled=1, last_sampled=7),
        ]
    )
    assert failures == {}
    assert plan.req_indices.tolist() == [0, 1]
    assert plan.query_start_loc.tolist() == [0, 3, 5]
    assert plan.positions.tolist() == [0, 1, 2, 0, 1]
    assert plan.seq_lens.tolist() == [3, 2]
    assert plan.next_prefill_tokens.tolist() == [[9], [0]]
    # The first draft comes from the query; only K-1 forwards need lookahead.
    assert plan.block_table_updates == {
        0: BlockTableUpdate(0, [1, 2, 3]),
        1: BlockTableUpdate(0, [4, 5]),
    }
    assert plan.slot_mapping.tolist() == [2, 3, 4, 8, 9]


def test_commit_keeps_only_the_accepted_prefix():
    planner = make_planner()
    planner.open(key(1))
    planner.commit(planner.plan([DraftStepRow(key(1), 4, num_sampled=1)])[0])
    # Verify one recovered + three drafts; two drafts are rejected.
    plan, _ = planner.plan(
        [DraftStepRow(key(1), 4, num_sampled=2, num_rejected=2, last_sampled=5)]
    )
    assert plan.positions.tolist() == [4, 5, 6, 7]
    planner.commit(plan)
    assert planner.num_computed(key(1)) == 6
    plan, _ = planner.plan([DraftStepRow(key(1), 4, num_sampled=4)])
    assert plan.positions.tolist() == [6, 7, 8, 9]
    assert plan.block_table_updates[0].start == 4


def test_rows_that_do_not_fit_fail_alone():
    planner = make_planner(num_blocks=5, max_model_len=8)
    for i in (1, 2, 3):
        planner.open(key(i))
    plan, failures = planner.plan(
        [
            DraftStepRow(key(1), 2, num_sampled=1),
            DraftStepRow(key(2), 9, num_sampled=1),
            DraftStepRow(key(3), 4, num_sampled=1),
            DraftStepRow(key(9), 1, num_sampled=1),
        ]
    )
    assert [row.key for row in plan.rows] == [key(1)]
    assert {i: status for i, (status, _) in failures.items()} == {
        1: SpeculatorStatusCode.OUT_OF_MEMORY,
        2: SpeculatorStatusCode.OUT_OF_MEMORY,
        3: SpeculatorStatusCode.SEQUENCE_DESYNC,
    }
    assert "max_model_len" in failures[1][1]


def test_request_slots_are_bounded_and_reused():
    planner = make_planner(max_num_reqs=1)
    assert planner.open(key(1))
    assert not planner.open(key(2))
    planner.commit(planner.plan([DraftStepRow(key(1), 3, num_sampled=1)])[0])
    planner.close(key(1))
    assert planner.open(key(2))
    plan, _ = planner.plan([DraftStepRow(key(2), 1, num_sampled=1)])
    assert plan.req_indices.tolist() == [0]
    assert plan.positions.tolist() == [0]
    assert plan.block_table_updates[0].start == 0


@pytest.mark.parametrize("failure", ["record", "ownership"])
def test_request_metadata_oom_preserves_slots(monkeypatch, failure):
    import vllm.v1.spec_decode.remote.runtime.step as step

    planner = make_planner(max_num_reqs=2)
    with monkeypatch.context() as patch:
        if failure == "record":

            def fail_record(*args):
                raise MemoryError("injected request allocation failure")

            patch.setattr(step, "_DraftSequence", fail_record)
        else:
            patch.setattr(planner, "_sequences", _FailNextInsert())
        with pytest.raises(MemoryError):
            planner.open(key(1))
    planner.close(key(1))
    assert planner.num_open == 0
    assert planner.open(key(2))
    assert planner.open(key(3))
    assert {planner.req_index(key(2)), planner.req_index(key(3))} == {0, 1}
    assert not planner.open(key(4))


def test_request_close_does_not_grow_free_storage():
    planner = make_planner(num_blocks=3, max_num_reqs=1)
    planner._free_slots = _NoGrowthList(planner._free_slots)
    planner.block_pool._free = _NoGrowthList(planner.block_pool._free)
    for sequence_id in range(3):
        current = key(sequence_id)
        assert planner.open(current)
        planner.block_pool.ensure(planner.req_index(current), 4)
        planner.close(current)
        planner.close(current)
        assert planner.num_open == 0
        assert planner.block_pool.num_free_blocks == 2


@pytest.mark.parametrize("failure", ["blocks", "slot"])
def test_failed_request_close_can_be_retried(monkeypatch, failure):
    planner = make_planner(num_blocks=3, max_num_reqs=1)
    planner.open(key(1))
    planner.block_pool.ensure(planner.req_index(key(1)), 4)
    with monkeypatch.context() as patch:
        if failure == "blocks":

            def fail_release(req_index):
                raise MemoryError("injected block return failure")

            patch.setattr(planner.block_pool, "release", fail_release)
        else:
            planner._free_slots = _FailReleaseWrite(planner._free_slots, 0)
        with pytest.raises(MemoryError):
            planner.close(key(1))
    assert key(1) in planner
    assert not planner.open(key(2))
    planner.close(key(1))
    assert planner.num_open == 0
    assert planner.block_pool.num_free_blocks == 2
    assert planner.open(key(2))
    assert planner.block_pool.ensure(planner.req_index(key(2)), 4) == [1, 2]


@pytest.mark.parametrize(
    "row",
    [
        DraftStepRow(key(1), 0, num_sampled=1),
        DraftStepRow(key(1), 2, num_sampled=1, num_rejected=2),
        DraftStepRow(key(1), 2, num_sampled=0),
        DraftStepRow(
            key(1), 2, num_sampled=0, num_rejected=1, next_prefill_tokens=(1,)
        ),
    ],
    ids=["empty", "all-rejected", "chunk-without-lookahead", "chunk-rejects"],
)
def test_malformed_rows_are_caller_bugs(row):
    planner = make_planner()
    planner.open(key(1))
    with pytest.raises(ValueError):
        planner.plan([row])


# ----------------------------------------------------------------------
# Adapter end to end
# ----------------------------------------------------------------------


def digest_row(prefix, next_token) -> list[int]:
    digest = sum((token + 1) * (position + 1) for position, token in enumerate(prefix))
    return [(digest + next_token + j) % VOCAB for j in range(K)]


class KVDigestExecutor(DraftStepExecutor):
    """CPU stand-in for a device executor.

    Writes (position, target token) into every token's KV slot and drafts
    from a digest of the committed prefix read back through its own block
    tables, so any position, slot, block table, or committed-length error
    in the runtime changes the drafts.
    """

    def __init__(self, pool: DraftBlockPool, max_num_tokens=None):
        self.pool = pool
        self._max_num_tokens = max_num_tokens
        self.kv = np.full(pool.num_blocks * pool.block_size, -1, dtype=np.int64)
        self.block_tables: dict[int, list[int]] = defaultdict(list)
        self.plans: list[DraftStepPlan] = []
        self.effective_ks: list[int] = []

    @property
    def max_num_tokens(self):
        return self._max_num_tokens

    def execute(self, plan, features, *, num_speculative_tokens=None):
        effective_k = K if num_speculative_tokens is None else num_speculative_tokens
        self.effective_ks.append(effective_k)
        if self.max_num_tokens is not None and plan.num_tokens > self.max_num_tokens:
            raise ValueError("draft step exceeds the configured feature-token budget")
        self.plans.append(plan)
        for req_index, update in plan.block_table_updates.items():
            table = self.block_tables[req_index][: update.start]
            self.block_tables[req_index] = table + update.block_ids
        tokens = features[0].numpy().astype(np.int64)
        assert tokens.shape == (plan.num_tokens,)
        self.kv[plan.slot_mapping] = plan.positions * STRIDE + tokens
        drafts = []
        for i in range(plan.num_reqs):
            committed = int(plan.seq_lens[i] - plan.num_rejected[i])
            table = np.array(self.block_tables[int(plan.req_indices[i])])
            positions = np.arange(committed)
            slots = table[positions // BLOCK_SIZE] * BLOCK_SIZE + positions % BLOCK_SIZE
            values = self.kv[slots]
            assert (values // STRIDE == positions).all()
            next_token = (
                plan.last_sampled[i]
                if plan.num_sampled[i]
                else plan.next_prefill_tokens[i][0]
            )
            drafts.append(
                digest_row((values % STRIDE).tolist(), int(next_token))[:effective_k]
                + [0] * (K - effective_k)
            )
        return torch.tensor(drafts)


SCHEMA = TargetFeatureSchema(
    schema_id=1,
    slots=(
        FeatureSlot(
            kind=TargetFeatureKind.TOKEN_IDS.value, dtype="int32", trailing_shape=()
        ),
        FeatureSlot(
            kind=TargetFeatureKind.NEXT_PREFILL_TOKENS.value,
            dtype="int32",
            trailing_shape=(1,),
            optional=True,
            per_sequence=True,
        ),
    ),
)
CAPABILITIES = SpeculatorPlacementCapabilities(
    state_dependency="own_kv",
    required_features=(
        TargetFeatureKind.TOKEN_IDS.value,
        TargetFeatureKind.NEXT_PREFILL_TOKENS.value,
    ),
    num_prefill_lookahead=1,
    standalone_weights="complete",
)
IDENTITY = VerifierIdentity(
    verifier_instance_id="verifier",
    target_fingerprint="",
    tokenizer_fingerprint="",
    method="mtp",
    num_speculative_tokens=K,
    parallel_drafting=False,
    provided_features=CAPABILITIES.required_features,
    max_num_seqs=4,
    max_num_batched_tokens=64,
    max_model_len=64,
    vocab_size=VOCAB,
)


@pytest.fixture
def runtime(request):
    executor_budget, wire_budget = getattr(request, "param", (None, 64))
    planner = make_planner(num_blocks=12)
    executor = KVDigestExecutor(planner.block_pool, executor_budget)
    adapter = TargetConditionedDraftAdapter(
        executor,
        planner,
        capabilities=CAPABILITIES,
        feature_schema=SCHEMA,
        methods={"mtp"},
    )
    server = RemoteDraftServer(
        adapter,
        limits=RemoteServerLimits(
            max_batch_size=4,
            max_feature_tokens=wire_budget,
            max_sequences=5,
            max_model_len=64,
        ),
    )
    server.start("tcp://127.0.0.1:0")
    session = RemoteDraftSession(
        server.endpoint,
        replace(IDENTITY, max_num_batched_tokens=wire_budget),
        request_timeout_ms=5000,
    )
    session.connect()
    yield session, executor
    session.close()
    server.stop()


def chunk(tokens, lookahead=None):
    return FeatureBatch(
        1,
        (
            torch.tensor(tokens, dtype=torch.int32),
            None
            if lookahead is None
            else torch.tensor([[lookahead]], dtype=torch.int32),
        ),
    )


def round_(
    session,
    batch_id,
    keys,
    query_lens,
    tokens,
    accepted,
    recovery,
    *,
    num_speculative_tokens=None,
):
    return session.collect(
        session.dispatch(
            RemoteProposalBatch(
                batch_id=batch_id,
                keys=tuple(keys),
                accepted_counts=torch.tensor(accepted, dtype=torch.int32),
                recovery_tokens=torch.tensor(recovery, dtype=torch.int64),
                features=chunk(tokens),
                query_lens=torch.tensor(query_lens, dtype=torch.int32),
                num_speculative_tokens=num_speculative_tokens,
            )
        )
    )


@pytest.mark.parametrize("runtime", [(None, 64), (4, 8)], indirect=True)
def test_adapter_drafts_from_the_verified_prefix(runtime):
    session, executor = runtime
    a = session.open_sequence(1)
    session.prefill(a, chunk([11, 12, 13], lookahead=14), is_final=False)
    b = session.open_sequence(2)
    session.prefill_many([(a, chunk([14, 15]), True), (b, chunk([21, 22]), True)])

    # The first round only delivers the tokens sampled from the prompts.
    result = round_(session, 0, [a, b], [0, 0], [], [0, 0], [30, 40])
    a_drafts = digest_row([11, 12, 13, 14, 15], 30)
    b_drafts = digest_row([21, 22], 40)
    assert result.row_statuses == (OK, OK)
    assert result.output.token_ids.tolist() == [a_drafts, b_drafts]

    # a accepts two drafts, b none; rejected positions must not leak into
    # the next prefix.
    result = round_(
        session,
        1,
        [a, b],
        [4, 4],
        [30, *a_drafts, 40, *b_drafts],
        [2, 0],
        [31, 41],
    )
    a_prefix = [11, 12, 13, 14, 15, 30, *a_drafts[:2]]
    assert result.output.token_ids.tolist() == [
        digest_row(a_prefix, 31),
        digest_row([21, 22, 40], 41),
    ]
    a_drafts = result.output.token_ids[0].tolist()

    # b finishes; its request slot and blocks go to a new sequence c.
    session.close_sequences((b,))
    c = session.open_sequence(3)
    session.prefill(c, chunk([7]), is_final=True)
    result = round_(session, 2, [a, c], [4, 0], [31, *a_drafts], [3, 0], [32, 50])
    assert result.output.token_ids.tolist() == [
        digest_row([*a_prefix, 31, *a_drafts], 32),
        digest_row([7], 50),
    ]
    assert result.output.valid_lengths.tolist() == [K, K]
    c_update = executor.plans[-1].block_table_updates[1]
    assert c_update.start == 0


@pytest.mark.parametrize("effective_k", [0, 1, K])
def test_round_budget_limits_generation_without_losing_own_kv(runtime, effective_k):
    session, executor = runtime
    a = session.open_sequence(1)
    session.prefill(a, chunk([11], lookahead=12), is_final=False)
    assert executor.effective_ks == [0]
    session.prefill(a, chunk([12, 13]), is_final=True)

    result = round_(
        session, 0, [a], [0], [], [0], [14], num_speculative_tokens=effective_k
    )
    used_drafts = digest_row([11, 12, 13], 14)[:effective_k]
    assert result.row_statuses == (OK,)
    assert result.output.valid_lengths.tolist() == [effective_k]
    assert result.output.token_ids.tolist() == [used_drafts + [0] * (K - effective_k)]
    assert session.sequence_state(a) is SequenceState.READY
    assert executor.effective_ks == [0, effective_k]

    # Even K=0 consumed the final prompt and committed its own-KV. Resume at
    # the first target-only query instead of skipping those context updates.
    verified = [14, *used_drafts]
    result = round_(session, 1, [a], [len(verified)], verified, [effective_k], [15])
    assert result.output.token_ids.tolist() == [digest_row([11, 12, 13, *verified], 15)]
    assert result.output.valid_lengths.tolist() == [K]
    assert executor.plans[-1].num_computed_tokens.tolist() == [3]
    assert executor.effective_ks == [0, effective_k, K]


def test_zero_budget_needs_context_kv_without_unused_draft_headroom():
    planner = make_planner(num_blocks=2)
    planner.open(key(1))
    row = DraftStepRow(key(1), 2, num_sampled=1)
    assert planner.plan([row])[1][0][0] is SpeculatorStatusCode.OUT_OF_MEMORY
    plan, failures = planner.plan([row], num_speculative_tokens=0)
    assert not failures
    planner.commit(plan)
    assert planner.num_computed(key(1)) == 2


@pytest.mark.parametrize("block_size", [4, 16, 32])
@pytest.mark.parametrize("effective_k", [0, 1])
def test_first_draft_does_not_need_kv_beyond_a_full_prompt_block(
    block_size, effective_k
):
    pool = DraftBlockPool(num_blocks=2, block_size=block_size)
    planner = DraftStepPlanner(
        pool, max_num_reqs=1, max_model_len=64, num_speculative_tokens=K
    )
    a = key(1)
    planner.open(a)
    plan, failures = planner.plan(
        [DraftStepRow(a, block_size, num_sampled=1)],
        num_speculative_tokens=effective_k,
    )
    assert not failures
    assert plan.positions.tolist() == list(range(block_size))
    assert pool.num_free_blocks == 0
    planner.commit(plan)
    assert planner.num_computed(a) == block_size


@pytest.mark.parametrize("rejected", [0, 2, 3])
@pytest.mark.parametrize("max_model_len", [8, 16])
def test_draft_kv_covers_query_and_only_executed_lookahead(rejected, max_model_len):
    pool = DraftBlockPool(num_blocks=3, block_size=4)
    planner = DraftStepPlanner(
        pool, max_num_reqs=1, max_model_len=max_model_len, num_speculative_tokens=K
    )
    a = key(1)
    planner.open(a)
    first, failures = planner.plan(
        [DraftStepRow(a, 4, num_sampled=1)], num_speculative_tokens=1
    )
    assert not failures
    planner.commit(first)
    plan, failures = planner.plan(
        [DraftStepRow(a, 4, num_sampled=4 - rejected, num_rejected=rejected)]
    )
    if rejected == 0 and max_model_len == 16:
        assert failures[0][0] is SpeculatorStatusCode.OUT_OF_MEMORY
        assert not plan.rows
        assert pool.blocks(planner.req_index(a)) == (1,)
        assert planner.num_computed(a) == 4
    else:
        assert not failures
        assert plan.positions.tolist() == [4, 5, 6, 7]
        assert pool.num_free_blocks == 0
        planner.commit(plan)
        assert planner.num_computed(a) == 8 - rejected


def test_round_must_not_resend_a_buffered_prompt_chunk(runtime):
    session, _ = runtime
    a = session.open_sequence(1)
    session.prefill(a, chunk([11, 12]), is_final=True)
    result = round_(session, 0, [a], [2], [11, 12], [0], [30])
    assert result.row_statuses == (SpeculatorStatusCode.SEQUENCE_DESYNC,)
    assert session.sequence_state(a) is SequenceState.DESYNCED


@pytest.mark.parametrize("runtime", [(4, 4)], indirect=True)
def test_buffered_final_chunks_fit_executor_budget_and_commit_each_row(runtime):
    session, executor = runtime
    a, b = session.open_sequence(1), session.open_sequence(2)
    session.prefill(a, chunk([1, 2, 3, 4]), is_final=True)
    session.prefill(b, chunk([5, 6, 7, 8]), is_final=True)

    result = round_(session, 0, [a, b], [0, 0], [], [0, 0], [9, 10])
    a_drafts, b_drafts = digest_row([1, 2, 3, 4], 9), digest_row([5, 6, 7, 8], 10)
    assert result.row_statuses == (OK, OK)
    assert result.output.valid_lengths.tolist() == [K, K]
    assert result.output.token_ids.tolist() == [a_drafts, b_drafts]
    assert [plan.num_tokens for plan in executor.plans] == [4, 4]

    # Each separate execution commits its prefix before the next round.
    result = round_(
        session, 1, [b, a], [2, 2], [10, b_drafts[0], 9, a_drafts[0]], [1, 0], [11, 12]
    )
    assert result.row_statuses == (OK, OK)
    assert result.output.valid_lengths.tolist() == [K, K]
    assert result.output.token_ids.tolist() == [
        digest_row([5, 6, 7, 8, 10, b_drafts[0]], 11),
        digest_row([1, 2, 3, 4, 9], 12),
    ]
    assert executor.plans[-1].num_tokens == 4


@pytest.mark.parametrize("runtime", [(4, 8)], indirect=True)
def test_individually_oversized_buffered_chunk_only_degrades_that_row(runtime):
    session, executor = runtime
    a, b = session.open_sequence(1), session.open_sequence(2)
    session.prefill(a, chunk([1, 2, 3, 4, 5]), is_final=True)
    session.prefill(b, chunk([6, 7]), is_final=True)

    result = round_(session, 0, [a, b], [0, 0], [], [0, 0], [8, 9])
    assert result.row_statuses == (SpeculatorStatusCode.OUT_OF_MEMORY, OK)
    assert result.output.valid_lengths.tolist() == [0, K]
    assert result.output.token_ids[1].tolist() == digest_row([6, 7], 9)
    assert session.sequence_state(a) is SequenceState.TARGET_ONLY
    assert [plan.num_tokens for plan in executor.plans] == [2]


def test_prompt_chunk_without_lookahead_is_refused(runtime):
    session, _ = runtime
    a = session.open_sequence(1)
    session.prefill(a, chunk([11, 12]), is_final=False)
    assert session.sequence_state(a) is SequenceState.TARGET_ONLY


@pytest.mark.parametrize("is_final", [False, True])
def test_empty_prompt_chunk_only_degrades_that_request(runtime, is_final):
    session, executor = runtime
    a, b = session.open_sequence(1), session.open_sequence(2)
    session.prefill_many(
        [
            (a, chunk([], lookahead=None if is_final else 9), is_final),
            (b, chunk([11, 12]), True),
        ]
    )
    assert session.sequence_state(a) is SequenceState.TARGET_ONLY
    assert session.sequence_state(b) is SequenceState.READY

    result = round_(session, 0, [a, b], [0, 0], [], [0, 0], [30, 40])
    assert result.row_statuses == (SpeculatorStatusCode.PROTOCOL_ERROR, OK)
    assert result.output.valid_lengths.tolist() == [0, K]
    assert result.output.token_ids[1].tolist() == digest_row([11, 12], 40)
    assert [plan.num_tokens for plan in executor.plans] == [2]


def test_exhausted_draft_kv_degrades_only_that_request(runtime):
    session, _ = runtime
    a = session.open_sequence(1)
    # 11 usable blocks of 2 slots cannot hold this prompt even at K=0.
    session.prefill(a, chunk(list(range(23)), lookahead=99), is_final=False)
    assert session.sequence_state(a) is SequenceState.TARGET_ONLY
    b = session.open_sequence(2)
    session.prefill(b, chunk([5]), is_final=True)
    result = round_(session, 0, [b], [0], [], [0], [6])
    assert result.row_statuses == (OK,)


def test_terminal_row_failure_returns_kv_without_disturbing_healthy_sibling(runtime):
    session, executor = runtime
    pool = executor.pool
    a, b = session.open_sequence(1), session.open_sequence(2)
    session.prefill(a, chunk(list(range(16)), lookahead=16), is_final=False)
    owned_before_failure = pool.num_free_blocks
    session.prefill(a, chunk(list(range(16, 24))), is_final=True)
    session.prefill(b, chunk([30]), is_final=True)

    result = round_(session, 0, [a, b], [0, 0], [], [0, 0], [24, 31])
    assert result.row_statuses == (SpeculatorStatusCode.OUT_OF_MEMORY, OK)
    assert session.sequence_state(a) is SequenceState.TARGET_ONLY
    assert session.sequence_state(b) is SequenceState.READY
    assert session.connected
    assert pool.num_free_blocks > owned_before_failure

    c = session.open_sequence(3)
    session.prefill(c, chunk(list(range(8))), is_final=True)
    assert round_(session, 1, [c], [0], [], [0], [8]).row_statuses == (OK,)
    assert round_(
        session, 2, [a], [0], [], [0], [25]
    ).output.valid_lengths.tolist() == [0]


@pytest.mark.parametrize("operation", ["close", "discard", "cleanup", "replace"])
@pytest.mark.parametrize("failure", [None, "once", "always"])
def test_server_retirement_preserves_real_kv_ownership(monkeypatch, operation, failure):
    """A failed host release retains its request slot and fences admission."""
    planner = make_planner(num_blocks=3, max_num_reqs=1)
    pool = planner.block_pool
    adapter = TargetConditionedDraftAdapter(
        KVDigestExecutor(pool),
        planner,
        capabilities=CAPABILITIES,
        feature_schema=SCHEMA,
        methods={"mtp"},
    )
    server = RemoteDraftServer(adapter)
    conn = Mock()
    session = _ServerSession(server, conn)
    session.session_id = "owner"
    session.verifier_instance_id = "verifier"
    current = key(1)
    assert server.registry.open(current, owner=session.session_id).created
    adapter.open_sequence(current)
    pool.ensure(planner.req_index(current), 4)
    assert pool.num_free_blocks == 0
    release = pool.release
    attempts = 0

    def release_blocks(req_index):
        nonlocal attempts
        attempts += 1
        if failure == "always" or (failure == "once" and attempts == 1):
            raise MemoryError("injected block return failure")
        release(req_index)

    monkeypatch.setattr(pool, "release", release_blocks)
    expected = (
        pytest.raises(MemoryError, match="block return") if failure else nullcontext()
    )
    with expected:
        if operation == "close":
            session._close(CloseSequence((current,)), 1)
        elif operation == "discard":
            with server.adapter_turn():
                session._discard_owned_sequences((current,))
        elif operation == "cleanup":
            session.cleanup()
        else:
            session._open(OpenSequence(SequenceKey("verifier", 1, 1)), 1)

    assert attempts == 1
    if failure:
        assert not server.healthy
        assert server.wait(0)
        assert server.registry.get(current) is not None
        assert current in planner
        assert pool.num_free_blocks == 0
        conn.send_control.assert_not_called()
        with pytest.raises(SequenceCleanupFailed):
            session.cleanup()
        with pytest.raises(SequenceCleanupFailed):
            session._open(OpenSequence(key(2)), 2)
        assert attempts == 1
        assert planner.num_open == 1
        assert not planner.open(key(2))
    else:
        assert server.healthy
        assert server.registry.get(current) is None
        assert current not in planner
        assert pool.num_free_blocks == 2
        session.cleanup()
        assert len(server.registry) == 0
        assert planner.num_open == 0
        assert planner.open(key(2))
        assert pool.ensure(planner.req_index(key(2)), 4) == [1, 2]


def test_repeated_unsent_close_timeouts_return_all_owned_kv(runtime, monkeypatch):
    session, executor = runtime
    pool = executor.pool
    free_blocks = pool.num_free_blocks
    reclaimed = threading.Event()
    release = pool.release

    def release_blocks(req_index):
        release(req_index)
        if pool.num_free_blocks == free_blocks:
            reclaimed.set()

    monkeypatch.setattr(pool, "release", release_blocks)
    send = session._send

    def expire_before_send(message, **kwargs):
        if isinstance(message, CloseSequence):
            kwargs["deadline"] = session._clock() - 1
        return send(message, **kwargs)

    previous_generation = -1
    for _ in range(3):
        key = session.open_sequence(1)
        assert key.generation > previous_generation
        previous_generation = key.generation
        session.prefill(key, chunk([1, 2, 3], lookahead=4), is_final=False)
        assert pool.num_free_blocks < free_blocks
        reclaimed.clear()
        with monkeypatch.context() as patch:
            patch.setattr(session, "_send", expire_before_send)
            session.close_sequences((key,))
        assert not session.connected
        assert reclaimed.wait(2), "connection cleanup retained draft KV"
        assert pool.num_free_blocks == free_blocks
        session.connect()

    fresh = session.open_sequence(1)
    session.prefill(fresh, chunk(list(range(20)), lookahead=20), is_final=False)
    assert session.sequence_state(fresh) is SequenceState.PREFILLING
    assert executor.plans[-1].num_tokens == 20
    session.close_sequences((fresh,))
    assert pool.num_free_blocks == free_blocks


def test_request_slots_beyond_the_runtime_are_queue_full(runtime):
    session, _ = runtime
    keys = [session.open_sequence(i) for i in range(5)]
    assert [session.sequence_state(k) for k in keys] == [
        SequenceState.PREFILLING
    ] * 4 + [SequenceState.TARGET_ONLY]
