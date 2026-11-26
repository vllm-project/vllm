# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Percy W1: the worker verifies every row delta and refuses drift.

Real SchedulerOutputs from a real Scheduler are applied to `RowVerifier`s in
the model runner's order. A correct stream is never refused, and the block
table the plans produce equals the scheduler's rows; a dropped delta, a
corrupted gen or a msg_seq gap is refused, identically on every "rank".
"""

import copy
import random
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor
from vllm.v1.core.grant_table import NULL_BLOCK_ID, NULL_GEN, row_digest
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.single_type_kv_cache_manager import (
    MambaManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.percy_rows import RowVerifier
from vllm.v1.worker.utils import copy_kv_cache_blocks_inplace

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16


class FakeWorker:
    """The model runner's W1 path: a RowVerifier plus a block table built
    only from the plans' writes (one contiguous write per row per group)."""

    def __init__(
        self, num_groups: int, num_blocks: int, state_groups: tuple[int, ...] = ()
    ) -> None:
        self.verifier = RowVerifier(num_groups, num_blocks, state_groups)
        self.table: dict[str, list[list[int]]] = {}
        self.num_groups = num_groups
        self.refused_per_step: list[set[str]] = []

    def _write(self, req_id: str, plan) -> None:
        rows = self.table.setdefault(req_id, [[] for _ in range(self.num_groups)])
        for group, write in enumerate(plan.writes):
            if write is not None:
                start, ids = write
                rows[group] = rows[group][:start] + list(ids)

    def apply(self, so: SchedulerOutput) -> set[str]:
        verdict = self.verifier.begin_step(so)
        for req_id in so.finished_req_ids | (so.preempted_req_ids or set()):
            self.table.pop(req_id, None)
        for new in so.scheduled_new_reqs:
            self.table.pop(new.req_id, None)
            self._write(
                new.req_id,
                self.verifier.add_new(
                    new.req_id,
                    new.block_ids,
                    new.block_gens,
                    new.expected_row_len,
                    new.row_digest,
                ),
            )
        cached = so.scheduled_cached_reqs
        checked = len(cached.expected_row_len) == len(cached.req_ids)
        for i, req_id in enumerate(cached.req_ids):
            self._write(
                req_id,
                self.verifier.apply_cached(
                    req_id,
                    cached.new_block_ids[i],
                    cached.new_block_gens[i] if checked else None,
                    cached.revoked[i] if checked else None,
                    cached.expected_row_len[i] if checked else (),
                    cached.row_digest[i] if checked else (),
                ),
            )
        self.refused_per_step.append(set(verdict.refused))
        return verdict.refused


def _model_output(
    so: SchedulerOutput, rng: random.Random, refused: set[str] = frozenset()
) -> ModelRunnerOutput:
    req_ids = list(so.num_scheduled_tokens)
    sampled = []
    for req_id in req_ids:
        num_drafts = len(so.scheduled_spec_decode_tokens.get(req_id, ()))
        sampled.append([1] * (1 + rng.randint(0, num_drafts)))
    return ModelRunnerOutput(
        req_ids=req_ids, sampled_token_ids=sampled, row_check_failures=set(refused)
    )


def _drive(
    scheduler,
    arrivals: dict[int, list],
    num_steps: int,
    *,
    depth: int = 1,
    num_drafts: int = 0,
    seed: int = 0,
    tamper=None,
    num_ranks: int = 1,
) -> list[FakeWorker]:
    """Run the scheduler like the engine and feed every dispatched output to
    `num_ranks` workers. `tamper(step, so)` may corrupt a rank-shared copy."""
    rng = random.Random(seed)
    num_groups = len(scheduler.kv_cache_manager.coordinator.single_type_managers)
    num_blocks = scheduler.kv_cache_manager.block_pool.num_gpu_blocks
    state_groups = tuple(
        g
        for g, m in enumerate(
            scheduler.kv_cache_manager.coordinator.single_type_managers
        )
        if isinstance(m, MambaManager)
    )
    workers = [
        FakeWorker(num_groups, num_blocks, state_groups) for _ in range(num_ranks)
    ]
    in_flight: deque[tuple[SchedulerOutput, set[str]]] = deque()
    for step in range(num_steps):
        for request in arrivals.get(step, ()):
            scheduler.add_request(request)
        if scheduler.has_requests():
            so = scheduler.schedule()
            sent = so
            if tamper is not None:
                sent = copy.deepcopy(so)
                tamper(step, sent)
            refused = [w.apply(sent) for w in workers]
            assert all(r == refused[0] for r in refused), "ranks must agree"
            in_flight.append((so, refused[0]))
        while in_flight and (len(in_flight) >= depth or not scheduler.has_requests()):
            done, refused = in_flight.popleft()
            scheduler.update_from_output(done, _model_output(done, rng, refused))
            if num_drafts and depth == 1:
                running = [r.request_id for r in scheduler.running]
                scheduler.update_draft_token_ids(
                    DraftTokenIds(running, [[2] * num_drafts for _ in running])
                )
    return workers


def _assert_rows_match_scheduler(scheduler, worker: FakeWorker) -> None:
    managers = scheduler.kv_cache_manager.coordinator.single_type_managers
    for req_id, rows in worker.table.items():
        if req_id not in scheduler.requests:
            continue
        for group, manager in enumerate(managers):
            blocks = manager.req_to_blocks.get(req_id)
            if blocks is not None:
                assert rows[group] == [b.block_id for b in blocks]


def _sliding_window_scheduler(**kwargs):
    return create_scheduler(
        block_size=BLOCK_SIZE,
        kv_cache_layer_config=SlidingWindowManager.layer_config(
            AttentionSpec(
                block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32
            ),
            sliding_window=BLOCK_SIZE,
        ),
        **kwargs,
    )


# --------------------------------------------------- (a) no false refusals
def test_prefill_decode_and_prefix_hits_never_refused():
    scheduler = create_scheduler(
        enable_prefix_caching=True, max_num_batched_tokens=48, block_size=BLOCK_SIZE
    )
    reqs = create_requests(
        4, num_tokens=100, max_tokens=12, same_prompt=True, block_size=BLOCK_SIZE
    )
    (worker,) = _drive(scheduler, {0: reqs[:1], 6: reqs[1:]}, num_steps=40)
    assert not any(worker.refused_per_step)
    assert all(r.is_finished() for r in reqs)


def test_preemption_and_resume_never_refused():
    scheduler = create_scheduler(num_blocks=14, block_size=BLOCK_SIZE)
    reqs = create_requests(3, num_tokens=60, max_tokens=40, block_size=BLOCK_SIZE)
    preempted: set[str] = set()
    orig_schedule = scheduler.schedule

    def schedule(*args, **kwargs):
        so = orig_schedule(*args, **kwargs)
        preempted.update(so.preempted_req_ids or ())
        return so

    scheduler.schedule = schedule
    (worker,) = _drive(scheduler, {0: reqs}, num_steps=80)
    assert preempted, "the scenario must preempt and resume"
    assert not any(worker.refused_per_step)
    assert all(r.is_finished() for r in reqs)


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_sliding_window_revokes_never_refused(async_scheduling: bool):
    scheduler = _sliding_window_scheduler(async_scheduling=async_scheduling)
    reqs = create_requests(2, num_tokens=100, max_tokens=40, block_size=BLOCK_SIZE)
    (worker,) = _drive(
        scheduler, {0: reqs}, num_steps=60, depth=2 if async_scheduling else 1
    )
    assert not any(worker.refused_per_step)
    assert all(r.is_finished() for r in reqs)


def test_sliding_window_plan_reproduces_scheduler_rows():
    scheduler = _sliding_window_scheduler()
    (request,) = create_requests(1, num_tokens=100, max_tokens=8)
    rng = random.Random(0)
    worker = FakeWorker(1, scheduler.kv_cache_manager.block_pool.num_gpu_blocks)
    scheduler.add_request(request)
    saw_revoke = False
    for _ in range(6):
        so = scheduler.schedule()
        saw_revoke |= any(so.scheduled_cached_reqs.revoked or ())
        assert not worker.apply(so)
        _assert_rows_match_scheduler(scheduler, worker)
        scheduler.update_from_output(so, _model_output(so, rng))
    assert saw_revoke
    assert NULL_BLOCK_ID in worker.table[request.request_id][0]


def test_spec_decode_with_rejections_never_refused():
    scheduler = create_scheduler(num_speculative_tokens=3, block_size=BLOCK_SIZE)
    reqs = create_requests(3, num_tokens=30, max_tokens=60, block_size=BLOCK_SIZE)
    (worker,) = _drive(scheduler, {0: reqs}, num_steps=60, num_drafts=3, seed=1)
    assert not any(worker.refused_per_step)
    assert all(r.is_finished() for r in reqs)


def test_async_scheduling_never_refused():
    scheduler = create_scheduler(
        async_scheduling=True, enable_prefix_caching=True, block_size=BLOCK_SIZE
    )
    reqs = create_requests(
        3, num_tokens=70, max_tokens=20, same_prompt=True, block_size=BLOCK_SIZE
    )
    (worker,) = _drive(scheduler, {0: reqs[:1], 3: reqs[1:]}, num_steps=60, depth=2)
    assert not any(worker.refused_per_step)
    assert all(r.is_finished() for r in reqs)


def test_plans_reproduce_scheduler_rows_every_step():
    scheduler = create_scheduler(
        enable_prefix_caching=True, max_num_batched_tokens=40, block_size=BLOCK_SIZE
    )
    reqs = create_requests(
        3, num_tokens=90, max_tokens=10, same_prompt=True, block_size=BLOCK_SIZE
    )
    rng = random.Random(0)
    worker = FakeWorker(1, scheduler.kv_cache_manager.block_pool.num_gpu_blocks)
    for r in reqs:
        scheduler.add_request(r)
    while scheduler.has_requests():
        so = scheduler.schedule()
        assert not worker.apply(so)
        _assert_rows_match_scheduler(scheduler, worker)
        scheduler.update_from_output(so, _model_output(so, rng))


# --------------------------------------------- (b) dropped delta, two ranks
def test_dropped_delta_is_refused_on_every_rank_only_for_its_request():
    scheduler = create_scheduler(block_size=BLOCK_SIZE)
    reqs = create_requests(3, num_tokens=40, max_tokens=40, block_size=BLOCK_SIZE)
    victim = reqs[1].request_id
    dropped = []

    def tamper(step, so):
        cached = so.scheduled_cached_reqs
        if dropped or victim not in cached.req_ids:
            return
        i = cached.req_ids.index(victim)
        if cached.new_block_ids[i] is not None and any(cached.new_block_ids[i]):
            cached.new_block_ids[i] = None
            cached.new_block_gens[i] = None
            dropped.append(step)

    workers = _drive(scheduler, {0: reqs}, num_steps=80, tamper=tamper, num_ranks=2)
    assert dropped, "the scenario must drop one append"
    refused = set().union(*workers[0].refused_per_step)
    assert refused == {victim}
    # Every request finished: the refused one was resynced and recomputed.
    assert all(r.is_finished() for r in reqs)
    assert workers[0].refused_per_step == workers[1].refused_per_step


# ------------------------------------------------------ (c) msg_seq checks
def test_msg_seq_gap_refuses_the_whole_step():
    scheduler = create_scheduler(block_size=BLOCK_SIZE)
    reqs = create_requests(3, num_tokens=20, max_tokens=30, block_size=BLOCK_SIZE)
    gap_step = []

    def tamper(step, so):
        if step == 4:
            so.msg_seq += 1
            gap_step.append(set(so.num_scheduled_tokens))

    (worker,) = _drive(scheduler, {0: reqs}, num_steps=80, tamper=tamper)
    assert worker.refused_per_step[4] == gap_step[0]
    assert all(r.is_finished() for r in reqs)


def test_msg_seq_zero_is_unchecked_and_first_seq_is_baseline():
    verifier = RowVerifier(1, 8)
    so = SimpleNamespace(
        msg_seq=0,
        gen_updates=None,
        finished_req_ids=set(),
        preempted_req_ids=None,
    )
    assert not verifier.begin_step(so).step_refused
    so.msg_seq = 41  # a restarted worker's first checked message
    assert not verifier.begin_step(so).step_refused
    so.msg_seq = 0
    assert not verifier.begin_step(so).step_refused
    so.msg_seq = 42
    assert not verifier.begin_step(so).step_refused
    so.msg_seq = 42  # a repeat
    assert verifier.begin_step(so).step_refused


def test_gen_updates_reach_the_cpu_mirror():
    verifier = RowVerifier(1, 8)
    assert verifier.gen_mirror[NULL_BLOCK_ID] == NULL_GEN
    assert verifier.gen_mirror[3] == 1
    so = SimpleNamespace(
        msg_seq=1,
        gen_updates=([3, 5], [2, 7]),
        finished_req_ids=set(),
        preempted_req_ids=None,
    )
    verifier.begin_step(so)
    assert verifier.gen_mirror.tolist()[3] == 2
    assert verifier.gen_mirror.tolist()[5] == 7


# ----------------------------------------------------- (d) corrupted gen
def test_corrupted_append_gen_is_refused():
    scheduler = create_scheduler(block_size=BLOCK_SIZE)
    reqs = create_requests(2, num_tokens=40, max_tokens=40, block_size=BLOCK_SIZE)
    victim = reqs[0].request_id
    corrupted = []

    def tamper(step, so):
        cached = so.scheduled_cached_reqs
        if corrupted or victim not in cached.req_ids:
            return
        i = cached.req_ids.index(victim)
        gens = cached.new_block_gens[i]
        if gens is not None and gens[0]:
            gens[0][0] += 1
            corrupted.append(step)

    (worker,) = _drive(scheduler, {0: reqs}, num_steps=80, tamper=tamper)
    assert corrupted
    assert worker.refused_per_step[corrupted[0]] == {victim}
    assert all(r.is_finished() for r in reqs)


def test_refused_row_points_at_the_null_block():
    verifier = RowVerifier(2, 16)
    plan = verifier.add_new(
        "r",
        ([3, 4], [5]),
        ([1, 1], [1]),
        expected_row_len=(2, 1),
        expected_digest=(0, 0),
    )
    assert plan.refused
    assert plan.writes == [(0, [NULL_BLOCK_ID] * 2), (0, [NULL_BLOCK_ID])]
    # It stays refused (rows stay nulled) until dropped or re-sent in full.
    plan = verifier.apply_cached("r", ([6], []), ([1], []))
    assert plan.refused and plan.writes[0] == (0, [NULL_BLOCK_ID] * 3)


# ------------------------------------------- (e) revokes before appends
def test_revoke_and_append_in_one_message_verify():
    verifier = RowVerifier(1, 16)
    verifier.add_new(
        "r", ([3, 4],), ([1, 1],), (2,), (row_digest([0, 1], [3, 4], [1, 1]),)
    )
    # Revoke column 1, then append block 5: [3, null, 5].
    expected = row_digest([0, 1, 2], [3, NULL_BLOCK_ID, 5], [1, NULL_GEN, 1])
    plan = verifier.apply_cached(
        "r",
        ([5],),
        ([1],),
        revoked=((0, 1),),
        expected_row_len=(3,),
        expected_digest=(expected,),
    )
    assert not plan.refused
    # One contiguous write from the revoked column through the appends.
    assert plan.writes == [(1, [NULL_BLOCK_ID, 5])]


def test_revoke_of_a_missing_column_is_refused():
    verifier = RowVerifier(1, 16)
    verifier.add_new("r", ([3],), ([1],), (1,), (row_digest([0], [3], [1]),))
    plan = verifier.apply_cached(
        "r",
        None,
        None,
        revoked=((0, 4),),
        expected_row_len=(1,),
        expected_digest=(row_digest([0], [3], [1]),),
    )
    assert plan.refused


# ------------------------------------------------- BlockTables.write_row
def test_block_tables_write_row_is_one_staged_write():
    staged: list[tuple[int, int, list[int]]] = []
    table = SimpleNamespace(
        gpu=torch.zeros(4, 8, dtype=torch.int32),
        stage_write=lambda idx, start, ids: staged.append((idx, start, list(ids))),
    )
    bt = BlockTables.__new__(BlockTables)
    bt.block_tables = [table]
    bt.num_blocks = SimpleNamespace(np=np.zeros((1, 4), dtype=np.int32))
    bt.num_kv_cache_groups = 1
    bt.write_row(2, 0, 1, [0, 9, 7])
    assert staged == [(2, 1, [0, 9, 7])]
    assert bt.num_blocks.np[0, 2] == 4
    # append_block_ids keeps its semantics on top of write_row.
    bt.append_block_ids(2, ([5],), overwrite=False)
    assert staged[-1] == (2, 4, [5]) and bt.num_blocks.np[0, 2] == 5
    with pytest.raises(RuntimeError):
        bt.write_row(2, 0, 6, [1, 2, 3])


# ===================================================================== W5
# Writers that address blocks by id (zeroing, CoW copies, recurrent state)
# are checked against the worker's gen mirror instead of slot_mapping.


def _w5_step(seq: int, **fields) -> SimpleNamespace:
    """A SchedulerOutput stand-in carrying only what `begin_step` reads."""
    base = dict(
        msg_seq=seq,
        gen_updates=None,
        finished_req_ids=set(),
        preempted_req_ids=None,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
        new_block_ids_to_zero=None,
        new_block_gens_to_zero=None,
        kv_cache_block_copies=None,
        kv_cache_block_copy_gens=None,
    )
    base.update(fields)
    return SimpleNamespace(**base)


def _held_row(verifier: RowVerifier, req_id: str, block_ids, gens) -> None:
    plan = verifier.add_new(req_id, (list(block_ids),), (list(gens),))
    assert not plan.refused


def test_stale_zero_is_skipped_reported_and_holder_refused():
    verifier = RowVerifier(1, 16)
    _held_row(verifier, "a", [3], [1])
    _held_row(verifier, "b", [5], [1])
    # Block 3 was revoked (gen 2) but the zero list still names gen 1.
    verdict = verifier.begin_step(
        _w5_step(
            1,
            gen_updates=([3], [2]),
            new_block_ids_to_zero=[3, 5],
            new_block_gens_to_zero=[1, 1],
        )
    )
    assert verdict.zero_block_ids == [5]
    assert verdict.skipped_block_ids == {3}
    assert verifier.num_skipped_zeros == 1
    # The request holding block 3 is refused when it is next applied; the
    # other proceeds.
    assert verifier.apply_cached("a", None).refused
    assert not verifier.apply_cached("b", None).refused
    assert verdict.refused == {"a"}


def test_zero_aimed_at_null_block_is_skipped():
    verifier = RowVerifier(1, 8)
    verdict = verifier.begin_step(
        _w5_step(1, new_block_ids_to_zero=[0, 4], new_block_gens_to_zero=[1, 1])
    )
    assert verdict.zero_block_ids == [4]
    assert verdict.skipped_block_ids == {0}


def test_unchecked_zero_and_copy_lists_pass_through():
    verifier = RowVerifier(1, 8)
    copies = [KVCacheBlockCopy(2, 3)]
    verdict = verifier.begin_step(
        _w5_step(1, new_block_ids_to_zero=[4, 5], kv_cache_block_copies=copies)
    )
    assert verdict.zero_block_ids == [4, 5]
    assert verdict.block_copies == copies
    assert not verdict.skipped_block_ids


@pytest.mark.parametrize("stale", ["src", "dst"])
def test_stale_cow_copy_is_skipped_and_block_unchanged(stale: str):
    num_blocks = 8
    verifier = RowVerifier(1, num_blocks)
    _held_row(verifier, "r", [3], [1])
    copies = [KVCacheBlockCopy(2, 3), KVCacheBlockCopy(4, 5)]
    stale_block = 2 if stale == "src" else 3
    verdict = verifier.begin_step(
        _w5_step(
            1,
            gen_updates=([stale_block], [9]),
            kv_cache_block_copies=copies,
            kv_cache_block_copy_gens=[(1, 1), (1, 1)],
        )
    )
    assert verdict.block_copies == [KVCacheBlockCopy(4, 5)]
    assert verdict.skipped_block_ids == {2, 3}
    assert verifier.num_skipped_copies == 1
    assert verifier.apply_cached("r", None).refused

    # Launch only the checked copies, as the runner does: the stale copy's
    # destination keeps its bytes, the checked one is copied.
    kv = torch.arange(num_blocks * 4, dtype=torch.float32).reshape(num_blocks, 4)
    before = kv.clone()
    copy_kv_cache_blocks_inplace([kv], num_blocks, verdict.block_copies)
    assert torch.equal(kv[3], before[3])
    assert torch.equal(kv[5], before[4])


def _mamba_scheduler(prefix_caching: bool, **kwargs):
    spec = MambaSpec(block_size=BLOCK_SIZE, shapes=((1, 1),), dtypes=(torch.float32,))
    layer = (
        MambaManager.layer_config(
            spec,
            enable_prefix_caching=True,
            num_prefill_checkpoint_blocks=1,
            prefill_checkpoint_alignment=BLOCK_SIZE,
        )
        if prefix_caching
        else MambaManager.layer_config(
            spec, enable_prefix_caching=False, num_speculative_blocks=0
        )
    )
    return create_scheduler(
        block_size=BLOCK_SIZE,
        enable_prefix_caching=prefix_caching,
        mamba_layer_config=layer,
        skip_tokenizer_init=True,
        **kwargs,
    )


@pytest.mark.parametrize("prefix_caching", [False, True])
def test_hybrid_state_rows_never_refused(prefix_caching: bool):
    """No false refusals on a hybrid (attention + Mamba) model: state rows
    only hold blocks at the gens they were granted."""
    scheduler = _mamba_scheduler(prefix_caching)
    managers = scheduler.kv_cache_manager.coordinator.single_type_managers
    assert any(isinstance(m, MambaManager) for m in managers)
    reqs = create_requests(
        6, num_tokens=70, max_tokens=24, same_prompt=True, block_size=BLOCK_SIZE
    )
    (worker,) = _drive(scheduler, {0: reqs[:3], 5: reqs[3:]}, num_steps=120)
    assert worker.verifier.state_groups
    assert not any(worker.refused_per_step)
    assert worker.verifier.num_stale_state_refusals == 0
    assert all(r.is_finished() for r in reqs)


def test_forced_revoke_of_held_state_block_redirects_state_indices():
    """T7 revokes a held published state block without a row revoke: the gen
    check refuses the request, its row (and so its state indices) point at
    the null block, which the state kernels skip; other rows are intact."""
    state = 1  # group 0 is attention
    verifier = RowVerifier(2, 32, state_groups=(state,))
    for req_id, (attn_ids, state_ids) in {
        "victim": ([4, 5], [0, 7, 8]),
        "other": ([9], [10, 11]),
    }.items():
        rows = (attn_ids, state_ids)
        exp_len = tuple(len(ids) for ids in rows)
        digests = tuple(
            row_digest(range(len(ids)), ids, [NULL_GEN if b == 0 else 1 for b in ids])
            for ids in rows
        )
        gens = tuple([NULL_GEN if b == 0 else 1 for b in ids] for ids in rows)
        assert not verifier.add_new(req_id, rows, gens, exp_len, digests).refused

    verifier.begin_step(_w5_step(1, gen_updates=([8], [2])))
    plan = verifier.apply_cached("victim", None)
    other = verifier.apply_cached("other", None)
    assert plan.refused and not other.refused
    assert verifier.num_stale_state_refusals == 1
    assert plan.writes[state] == (0, [NULL_BLOCK_ID] * 3)

    # Gather state indices from the resulting block table as the Mamba
    # metadata builders do (prefix caching: the last block of each row).
    table = torch.tensor([[NULL_BLOCK_ID] * 3, [10, 11, 0]], dtype=torch.int32)
    seq_lens = torch.tensor([3 * BLOCK_SIZE, 2 * BLOCK_SIZE], dtype=torch.int32)
    spec = MambaSpec(block_size=BLOCK_SIZE, shapes=((1, 1),), dtypes=(torch.float32,))
    indices = mamba_get_block_table_tensor(table, seq_lens, spec, True, 0)
    assert indices.flatten().tolist() == [NULL_BLOCK_ID, 11]


def test_stale_check_ignores_attention_groups():
    """Only recurrent-state groups get the row-wide gen check; attention
    writes are checked per token by W2."""
    verifier = RowVerifier(1, 16)
    _held_row(verifier, "r", [3], [1])
    verifier.begin_step(_w5_step(1, gen_updates=([3], [2])))
    assert not verifier.apply_cached("r", None).refused


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_state_update_skips_redirected_index_on_gpu():
    """Real Triton conv-state update: an index redirected to the null block
    leaves the revoked block's state untouched."""
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )

    dim, width, num_blocks = 8, 4, 6
    conv_state = torch.randn(num_blocks, dim, width - 1, device="cuda")
    before = conv_state.clone()
    x = torch.randn(2, dim, device="cuda")
    weight = torch.randn(dim, width, device="cuda")
    indices = torch.tensor([NULL_BLOCK_ID, 4], dtype=torch.int32, device="cuda")
    causal_conv1d_update(x, conv_state, weight, conv_state_indices=indices)
    assert torch.equal(conv_state[1:4], before[1:4])
    assert not torch.equal(conv_state[4], before[4])
