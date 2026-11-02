# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recurrent-state blocks for draft rows follow the drafts each step verifies.

`MambaManager.allocate_new_blocks(..., num_draft_tokens=n)` holds one
speculative state block per draft, at most `num_speculative_blocks`, and never
fewer than reach the end of the previous step's span: the next step reads the
state the previous verify accepted, which may sit in any of its draft columns.
The tests replay random draft counts against a model of the worker's block
table row (truncate to the reported count, then append the new blocks).
"""

import random
from collections.abc import Callable

import pytest
import torch

from tests.v1.core.test_async_scheduler import PipelinedEngine
from tests.v1.core.test_prefix_caching import make_request
from tests.v1.core.utils import (
    create_requests,
    create_scheduler,
    new_mamba_spec,
    runner_output,
)
from vllm.utils.hashing import sha256
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import DraftTokenIds

pytestmark = pytest.mark.cpu_test


def _manager(
    mode: str, num_spec: int, block_size: int, num_blocks: int = 256
) -> tuple[MambaManager, BlockPool]:
    spec = new_mamba_spec(
        block_size=block_size,
        shapes=((1, 1),),
        mamba_cache_mode=mode,
        num_speculative_blocks=num_spec,
    )
    pool = BlockPool(
        num_gpu_blocks=num_blocks, enable_caching=False, hash_block_size=block_size
    )
    manager = MambaManager(
        spec,
        block_pool=pool,
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=block_size,
    )
    return manager, pool


class _Request:
    """One request driven through the manager plus its worker table row."""

    def __init__(self, manager: MambaManager, req_id: str, num_computed: int):
        self.manager = manager
        self.req_id = req_id
        self.num_computed = num_computed
        self.row: list[int] = []
        self.started = False
        # Columns the next step may read (the accepted state), and the end of
        # the last step's span, which an async scheduler must keep: it
        # allocates the next step before the acceptance is known.
        self.read_cols: range = range(0)
        self.span_end = 0

    @property
    def blocks(self):
        return self.manager.req_to_blocks[self.req_id]

    def _allocate(
        self, num_drafts: int | None, kept: list[tuple[int, int]]
    ) -> tuple[int, set[int]]:
        """Allocate one step verifying `num_drafts` drafts (prefill when not
        started) and apply the manager's report to the worker row. The row
        must mirror the manager's blocks and keep each `kept` (column, block
        id). Returns the step's state column and the blocks the request owns."""
        manager = self.manager
        num_tokens = self.num_computed + 1 + (num_drafts or 0)
        prev_owned = {b.block_id for b in self.blocks}
        predicted = manager.get_num_blocks_to_allocate(
            self.req_id,
            num_tokens,
            new_computed_blocks=[],
            total_computed_tokens=self.num_computed,
            num_local_computed_tokens=0,
            num_tokens_main_model=num_tokens,
            num_draft_tokens=num_drafts,
        )
        new_blocks = manager.allocate_new_blocks(
            self.req_id, num_tokens, num_tokens, num_draft_tokens=num_drafts
        )
        counts = manager.take_spec_state_block_counts()
        if not self.started:
            assert self.req_id not in counts
            self.row = []
            self.started = True
        if self.req_id in counts:
            # The worker cuts the row, then appends.
            assert counts[self.req_id] <= len(self.row)
            del self.row[counts[self.req_id] :]
        self.row.extend(b.block_id for b in new_blocks)

        owned = {b.block_id for b in self.blocks if not b.is_null}
        assert predicted >= len(owned - prev_owned)
        # The worker row mirrors the manager's list wherever it is not null.
        assert len(self.row) == len(self.blocks)
        for col, block in enumerate(self.blocks):
            if not block.is_null:
                assert self.row[col] == block.block_id
        for col, block_id in kept:
            assert self.row[col] == block_id
            assert block_id in owned
        return cdiv(num_tokens, manager.block_size) - 1, owned

    def step(self, num_drafts: int | None, num_accepted: int | None = None) -> None:
        """Allocate one step verifying `num_drafts` drafts, then accept
        `num_accepted` of its tokens (all by default)."""
        manager = self.manager
        prev_state = manager._spec_state_idx.get(self.req_id)
        # The state the previous step left stays where the worker reads it.
        state, owned = self._allocate(
            num_drafts, [(col, self.row[col]) for col in self.read_cols]
        )
        # This step writes its state and one column per draft.
        num_spec = manager.num_speculative_blocks
        span = 1 + min(num_drafts if num_drafts is not None else num_spec, num_spec)
        for col in range(state, state + span):
            assert self.row[col] in owned
        if num_drafts is not None and prev_state is not None:
            # Held no more than this step's and the last step's drafts need,
            # plus scratch blocks relocated past a crossed block boundary. A
            # rejected draft can move the state back a block; the manager
            # keeps the block it laid the state out in.
            relocated = max(state - prev_state - 1, 0)
            need = max(state + span, self.span_end, prev_state + 1)
            assert len(self.blocks) <= need + relocated
        if manager.mamba_cache_mode != "align":
            assert len(self.blocks) == max(state + span, self.span_end)

        self.span_end = state + span
        if num_accepted is None:
            num_accepted = 1 + (num_drafts or 0)
        self.num_computed += num_accepted
        # The next step reads the accepted state, column state + a - 1.
        self.read_cols = range(state + num_accepted - 1, state + num_accepted)

    def free(self) -> None:
        self.manager.free(self.req_id)
        self.started = False
        self.read_cols = range(0)
        self.span_end = 0


def _held(*managers) -> int:
    """Spec blocks the requests hold past their state block, over the managers."""
    return sum(
        len(manager.req_to_blocks[req_id]) - 1 - state_idx
        for manager in managers
        if isinstance(manager, MambaManager)
        for req_id, state_idx in manager._spec_state_idx.items()
    )


def _check_invariants(manager: MambaManager, pool: BlockPool, requests, free0):
    held = {
        r.req_id: len(r.blocks) - 1 - manager._spec_state_idx[r.req_id]
        for r in requests
        if r.started
    }
    assert {r: manager._num_spec_blocks(r) for r in held} == held
    num_owned = sum(sum(not b.is_null for b in r.blocks) for r in requests if r.started)
    assert pool.get_num_free_blocks() == free0 - num_owned


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("mode", ["none", "align"])
def test_spec_blocks_follow_random_drafts(seed: int, mode: str):
    rng = random.Random(seed)
    num_spec = rng.randint(1, 4)
    # `none` keeps the whole sequence in one block (block_size is the max
    # model length); `align` crosses a block boundary every few steps.
    block_size = 4096 if mode == "none" else rng.choice([num_spec + 1, 8])
    manager, pool = _manager(mode, num_spec, block_size)
    free0 = pool.get_num_free_blocks()
    requests = [_Request(manager, f"r{i}", 0) for i in range(3)]
    for _ in range(60):
        request = rng.choice(requests)
        if request.started and rng.random() < 0.1:
            # Preemption, possibly right after a shrink the worker never saw.
            request.free()
        elif not request.started:
            # Prefill: no drafts verified yet.
            # `none` sequences stay within their one block.
            max_prompt = 64 if mode == "none" else 3 * block_size // 2
            request.num_computed = rng.randint(0, max_prompt)
            request.step(0)
        else:
            num_drafts = rng.randint(0, num_spec)
            num_accepted = rng.randint(1, num_drafts + 1)
            request.step(num_drafts, num_accepted)
        _check_invariants(manager, pool, requests, free0)
    for request in requests:
        if request.started:
            request.free()
    assert _held(manager) == 0
    assert pool.get_num_free_blocks() == free0


class _InFlightRequest(_Request):
    """A request under async scheduling: each step is allocated while the
    previous one is still running, so the scheduler counts that step's drafts
    as accepted and rolls the rejected ones back when its output arrives."""

    def __init__(self, manager: MambaManager, req_id: str, num_computed: int):
        super().__init__(manager, req_id, num_computed)
        # Drafts of the steps allocated but not yet returned, oldest first.
        self.in_flight: list[int] = []
        # (column, block id) of the last allocation's span.
        self.last_span: list[tuple[int, int]] = []

    def allocate(self, num_drafts: int) -> None:
        # The step still running wrote its states into its span, and this
        # step reads one of them: none of those blocks went back to the pool
        # or moved, whatever the running step accepts.
        state, owned = self._allocate(num_drafts, self.last_span)
        manager = self.manager
        num_spec = manager.num_speculative_blocks
        span = range(state, state + 1 + min(num_drafts, num_spec))
        for col in span:
            assert self.row[col] in owned
        self.last_span = [(col, self.row[col]) for col in span]
        self.num_computed += 1 + num_drafts
        self.in_flight.append(num_drafts)

    def finish_oldest(self, num_accepted: int) -> None:
        """The oldest step returns: roll back its rejected drafts."""
        num_drafts = self.in_flight.pop(0)
        assert 1 <= num_accepted <= 1 + num_drafts
        self.num_computed -= 1 + num_drafts - num_accepted

    def free(self) -> None:
        super().free()
        self.in_flight = []
        self.last_span = []


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("mode", ["none", "align"])
def test_spec_blocks_with_a_step_in_flight(seed: int, mode: str):
    """Async scheduling: step N+1 is allocated before step N's acceptance is
    known, and step N's rejected drafts are rolled back before step N+2. Every
    allocation keeps the running step's span, and the worker row, the held
    sum and the pool stay consistent across rollbacks and preemptions."""
    rng = random.Random(seed)
    num_spec = rng.randint(1, 4)
    # `align` blocks small enough that drafts and rollbacks cross boundaries.
    block_size = 4096 if mode == "none" else rng.choice([num_spec + 1, 8])
    manager, pool = _manager(mode, num_spec, block_size)
    free0 = pool.get_num_free_blocks()
    requests = [_InFlightRequest(manager, f"r{i}", 0) for i in range(3)]
    for _ in range(120):
        request = rng.choice(requests)
        if request.started and rng.random() < 0.05:
            request.free()
        elif not request.started:
            max_prompt = 64 if mode == "none" else 3 * block_size // 2
            request.num_computed = rng.randint(0, max_prompt)
            request.allocate(0)
        else:
            # Budgets K+1, 2 and 1 (drafts K, 1 and 0) and anything between.
            request.allocate(rng.choice([num_spec, 1, 0, rng.randint(0, num_spec)]))
        if len(request.in_flight) == 2:
            num_drafts = request.in_flight[0]
            request.finish_oldest(rng.randint(1, num_drafts + 1))
        _check_invariants(manager, pool, requests, free0)
    for request in requests:
        if request.started:
            request.free()
    assert _held(manager) == 0
    assert pool.get_num_free_blocks() == free0


@pytest.mark.parametrize("mode", ["none", "align"])
def test_static_layout_without_draft_counts(mode: str):
    """`num_draft_tokens=None` (spec decode off, or a gated-off scheduler)
    allocates the static `1 + num_speculative_blocks` tail."""
    manager, pool = _manager(mode, 3, 4096 if mode == "none" else 4)
    request = _Request(manager, "r", 4)
    request.step(None)
    state = 1 if mode == "align" else 0
    assert len(request.blocks) == state + 1 + 3
    assert manager.take_spec_state_block_counts() == {}
    request.step(None)
    assert len(request.blocks) >= state + 1 + 3
    assert manager.take_spec_state_block_counts() == {}


@pytest.mark.parametrize("mode", ["none", "align"])
def test_no_spec_blocks_without_speculation(mode: str):
    manager, pool = _manager(mode, 0, 4096 if mode == "none" else 4)
    free0 = pool.get_num_free_blocks()
    request = _Request(manager, "r", 2)
    for num_drafts in (0, 2, 0):
        request.step(num_drafts, 1)
        assert manager.take_spec_state_block_counts() == {}
    assert manager._spec_state_idx == {}
    request.free()
    assert pool.get_num_free_blocks() == free0


def test_shrink_keeps_the_previous_span_one_step():
    """A step with fewer drafts keeps the columns the last verify wrote until
    the step after it has read the accepted state."""
    manager, _ = _manager("none", 3, 4096)
    request = _Request(manager, "r", 4)
    request.step(0)
    assert len(request.blocks) == 1
    request.step(3, 4)
    assert len(request.blocks) == 4
    assert _held(manager) == 3
    request.step(1, 2)
    assert len(request.blocks) == 4
    request.step(0, 1)
    assert len(request.blocks) == 2
    assert _held(manager) == 1
    request.step(0, 1)
    assert len(request.blocks) == 1
    assert _held(manager) == 0


def test_boundary_crossing_while_shrinking():
    """Shrinking the tail in the step whose state crosses into the next block
    keeps the accepted state and moves the state block forward."""
    manager, _ = _manager("align", 3, 4)
    request = _Request(manager, "r", 2)
    request.step(0)  # prefill 3 tokens: state block 0
    request.step(3, 4)  # tokens 3..6: state block 1, three spec blocks
    assert len(request.blocks) == 5
    request.step(0, 1)  # token 7: keeps the span the last verify wrote
    assert len(request.blocks) == 5
    assert manager.take_spec_state_block_counts() == {}
    state_block = request.row[1]
    # Token 8 crosses into block 2 while the tail shrinks to nothing.
    assert request.num_computed == 8
    manager.allocate_new_blocks("r", 9, 9, num_draft_tokens=0)
    assert manager.take_spec_state_block_counts() == {"r": 3}
    assert len(request.blocks) == 3
    assert manager.last_state_block_idx["r"] == 1
    assert request.blocks[1].block_id == state_block
    assert manager._spec_state_idx["r"] == 2
    assert _held(manager) == 0


@pytest.mark.parametrize("mode", ["none", "align"])
def test_preempt_mid_shrink(mode: str):
    """A request freed in the step that shrank it leaves no report behind and
    returns every block."""
    manager, pool = _manager(mode, 3, 4096 if mode == "none" else 8)
    free0 = pool.get_num_free_blocks()
    request = _Request(manager, "r", 2)
    request.step(0)
    request.step(3, 1)
    request.step(0, 1)
    num_tokens = request.num_computed + 1
    manager.allocate_new_blocks("r", num_tokens, num_tokens, num_draft_tokens=0)
    assert len(manager.req_to_blocks["r"]) < len(request.row)  # shrank
    manager.free("r")
    assert manager.take_spec_state_block_counts() == {}
    assert _held(manager) == 0
    assert pool.get_num_free_blocks() == free0
    # Resumed, it starts over from its prefill layout.
    resumed = _Request(manager, "r", num_tokens)
    resumed.step(0)
    assert _held(manager) == 0


def test_kv_cache_manager_threads_draft_counts():
    init_none_hash(sha256)
    block_size = 64
    config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["mamba_layer"],
                new_mamba_spec(
                    block_size=block_size,
                    shapes=((1, 1),),
                    mamba_cache_mode="none",
                    num_speculative_blocks=2,
                ),
            )
        ],
    )
    manager = KVCacheManager(
        config,
        max_model_len=block_size,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
        enable_caching=False,
    )
    request = make_request("r", list(range(5)), block_size, sha256)
    free0 = manager.block_pool.get_num_free_blocks()
    assert manager.allocate_slots(request, 5, num_draft_tokens=0) is not None
    assert manager.get_block_ids("r") and len(manager.get_block_ids("r")[0]) == 1
    request.num_computed_tokens = 5
    manager.allocate_slots(request, 3, num_draft_tokens=2)
    assert len(manager.get_block_ids("r")[0]) == 3
    assert _held(*manager.coordinator.single_type_managers) == 2
    request.num_computed_tokens = 6
    manager.allocate_slots(request, 1, num_draft_tokens=0)
    request.num_computed_tokens = 7
    manager.allocate_slots(request, 1, num_draft_tokens=0)
    assert manager.take_spec_state_block_counts() == {"r": {0: 1}}
    assert _held(*manager.coordinator.single_type_managers) == 0
    # Without a draft count the request takes the static tail back.
    request.num_computed_tokens = 8
    manager.allocate_slots(request, 1)
    assert len(manager.get_block_ids("r")[0]) == 3
    manager.free(request)
    assert manager.block_pool.get_num_free_blocks() == free0
    assert _held(*manager.coordinator.single_type_managers) == 0


def _run_hybrid_with_prefix_caching(seed: int, dynamic: bool) -> tuple[int, int]:
    """Prefill then decode one align-mode request with random drafts and
    acceptances; return the prefix a sibling with its tokens hits, in tokens,
    and how many Mamba blocks the decode held at most."""
    init_none_hash(sha256)
    rng = random.Random(seed)
    attn_block_size, mamba_block_size = 16, 64
    config = KVCacheConfig(
        num_blocks=512,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=attn_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                new_mamba_spec(
                    block_size=mamba_block_size,
                    shapes=((1, 1),),
                    mamba_cache_mode="align",
                    num_speculative_blocks=3,
                ),
            ),
        ],
    )
    manager = KVCacheManager(
        config,
        max_model_len=4096,
        scheduler_block_size=mamba_block_size,
        hash_block_size=attn_block_size,
        enable_caching=True,
    )
    prompt = [rng.randrange(1000) for _ in range(mamba_block_size)]
    request = make_request("a", prompt, attn_block_size, sha256)
    computed, num_computed, _ = manager.get_computed_blocks(request)
    assert num_computed == 0
    assert manager.allocate_slots(
        request,
        len(prompt),
        new_computed_blocks=computed,
        num_draft_tokens=0 if dynamic else None,
    )
    request.num_computed_tokens = len(prompt)
    request.append_output_token_ids(rng.randrange(1000))
    max_blocks = 0
    for _ in range(80):
        num_drafts = rng.randint(0, 3)
        num_accepted = rng.randint(1, num_drafts + 1)
        assert manager.allocate_slots(
            request,
            1 + num_drafts,
            num_draft_tokens=num_drafts if dynamic else None,
        )
        manager.take_spec_state_block_counts()
        max_blocks = max(max_blocks, len(manager.get_block_ids("a")[1]))
        request.num_computed_tokens += num_accepted
        request.append_output_token_ids(
            [rng.randrange(1000) for _ in range(num_accepted)]
        )
    sibling = make_request("b", list(request.all_token_ids), attn_block_size, sha256)
    _, num_hit, _ = manager.get_computed_blocks(sibling)
    manager.free(request)
    assert _held(*manager.coordinator.single_type_managers) == 0
    return num_hit, max_blocks


@pytest.mark.parametrize("seed", range(6))
def test_prefix_caching_hits_unchanged(seed: int):
    """Prefix-cache hits after a decode with shrinking and growing draft tails
    match the static layout's."""
    static_hit, static_blocks = _run_hybrid_with_prefix_caching(seed, dynamic=False)
    dynamic_hit, dynamic_blocks = _run_hybrid_with_prefix_caching(seed, dynamic=True)
    assert static_hit > 0
    assert dynamic_hit == static_hit
    assert dynamic_blocks <= static_blocks


# --- Scheduler ---------------------------------------------------------------


def _scheduler_steps(
    budgets: list[int], use_v2_model_runner: bool = True, num_lookahead: int = 0
):
    """Drive two decode requests of a recurrent model through one step per
    candidate budget, every draft accepted. Returns per step the scheduled
    drafts, the requests' recurrent block counts, the held sum, and the
    output's `spec_state_block_counts`."""
    scheduler = create_scheduler(
        num_speculative_tokens=3,
        max_num_batched_tokens=64,
        max_model_len=256,
        block_size=256,
        kv_cache_spec=new_mamba_spec(
            block_size=256,
            shapes=((1, 1),),
            mamba_cache_mode="none",
            num_speculative_blocks=3,
        ),
        use_v2_model_runner=use_v2_model_runner,
    )
    # Draft KV slots a drafter like EAGLE reserves; n-gram reserves none.
    scheduler.num_lookahead_tokens = num_lookahead
    budget_iter = iter(budgets)
    views = []

    def candidate_budget(view):
        views.append(view)
        return next(budget_iter)

    scheduler.speculator_config.candidate_budget = candidate_budget
    for request in create_requests(2, num_tokens=8, max_tokens=64):
        scheduler.add_request(request)
    manager = scheduler.kv_cache_manager
    steps = []
    for _ in budgets:
        output = scheduler.schedule()
        req_ids = list(output.num_scheduled_tokens)
        drafts = {
            req_id: len(output.scheduled_spec_decode_tokens.get(req_id, ()))
            for req_id in req_ids
        }
        steps.append(
            (
                drafts,
                {req_id: len(manager.get_block_ids(req_id)[0]) for req_id in req_ids},
                _held(*manager.coordinator.single_type_managers),
                output.spec_state_block_counts,
            )
        )
        scheduler.update_from_output(
            output, runner_output(req_ids, [[100] * (1 + drafts[r]) for r in req_ids])
        )
        k = output.num_spec_tokens_to_schedule
        scheduler.update_draft_token_ids(
            DraftTokenIds(req_ids, [[7] * k for _ in req_ids])
        )
    return steps, views


def test_scheduler_spec_blocks_follow_the_budget():
    steps, views = _scheduler_steps([4, 4, 2, 1, 1, 4, 4])
    drafts = [set(d.values()) for d, *_ in steps]
    blocks = [set(b.values()) for _, b, *_ in steps]
    held = [h for *_, h, _ in steps]
    counts = [c for *_, c in steps]
    # Prefill, then each step verifies the drafts the last step's budget asked.
    assert drafts == [{0}, {3}, {3}, {1}, {0}, {0}, {3}]
    # One state block plus one per draft, keeping the last step's span a step.
    assert blocks == [{1}, {4}, {4}, {4}, {2}, {1}, {4}]
    assert held == [0, 6, 6, 6, 2, 0, 6]
    # The worker hears of each shrink to cut its table rows.
    assert counts == [None] * 4 + [{"0": {0: 2}, "1": {0: 2}}] + [
        {"0": {0: 1}, "1": {0: 1}},
        None,
    ]
    assert all(v.block_size == 256 for v in views)


def _run_budget_policy(
    budget_at: Callable[[int], int], async_scheduling: bool, seed: int = 0
) -> tuple[dict[int, int], list[SchedulerOutput], MambaManager]:
    """Run two requests of a recurrent model to completion under a candidate
    budget that depends only on the step, with an EAGLE-like draft lookahead
    and random acceptance. Async keeps one step in flight: each step is
    scheduled before the previous one's output is known.

    Returns the budget asked at each step, every step's output, and the
    recurrent group's manager.
    """
    num_spec = 3
    scheduler = create_scheduler(
        num_speculative_tokens=num_spec,
        # The V1 runner's n-gram drafter does not run async.
        speculative_method="ngram_gpu",
        max_num_batched_tokens=64,
        max_model_len=256,
        block_size=256,
        kv_cache_spec=new_mamba_spec(
            block_size=256,
            shapes=((1, 1),),
            mamba_cache_mode="none",
            num_speculative_blocks=num_spec,
        ),
        use_v2_model_runner=True,
        async_scheduling=async_scheduling,
    )
    scheduler.num_lookahead_tokens = num_spec
    assert scheduler.dynamic_spec_blocks
    outputs: list[SchedulerOutput] = []
    asked: dict[int, int] = {}
    schedule = scheduler.schedule

    def recording_schedule() -> SchedulerOutput:
        output = schedule()
        outputs.append(output)
        return output

    def candidate_budget(view) -> int:
        # Called inside schedule(), before its output is recorded.
        asked[len(outputs)] = budget_at(len(outputs))
        return asked[len(outputs)]

    scheduler.schedule = recording_schedule
    scheduler.speculator_config.candidate_budget = candidate_budget
    for request in create_requests(2, num_tokens=8, max_tokens=120, ignore_eos=True):
        scheduler.add_request(request)
    rng = random.Random(seed)

    def accept(step: int, req_id: str, num_drafts: int) -> int:
        return rng.randint(0, num_drafts)

    if async_scheduling:
        PipelinedEngine(scheduler, queue_size=2, accept_drafts=accept).run()
    else:
        while scheduler.has_requests():
            output = scheduler.schedule()
            req_ids = list(output.num_scheduled_tokens)
            sampled = []
            for req_id in req_ids:
                num_drafts = len(output.scheduled_spec_decode_tokens.get(req_id, ()))
                sampled.append([100] * (1 + accept(len(outputs), req_id, num_drafts)))
            scheduler.update_from_output(output, runner_output(req_ids, sampled))
            k = output.num_spec_tokens_to_schedule
            scheduler.update_draft_token_ids(
                DraftTokenIds(req_ids, [[7] * k for _ in req_ids])
            )
    (manager,) = scheduler.kv_cache_manager.coordinator.single_type_managers
    assert isinstance(manager, MambaManager)
    return asked, outputs, manager


def _random_budgets(seed: int) -> Callable[[int], int]:
    rng = random.Random(seed)
    budgets = [rng.randint(1, 4) for _ in range(512)]
    return budgets.__getitem__


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize(
    "policy",
    [
        pytest.param(lambda t: 4 if t % 2 == 0 else 1, id="alternate-K+1-1"),
        pytest.param(lambda t: 1 if t % 3 == 2 else 4, id="K+1-K+1-1"),
        pytest.param(lambda t: 1 if t % 2 == 0 else 4, id="alternate-1-K+1"),
        pytest.param(lambda t: [4, 2, 1][t % 3], id="K+1-2-1"),
        *(pytest.param(_random_budgets(s), id=f"random-{s}") for s in range(8)),
    ],
)
def test_scheduler_honors_every_budget(
    policy: Callable[[int], int], async_scheduling: bool
):
    """The drafts a step verifies are the ones the previous step's budget
    asked for, whatever the budget was the step before (no step is forced to
    budget 1 after a budget-1 step), with rejected drafts rolled back."""
    asked, outputs, manager = _run_budget_policy(policy, async_scheduling)
    num_checked = num_drafting = 0
    for step in range(1, len(outputs)):
        prev, output = outputs[step - 1], outputs[step]
        for req_id in output.num_scheduled_tokens:
            if req_id not in prev.num_scheduled_tokens:
                continue
            drafts = len(output.scheduled_spec_decode_tokens.get(req_id, ()))
            assert drafts == asked[step - 1] - 1, (step, req_id, asked, drafts)
            num_checked += 1
            num_drafting += drafts > 0
    assert num_checked > 40 and num_drafting > 10
    assert _held(manager) == 0


def test_scheduler_reserves_draft_lookahead_after_budget_one(monkeypatch):
    """Every step reserves the drafter's lookahead slots, including one after a
    budget-1 step: its budget is chosen after allocation and may ask for
    drafts, which the drafter writes KV for in that same step."""
    lookaheads = []
    allocate_slots = KVCacheManager.allocate_slots

    def recording_allocate_slots(self, request, num_new_tokens, *args, **kwargs):
        lookaheads.append(kwargs.get("num_lookahead_tokens"))
        return allocate_slots(self, request, num_new_tokens, *args, **kwargs)

    monkeypatch.setattr(KVCacheManager, "allocate_slots", recording_allocate_slots)
    steps, _ = _scheduler_steps([4, 1, 4, 1, 1, 4, 4], num_lookahead=3)
    assert [set(d.values()) for d, *_ in steps] == [
        {0},
        {3},
        {0},
        {3},
        {0},
        {0},
        {3},
    ]
    assert lookaheads and all(n == 3 for n in lookaheads)


def test_scheduler_keeps_static_spec_blocks_without_the_v2_runner():
    steps, _ = _scheduler_steps([4, 2, 1, 1], use_v2_model_runner=False)
    assert all(c is None for *_, c in steps)
    assert all(b == {"0": 4, "1": 4} for _, b, *_ in steps)


def test_scheduler_without_recurrent_layers_is_unchanged():
    scheduler = create_scheduler(num_speculative_tokens=3, use_v2_model_runner=True)
    assert not scheduler.dynamic_spec_blocks
    scheduler = create_scheduler(use_v2_model_runner=True)
    assert not scheduler.dynamic_spec_blocks
