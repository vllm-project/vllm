# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Short-uncached-prefill-first ordering with aging (prefill_order.py) and the
side-effect free prefix cache peek it ranks with."""

import copy
import random
from types import SimpleNamespace

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.prefill_order import (
    is_new_prompt,
    make_uncached_fn,
    note_admitted,
    oldest_new_prompt,
    order_new_prompts,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request, RequestStatus

pytestmark = pytest.mark.cpu_test

# TokenSpeed kMaxPrefillBypasses (forward.cpp:78).
TS_MAX_BYPASS = 8


def fake_request(
    name: str,
    arrival: float,
    work: int = 0,
    *,
    status: RequestStatus = RequestStatus.WAITING,
    num_computed_tokens: int = 0,
    num_preemptions: int = 0,
    prefill_bypassed: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=name,
        arrival_time=arrival,
        work=work,
        status=status,
        num_computed_tokens=num_computed_tokens,
        num_preemptions=num_preemptions,
        prefill_bypassed=prefill_bypassed,
    )


def by_work(request) -> int:
    return request.work


def ids(requests) -> list[str]:
    return [r.request_id for r in requests]


# ---------------------------------------------------------------------------
# order_new_prompts
# ---------------------------------------------------------------------------


def test_shortest_uncached_work_first():
    reqs = [
        fake_request("a", 0.0, 500),
        fake_request("b", 1.0, 100),
        fake_request("c", 2.0, 300),
        fake_request("d", 3.0, 0),
    ]
    assert ids(order_new_prompts(reqs, by_work, 8)) == ["d", "b", "c", "a"]
    # The input is not modified.
    assert ids(reqs) == ["a", "b", "c", "d"]


def test_ties_keep_arrival_order():
    reqs = [
        fake_request("c", 2.0, 100),
        fake_request("a", 0.0, 100),
        fake_request("b", 1.0, 100),
        fake_request("x", 0.5, 50),
    ]
    assert ids(order_new_prompts(reqs, by_work, 8)) == ["x", "a", "b", "c"]


def test_equal_arrival_ties_by_request_id():
    reqs = [
        fake_request("q1", 5.0, 100),
        fake_request("q0", 5.0, 100),
        fake_request("q2", 5.0, 100),
    ]
    assert ids(order_new_prompts(reqs, by_work, 8)) == ["q0", "q1", "q2"]
    assert ids(order_new_prompts(reversed(reqs), by_work, 8)) == ["q0", "q1", "q2"]


def test_oldest_independent_of_queue_order():
    # Equal arrivals: the oldest must not depend on whether it is taken before
    # or after reordering, else the bypass is charged to the wrong request.
    a = fake_request("a", 1.0, 100)
    b = fake_request("b", 1.0, 1)
    ordered = order_new_prompts([a, b], by_work, 8)
    assert ids(ordered) == ["b", "a"]
    assert oldest_new_prompt([a, b]) is a
    assert oldest_new_prompt(ordered) is a
    note_admitted(ordered[:1], oldest_new_prompt(ordered))
    assert a.prefill_bypassed == 1


def test_submission_order_is_arrival_not_queue_position():
    # A queue already reordered by an earlier step: the oldest is still found by
    # arrival time, so its aging is not lost.
    reqs = [
        fake_request("short", 3.0, 10),
        fake_request("mid", 2.0, 20),
        fake_request("old", 0.0, 1000, prefill_bypassed=8),
    ]
    assert ids(order_new_prompts(reqs, by_work, 8)) == ["old", "short", "mid"]


def test_started_requests_keep_fcfs_and_go_first():
    reqs = [
        fake_request("new_long", 0.0, 900),
        fake_request("preempted", 5.0, 900, num_preemptions=1),
        fake_request("new_short", 1.0, 10),
        fake_request("loaded", 6.0, 900, num_computed_tokens=64),
        fake_request(
            "grammar",
            7.0,
            900,
            status=RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
        ),
        fake_request("remote", 8.0, 900, status=RequestStatus.WAITING_FOR_REMOTE_KVS),
        fake_request("resumed", 9.0, 900, status=RequestStatus.PREEMPTED),
    ]
    assert ids(order_new_prompts(reqs, by_work, 8)) == [
        "preempted",
        "loaded",
        "grammar",
        "remote",
        "resumed",
        "new_short",
        "new_long",
    ]


def test_uncached_fn_called_once_per_new_prompt_only():
    calls: list[str] = []

    def uncached(request) -> int:
        calls.append(request.request_id)
        return request.work

    reqs = [
        fake_request("p", 0.0, 1, num_preemptions=1),
        fake_request("a", 1.0, 3),
        fake_request("b", 2.0, 2),
        fake_request("c", 3.0, 1),
    ]
    order_new_prompts(reqs, uncached, 8)
    assert sorted(calls) == ["a", "b", "c"]


@pytest.mark.parametrize("num_new", [0, 1])
def test_trivial_queues(num_new):
    started = fake_request("s", 9.0, num_preemptions=1)
    new = [fake_request("n", 0.0, 5)][:num_new]
    assert order_new_prompts([*new, started], by_work, 8) == [started, *new]
    assert order_new_prompts([], by_work, 8) == []


def test_aged_oldest_goes_first():
    reqs = [
        fake_request("old", 0.0, 1000, prefill_bypassed=TS_MAX_BYPASS),
        fake_request("s1", 1.0, 1),
        fake_request("s2", 2.0, 2),
    ]
    assert ids(order_new_prompts(reqs, by_work, TS_MAX_BYPASS)) == ["old", "s1", "s2"]


def test_aging_within_one_order():
    # The oldest has 1 bypass and ages at 3: two shorter prompts may still pass
    # it, then it goes, then short work first again.
    reqs = [fake_request("old", 0.0, 1000, prefill_bypassed=1)] + [
        fake_request(f"s{i}", 1.0 + i, 10 + i) for i in range(5)
    ]
    assert ids(order_new_prompts(reqs, by_work, 3)) == [
        "s0",
        "s1",
        "old",
        "s2",
        "s3",
        "s4",
    ]


def test_next_oldest_starts_from_its_own_counter():
    # After "old" goes, "next" (the new oldest, 0 bypasses) is passed by
    # max_bypass shorter prompts before it goes.
    reqs = [
        fake_request("old", 0.0, 1000, prefill_bypassed=2),
        fake_request("next", 1.0, 900),
    ] + [fake_request(f"s{i}", 2.0 + i, i) for i in range(6)]
    assert ids(order_new_prompts(reqs, by_work, 2)) == [
        "old",
        "s0",
        "s1",
        "next",
        "s2",
        "s3",
        "s4",
        "s5",
    ]


def test_aged_non_oldest_goes_first_in_submission_order():
    # Not reachable through note_admitted alone, but TokenSpeed ranks any aged
    # prompt first; passing the un-aged oldest still ages it.
    reqs = [
        fake_request("oldest", 0.0, 1000, prefill_bypassed=1),
        fake_request("aged_b", 2.0, 900, prefill_bypassed=2),
        fake_request("aged_a", 1.0, 950, prefill_bypassed=5),
        fake_request("short", 3.0, 1),
    ]
    assert ids(order_new_prompts(reqs, by_work, 2)) == [
        "aged_a",
        "oldest",
        "aged_b",
        "short",
    ]


# ---------------------------------------------------------------------------
# note_admitted / oldest_new_prompt
# ---------------------------------------------------------------------------


def test_oldest_new_prompt():
    reqs = [
        fake_request("p", 0.0, num_preemptions=1),
        fake_request("b", 2.0),
        fake_request("a", 1.0),
        fake_request("a2", 1.0),
    ]
    assert oldest_new_prompt(reqs).request_id == "a"
    assert oldest_new_prompt(reqs[:1]) is None
    assert oldest_new_prompt([]) is None


def test_is_new_prompt():
    assert is_new_prompt(fake_request("n", 0.0))
    assert not is_new_prompt(fake_request("p", 0.0, num_preemptions=1))
    assert not is_new_prompt(fake_request("c", 0.0, num_computed_tokens=16))
    assert not is_new_prompt(
        fake_request("r", 0.0, status=RequestStatus.WAITING_FOR_REMOTE_KVS)
    )


def test_note_admitted():
    old = fake_request("old", 0.0)
    a, b, c = (fake_request(n, 1.0) for n in "abc")
    note_admitted([], old)
    assert old.prefill_bypassed == 0
    note_admitted([a, b], old)
    assert old.prefill_bypassed == 2
    # Admissions after the oldest itself are not charged to it.
    note_admitted([a, old, b, c], old)
    assert old.prefill_bypassed == 3
    note_admitted([a], None)
    assert a.prefill_bypassed == 0


def test_note_admitted_charges_next_oldest():
    # After the oldest is admitted, later admissions bypass the next oldest
    # (TokenSpeed recordPrefillBypass), also with one call per step.
    o = fake_request("o", 0.0, 1000, prefill_bypassed=7)
    n = fake_request("n", 1.0, 1000)
    s1 = fake_request("s1", 2.0, 1)
    s2 = fake_request("s2", 3.0, 2)
    ordered = order_new_prompts([o, n, s1, s2], by_work, 8)
    assert ids(ordered) == ["s1", "o", "s2", "n"]
    note_admitted(ordered[:3], o, new_prompts=ordered)
    assert (o.prefill_bypassed, n.prefill_bypassed) == (8, 1)
    assert s1.prefill_bypassed == s2.prefill_bypassed == 0
    # Without new_prompts only the bypasses of `oldest` are counted.
    o.prefill_bypassed, n.prefill_bypassed = 7, 0
    note_admitted(ordered[:3], o)
    assert (o.prefill_bypassed, n.prefill_bypassed) == (8, 0)


def test_aging_bounds_wait_across_steps():
    # One admission per step and a steady stream of short prompts: the long
    # oldest prompt is admitted after exactly max_bypass bypasses.
    max_bypass = 4
    long_req = fake_request("long", 0.0, 10_000)
    waiting = [long_req]
    admitted_at = None
    for step in range(20):
        waiting.append(fake_request(f"s{step}", 1.0 + step, 1))
        oldest = oldest_new_prompt(waiting)
        waiting = order_new_prompts(waiting, by_work, max_bypass)
        admitted = [waiting.pop(0)]
        note_admitted(admitted, oldest)
        if admitted[0] is long_req:
            admitted_at = step
            break
    assert admitted_at == max_bypass
    assert long_req.prefill_bypassed == max_bypass


def test_held_steps_do_not_age():
    old = fake_request("old", 0.0, 1000)
    waiting = [old, fake_request("s", 1.0, 1)]
    for _ in range(50):
        oldest = oldest_new_prompt(waiting)
        waiting = order_new_prompts(waiting, by_work, 2)
        note_admitted([], oldest)
    assert old.prefill_bypassed == 0
    assert ids(waiting) == ["s", "old"]


def _ts_admission_order(reqs, max_bypass):
    """Reference: TokenSpeed popShortPrefill + recordPrefillBypass
    (forward.cpp:87-98, 1796-1814), admitting every candidate one by one."""
    candidates = sorted(reqs, key=lambda r: (r.arrival_time, r.request_id))
    pending = list(candidates)
    submitted = set(map(id, candidates))
    order = []
    while pending:

        def rank(r):
            aged = r.prefill_bypassed >= max_bypass
            return (0, 0) if aged else (1, r.work)

        # min() keeps the first of equal keys, like std::min_element.
        selected = min(pending, key=rank)
        pending.remove(selected)
        for older in candidates:
            if older is selected:
                break
            if id(older) in submitted:
                older.prefill_bypassed = min(max_bypass, older.prefill_bypassed + 1)
                break
        submitted.discard(id(selected))
        order.append(selected)
    return order


@pytest.mark.parametrize("seed", range(40))
def test_matches_tokenspeed_reference(seed):
    rng = random.Random(seed)
    num = rng.randint(2, 30)
    max_bypass = rng.choice([1, 2, 3, 8])
    reqs = [
        fake_request(
            f"r{i}",
            float(rng.randint(0, 10)),
            rng.choice([0, 16, 64, 64, 128, 4096, 140_000]),
            prefill_bypassed=rng.choice([0, 0, 0, 1, max_bypass]),
        )
        for i in range(num)
    ]
    rng.shuffle(reqs)
    ts_reqs = copy.deepcopy(reqs)
    order = order_new_prompts(reqs, by_work, max_bypass)
    assert ids(order) == ids(_ts_admission_order(ts_reqs, max_bypass))

    # Per-admission note_admitted with the current oldest reproduces the
    # TokenSpeed counters (up to its clamp at max_bypass).
    waiting = list(reqs)
    for request in order:
        oldest = oldest_new_prompt(waiting)
        note_admitted([request], oldest)
        waiting.remove(request)
        request.status = RequestStatus.RUNNING
    ts_by_id = {r.request_id: r.prefill_bypassed for r in ts_reqs}
    for request in reqs:
        assert min(request.prefill_bypassed, max_bypass) == ts_by_id[request.request_id]


def _ts_admit(waiting, num_admit, max_bypass):
    """Reference: TokenSpeed admitting `num_admit` of the waiting new prompts
    one by one (popShortPrefill + recordPrefillBypass); returns them."""
    candidates = sorted(waiting, key=lambda r: (r.arrival_time, r.request_id))
    selected_all = []
    for _ in range(num_admit):
        pending = [r for r in candidates if r not in selected_all]

        def rank(r):
            aged = r.prefill_bypassed >= max_bypass
            return (0, 0) if aged else (1, r.work)

        selected = min(pending, key=rank)
        if pending[0] is not selected:
            pending[0].prefill_bypassed = min(
                max_bypass, pending[0].prefill_bypassed + 1
            )
        selected_all.append(selected)
    return selected_all


@pytest.mark.parametrize("seed", range(40))
def test_one_call_per_step_matches_tokenspeed(seed):
    # Multi-step: each step orders the queue (taking the oldest after
    # reordering, as the scheduler does), admits a prefix of the order and
    # calls note_admitted once with new_prompts. Admissions and counters match
    # TokenSpeed admitting the same number of prompts one by one.
    rng = random.Random(seed)
    max_bypass = rng.choice([1, 2, 3, 8])
    pending = sorted(
        (
            fake_request(f"r{i}", float(rng.randint(0, 20)), rng.randint(0, 50))
            for i in range(rng.randint(2, 40))
        ),
        key=lambda r: r.arrival_time,
    )
    ts_pending = copy.deepcopy(pending)
    waiting: list = []
    ts_waiting: list = []
    while pending or waiting:
        for _ in range(rng.randint(0, 4)):
            if pending:
                waiting.append(pending.pop(0))
                ts_waiting.append(ts_pending.pop(0))
        ordered = order_new_prompts(waiting, by_work, max_bypass)
        num_admit = rng.randint(0, len(ordered))
        admitted = ordered[:num_admit]
        note_admitted(admitted, oldest_new_prompt(ordered), new_prompts=ordered)
        ts_admitted = _ts_admit(ts_waiting, num_admit, max_bypass)
        assert ids(admitted) == ids(ts_admitted)
        for request in admitted:
            request.status = RequestStatus.RUNNING
        waiting = ordered[num_admit:]
        ts_waiting = [r for r in ts_waiting if r not in ts_admitted]
        ts_by_id = {r.request_id: r.prefill_bypassed for r in ts_waiting}
        for request in waiting:
            assert (
                min(request.prefill_bypassed, max_bypass)
                == ts_by_id[request.request_id]
            )


def test_make_uncached_fn():
    class FakeManager:
        def peek_cached_prefix_len(self, request) -> int:
            return request.cached

    req = SimpleNamespace(num_tokens=1000, cached=600)
    assert make_uncached_fn(FakeManager())(req) == 400
    assert make_uncached_fn(FakeManager(), lambda r, local: 300)(req) == 100
    assert make_uncached_fn(FakeManager(), lambda r, local: 900)(req) == 0


# ---------------------------------------------------------------------------
# KVCacheManager.peek_cached_prefix_len on a real manager
# ---------------------------------------------------------------------------

BLOCK_SIZE = 16


@pytest.fixture(autouse=True)
def _init_none_hash():
    init_none_hash(sha256)


def make_request(
    request_id: str,
    prompt_token_ids: list[int],
    *,
    prompt_logprobs: int | None = None,
    report_mode: str | None = None,
    arrival_time: float | None = None,
) -> Request:
    sampling_params = SamplingParams(
        max_tokens=17,
        prompt_logprobs=prompt_logprobs,
        extra_args={"kv_cache_report_mode": report_mode} if report_mode else None,
    )
    sampling_params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
        arrival_time=arrival_time,
    )


def _full_spec():
    return FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32
    )


def make_kv_cache_config(kind: str, num_blocks: int = 64) -> KVCacheConfig:
    swa = SlidingWindowSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=2 * BLOCK_SIZE,
    )
    mamba = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    groups = {
        "full": [_full_spec()],
        "swa": [_full_spec(), swa],
        "mamba": [_full_spec(), mamba],
        "full_swa_mamba": [_full_spec(), swa, mamba],
    }[kind]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer{i}"], spec) for i, spec in enumerate(groups)
        ],
    )


def make_manager(kind: str = "full", **kwargs) -> KVCacheManager:
    kwargs.setdefault("max_model_len", 8192)
    kwargs.setdefault("enable_caching", True)
    return KVCacheManager(
        make_kv_cache_config(kind),
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        **kwargs,
    )


def run_prefill(manager: KVCacheManager, request: Request) -> None:
    """Admit the request the way the scheduler does and compute its prompt in
    3-block chunks: align-mode Mamba keeps a state only where a chunk ends."""
    blocks, num_computed, _ = manager.get_computed_blocks(request)
    request.num_computed_tokens = num_computed
    while request.num_computed_tokens < request.num_tokens:
        num_new = min(3 * BLOCK_SIZE, request.num_tokens - request.num_computed_tokens)
        assert manager.allocate_slots(request, num_new, num_computed, blocks)
        request.num_computed_tokens += num_new
        blocks, num_computed = None, 0
        manager.new_step_starts()


def snapshot(manager: KVCacheManager):
    pool = manager.block_pool
    return (
        [b.ref_cnt for b in pool.blocks],
        [b.block_id for b in pool.free_block_queue.get_all_free_blocks()],
        pool.get_num_free_blocks(),
        len(pool.cached_block_hash_to_block),
    )


PREFIX = [i for i in range(6) for _ in range(BLOCK_SIZE)]  # 6 full blocks


@pytest.mark.parametrize("kind", ["full", "swa", "mamba", "full_swa_mamba"])
@pytest.mark.parametrize("use_eagle", [False, True])
def test_peek_matches_get_computed_blocks(kind, use_eagle):
    manager = make_manager(kind, use_eagle=use_eagle)
    run_prefill(manager, make_request("running", PREFIX + [7] * 5))
    freed = make_request("freed", [9] * BLOCK_SIZE * 3 + [8] * 3)
    run_prefill(manager, freed)
    manager.free(freed)

    probes = [
        make_request("hit", PREFIX + [3] * 20),
        make_request("partial", PREFIX[: 3 * BLOCK_SIZE] + [5] * 40),
        make_request("exact", list(PREFIX)),
        make_request("freed_hit", [9] * BLOCK_SIZE * 3 + [1] * 30),
        make_request("miss", [11] * 100),
        make_request("short", [12] * 3),
    ]
    for probe in probes:
        expected = manager.get_computed_blocks(probe)[1]
        assert manager.peek_cached_prefix_len(probe) == expected, probe.request_id
        for bound in (0, BLOCK_SIZE, 3 * BLOCK_SIZE + 1):
            assert (
                manager.peek_cached_prefix_len(probe, bound)
                == manager.get_computed_blocks(probe, bound)[1]
            )


def test_peek_values_full_attention():
    manager = make_manager()
    run_prefill(manager, make_request("running", PREFIX + [7] * 5))
    assert manager.peek_cached_prefix_len(make_request("a", PREFIX + [3] * 20)) == len(
        PREFIX
    )
    # A full hit leaves the last token to recompute: one block less.
    assert (
        manager.peek_cached_prefix_len(make_request("b", list(PREFIX)))
        == len(PREFIX) - BLOCK_SIZE
    )
    assert manager.peek_cached_prefix_len(make_request("c", [11] * 100)) == 0
    # Requests that skip the prefix cache read (prompt logprobs) peek 0.
    assert (
        manager.peek_cached_prefix_len(
            make_request("d", PREFIX + [3] * 20, prompt_logprobs=1)
        )
        == 0
    )


def test_peek_with_caching_disabled():
    manager = make_manager(enable_caching=False)
    run_prefill(manager, make_request("running", PREFIX + [7] * 5))
    assert manager.peek_cached_prefix_len(make_request("a", PREFIX + [3] * 20)) == 0


@pytest.mark.parametrize("kind", ["full", "full_swa_mamba"])
def test_peek_has_no_side_effects(kind):
    manager = make_manager(kind, log_stats=True, enable_kv_cache_events=True)
    running = make_request("running", PREFIX + [7] * 5)
    run_prefill(manager, running)
    # Cached but free (evictable) blocks: a touch would pull them out of the
    # free queue.
    freed = make_request("freed", [9] * BLOCK_SIZE * 3 + [8] * 3)
    run_prefill(manager, freed)
    manager.free(freed)
    manager.take_events()
    manager.make_prefix_cache_stats()

    before = snapshot(manager)
    probes = [
        make_request("hit", PREFIX + [3] * 20, report_mode="full"),
        make_request("freed_hit", [9] * BLOCK_SIZE * 3 + [1] * 30, report_mode="full"),
        make_request("miss", [11] * 100),
    ]
    for _ in range(3):
        for probe in probes:
            manager.peek_cached_prefix_len(probe)

    assert snapshot(manager) == before
    assert manager.take_events() == []
    stats = manager.make_prefix_cache_stats()
    assert stats is not None and stats.requests == 0 and stats.queries == 0
    for probe in probes:
        assert probe.num_computed_tokens == 0
        assert probe.status == RequestStatus.WAITING
        assert manager.get_blocks(probe.request_id).get_block_ids() == tuple(
            [] for _ in manager.kv_cache_config.kv_cache_groups
        )

    # Sanity: the real lookup in "full" report mode does emit events, so the
    # assertion above is not vacuous.
    manager.get_computed_blocks(probes[0])
    assert manager.take_events() != []


def test_rank_cached_prefix_before_older_cold_prompt():
    manager = make_manager()
    run_prefill(manager, make_request("prev_turn", PREFIX + [7] * 5))

    cold = make_request("cold", [11] * 120, arrival_time=1.0)
    warm = make_request("warm", PREFIX + [3] * 30, arrival_time=2.0)
    preempted = make_request("preempted", [13] * 50, arrival_time=3.0)
    preempted.num_preemptions = 1
    waiting = [cold, warm, preempted]

    oldest = oldest_new_prompt(waiting)
    assert oldest is cold
    order = order_new_prompts(waiting, make_uncached_fn(manager), TS_MAX_BYPASS)
    assert ids(order) == ["preempted", "warm", "cold"]
    note_admitted([warm], oldest)
    assert cold.prefill_bypassed == 1
    assert warm.prefill_bypassed == 0
