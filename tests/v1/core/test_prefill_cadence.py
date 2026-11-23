# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TokenSpeed-style two-clock prefill cadence (prefill_decode_interval N /
prefill_decode_min_interval M) and its scheduler wiring, plus the wiring of
the short_uncached_first prefill order."""

import pytest

from tests.v1.core.utils import create_requests, create_scheduler, mock_kv
from vllm.config import SchedulerConfig
from vllm.v1.core.sched.prefill_cadence import PrefillCadence, note_deferred
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _feed_back(scheduler, output) -> None:
    """Complete the step: one sampled token per request that finished its
    prompt, none for partial prefill chunks."""
    req_ids = list(output.num_scheduled_tokens)
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=[
                [] if scheduler.requests[r].is_prefill_chunk else [0] for r in req_ids
            ],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def _step(scheduler):
    output = scheduler.schedule()
    _feed_back(scheduler, output)
    return output


def _add(scheduler, req_id: str, num_tokens: int = 8, max_tokens: int = 100):
    (request,) = create_requests(
        num_requests=1, num_tokens=num_tokens, max_tokens=max_tokens, req_ids=[req_id]
    )
    scheduler.add_request(request)
    return request


def _start_decoding(scheduler, req_id: str = "run0"):
    """Prefill one request and take it into the decode state."""
    _add(scheduler, req_id)
    output = _step(scheduler)
    assert req_id in output.num_scheduled_tokens
    assert not scheduler.requests[req_id].is_prefill_chunk
    return output


# ---------------------------------------------------------------------------
# 1. PrefillCadence (forward.cpp buildFusedPlan semantics)
# ---------------------------------------------------------------------------


def _drive(cadence: PrefillCadence, script):
    """Run (max_deferred_rounds, pushed_prefill, pushed_decode) steps and
    return the `owes` decisions."""
    owes = []
    for max_deferred, pushed_prefill, pushed_decode in script:
        owes.append(cadence.decide(max_deferred).owes)
        cadence.commit(pushed_prefill, pushed_decode)
    return owes


def test_cadence_disabled_by_default():
    assert not PrefillCadence(0, 0).enabled
    assert PrefillCadence(3, 0).enabled
    assert PrefillCadence(0, 3).enabled
    # A disabled cadence never owes anything.
    assert _drive(PrefillCadence(0, 0), [(0, True, True), (0, False, True)]) == [
        False,
        False,
    ]


def test_cadence_interval_only():
    cadence = PrefillCadence(3, 0)
    # The first step is never gated; a prefill step then owes 3 decode rounds.
    script = [(0, True, False)] + [(0, False, True)] * 3 + [(0, True, True)]
    assert _drive(cadence, script) == [False, True, True, True, False]
    assert cadence.interval_left == 3
    assert cadence.num_prefill_steps == 2
    assert cadence.num_held_steps == 3


def test_cadence_min_interval_only():
    cadence = PrefillCadence(0, 2)
    script = [(0, True, True), (99, False, True), (99, False, True), (99, True, True)]
    # Aging never lifts M.
    assert _drive(cadence, script) == [False, True, True, False]


def test_cadence_rounds_count_only_with_decode():
    cadence = PrefillCadence(2, 1)
    cadence.decide(0)
    cadence.commit(pushed_prefill=True, pushed_decode=False)
    # Empty rounds (and rounds that ran neither) pay nothing.
    for _ in range(5):
        assert cadence.decide(0).owes
        cadence.commit(pushed_prefill=False, pushed_decode=False)
    assert (cadence.interval_left, cadence.min_interval_left) == (2, 1)
    cadence.decide(0)
    cadence.commit(pushed_prefill=False, pushed_decode=True)
    assert (cadence.interval_left, cadence.min_interval_left) == (1, 0)


def test_cadence_aging_lifts_interval_at_exactly_n():
    cadence = PrefillCadence(4, 0)
    cadence.decide(0)
    cadence.commit(True, True)
    # A prompt held N-1 rounds still owes; held N rounds lifts the clock.
    assert cadence.decide(3).owes
    decision = cadence.decide(4)
    assert not decision.owes and decision.n_lifted
    cadence.commit(True, True)
    assert cadence.num_aging_lifts == 1


def test_cadence_min_interval_survives_aging():
    cadence = PrefillCadence(10, 2)
    cadence.decide(0)
    cadence.commit(True, True)
    # N is lifted by aging, M is not: still 2 decode rounds owed.
    assert _drive(
        cadence, [(10, False, True), (10, False, True), (10, True, True)]
    ) == [
        True,
        True,
        False,
    ]


def test_cadence_release_on_full_budget():
    cadence = PrefillCadence(5, 1, release_on_full_budget=True)
    cadence.decide(0)
    cadence.commit(True, True)
    cadence.decide(0)
    cadence.commit(False, True)  # pays M off, N has 4 left
    assert cadence.decide(0, pending_prefill_tokens=63, token_budget=64).owes
    decision = cadence.decide(0, pending_prefill_tokens=64, token_budget=64)
    assert not decision.owes and decision.n_lifted
    # Never lifts M.
    cadence.commit(True, True)
    assert cadence.decide(0, pending_prefill_tokens=10**6, token_budget=64).owes
    # Off by default.
    plain = PrefillCadence(5, 0)
    plain.decide(0)
    plain.commit(True, True)
    assert plain.decide(0, pending_prefill_tokens=10**6, token_budget=64).owes


def test_note_deferred():
    requests = create_requests(num_requests=2)
    note_deferred(requests)
    note_deferred(requests[:1])
    assert [r.prefill_deferred_rounds for r in requests] == [2, 1]


# ---------------------------------------------------------------------------
# 2. Defaults: byte-identical scheduling
# ---------------------------------------------------------------------------

# (arrival step, request id, prompt token id, prompt tokens); "d" shares a
# prefix with "a".
_ARRIVALS = [
    (0, "a", 1, 100),
    (0, "b", 2, 30),
    (3, "c", 3, 150),
    (5, "d", 1, 120),
    (5, "e", 5, 70),
    (8, "f", 6, 40),
    (9, "g", 7, 200),
]

# Generated by `_run_scripted_workload()` on the scheduler without the prefill
# cadence (k31-amd-tracking @712d9eb).
_GOLDEN_TRACE = [
    [("a", 64)],
    [("a", 36), ("b", 28)],
    [("a", 1), ("b", 2)],
    [("a", 1), ("b", 1), ("c", 62)],
    [("a", 1), ("b", 1), ("c", 62)],
    [("a", 1), ("b", 1), ("c", 26), ("d", 24)],
    [("a", 1), ("b", 1), ("c", 1), ("d", 1)],
    [("a", 1), ("b", 1), ("c", 1), ("d", 1)],
    [("a", 1), ("b", 1), ("c", 1), ("d", 1)],
    [("b", 1), ("c", 1), ("d", 1), ("e", 61)],
    [("c", 1), ("d", 1), ("e", 9), ("f", 40)],
    [("c", 1), ("d", 1), ("e", 1), ("f", 1)],
    [("c", 1), ("d", 1), ("e", 1), ("f", 1)],
    [("e", 1), ("f", 1), ("g", 62)],
    [("e", 1), ("f", 1), ("g", 62)],
    [("e", 1), ("f", 1), ("g", 62)],
    [("e", 1), ("f", 1), ("g", 14)],
    [("e", 1), ("f", 1), ("g", 1)],
    [("g", 1)],
    [("g", 1)],
    [("g", 1)],
    [("g", 1)],
    [("g", 1)],
    [("g", 1)],
]


def _run_scripted_workload(**kwargs) -> list[list[tuple[str, int]]]:
    scheduler = create_scheduler(
        max_num_seqs=4,
        max_num_batched_tokens=64,
        max_model_len=512,
        enable_prefix_caching=True,
        **kwargs,
    )
    trace = []
    for step in range(80):
        for arrival, req_id, token_id, num_tokens in _ARRIVALS:
            if arrival == step:
                # create_requests fills prompt i with token i.
                request = create_requests(
                    num_requests=token_id + 1,
                    num_tokens=num_tokens,
                    max_tokens=8,
                    req_ids=[f"{req_id}{i}" for i in range(token_id)] + [req_id],
                )[-1]
                scheduler.add_request(request)
        output = _step(scheduler)
        trace.append(sorted(output.num_scheduled_tokens.items()))
    assert not scheduler.requests
    while trace and not trace[-1]:
        trace.pop()
    return trace


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        # Knobs that only matter once N or M is set.
        dict(
            prefill_cadence_mixed=False,
            prefill_cadence_release_on_full_budget=True,
            prefill_cadence_count_slotless=False,
            prefill_cadence_lookahead_loads=4,
            prefill_order_max_bypass=2,
        ),
    ],
)
def test_defaults_unchanged(kwargs):
    assert _run_scripted_workload(**kwargs) == _GOLDEN_TRACE


def test_defaults_leave_cadence_off():
    scheduler = create_scheduler()
    assert not scheduler.prefill_cadence.enabled
    assert not scheduler.prefill_order_short_uncached_first
    _start_decoding(scheduler)
    new_req = _add(scheduler, "new0")
    output = _step(scheduler)
    assert "new0" in output.num_scheduled_tokens
    assert new_req.prefill_deferred_rounds == 0
    assert new_req.prefill_bypassed == 0


# ---------------------------------------------------------------------------
# 3-6. Hold, release, aging, min interval, never idle
# ---------------------------------------------------------------------------


def _admission_step(scheduler, req_id: str, max_steps: int = 50) -> int:
    """Steps until `req_id` is first scheduled (1-based); asserts the decodes
    kept running meanwhile."""
    for step in range(1, max_steps + 1):
        output = _step(scheduler)
        if req_id in output.num_scheduled_tokens:
            return step
        assert "run0" in output.num_scheduled_tokens
        assert scheduler.requests[req_id].status == RequestStatus.WAITING
    raise AssertionError(f"{req_id} not scheduled in {max_steps} steps")


@pytest.mark.parametrize(
    "scheduler_kwargs",
    [
        {},
        dict(async_scheduling=True),
        dict(num_speculative_tokens=5),
    ],
    ids=["sync", "async", "spec"],
)
def test_hold_n_then_release(scheduler_kwargs):
    scheduler = create_scheduler(prefill_decode_interval=5, **scheduler_kwargs)
    _start_decoding(scheduler)
    assert scheduler.prefill_cadence.interval_left == 5
    new_req = _add(scheduler, "new0")
    # Held for the 5 owed decode rounds, admitted on the 6th step.
    assert _admission_step(scheduler, "new0") == 6
    assert new_req.prefill_deferred_rounds == 5
    assert scheduler.prefill_cadence.num_held_steps == 5
    # The admission restarted the interval.
    assert scheduler.prefill_cadence.interval_left == 5


def test_aging_lifts_the_interval_early():
    scheduler = create_scheduler(prefill_decode_interval=5)
    _start_decoding(scheduler)
    new_req = _add(scheduler, "new0")
    # Held 3 rounds before (e.g. behind other prompts' intervals).
    new_req.prefill_deferred_rounds = 3
    assert _admission_step(scheduler, "new0") == 3
    assert new_req.prefill_deferred_rounds == 5
    assert scheduler.prefill_cadence.num_aging_lifts == 1


def test_min_interval_never_lifted():
    scheduler = create_scheduler(
        prefill_decode_interval=50, prefill_decode_min_interval=3
    )
    _start_decoding(scheduler)
    new_req = _add(scheduler, "new0")
    new_req.prefill_deferred_rounds = 50  # N is lifted from the start
    assert _admission_step(scheduler, "new0") == 4


def test_min_interval_alternates_long_prompt_chunks():
    """An overdue long prompt's chunks alternate with exactly M decode rounds
    instead of running back to back (the long-cold-episode fix)."""
    budget, min_interval = 64, 3
    scheduler = create_scheduler(
        max_num_batched_tokens=budget,
        max_model_len=1024,
        prefill_decode_interval=50,
        prefill_decode_min_interval=min_interval,
    )
    _start_decoding(scheduler)
    long_req = _add(scheduler, "long", num_tokens=4 * budget, max_tokens=4)
    long_req.prefill_deferred_rounds = 50
    chunk_steps = []
    for step in range(40):
        output = _step(scheduler)
        assert "run0" in output.num_scheduled_tokens
        if output.num_scheduled_tokens.get("long", 0) > 1:
            chunk_steps.append(step)
    assert len(chunk_steps) == 5  # 256 tokens in chunks of 63
    assert {b - a for a, b in zip(chunk_steps, chunk_steps[1:])} == {min_interval + 1}


def test_aging_bounds_ttft():
    """Under a queue of prompts, every prompt is admitted within
    N + num_prompts * (M + 1) steps: aging lifts N, M then paces the prefill
    steps, and every released step starts at least one prompt."""
    n, m, num_prompts = 8, 2, 10
    scheduler = create_scheduler(
        max_num_seqs=12,
        max_num_batched_tokens=64,
        prefill_decode_interval=n,
        prefill_decode_min_interval=m,
    )
    _add(scheduler, "run0")
    _add(scheduler, "run1")
    _step(scheduler)
    for i in range(num_prompts):
        _add(scheduler, f"p{i}", num_tokens=32)
    first_scheduled: dict[str, int] = {}
    for step in range(1, n + num_prompts * (m + 1) + 1):
        output = _step(scheduler)
        for req_id in output.num_scheduled_tokens:
            first_scheduled.setdefault(req_id, step)
    prompts = {f"p{i}" for i in range(num_prompts)}
    assert prompts <= first_scheduled.keys()
    # Nothing was admitted before the hold had to give way.
    assert min(first_scheduled[p] for p in prompts) > m


def test_never_idle_without_decodes():
    scheduler = create_scheduler(
        max_num_batched_tokens=64,
        max_model_len=1024,
        prefill_decode_interval=5,
        prefill_decode_min_interval=5,
    )
    _add(scheduler, "long", num_tokens=200)
    # Nothing decodes: the chunks run back to back.
    for _ in range(4):
        output = _step(scheduler)
        assert "long" in output.num_scheduled_tokens
    assert scheduler.prefill_cadence.num_held_steps == 0
    # A prompt arriving while nothing decodes is admitted at once, although
    # the last step owes decode rounds.
    scheduler2 = create_scheduler(prefill_decode_interval=5)
    _add(scheduler2, "a", max_tokens=1)
    _step(scheduler2)
    assert not scheduler2.requests
    assert scheduler2.prefill_cadence.interval_left == 5
    _add(scheduler2, "b")
    assert "b" in _step(scheduler2).num_scheduled_tokens


def test_empty_step_neither_counts_nor_ages():
    scheduler = create_scheduler(prefill_decode_interval=5)
    _start_decoding(scheduler)
    new_req = _add(scheduler, "new0")
    # A decode step left in flight.
    output = scheduler.schedule()
    assert "run0" in output.num_scheduled_tokens
    assert scheduler.prefill_cadence.interval_left == 4
    assert new_req.prefill_deferred_rounds == 1
    # Nothing can be scheduled until it completes.
    empty = scheduler.schedule()
    assert empty.total_num_scheduled_tokens == 0
    assert scheduler.prefill_cadence.interval_left == 4
    assert new_req.prefill_deferred_rounds == 1
    _feed_back(scheduler, output)
    _feed_back(scheduler, empty)


def test_count_slotless():
    """Prompts without a request slot age only with count_slotless (default),
    as in TokenSpeed."""
    for count_slotless in (True, False):
        scheduler = create_scheduler(
            max_num_seqs=2,
            prefill_decode_interval=5,
            prefill_cadence_count_slotless=count_slotless,
        )
        _start_decoding(scheduler, "run0")
        has_slot = _add(scheduler, "has_slot")
        slotless = _add(scheduler, "slotless")
        _step(scheduler)
        assert has_slot.prefill_deferred_rounds == 1
        assert slotless.prefill_deferred_rounds == int(count_slotless)


# ---------------------------------------------------------------------------
# 7. Non-mixed prefill steps
# ---------------------------------------------------------------------------


def test_non_mixed_prefill_step_has_no_decode_rows():
    scheduler = create_scheduler(prefill_decode_interval=4, prefill_cadence_mixed=False)
    _start_decoding(scheduler)
    _add(scheduler, "new0")
    for _ in range(4):
        output = _step(scheduler)
        assert set(output.num_scheduled_tokens) == {"run0"}
    output = _step(scheduler)
    assert set(output.num_scheduled_tokens) == {"new0"}
    # The decodes resume on the next (held) step.
    output = _step(scheduler)
    assert set(output.num_scheduled_tokens) == {"run0", "new0"}


def test_non_mixed_fallback_on_alloc_failure():
    """A prefill-only step whose allocation fails runs nothing; released steps
    then run mixed until prefill runs again, so the decodes are not starved."""
    scheduler = create_scheduler(
        num_blocks=12,
        prefill_decode_min_interval=2,
        prefill_cadence_mixed=False,
    )
    _start_decoding(scheduler)
    big = _add(scheduler, "big", num_tokens=400)  # needs 25 blocks
    num_empty = 0
    for _ in range(90):
        output = _step(scheduler)
        assert "big" not in output.num_scheduled_tokens
        num_empty += output.total_num_scheduled_tokens == 0
    assert num_empty == 1
    assert scheduler._prefill_cadence_fallback
    assert big.status == RequestStatus.WAITING


def test_non_mixed_fallback_clears_after_prefill():
    scheduler = create_scheduler(prefill_cadence_mixed=False, prefill_decode_interval=2)
    _start_decoding(scheduler)
    scheduler._prefill_cadence_fallback = True
    _add(scheduler, "new0")
    for _ in range(2):
        _step(scheduler)
    # Released mixed step.
    output = _step(scheduler)
    assert set(output.num_scheduled_tokens) == {"run0", "new0"}
    assert not scheduler._prefill_cadence_fallback


# ---------------------------------------------------------------------------
# 9. A full prefix-cache hit (one token left) does not restart the clock
# ---------------------------------------------------------------------------


def test_full_prefix_hit_does_not_reset_clock():
    scheduler = create_scheduler(enable_prefix_caching=True, prefill_decode_interval=3)
    # Cache two full blocks of prompt 0, then free them.
    _add(scheduler, "warm", num_tokens=32, max_tokens=1)
    _step(scheduler)
    assert "warm" not in scheduler.requests
    _start_decoding(scheduler, "run0")
    assert scheduler.prefill_cadence.interval_left == 3
    # The longest hit stops one token short of the prompt.
    hit = _add(scheduler, "hit", num_tokens=33)
    # Admitted on a held step with one token left: a decode row.
    output = _step(scheduler)
    assert output.num_scheduled_tokens.get("hit") == 1
    assert scheduler.prefill_cadence.interval_left == 2
    assert hit.prefill_deferred_rounds == 0


# ---------------------------------------------------------------------------
# 10. Lookahead connector loads on held steps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lookahead", [0, 2])
def test_lookahead_loads_start_for_k_requests(lookahead):
    scheduler = create_scheduler(
        use_kv_connector=mock_kv(matched_tokens=32, is_async=True),
        prefill_decode_interval=10,
        prefill_cadence_lookahead_loads=lookahead,
    )
    # Neither the decoding request nor the queue head have an external hit.
    (run0,) = create_requests(num_requests=1, num_tokens=8, req_ids=["run0"])
    run0.prefix_cache_hit_limit = 0
    scheduler.add_request(run0)
    _step(scheduler)
    (head,) = create_requests(num_requests=1, num_tokens=64, req_ids=["head"])
    head.prefix_cache_hit_limit = 0
    scheduler.add_request(head)
    loads = [_add(scheduler, f"w{i}", num_tokens=64) for i in range(3)]

    output = _step(scheduler)
    assert set(output.num_scheduled_tokens) == {"run0"}  # a held step
    assert head.status == RequestStatus.WAITING
    statuses = [r.status for r in loads]
    assert statuses == [RequestStatus.WAITING_FOR_REMOTE_KVS] * lookahead + [
        RequestStatus.WAITING
    ] * (3 - lookahead)
    # The held head keeps its place.
    assert scheduler.waiting.peek_request() is head


# ---------------------------------------------------------------------------
# 11. Config
# ---------------------------------------------------------------------------


def _scheduler_config(**kwargs) -> SchedulerConfig:
    return SchedulerConfig(max_model_len=8192, is_encoder_decoder=False, **kwargs)


def test_config_mutual_exclusion():
    with pytest.raises(ValueError, match="prefill_schedule_interval"):
        _scheduler_config(prefill_decode_interval=8, prefill_schedule_interval=4)
    with pytest.raises(ValueError, match="prefill_schedule_interval"):
        _scheduler_config(prefill_decode_min_interval=2, prefill_schedule_interval=4)
    _scheduler_config(prefill_decode_interval=8, prefill_decode_min_interval=2)
    _scheduler_config(prefill_schedule_interval=4)


def test_config_no_escape_is_ignored_with_warning(monkeypatch):
    monkeypatch.setenv("VLLM_PREFILL_CADENCE_NO_ESCAPE", "1")
    _scheduler_config(prefill_decode_interval=8)


def test_config_prefill_order_requires_fcfs():
    with pytest.raises(ValueError, match="fcfs"):
        _scheduler_config(prefill_order="short_uncached_first", policy="priority")
    _scheduler_config(prefill_order="short_uncached_first")


def test_dp_rank_does_not_apply_cadence_locally():
    scheduler = create_scheduler(prefill_decode_interval=8, data_parallel_size=2)
    assert not scheduler.prefill_cadence.enabled


def test_cli_flags():
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(
        [
            "--prefill-decode-interval",
            "80",
            "--prefill-decode-min-interval",
            "4",
            "--no-prefill-cadence-mixed",
            "--prefill-cadence-release-on-full-budget",
            "--no-prefill-cadence-count-slotless",
            "--prefill-cadence-lookahead-loads",
            "4",
            "--prefill-order",
            "short_uncached_first",
            "--prefill-order-max-bypass",
            "6",
        ]
    )
    engine_args = EngineArgs.from_cli_args(args)
    assert engine_args.prefill_decode_interval == 80
    assert engine_args.prefill_decode_min_interval == 4
    assert engine_args.prefill_cadence_mixed is False
    assert engine_args.prefill_cadence_release_on_full_budget is True
    assert engine_args.prefill_cadence_count_slotless is False
    assert engine_args.prefill_cadence_lookahead_loads == 4
    assert engine_args.prefill_order == "short_uncached_first"
    assert engine_args.prefill_order_max_bypass == 6
    defaults = EngineArgs.from_cli_args(parser.parse_args([]))
    assert defaults.prefill_decode_interval == 0
    assert defaults.prefill_decode_min_interval == 0
    assert defaults.prefill_cadence_mixed is True
    assert defaults.prefill_order == "fcfs"


# ---------------------------------------------------------------------------
# Prefill order (WP-A2) wiring
# ---------------------------------------------------------------------------


def test_short_uncached_first_admits_cached_prompt_first():
    scheduler = create_scheduler(
        max_num_seqs=2,
        enable_prefix_caching=True,
        prefill_order="short_uncached_first",
    )
    # Cache prompt 1 (64 tokens), then free it.
    warm = create_requests(
        num_requests=2, num_tokens=64, max_tokens=1, req_ids=["x", "warm"]
    )[-1]
    scheduler.add_request(warm)
    _step(scheduler)
    assert not scheduler.requests
    _start_decoding(scheduler, "run0")  # one slot left
    cold = create_requests(num_requests=3, num_tokens=64, req_ids=["x", "y", "cold"])[
        -1
    ]
    scheduler.add_request(cold)
    cached = create_requests(num_requests=2, num_tokens=64, req_ids=["x", "cached"])[-1]
    scheduler.add_request(cached)
    output = _step(scheduler)
    assert "cached" in output.num_scheduled_tokens
    assert "cold" not in output.num_scheduled_tokens
    # The older prompt was bypassed once.
    assert cold.prefill_bypassed == 1
    assert scheduler.waiting.peek_request() is cold


def test_fcfs_order_by_default():
    scheduler = create_scheduler(max_num_seqs=2, enable_prefix_caching=True)
    warm = create_requests(
        num_requests=2, num_tokens=64, max_tokens=1, req_ids=["x", "warm"]
    )[-1]
    scheduler.add_request(warm)
    _step(scheduler)
    _start_decoding(scheduler, "run0")
    cold = create_requests(num_requests=3, num_tokens=64, req_ids=["x", "y", "cold"])[
        -1
    ]
    scheduler.add_request(cold)
    cached = create_requests(num_requests=2, num_tokens=64, req_ids=["x", "cached"])[-1]
    scheduler.add_request(cached)
    output = _step(scheduler)
    assert "cold" in output.num_scheduled_tokens
    assert cold.prefill_bypassed == 0


def test_short_uncached_first_ages_oldest():
    """After max_bypass admissions went ahead of it, the oldest prompt goes
    first."""
    scheduler = create_scheduler(
        max_num_seqs=2,
        enable_prefix_caching=True,
        prefill_order="short_uncached_first",
        prefill_order_max_bypass=1,
    )
    _start_decoding(scheduler, "run0")
    long_req = _add(scheduler, "long", num_tokens=64, max_tokens=1)
    short1 = create_requests(num_requests=2, num_tokens=16, max_tokens=1)[-1]
    scheduler.add_request(short1)
    output = _step(scheduler)
    assert short1.request_id in output.num_scheduled_tokens
    assert long_req.prefill_bypassed == 1
    short2 = create_requests(
        num_requests=3, num_tokens=16, max_tokens=1, req_ids=["a", "b", "short2"]
    )[-1]
    scheduler.add_request(short2)
    output = _step(scheduler)
    assert "long" in output.num_scheduled_tokens
    assert "short2" not in output.num_scheduled_tokens


def test_short_uncached_first_charges_next_oldest_after_aged_oldest():
    """An admission that lands after the aged oldest prompt in the same step
    bypasses the next-oldest prompt, which is charged for it."""
    scheduler = create_scheduler(
        max_num_seqs=3,
        prefill_order="short_uncached_first",
        prefill_order_max_bypass=1,
    )
    _start_decoding(scheduler, "run0")  # two slots left
    oldest = _add(scheduler, "o", num_tokens=64)
    oldest.prefill_bypassed = 1  # aged: goes first
    next_oldest = _add(scheduler, "p", num_tokens=48)
    short = _add(scheduler, "s", num_tokens=8)
    output = _step(scheduler)
    assert {"o", "s"} <= set(output.num_scheduled_tokens)
    assert "p" not in output.num_scheduled_tokens
    assert oldest.prefill_bypassed == 1
    assert next_oldest.prefill_bypassed == 1
    assert short.prefill_bypassed == 0


def test_short_uncached_first_keeps_started_requests_first():
    """Requests that are not new prompts (here preempted) keep FCFS order
    ahead of the new prompts."""
    scheduler = create_scheduler(
        max_num_seqs=2,
        prefill_order="short_uncached_first",
    )
    _start_decoding(scheduler, "run0")
    preempted = _add(scheduler, "pre", num_tokens=64)
    short = create_requests(num_requests=2, num_tokens=8, req_ids=["x", "short"])[-1]
    scheduler.add_request(short)
    # Mark "pre" as a previously preempted request.
    preempted.num_preemptions = 1
    output = _step(scheduler)
    assert "pre" in output.num_scheduled_tokens
    assert "short" not in output.num_scheduled_tokens


def test_short_uncached_first_with_cadence_orders_on_release_only():
    scheduler = create_scheduler(
        max_num_seqs=3,
        prefill_decode_interval=2,
        prefill_order="short_uncached_first",
    )
    _start_decoding(scheduler, "run0")
    long_req = _add(scheduler, "long", num_tokens=64)
    short = create_requests(num_requests=2, num_tokens=8, req_ids=["x", "short"])[-1]
    scheduler.add_request(short)
    for _ in range(2):
        output = _step(scheduler)
        assert set(output.num_scheduled_tokens) == {"run0"}
        # Held steps neither reorder nor age.
        assert list(scheduler.waiting) == [long_req, short]
    output = _step(scheduler)
    assert {"short", "long"} <= set(output.num_scheduled_tokens)
    assert long_req.prefill_bypassed == 1


# ---------------------------------------------------------------------------
# Async KV loads, runnable decodes, lookahead limits
# ---------------------------------------------------------------------------


def _async_load_scheduler(**kwargs):
    """A scheduler whose prompts hit 32 tokens in an async connector, with
    `run0` (no external hit) decoding."""
    scheduler = create_scheduler(
        use_kv_connector=mock_kv(matched_tokens=32, is_async=True), **kwargs
    )
    (run0,) = create_requests(num_requests=1, num_tokens=8, req_ids=["run0"])
    run0.prefix_cache_hit_limit = 0
    scheduler.add_request(run0)
    _step(scheduler)
    return scheduler


def test_loaded_prompt_is_held():
    """A prompt whose async load just finished is still owed the decode
    rounds: it is promoted only inside the waiting loop."""
    scheduler = _async_load_scheduler(
        prefill_decode_interval=5, prefill_decode_min_interval=5
    )
    loaded = _add(scheduler, "w", num_tokens=64)
    output = _step(scheduler)
    assert set(output.num_scheduled_tokens) == {"run0"}
    assert loaded.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.finished_recving_kv_req_ids.add("w")
    for _ in range(4):
        output = _step(scheduler)
        assert set(output.num_scheduled_tokens) == {"run0"}
    # Aged on the held step that started its load too.
    assert loaded.prefill_deferred_rounds == 5
    output = _step(scheduler)
    assert output.num_scheduled_tokens.get("w") == 32


def test_non_mixed_with_async_loads():
    """A released prefill-only step whose prompt turns into an async load
    only starts the load (a zero-token step); that does not set the sticky
    fallback, and the prefill after the load is still prefill-only."""
    scheduler = _async_load_scheduler(
        prefill_decode_interval=2, prefill_cadence_mixed=False
    )
    for _ in range(2):
        _step(scheduler)  # pay the interval down
    loaded = _add(scheduler, "w", num_tokens=64)
    output = _step(scheduler)
    assert not output.num_scheduled_tokens
    assert loaded.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert not scheduler._prefill_cadence_fallback
    # While the load is in flight run0 decodes (no prompt has compute).
    output = _step(scheduler)
    assert set(output.num_scheduled_tokens) == {"run0"}
    scheduler.finished_recving_kv_req_ids.add("w")
    # The prefill carries no decode rows.
    output = _step(scheduler)
    assert dict(output.num_scheduled_tokens) == {"w": 32}
    assert not scheduler._prefill_cadence_fallback


def test_non_mixed_with_connector_cold_prompt():
    """With a KV connector configured, a cold prompt (no external hit) still
    prefills without decode rows."""
    scheduler = create_scheduler(
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
        prefill_decode_interval=3,
        prefill_decode_min_interval=1,
        prefill_cadence_mixed=False,
    )
    _add(scheduler, "run0", num_tokens=8, max_tokens=100)
    _step(scheduler)
    _add(scheduler, "cold", num_tokens=40)
    for _ in range(6):
        scheduled = dict(_step(scheduler).num_scheduled_tokens)
        if scheduled.get("cold", 0) > 1:
            assert set(scheduled) == {"cold"}
            break
    else:
        raise AssertionError("cold was never prefilled")


def test_hold_needs_a_runnable_decode():
    """Async scheduling: the only decode has its last token in flight, so a
    held step would be empty; the prompt is admitted instead."""
    scheduler = create_scheduler(async_scheduling=True, prefill_decode_interval=5)
    _add(scheduler, "run0", max_tokens=2)
    first = scheduler.schedule()  # prefill run0
    scheduler.schedule()  # its last decode, in flight
    _feed_back(scheduler, first)
    _add(scheduler, "new0")
    output = scheduler.schedule()
    assert set(output.num_scheduled_tokens) == {"new0"}


def test_hold_with_overlapped_async_steps():
    """Async scheduling with step t+1 scheduled before step t returns."""
    scheduler = create_scheduler(async_scheduling=True, prefill_decode_interval=3)
    _add(scheduler, "run0", max_tokens=100)
    in_flight = scheduler.schedule()
    _add(scheduler, "new0")
    admitted_at = None
    for step in range(1, 10):
        output = scheduler.schedule()
        _feed_back(scheduler, in_flight)
        in_flight = output
        if "new0" in output.num_scheduled_tokens:
            admitted_at = step
            break
        assert "run0" in output.num_scheduled_tokens
    # The prefill of run0 restarted N=3; three decode rounds, then new0.
    assert admitted_at == 4
    _feed_back(scheduler, in_flight)


def test_hold_with_spec_decode_drafts():
    from vllm.v1.outputs import DraftTokenIds

    scheduler = create_scheduler(num_speculative_tokens=3, prefill_decode_interval=3)
    _start_decoding(scheduler)
    _add(scheduler, "new0")
    for _ in range(3):
        scheduler.update_draft_token_ids(DraftTokenIds(["run0"], [[1, 2, 3]]))
        output = _step(scheduler)
        assert set(output.num_scheduled_tokens) == {"run0"}
        assert output.num_scheduled_tokens["run0"] == 4
    scheduler.update_draft_token_ids(DraftTokenIds(["run0"], [[1, 2, 3]]))
    output = _step(scheduler)
    assert "new0" in output.num_scheduled_tokens


def test_slot_limited_aging_paces_with_min_interval():
    """max_num_seqs below the offered load: waiters age past N while they
    wait for a slot, so the prefill steps come every M + 1 steps."""
    n, m = 20, 2
    scheduler = create_scheduler(
        max_num_seqs=2, prefill_decode_interval=n, prefill_decode_min_interval=m
    )
    _add(scheduler, "run0", max_tokens=1000)
    _add(scheduler, "short0", max_tokens=1)
    _step(scheduler)
    prompts = [_add(scheduler, f"p{i}", max_tokens=1) for i in range(4)]
    for prompt in prompts:
        prompt.prefill_deferred_rounds = n  # aged while slotless
    prefill_steps = []
    for step in range(30):
        output = _step(scheduler)
        assert "run0" in output.num_scheduled_tokens
        if any(p.request_id in output.num_scheduled_tokens for p in prompts):
            prefill_steps.append(step)
    assert len(prefill_steps) == 4
    assert {b - a for a, b in zip(prefill_steps, prefill_steps[1:])} == {m + 1}


def test_release_on_full_budget_counts_cached_prefix():
    """A long prompt that is mostly cached does not fill the budget."""
    for release, admitted in ((False, False), (True, False)):
        scheduler = create_scheduler(
            enable_prefix_caching=True,
            max_num_batched_tokens=256,
            max_model_len=2048,
            num_blocks=500,
            prefill_decode_interval=20,
            prefill_cadence_release_on_full_budget=release,
        )
        _add(scheduler, "warm", num_tokens=592, max_tokens=1)
        for _ in range(4):
            _step(scheduler)
        assert "warm" not in scheduler.requests
        _start_decoding(scheduler)
        _add(scheduler, "hit", num_tokens=600)
        output = _step(scheduler)
        assert ("hit" in output.num_scheduled_tokens) == admitted
    # Uncached, the same prompt fills the budget and lifts N.
    scheduler = create_scheduler(
        max_num_batched_tokens=256,
        max_model_len=2048,
        prefill_decode_interval=20,
        prefill_cadence_release_on_full_budget=True,
    )
    _start_decoding(scheduler)
    _add(scheduler, "cold", num_tokens=600)
    assert "cold" in _step(scheduler).num_scheduled_tokens


def test_lookahead_keeps_the_held_head_first():
    """Lookahead-loaded requests do not prefill ahead of the cold prompt
    they were loaded behind, and the loads are budgeted per hold window."""
    from vllm.v1.outputs import KVConnectorOutput

    scheduler = create_scheduler(
        use_kv_connector=mock_kv(matched_tokens=32, is_async=True),
        max_num_seqs=64,
        max_num_batched_tokens=64,
        max_model_len=512,
        num_blocks=2000,
        prefill_decode_interval=6,
        prefill_decode_min_interval=2,
        prefill_cadence_lookahead_loads=4,
    )

    def step():
        output = scheduler.schedule()
        loading = {
            r.request_id
            for r in scheduler.requests.values()
            if r.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        }
        req_ids = list(output.num_scheduled_tokens)
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index={r: i for i, r in enumerate(req_ids)},
                sampled_token_ids=[
                    [] if scheduler.requests[r].is_prefill_chunk else [0]
                    for r in req_ids
                ],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
                kv_connector_output=KVConnectorOutput(finished_recving=loading)
                if loading
                else None,
            ),
        )
        return output, loading

    (run0,) = create_requests(
        num_requests=1, num_tokens=8, max_tokens=500, req_ids=["run0"]
    )
    run0.prefix_cache_hit_limit = 0
    scheduler.add_request(run0)
    step()
    (head,) = create_requests(
        num_requests=1, num_tokens=64, max_tokens=1, req_ids=["head"]
    )
    head.prefix_cache_hit_limit = 0
    scheduler.add_request(head)
    for i in range(12):
        _add(scheduler, f"w{i}", num_tokens=64, max_tokens=1)
    first: dict[str, int] = {}
    loads_before_head: set[str] = set()
    for i in range(80):
        output, loading = step()
        if "head" not in first:
            loads_before_head |= loading
        for req_id, n in output.num_scheduled_tokens.items():
            if n > 1:
                first.setdefault(req_id, i)
    assert all(first[f"w{i}"] > first["head"] for i in range(12))
    # One hold window before the head: at most K loads.
    assert len(loads_before_head) <= 4


def test_lookahead_does_not_defer_held_prompts():
    scheduler = _async_load_scheduler(
        prefill_decode_interval=10, prefill_cadence_lookahead_loads=2
    )
    (head,) = create_requests(num_requests=1, num_tokens=64, req_ids=["head"])
    head.prefix_cache_hit_limit = 0
    scheduler.add_request(head)
    for i in range(2):
        (held,) = create_requests(num_requests=1, num_tokens=64, req_ids=[f"h{i}"])
        held.prefix_cache_hit_limit = 0
        scheduler.add_request(held)
    _step(scheduler)
    # Held prompts are neither deferred nor skipped in the stats.
    assert head not in scheduler.deferred_waiting
    assert not any(r.request_id.startswith("h") for r in scheduler.deferred_waiting)
    assert scheduler.waiting.peek_request() is head


def test_lookahead_capped_by_free_slots():
    # Slots: run0 and the held head; one more for one load.
    scheduler = _async_load_scheduler(
        max_num_seqs=3, prefill_decode_interval=10, prefill_cadence_lookahead_loads=4
    )
    (head,) = create_requests(num_requests=1, num_tokens=64, req_ids=["head"])
    head.prefix_cache_hit_limit = 0
    scheduler.add_request(head)
    loads = [_add(scheduler, f"w{i}", num_tokens=64) for i in range(4)]
    _step(scheduler)
    assert [r.status for r in loads] == [RequestStatus.WAITING_FOR_REMOTE_KVS] + [
        RequestStatus.WAITING
    ] * 3


def test_lookahead_stops_at_projected_kv_usage():
    # 100 usable blocks of 16 tokens; each load takes 32 blocks for its hit.
    # The third would take usage from 0.65 to 0.97: it does not start,
    # although usage before it is below 0.9.
    scheduler = create_scheduler(
        use_kv_connector=mock_kv(matched_tokens=512, is_async=True),
        num_blocks=101,
        max_model_len=1024,
        prefill_decode_interval=10,
        prefill_cadence_lookahead_loads=8,
    )
    (run0,) = create_requests(num_requests=1, num_tokens=8, req_ids=["run0"])
    run0.prefix_cache_hit_limit = 0
    scheduler.add_request(run0)
    _step(scheduler)
    (head,) = create_requests(num_requests=1, num_tokens=520, req_ids=["head"])
    head.prefix_cache_hit_limit = 0
    scheduler.add_request(head)
    loads = [_add(scheduler, f"w{i}", num_tokens=520) for i in range(4)]
    _step(scheduler)
    assert [r.status for r in loads] == [RequestStatus.WAITING_FOR_REMOTE_KVS] * 2 + [
        RequestStatus.WAITING
    ] * 2
    assert scheduler.kv_cache_manager.usage <= 0.9


def test_dp_ignored_knobs_warn():
    from unittest.mock import patch

    from vllm.v1.core.sched import scheduler as scheduler_module

    with patch.object(scheduler_module.logger, "warning_once") as warning_once:
        create_scheduler(prefill_decode_interval=8, data_parallel_size=2)
    assert any("data_parallel_size" in c.args[0] for c in warning_once.call_args_list)


def test_config_non_mixed_without_min_interval_warns():
    from unittest.mock import patch

    from vllm.config import scheduler as config_module

    with patch.object(config_module.logger, "warning_once") as warning_once:
        _scheduler_config(prefill_decode_interval=8, prefill_cadence_mixed=False)
        assert warning_once.call_count == 1
        _scheduler_config(
            prefill_decode_interval=8,
            prefill_decode_min_interval=2,
            prefill_cadence_mixed=False,
        )
        assert warning_once.call_count == 1


def test_short_uncached_first_skips_ranking_without_free_slot():
    scheduler = create_scheduler(
        max_num_seqs=2, enable_prefix_caching=True, prefill_order="short_uncached_first"
    )
    calls = []
    uncached_fn = scheduler._prefill_order_uncached_fn

    def counting_fn(request):
        calls.append(request.request_id)
        return uncached_fn(request)

    scheduler._prefill_order_uncached_fn = counting_fn
    _add(scheduler, "run0", max_tokens=50)
    _add(scheduler, "run1", max_tokens=50)
    _step(scheduler)
    calls.clear()
    for i in range(4):
        _add(scheduler, f"w{i}")
    for _ in range(10):
        _step(scheduler)
    assert not calls
