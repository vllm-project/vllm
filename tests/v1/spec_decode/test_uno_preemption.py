# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Uno survives scheduler preemption, proved on CPU without a growth race.

``tests/v1/e2e/spec_decode/test_uno.py`` shows end to end that a Uno request
preempted mid-generation resumes with the same token IDs as its solo run, but it
gets there by making concurrent requests outgrow a pinned KV pool, which needs a
GPU and a model. These tests drive the real ``Scheduler``/``AsyncScheduler``
over a hand-sized pool instead, so the production ``allocate_slots`` failure and
the production preempt/resume path run on every card:

* ``test_uno_running_request_is_preempted_when_slots_run_out`` forces the
  failure by construction and pins what a preempted Uno request keeps.
* ``test_uno_preempted_request_resumes_with_identical_tokens`` replays a fixed
  sampling script with and without the preemption and requires the committed
  token sequence to be identical.
* ``test_uno_survivor_geometry_forces_preemption_for_any_acceptance`` runs the
  e2e's pinned geometry at several acceptance ratios per peer. This is the case
  that fails if the survivor budget ever again only crosses when the peers grow
  at the same rate: two cards skipped that way with every prompt distinct.
"""

import json

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from tests.v1.e2e.spec_decode import uno_kv_budget as budget
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

pytestmark = pytest.mark.cpu_test

# K for every scheduler in this module, read from the same constant the e2e
# passes to `speculative_config` and the resident-block arithmetic reserves, so
# a K change cannot leave the twin and the pre-gate describing different
# engines.
NUM_SPECULATIVE_TOKENS = budget.SURVIVOR_NUM_SPECULATIVE_TOKENS


@pytest.fixture
def uno_scheduler(tmp_path, monkeypatch):
    """A real Uno scheduler on CPU, with no hub access and no model weights."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setattr("vllm.config.vllm.HAS_TRITON", True)
    from vllm.platforms import current_platform

    # Uno's config contract is CUDA-only; the scheduler under test is not.
    monkeypatch.setattr(
        current_platform, "apply_config_platform_defaults", lambda _config: None
    )
    monkeypatch.setattr(
        current_platform, "check_and_update_config", lambda _config: None
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["OPTForCausalLM"], "model_type": "opt"})
    )

    def create(**kwargs):
        options = dict(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            use_v2_model_runner=True,
            # Uno refuses to configure without async scheduling.
            async_scheduling=True,
            num_speculative_tokens=NUM_SPECULATIVE_TOKENS,
            speculative_method="uno",
            device="cpu",
            block_size=budget.BLOCK_SIZE,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
        )
        options.update(kwargs)
        return create_scheduler(**options)

    return create


def _context_token(context: tuple[int, ...]) -> int:
    """A deterministic greedy "model": the next token is a hash of the context.

    Chunking-invariant and context-sensitive, which is what makes the resume
    comparison non-vacuous. A counter would match across runs however the
    resume mangled the request, and a position-indexed token would too; a token
    derived from every committed id diverges as soon as one is dropped,
    duplicated or reordered.
    """
    digest = 0
    for token_id in context:
        digest = (digest * 31 + token_id) % 1_000_003
    return 600_000 + digest % 100_000


def _sample_tokens(request: Request, count: int) -> list[int]:
    context = tuple(request.prompt_token_ids) + tuple(request.output_token_ids)
    tokens: list[int] = []
    for _ in range(count):
        token_id = _context_token(context)
        tokens.append(token_id)
        context += (token_id,)
    return tokens


def _drive_step(scheduler, accepted_per_request: dict[str, int]):
    """Run one scheduler step, sampling ``accepted + 1`` tokens per decode.

    Prompt chunks sample nothing, which is what the model runner reports for a
    request whose prefill is not finished. A decode returns the bonus token plus
    however many of its scheduled drafts this request "accepted", so the
    scheduler's rejection rollback runs exactly as it does on a GPU.
    """
    scheduler_output = scheduler.schedule()
    request_ids = list(scheduler_output.num_scheduled_tokens)
    sampled: list[list[int]] = []
    for request_id in request_ids:
        request = scheduler.requests[request_id]
        if request.is_prefill_chunk:
            sampled.append([])
            continue
        drafts = scheduler_output.scheduled_spec_decode_tokens.get(request_id, ())
        accepted = min(accepted_per_request.get(request_id, 0), len(drafts))
        sampled.append(_sample_tokens(request, accepted + 1))
    scheduler.update_from_output(
        scheduler_output,
        ModelRunnerOutput(
            req_ids=request_ids,
            req_id_to_index={
                request_id: index for index, request_id in enumerate(request_ids)
            },
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    return scheduler_output


def _held_blocks(scheduler, request_id: str) -> int:
    return len(scheduler.kv_cache_manager.get_block_ids(request_id)[0])


def _apply_output(scheduler, scheduler_output, accepted_per_request: dict[str, int]):
    """Feed one already-scheduled batch back, sampling per request."""
    request_ids = list(scheduler_output.num_scheduled_tokens)
    sampled: list[list[int]] = []
    for request_id in request_ids:
        request = scheduler.requests.get(request_id)
        if request is None or request.is_prefill_chunk:
            sampled.append([])
            continue
        drafts = scheduler_output.scheduled_spec_decode_tokens.get(request_id, ())
        accepted = min(accepted_per_request.get(request_id, 0), len(drafts))
        sampled.append(_sample_tokens(request, accepted + 1))
    scheduler.update_from_output(
        scheduler_output,
        ModelRunnerOutput(
            req_ids=request_ids,
            req_id_to_index={
                request_id: index for index, request_id in enumerate(request_ids)
            },
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def test_uno_running_request_is_preempted_when_slots_run_out(uno_scheduler):
    """The production allocate_slots failure preempts, and Uno state survives.

    Block math, pinned rather than raced: block_size 16, ``num_blocks=9`` so 8
    blocks are allocatable (one is the pool's null block). Two 16-token requests
    are admitted holding ``cdiv(16 + 1 + K, 16) = 2`` blocks each; decoding
    grows each one to 3 blocks, which is 6 of 8. The next step needs a seventh
    and an eighth block, and the step after that has nothing left, so
    ``allocate_slots`` returns None for a running request and the scheduler
    preempts ``running[-1]``.
    """
    scheduler = uno_scheduler(num_blocks=9, max_num_seqs=2, max_num_batched_tokens=64)
    pool = scheduler.kv_cache_manager.block_pool
    assert pool.get_num_free_blocks() == 8, "one block is the null block"

    requests = create_requests(
        num_requests=2,
        num_tokens=16,
        max_tokens=512,
        ignore_eos=True,
        block_size=budget.BLOCK_SIZE,
        req_ids=["uno-lead", "uno-victim"],
    )
    for request in requests:
        scheduler.add_request(request)

    victim = scheduler.requests["uno-victim"]
    for _ in range(64):
        _drive_step(scheduler, {})
        if victim.num_preemptions:
            break

    assert victim.num_preemptions == 1, (
        "the scheduler never hit an allocate_slots failure: free="
        f"{pool.get_num_free_blocks()}, "
        f"running={[r.request_id for r in scheduler.running]}"
    )
    # The last running request is the victim, and the leader keeps running.
    assert victim.status == RequestStatus.PREEMPTED
    assert [r.request_id for r in scheduler.running] == ["uno-lead"]
    # What a preempted Uno request keeps and drops.
    assert victim.num_computed_tokens == 0, "recompute must start from scratch"
    assert not victim.spec_token_ids, "stale drafts must not survive preemption"
    assert _held_blocks(scheduler, "uno-victim") == 0, "blocks must be released"
    assert victim.num_output_placeholders == 0
    assert len(victim.output_token_ids) > 0, "committed tokens must be kept"
    kept_tokens = list(victim.output_token_ids)

    # Resume: the recompute is scheduled as a prefill of prompt plus generated
    # tokens, with no draft tokens for the first step back.
    expected_resume_tokens = victim.num_prompt_tokens + len(kept_tokens)
    assert victim.num_tokens == expected_resume_tokens

    # Retire the leader so the pool has room; otherwise the victim waits for the
    # leader's 512-token cap, which is the scheduler working as intended but
    # says nothing about the resume.
    scheduler.finish_requests("uno-lead", RequestStatus.FINISHED_ABORTED)
    resumed = None
    for _ in range(64):
        scheduler_output = _drive_step(scheduler, {})
        if "uno-victim" in scheduler_output.num_scheduled_tokens:
            resumed = scheduler_output
            break
    assert resumed is not None, "the preempted request was never rescheduled"
    # The recompute is a prefill of prompt plus generated tokens (chunked by the
    # token budget), and it carries no stale draft tokens.
    assert "uno-victim" not in resumed.scheduled_spec_decode_tokens
    assert resumed.num_scheduled_tokens["uno-victim"] == min(
        expected_resume_tokens, 64
    ), resumed.num_scheduled_tokens
    assert list(victim.output_token_ids)[: len(kept_tokens)] == kept_tokens


@pytest.mark.parametrize("in_flight", [False, True])
def test_uno_preemption_drains_in_flight_output(uno_scheduler, in_flight):
    """A preemption with a batch still in flight drains its stale share.

    The engine schedules the next step before the previous step's output comes
    back, so a request can be preempted while it still has tokens in flight.
    ``_preempt_request`` then parks that share in ``num_stale_output_tokens``
    and ``update_from_output`` drains it in lockstep, guarding the async
    placeholder accounting. A synchronous schedule/apply loop never reaches
    that path: with ``in_flight=False`` the stale share at preemption is zero,
    which is exactly what this case asserts, so the parameterisation is its own
    inverted control.
    """
    scheduler = uno_scheduler(num_blocks=9, max_num_seqs=2, max_num_batched_tokens=64)
    requests = create_requests(
        num_requests=2,
        num_tokens=16,
        max_tokens=512,
        ignore_eos=True,
        block_size=budget.BLOCK_SIZE,
        req_ids=["uno-lead", "uno-victim"],
    )
    for request in requests:
        scheduler.add_request(request)
    victim = scheduler.requests["uno-victim"]

    pending: list = []
    stale_at_preemption: int | None = None
    in_flight_at_preemption: int | None = None
    for _ in range(64):
        scheduler_output = scheduler.schedule()
        pending.append(scheduler_output)
        if victim.num_preemptions and stale_at_preemption is None:
            stale_at_preemption = victim.num_stale_output_tokens
            in_flight_at_preemption = victim.num_in_flight_tokens
        # With in_flight, keep one batch outstanding while the next is
        # scheduled, which is what makes a preemption land on a request that
        # still has tokens in flight.
        if in_flight and len(pending) < 2 and not victim.num_preemptions:
            continue
        _apply_output(scheduler, pending.pop(0), {})
        if victim.num_preemptions:
            break
    assert victim.num_preemptions == 1, (
        f"the victim was never preempted: free="
        f"{scheduler.kv_cache_manager.block_pool.get_num_free_blocks()}"
    )
    assert stale_at_preemption is not None

    if in_flight:
        assert in_flight_at_preemption and in_flight_at_preemption > 0, (
            "no batch was in flight at the preemption, so the drain was not "
            f"exercised: num_in_flight_tokens={in_flight_at_preemption}"
        )
        assert stale_at_preemption == in_flight_at_preemption, (
            "the preemption did not park the in-flight share: "
            f"stale={stale_at_preemption} vs in_flight={in_flight_at_preemption}"
        )
        # Drain every outstanding batch; the stale share must go to zero
        # without the placeholder accounting underflowing.
        while pending:
            _apply_output(scheduler, pending.pop(0), {})
        assert victim.num_stale_output_tokens == 0, (
            "the stale output share was not drained: "
            f"{victim.num_stale_output_tokens} left"
        )
        assert victim.num_output_placeholders == 0
    else:
        assert stale_at_preemption == 0, (
            "a synchronous driver should leave nothing in flight at the "
            f"preemption, but parked {stale_at_preemption} tokens"
        )


def test_scheduler_is_keyed_by_the_internal_request_id(uno_scheduler):
    """A request is not in `Scheduler.requests` under the id the caller passed.

    `InputProcessor.assign_request_id` rewrites `request_id` to
    ``f"{external}-{random_uuid():.8}"`` unless
    VLLM_DISABLE_REQUEST_ID_RANDOMIZATION is set, and `Request` keeps only that
    internal id. The e2e driver adds its peers through the engine, so every
    scheduler lookup it makes has to go through the resolver; this case builds
    the same shape directly (the twin adds to the scheduler itself, which is why
    it could not reproduce the two GPU no-fires) and pins both halves: equality
    with the external id finds nothing, and the resolver finds exactly one.
    """
    scheduler = uno_scheduler(num_blocks=32, max_num_seqs=4)
    external_ids = ["uno-finish-peer-0", "uno-finish-peer-1"]
    internal_ids = [
        f"{external}-{index:08x}" for index, external in enumerate(external_ids)
    ]
    for request, request_id in zip(_victim_pair(64), internal_ids):
        request.request_id = request_id
        scheduler.add_request(request)

    for external in external_ids:
        assert scheduler.requests.get(external) is None, (
            "the scheduler answered to the external id, so this case no longer "
            "reproduces the id path the engine takes"
        )

    resolved, problems = budget.resolve_internal_request_ids(
        scheduler.requests, external_ids
    )
    assert not problems, problems
    assert resolved == dict(zip(external_ids, internal_ids))
    for external, internal in resolved.items():
        request = scheduler.requests[internal]
        assert request is not None and request.request_id == internal, external


def test_preemption_must_be_recorded_where_it_happens(uno_scheduler):
    """Polling the scheduler after the fact loses preemptions; a hook does not.

    A preempted request is freed from ``Scheduler.requests`` as soon as it
    finishes, and under async scheduling that can be the same step: the victim's
    stale output is still delivered and can reach its stop. Anything that reads
    the request afterwards then sees nothing at all, which is why the e2e
    records at ``_preempt_request`` instead of polling.
    """
    scheduler = uno_scheduler(num_blocks=9, max_num_seqs=2, max_num_batched_tokens=64)
    requests = create_requests(
        num_requests=2,
        num_tokens=16,
        max_tokens=96,
        ignore_eos=True,
        block_size=budget.BLOCK_SIZE,
        req_ids=["uno-lead", "uno-victim"],
    )
    for request in requests:
        scheduler.add_request(request)

    recorded: list[tuple[str, int, str]] = []
    original = scheduler._preempt_request

    def _record(request, timestamp, drop_stale_output=False):
        recorded.append(
            (request.request_id, len(request.output_token_ids), request.status.name)
        )
        return original(request, timestamp, drop_stale_output=drop_stale_output)

    scheduler._preempt_request = _record
    try:
        for _ in range(16 * 96):
            if not scheduler.has_unfinished_requests():
                break
            _drive_step(scheduler, {"uno-lead": 3, "uno-victim": 3})
    finally:
        scheduler._preempt_request = original

    assert recorded, "this pool must preempt the victim at least once"
    victim_events = [event for event in recorded if event[0] == "uno-victim"]
    assert victim_events, recorded
    # Every hook event catches the request RUNNING with its committed tokens,
    # which is the state the receipt needs and the state a later poll cannot
    # reconstruct.
    for _request_id, generated, status in victim_events:
        assert status == "RUNNING", status
        assert generated >= 0
    # And by the end the requests are gone, so a poll would have nothing left
    # to read.
    assert not scheduler.requests, sorted(scheduler.requests)


def test_internal_id_resolution_refuses_what_it_cannot_pin():
    """Ambiguity and absence must be reported, never silently resolved."""
    external = "uno-finish-peer-0"
    # Randomization disabled: the key is the external id itself.
    resolved, problems = budget.resolve_internal_request_ids([external], [external])
    assert resolved == {external: external} and not problems

    # Nothing there at all.
    resolved, problems = budget.resolve_internal_request_ids(["other"], [external])
    assert resolved == {} and problems == {external: []}

    # Two candidates: refuse rather than pick one.
    twins = [f"{external}-aaaaaaaa", f"{external}-bbbbbbbb"]
    resolved, problems = budget.resolve_internal_request_ids(twins, [external])
    assert resolved == {} and problems == {external: twins}

    # A longer id that merely starts with the external one is not a match.
    resolved, problems = budget.resolve_internal_request_ids(
        [f"{external}-0", f"{external}-toolongsuffix"], [external]
    )
    assert resolved == {} and problems == {external: []}


def test_mid_generation_predicate_rejects_prefill_preemptions():
    """A preemption at zero generated tokens must not satisfy the gate.

    A request preempted while its prompt is still being chunked recomputes a
    prefill; the survivor claim is about a request that had already generated.
    The e2e counts its observed preemptions through this predicate, so the two
    cases are distinguished by one function both sides read.
    """
    counts = budget.mid_generation_preemption_counts(
        [("uno-finish-peer-0", 0), ("uno-finish-peer-1", 0)]
    )
    assert counts == {}, counts

    counts = budget.mid_generation_preemption_counts(
        [("uno-finish-peer-0", 0), ("uno-finish-peer-0", 37), ("x", 1)]
    )
    assert counts == {"uno-finish-peer-0": 1, "x": 1}, counts


def _victim_pair(cap: int) -> list[Request]:
    """Two requests whose prompts are fixed by index, so runs are comparable.

    ``create_requests`` fills prompt i with the token id ``i``, so the compared
    request must keep the same index in both the solo and the mixed run.
    """
    return create_requests(
        num_requests=2,
        num_tokens=16,
        max_tokens=cap,
        ignore_eos=True,
        block_size=budget.BLOCK_SIZE,
        req_ids=["uno-lead", "uno-victim"],
    )


def test_uno_preempted_request_resumes_with_identical_tokens(uno_scheduler):
    """Preempt/recompute must not drop, duplicate or reorder committed tokens.

    The engine-level version of this claim is the e2e survivor case; here the
    same request runs twice against the same deterministic "model", once alone
    in a roomy pool and once in a pool small enough that it is preempted, and
    the committed sequences must match. Each sampled token is a hash of the
    request's whole context, so a resume that mangled the context diverges.
    """
    cap = 96
    accepted = 3

    # Non-vacuity: the oracle must actually depend on the context, or the
    # comparison below could not fail (build checklist row 2).
    assert _context_token((1, 2, 3)) != _context_token((1, 2, 3, 4))
    assert _context_token((1, 2, 3)) != _context_token((3, 2, 1))

    solo_scheduler = uno_scheduler(
        num_blocks=64, max_num_seqs=2, max_num_batched_tokens=64
    )
    solo_request = _victim_pair(cap)[1]
    solo_scheduler.add_request(solo_request)
    for _ in range(16 * cap):
        if not solo_scheduler.has_unfinished_requests():
            break
        _drive_step(solo_scheduler, {"uno-victim": accepted})
    solo_tokens = list(solo_request.output_token_ids)
    assert solo_request.num_preemptions == 0, "the solo pool must not preempt"
    assert len(solo_tokens) >= cap, (len(solo_tokens), cap)

    mixed_scheduler = uno_scheduler(
        num_blocks=9, max_num_seqs=2, max_num_batched_tokens=64
    )
    for request in _victim_pair(cap):
        mixed_scheduler.add_request(request)
    victim = mixed_scheduler.requests["uno-victim"]
    rates = {"uno-lead": accepted, "uno-victim": accepted}
    for _ in range(16 * cap):
        if not mixed_scheduler.has_unfinished_requests():
            break
        _drive_step(mixed_scheduler, rates)

    assert victim.num_preemptions >= 1, "this pool must preempt the victim"
    assert victim.is_finished(), victim.status
    mixed_tokens = list(victim.output_token_ids)
    assert mixed_tokens == solo_tokens, (
        f"preemption/resume changed the committed tokens after "
        f"{victim.num_preemptions} preemptions: solo={len(solo_tokens)} tokens, "
        f"mixed={len(mixed_tokens)} tokens"
    )


# Accepted drafts per step for (finish-peer-0, finish-peer-1). The asymmetric
# rows are the failure class this case exists for: a peer that outruns its twin
# far enough makes a grow-together budget uncrossable.
_ACCEPTANCE_RATIOS = [(8, 8), (8, 1), (1, 8), (6, 3), (0, 0)]


@pytest.mark.parametrize(("lead_accepted", "trail_accepted"), _ACCEPTANCE_RATIOS)
def test_uno_survivor_geometry_forces_preemption_for_any_acceptance(
    uno_scheduler, lead_accepted, trail_accepted
):
    """The e2e survivor geometry must preempt a long peer at any rate ratio.

    Prompt lengths, pool, caps and ``max_model_len`` all come from
    ``uno_kv_budget``, so this replay and the e2e pre-gate cannot describe
    different engines. The abort peer is retired
    after two tokens and the seed finishes early, exactly as in the e2e, so the
    only requests left to preempt are the two long peers.
    """
    budget_blocks = budget.survivor_kv_budget() // budget.kv_bytes_per_block()
    scheduler = uno_scheduler(
        num_blocks=budget_blocks,
        max_model_len=budget.SURVIVOR_MAX_MODEL_LEN,
        max_num_seqs=4,
        max_num_batched_tokens=256,
    )
    cap = budget.SURVIVOR_FINISH_MAX_TOKENS
    seed_prompt, peer_prompt, _, abort_prompt = budget.SURVIVOR_PROMPT_TOKENS
    specs = (
        ("uno-seed", seed_prompt, 96),
        ("uno-finish-peer-0", peer_prompt, cap),
        ("uno-finish-peer-1", peer_prompt, cap),
        ("uno-abort-peer", abort_prompt, 64),
    )
    requests = {}
    for request_id, prompt_tokens, max_tokens in specs:
        (request,) = create_requests(
            num_requests=1,
            num_tokens=prompt_tokens,
            max_tokens=max_tokens,
            ignore_eos=True,
            block_size=budget.BLOCK_SIZE,
            req_ids=[request_id],
        )
        requests[request_id] = request

    finish_ids = ("uno-finish-peer-0", "uno-finish-peer-1")
    rates = {
        "uno-seed": lead_accepted,
        "uno-finish-peer-0": lead_accepted,
        "uno-finish-peer-1": trail_accepted,
        "uno-abort-peer": lead_accepted,
    }
    scheduler.add_request(requests["uno-seed"])
    injected = False
    aborted = False
    peak_usage = 0.0
    max_lag = 0
    preempted_while_active: dict[str, int] = {}

    for _ in range(24 * cap):
        if not scheduler.has_unfinished_requests():
            break
        _drive_step(scheduler, rates)
        peak_usage = max(peak_usage, scheduler.get_kv_cache_usage())

        live = [scheduler.requests.get(request_id) for request_id in finish_ids]
        if all(request is not None for request in live):
            lengths = [len(request.output_token_ids) for request in live]
            max_lag = max(max_lag, max(lengths) - min(lengths))
        for request_id in finish_ids:
            request = scheduler.requests.get(request_id)
            if request is not None and request.num_preemptions:
                preempted_while_active[request_id] = request.num_preemptions

        seed = scheduler.requests.get("uno-seed")
        if not injected and seed is not None and len(seed.output_token_ids) >= 4:
            for request_id in ("uno-finish-peer-0", "uno-finish-peer-1"):
                scheduler.add_request(requests[request_id])
            scheduler.add_request(requests["uno-abort-peer"])
            injected = True
        abort = scheduler.requests.get("uno-abort-peer")
        if (
            injected
            and not aborted
            and abort is not None
            and len(abort.output_token_ids) >= 2
        ):
            scheduler.finish_requests("uno-abort-peer", RequestStatus.FINISHED_ABORTED)
            aborted = True

    assert injected and aborted, (injected, aborted)
    assert preempted_while_active, (
        "the survivor geometry did not preempt either long peer at acceptance "
        f"({lead_accepted}, {trail_accepted}): pool="
        f"{budget.allocatable_blocks(budget_blocks)} blocks, "
        f"peak_kv_cache_usage={peak_usage:.3%}, max_generation_lag={max_lag}"
    )
    for request_id in finish_ids:
        request = scheduler.requests.get(request_id)
        assert request is None or request.is_finished(), (request_id, request.status)
