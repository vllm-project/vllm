# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for acceptance-adaptive speculative length (Dynamic SD)."""

import logging

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.config import SpeculativeConfig
from vllm.v1.core.sched.output import DraftTokenIds
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.spec_decode.dynamic.adaptive import (
    AcceptanceAdaptiveK,
    possible_num_speculative_tokens,
)

# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


def _feed(controller: AcceptanceAdaptiveK, accepted_counts: list[int], k: int):
    for accepted in accepted_counts:
        controller.observe(num_draft_tokens=k, num_accepted_tokens=accepted)


def test_controller_uses_max_k_during_warmup():
    controller = AcceptanceAdaptiveK(
        max_num_speculative_tokens=5, threshold=0.4, window=10, probe_interval=0
    )
    # Terrible acceptance, but fewer than `window` drafts observed.
    _feed(controller, [0] * 9, k=5)
    assert controller.next_num_speculative_tokens() == 5
    controller.observe(5, 0)
    assert controller.next_num_speculative_tokens() == 0


def test_controller_keeps_leading_positions_above_threshold():
    controller = AcceptanceAdaptiveK(
        max_num_speculative_tokens=5, threshold=0.4, window=100, probe_interval=0
    )
    # Unconditional acceptance per position ~ [0.70, 0.45, 0.35, 0.25, 0.20]:
    # the measured Qwen2.5-3B / 0.5B curve. Position 1 is the last >= 0.4.
    # accepted=5 -> all positions accepted; accepted=2 -> positions 0,1; etc.
    _feed(controller, [5] * 20 + [4] * 5 + [3] * 10 + [2] * 10 + [1] * 25 + [0] * 30, k=5)
    rates = controller.acceptance_rate
    assert rates[0] == pytest.approx(0.70)
    assert rates[1] == pytest.approx(0.45)
    assert rates[2] == pytest.approx(0.35)
    assert controller.recommended_k() == 2
    assert controller.next_num_speculative_tokens() == 2


def test_controller_respects_min_k_and_can_disable():
    strong_floor = AcceptanceAdaptiveK(
        max_num_speculative_tokens=4,
        threshold=0.9,
        min_num_speculative_tokens=1,
        window=4,
        probe_interval=0,
    )
    _feed(strong_floor, [0] * 4, k=4)
    assert strong_floor.next_num_speculative_tokens() == 1

    no_floor = AcceptanceAdaptiveK(
        max_num_speculative_tokens=4, threshold=0.9, window=4, probe_interval=0
    )
    _feed(no_floor, [0] * 4, k=4)
    assert no_floor.next_num_speculative_tokens() == 0


def test_controller_probes_full_k_periodically_and_recovers():
    controller = AcceptanceAdaptiveK(
        max_num_speculative_tokens=3, threshold=0.5, window=8, probe_interval=4
    )
    _feed(controller, [0] * 8, k=3)
    steps = [controller.next_num_speculative_tokens() for _ in range(8)]
    # Steps 4 and 8 are probes at the full K; the rest are the recommended 0.
    assert steps == [0, 0, 0, 3, 0, 0, 0, 3]

    # Workload changes: probes now see perfect acceptance. Because observe()
    # only updates drafted positions, the probes are the only way higher
    # positions get fresh samples. EMA decay 1/8 -> a handful of hits suffice.
    for _ in range(12):
        controller.observe(num_draft_tokens=3, num_accepted_tokens=3)
    assert controller.recommended_k() == 3


def test_controller_ignores_positions_never_drafted():
    controller = AcceptanceAdaptiveK(
        max_num_speculative_tokens=5, threshold=0.4, window=4, probe_interval=0
    )
    # Only 2 tokens were ever drafted (e.g. capped by a batch-size schedule),
    # both always accepted: positions 2..4 have no estimate and stay enabled.
    _feed(controller, [2] * 4, k=2)
    assert controller.recommended_k() == 5


def test_controller_hysteresis_prevents_flapping():
    # window=20 -> EMA step 0.05, fine enough to land inside the hysteresis band.
    controller = AcceptanceAdaptiveK(
        max_num_speculative_tokens=2,
        threshold=0.5,
        window=20,
        probe_interval=0,
        hysteresis=0.1,
    )
    # Position 1 exactly at threshold: enabled -> stays enabled (>= threshold).
    _feed(controller, [2, 1] * 10, k=2)
    assert controller.acceptance_rate[1] == pytest.approx(0.5)
    assert controller.recommended_k() == 2
    # Dips below: disabled.
    controller.observe(2, 1)
    assert controller.acceptance_rate[1] < 0.5
    assert controller.recommended_k() == 1
    # Climbs back above threshold but below threshold + hysteresis (0.6):
    # stays disabled instead of flapping.
    while controller.acceptance_rate[1] < 0.55:
        controller.observe(2, 2)
    assert 0.5 < controller.acceptance_rate[1] < 0.6
    assert controller.recommended_k() == 1
    # Clears the hysteresis band: re-enabled.
    while controller.acceptance_rate[1] < 0.6:
        controller.observe(2, 2)
    assert controller.recommended_k() == 2


def test_controller_rejects_bad_arguments():
    with pytest.raises(ValueError):
        AcceptanceAdaptiveK(0, 0.4)
    with pytest.raises(ValueError):
        AcceptanceAdaptiveK(3, 0.0)
    with pytest.raises(ValueError):
        AcceptanceAdaptiveK(3, 1.5)
    with pytest.raises(ValueError):
        AcceptanceAdaptiveK(3, 0.4, min_num_speculative_tokens=4)
    with pytest.raises(ValueError):
        AcceptanceAdaptiveK(3, 0.4, window=0)
    with pytest.raises(ValueError):
        AcceptanceAdaptiveK(3, 0.4, probe_interval=-1)
    with pytest.raises(ValueError):
        AcceptanceAdaptiveK(3, 0.4, hysteresis=-0.1)


def test_possible_num_speculative_tokens_shapes_cudagraph_tiers():
    # Schedule only: one K per batch size, widest batch per K.
    dense = [0, 3, 3, 2, 2, 0, 0]
    assert possible_num_speculative_tokens(dense, None, 3, 6) == {3: 2, 2: 4, 0: 6}
    # Adaptive only: every K in [min, max] possible at every batch size.
    assert possible_num_speculative_tokens(None, 0, 3, 4) == {0: 4, 1: 4, 2: 4, 3: 4}
    # Both: K in [min, schedule_k(batch)] at each batch size. Where the schedule
    # caps at 0 the adaptive floor cannot lift it, so K=1 tops out at batch 4.
    assert possible_num_speculative_tokens(dense, 1, 3, 6) == {1: 4, 2: 4, 3: 2, 0: 6}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _spec(**overrides) -> SpeculativeConfig:
    kwargs = dict(
        model="ngram",
        num_speculative_tokens=3,
        prompt_lookup_max=3,
        prompt_lookup_min=1,
        method="ngram",
    )
    kwargs.update(overrides)
    return SpeculativeConfig(**kwargs)


def test_config_reports_dynamic_for_adaptive_mode():
    assert not _spec().uses_dynamic_speculative_decoding()
    assert _spec(adaptive_num_speculative_tokens=True).uses_dynamic_speculative_decoding()


@pytest.mark.parametrize(
    "bad",
    [
        {"adaptive_acceptance_threshold": 0.0},
        {"adaptive_acceptance_threshold": 1.5},
        {"adaptive_min_num_speculative_tokens": 4},
        {"adaptive_min_num_speculative_tokens": -1},
        {"adaptive_window_drafts": 0},
        {"adaptive_probe_interval": -1},
        {"adaptive_hysteresis": -0.1},
    ],
)
def test_config_validates_adaptive_fields(bad):
    with pytest.raises(ValueError):
        _spec(adaptive_num_speculative_tokens=True, **bad)


def test_config_ignores_adaptive_fields_when_disabled():
    # Validation only applies once the mode is on.
    _spec(adaptive_acceptance_threshold=5.0)


# ---------------------------------------------------------------------------
# Scheduler integration
# ---------------------------------------------------------------------------


def _make_adaptive_scheduler(
    *,
    num_speculative_tokens: int = 3,
    threshold: float = 0.5,
    window: int = 4,
    probe_interval: int = 0,
    schedule: list[tuple[int, int, int]] | None = None,
    max_num_seqs: int = 16,
) -> Scheduler:
    # log_stats=False: adaptation must not depend on stats logging.
    return create_scheduler(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=8192,
        num_speculative_tokens=num_speculative_tokens,
        num_speculative_tokens_per_batch_size=schedule,
        adaptive_num_speculative_tokens=True,
        adaptive_speculative_kwargs=dict(
            adaptive_acceptance_threshold=threshold,
            adaptive_window_drafts=window,
            adaptive_probe_interval=probe_interval,
        ),
    )


def _runner_output(req_ids, sampled):
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        sampled_token_ids=sampled,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _decode_step_with_drafts(scheduler: Scheduler, req_ids, drafts, accepted):
    """Attach `drafts`, schedule the verification step, return accepted prefixes.

    `accepted[i]` draft tokens of request i are accepted; the model also emits
    one bonus token, so the sampled list has accepted[i] + 1 entries.
    """
    scheduler.update_draft_token_ids(DraftTokenIds(req_ids, drafts))
    output = scheduler.schedule()
    sampled = [
        drafts[i][: accepted[i]] + [999] for i in range(len(req_ids))
    ]
    scheduler.update_from_output(output, _runner_output(req_ids, sampled))
    return output


def test_scheduler_shrinks_k_from_observed_acceptance():
    scheduler = _make_adaptive_scheduler(num_speculative_tokens=3, threshold=0.5, window=4)
    requests = create_requests(num_requests=2, num_tokens=1)
    req_ids = [r.request_id for r in requests]
    for request in requests:
        scheduler.add_request(request)

    # Prefill step (no drafts yet).
    output = scheduler.schedule()
    assert output.num_spec_tokens_to_schedule == 3
    scheduler.update_from_output(output, _runner_output(req_ids, [[0], [0]]))

    # Two verification steps where only the first draft token is ever accepted:
    # 4 drafts observed -> warm-up complete, per-position acceptance [1, 0, 0].
    for _ in range(2):
        output = _decode_step_with_drafts(
            scheduler, req_ids, drafts=[[1, 2, 3], [1, 2, 3]], accepted=[1, 1]
        )
        assert output.num_spec_tokens_to_schedule == 3  # still warming up

    assert scheduler.adaptive_sd is not None
    assert scheduler.adaptive_sd.acceptance_rate == pytest.approx([1.0, 0.0, 0.0])

    output = _decode_step_with_drafts(
        scheduler, req_ids, drafts=[[1, 2, 3], [1, 2, 3]], accepted=[1, 1]
    )
    assert output.num_spec_tokens_to_schedule == 1


def test_scheduler_takes_min_of_schedule_and_adaptive_k():
    # Schedule allows 3 at this batch size; acceptance says 1 -> 1 wins.
    scheduler = _make_adaptive_scheduler(
        num_speculative_tokens=3, threshold=0.5, window=2, schedule=[(1, 16, 3)]
    )
    requests = create_requests(num_requests=1, num_tokens=1)
    req_ids = [requests[0].request_id]
    scheduler.add_request(requests[0])
    output = scheduler.schedule()
    scheduler.update_from_output(output, _runner_output(req_ids, [[0]]))
    for _ in range(2):
        _decode_step_with_drafts(scheduler, req_ids, drafts=[[1, 2, 3]], accepted=[1])
    output = _decode_step_with_drafts(scheduler, req_ids, drafts=[[1, 2, 3]], accepted=[1])
    assert output.num_spec_tokens_to_schedule == 1

    # Schedule caps at 0 for this batch size -> 0 wins even with perfect acceptance.
    scheduler = _make_adaptive_scheduler(
        num_speculative_tokens=3, threshold=0.5, window=1, schedule=[(1, 16, 0)]
    )
    requests = create_requests(num_requests=1, num_tokens=1)
    req_ids = [requests[0].request_id]
    scheduler.add_request(requests[0])
    output = scheduler.schedule()
    assert output.num_spec_tokens_to_schedule == 0


def test_scheduler_disables_uniform_padding_in_adaptive_mode():
    scheduler = _make_adaptive_scheduler()
    assert scheduler.uses_dynamic_sd
    static = create_scheduler(num_speculative_tokens=3)
    assert not static.uses_dynamic_sd


def test_adaptive_mode_is_disabled_with_data_parallel(caplog_vllm):
    with caplog_vllm.at_level(logging.WARNING, logger="vllm"):
        scheduler = create_scheduler(
            max_num_seqs=16,
            num_speculative_tokens=3,
            adaptive_num_speculative_tokens=True,
            data_parallel_size=2,
        )
    spec = scheduler.vllm_config.speculative_config
    assert spec is not None
    assert not spec.adaptive_num_speculative_tokens
    assert not spec.uses_dynamic_speculative_decoding()
    assert scheduler.adaptive_sd is None
    assert not scheduler.uses_dynamic_sd
    assert "Dynamic speculative decoding is not supported with data parallelism" in (
        caplog_vllm.text
    )
