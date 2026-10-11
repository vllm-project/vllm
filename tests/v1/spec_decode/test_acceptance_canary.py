# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the near-zero draft-acceptance canary (#61072).

A mismatched draft/target pairing can silently run at ~0% acceptance,
making speculative decoding pure overhead. These tests cover the
canary's window/threshold logic, its one-shot bound, and its wiring into
``SpecDecodingLogging`` (no GPU / no model involved).
"""

from vllm.v1.spec_decode.metrics import (
    NearZeroAcceptanceCanary,
    SpecDecodingLogging,
    SpecDecodingStats,
)


def _stats(drafted: int, accepted: int, num_spec_tokens: int = 15):
    """A per-step aggregate as the scheduler hands it to the loggers.

    ``observe_draft`` performs per-position bookkeeping that the canary does
    not read; assign the totals directly so steps can carry any drafted count.
    """
    stats = SpecDecodingStats.new(num_spec_tokens=num_spec_tokens)
    stats.num_drafts = 1 if drafted else 0
    stats.num_draft_tokens = drafted
    stats.num_accepted_tokens = accepted
    return stats


class RecordingLog:
    def __init__(self):
        self.records: list[tuple] = []

    def __call__(self, fmt, *args):
        self.records.append((fmt, args))


def test_warns_once_after_window_of_near_zero_acceptance():
    log = RecordingLog()
    canary = NearZeroAcceptanceCanary(window_draft_tokens=1000, log_fn=log)
    # The window fills exactly at 1000 drafted tokens with none accepted.
    for _ in range(10):
        canary.observe(_stats(drafted=100, accepted=0))
    assert len(log.records) == 1
    fmt, args = log.records[0]
    assert args[0] == 0  # accepted
    assert args[1] == 1000  # drafted
    assert "0.00%" in (fmt % args)
    # The message must frame near-zero acceptance as a symptom, not proof
    # that the draft/target pairing is invalid.
    message = fmt % args
    assert "does not prove the pairing is invalid" in message
    # Bounded: continued zero acceptance must not warn again.
    for _ in range(50):
        canary.observe(_stats(drafted=100, accepted=0))
    assert len(log.records) == 1


def test_no_warning_with_healthy_acceptance():
    log = RecordingLog()
    canary = NearZeroAcceptanceCanary(window_draft_tokens=1000, log_fn=log)
    # 3 of 15 accepted per step (20%): far above the near-zero threshold.
    for _ in range(100):
        canary.observe(_stats(drafted=15, accepted=3))
    assert log.records == []


def test_no_warning_before_window_fills():
    log = RecordingLog()
    canary = NearZeroAcceptanceCanary(window_draft_tokens=1000, log_fn=log)
    for _ in range(9):
        canary.observe(_stats(drafted=100, accepted=0))
    assert log.records == []


def test_threshold_is_inclusive():
    # Exactly at the threshold -> no warning; just below -> warning.
    at_threshold_log = RecordingLog()
    at_threshold = NearZeroAcceptanceCanary(
        window_draft_tokens=1000, log_fn=at_threshold_log
    )
    at_threshold.observe(_stats(drafted=1000, accepted=10))
    assert at_threshold_log.records == []

    below_log = RecordingLog()
    below = NearZeroAcceptanceCanary(window_draft_tokens=1000, log_fn=below_log)
    below.observe(_stats(drafted=1000, accepted=9))
    assert len(below_log.records) == 1


def test_evaluates_with_full_counts_when_window_overshoots():
    log = RecordingLog()
    canary = NearZeroAcceptanceCanary(window_draft_tokens=1000, log_fn=log)
    canary.observe(_stats(drafted=2500, accepted=0))
    assert len(log.records) == 1
    _, args = log.records[0]
    assert args[:2] == (0, 2500)


def test_zero_draft_steps_do_not_trigger_evaluation():
    log = RecordingLog()
    canary = NearZeroAcceptanceCanary(window_draft_tokens=1000, log_fn=log)
    # Idle steps (no speculative drafts) carry zero counters and must not
    # divide by zero or evaluate early.
    for _ in range(5):
        canary.observe(_stats(drafted=0, accepted=0))
    assert log.records == []
    canary.observe(_stats(drafted=1000, accepted=0))
    assert len(log.records) == 1


def test_logging_integration_warns_across_log_intervals():
    logger = SpecDecodingLogging()
    assert logger.acceptance_canary is not None
    log = RecordingLog()
    logger.acceptance_canary._log_fn = log
    # The default canary window is 4096 drafted tokens. Drain the interval
    # buffers partway through via log() to prove the canary accumulates
    # across intervals.
    for step in range(300):
        logger.observe(_stats(drafted=15, accepted=0))
        if step == 100:
            logger.log(log_fn=lambda *a, **k: None)
    assert len(log.records) == 1
    assert "accepted 0 of the first" in (log.records[0][0] % log.records[0][1])


def test_diffusion_logging_has_no_canary():
    logger = SpecDecodingLogging(is_diffusion=True)
    assert logger.acceptance_canary is None
    for _ in range(300):
        logger.observe(_stats(drafted=15, accepted=0))
    # No canary to warn; observe() still aggregates for the diffusion log.
    assert sum(logger.num_draft_tokens) == 4500


def test_default_window_matches_issue_scenario():
    # The reporter's broken 27B pairing rejected every draft token (0 of
    # 4485); with the default window this must warn. A healthy pairing
    # (16%+ acceptance) must not, and 41 of 4096 (1.001%) sits just above
    # the 1% threshold.
    cases = [
        (0, True),  # the reported 0.0% run
        (1, True),  # ~0.02%: also near-zero
        (66, False),  # 1.6%: low, but not near-zero
        (656, False),  # 16%: the mismatched-but-working Swift pairing
    ]
    for accepted, expect_warn in cases:
        canary = NearZeroAcceptanceCanary()
        assert canary.window_draft_tokens == 4096
        assert canary.min_acceptance_rate == 0.01
        log = RecordingLog()
        canary._log_fn = log
        remaining = accepted
        drafted = 0
        while drafted < 4096:
            take = min(15, 4096 - drafted)
            accept_take = min(remaining, take)
            remaining -= accept_take
            canary.observe(_stats(drafted=take, accepted=accept_take))
            drafted += take
        assert bool(log.records) is expect_warn, accepted
