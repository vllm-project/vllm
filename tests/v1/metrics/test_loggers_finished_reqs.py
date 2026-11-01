# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The periodic serving-stats line must show progress, not just motion.

Under preemption churn the line reported high generation throughput while
zero requests finished (measured: 798 preemptions, 2 completions, healthy
tok/s). Two failure classes are guarded here:
- interval counters were reset before the line was built, so the
  Preemptions field could never appear (dead since #28522);
- no Finished count existed at all, so motion-without-progress was
  invisible.
"""

import logging

from vllm.config import VllmConfig
from vllm.v1.metrics.loggers import LoggingStatLogger
from vllm.v1.metrics.stats import IterationStats


def _storm_iteration(preemptions=100, finished=0, gen_tokens=50_000):
    it = IterationStats()
    it.num_preempted_reqs = preemptions
    it.num_generation_tokens = gen_tokens
    it.finished_requests = [object()] * finished
    return it


def _make_logger():
    return LoggingStatLogger(VllmConfig())


def _log_line(logger, caplog):
    with caplog.at_level(logging.INFO, logger="vllm.v1.metrics.loggers"):
        logger.log()
    lines = [r.getMessage() for r in caplog.records if "Avg prompt" in r.getMessage()]
    assert lines, "expected the periodic serving-stats line"
    return lines[-1]


def test_preemptions_survive_the_interval_reset(caplog):
    # A storm interval: 100 preemptions with high generation throughput.
    # The field was dead since #28522: _reset cleared the counter before
    # log() read it, so Preemptions never appeared.
    logger = _make_logger()
    logger._track_iteration_stats(_storm_iteration())
    line = _log_line(logger, caplog)
    assert "Preemptions: 100" in line


def test_finished_zero_is_visible_during_churn(caplog):
    # Motion without progress: high tok/s, nothing finishing. The zero must
    # be printed — it is the thrashing signature.
    logger = _make_logger()
    logger._track_iteration_stats(_storm_iteration(finished=0))
    line = _log_line(logger, caplog)
    assert "Finished: 0 reqs" in line


def test_finished_counts_carry_and_reset_per_interval(caplog):
    logger = _make_logger()
    logger._track_iteration_stats(_storm_iteration(finished=3, gen_tokens=10))
    assert "Finished: 3 reqs" in _log_line(logger, caplog)
    # Next interval starts from zero: a second quiet interval reports 0,
    # not the previous interval's count.
    logger._track_iteration_stats(_storm_iteration(finished=0, gen_tokens=10))
    assert "Finished: 0 reqs" in _log_line(logger, caplog)


def test_corrupted_reads_the_interval_snapshot(caplog, monkeypatch):
    monkeypatch.setattr(
        "vllm.v1.metrics.loggers.envs.VLLM_COMPUTE_NANS_IN_LOGITS", True
    )
    logger = _make_logger()
    it = _storm_iteration()
    it.num_corrupted_reqs = 4
    logger._track_iteration_stats(it)
    line = _log_line(logger, caplog)
    assert "Corrupted: 4 reqs" in line


def test_idle_engine_stays_quiet(caplog):
    # No traffic at all: the line drops to debug level (no info records).
    logger = _make_logger()
    with caplog.at_level(logging.INFO, logger="vllm.v1.metrics.loggers"):
        logger.log()
    assert not [r for r in caplog.records if "Avg prompt" in r.getMessage()]
