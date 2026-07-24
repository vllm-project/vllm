# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from types import SimpleNamespace
from typing import cast

import prometheus_client
import pytest

from vllm.config import SpeculativeConfig
from vllm.v1.spec_decode.metrics import (
    SpecDecodingLogging,
    SpecDecodingProm,
    SpecDecodingStats,
)


def test_new_initializes_zeroed_counters():
    stats = SpecDecodingStats.new(num_spec_tokens=4)
    assert stats.num_spec_tokens == 4
    assert stats.num_drafts == 0
    assert stats.num_draft_tokens == 0
    assert stats.num_accepted_tokens == 0
    assert stats.num_accepted_tokens_per_pos == [0, 0, 0, 0]
    assert stats.num_draft_tokens_per_pos == [0, 0, 0, 0]


def test_new_creates_independent_per_pos_lists():
    a = SpecDecodingStats.new(num_spec_tokens=3)
    b = SpecDecodingStats.new(num_spec_tokens=3)
    a.observe_draft(num_draft_tokens=3, num_accepted_tokens=2)
    # Mutating one instance must not leak into another.
    assert b.num_accepted_tokens_per_pos == [0, 0, 0]
    assert b.num_draft_tokens_per_pos == [0, 0, 0]


def test_observe_draft_records_single_draft():
    stats = SpecDecodingStats.new(num_spec_tokens=4)
    stats.observe_draft(num_draft_tokens=4, num_accepted_tokens=2)
    assert stats.num_drafts == 1
    assert stats.num_draft_tokens == 4
    assert stats.num_accepted_tokens == 2
    # Accepted tokens fill the first `num_accepted_tokens` positions.
    assert stats.num_accepted_tokens_per_pos == [1, 1, 0, 0]
    # Draft tokens fill the first `num_draft_tokens` positions.
    assert stats.num_draft_tokens_per_pos == [1, 1, 1, 1]


def test_observe_draft_accumulates_across_drafts():
    stats = SpecDecodingStats.new(num_spec_tokens=3)
    stats.observe_draft(num_draft_tokens=3, num_accepted_tokens=3)
    stats.observe_draft(num_draft_tokens=3, num_accepted_tokens=1)
    stats.observe_draft(num_draft_tokens=2, num_accepted_tokens=0)
    assert stats.num_drafts == 3
    assert stats.num_draft_tokens == 8
    assert stats.num_accepted_tokens == 4
    # pos 0 accepted in 2 drafts; pos 1 and 2 only in the first draft.
    assert stats.num_accepted_tokens_per_pos == [2, 1, 1]
    # pos 0 and 1 drafted every time; pos 2 drafted in 2 of 3 drafts.
    assert stats.num_draft_tokens_per_pos == [3, 3, 2]


def test_observe_draft_accepts_all_spec_tokens():
    stats = SpecDecodingStats.new(num_spec_tokens=2)
    stats.observe_draft(num_draft_tokens=2, num_accepted_tokens=2)
    assert stats.num_accepted_tokens_per_pos == [1, 1]


def test_observe_draft_rejects_more_accepted_than_spec_tokens():
    stats = SpecDecodingStats.new(num_spec_tokens=2)
    with pytest.raises(AssertionError):
        stats.observe_draft(num_draft_tokens=3, num_accepted_tokens=3)


def test_logging_log_is_noop_without_observations():
    logging_stats = SpecDecodingLogging()
    captured = []
    logging_stats.log(log_fn=lambda *args: captured.append(args))
    assert captured == []


def test_logging_includes_bonus_token_in_mean_acceptance_length():
    logging_stats = SpecDecodingLogging()
    stats = SpecDecodingStats.new(num_spec_tokens=3)
    stats.observe_draft(num_draft_tokens=3, num_accepted_tokens=3)
    stats.observe_draft(num_draft_tokens=3, num_accepted_tokens=0)
    logging_stats.observe(stats)

    captured = []
    logging_stats.log(log_fn=lambda *args: captured.append(args))

    # A single record is emitted; the mean acceptance length is the first
    # interpolated value and includes the bonus token: 1 + accepted / drafts.
    assert len(captured) == 1
    mean_acceptance_length = captured[0][1]
    assert mean_acceptance_length == pytest.approx(1 + 3 / 2)


def test_logging_resets_after_log():
    logging_stats = SpecDecodingLogging()
    stats = SpecDecodingStats.new(num_spec_tokens=2)
    stats.observe_draft(num_draft_tokens=2, num_accepted_tokens=1)
    logging_stats.observe(stats)
    logging_stats.log(log_fn=lambda *args: None)

    # After logging, accumulated stats are cleared, so the next log() is a
    # no-op until new stats are observed.
    captured = []
    logging_stats.log(log_fn=lambda *args: captured.append(args))
    assert captured == []


def test_logging_diffusion_path_reports_committed_metrics():
    logging_stats = SpecDecodingLogging(is_diffusion=True)
    stats = SpecDecodingStats.new(num_spec_tokens=4)
    stats.observe_draft(num_draft_tokens=4, num_accepted_tokens=2)
    stats.observe_draft(num_draft_tokens=4, num_accepted_tokens=1)
    logging_stats.observe(stats)

    captured = []
    logging_stats.log(log_fn=lambda *args: captured.append(args))

    # Diffusion reuses the spec-decode counters: drafts -> denoising steps,
    # draft tokens -> canvas positions, accepted tokens -> committed tokens.
    assert len(captured) == 1
    record = captured[0]
    assert record[0].startswith("DiffusionDecoding metrics:")
    assert record[2] == pytest.approx(8 / 3)  # mean denoising steps per canvas
    assert record[3] == pytest.approx(3 / 2)  # mean committed per denoising step
    assert record[4] == 3  # committed tokens
    assert record[5] == 2  # denoising steps
    assert record[6] == 8  # canvas positions evaluated


def test_prom_disabled_when_no_speculative_config():
    prom = SpecDecodingProm(
        speculative_config=None, labelnames=[], per_engine_labelvalues={}
    )
    assert prom.spec_decoding_enabled is False
    stats = SpecDecodingStats.new(num_spec_tokens=2)
    stats.observe_draft(num_draft_tokens=2, num_accepted_tokens=1)
    prom.observe(stats)  # disabled -> no-op, must not raise


def test_prom_observe_increments_counters(monkeypatch):
    registry = prometheus_client.CollectorRegistry()
    monkeypatch.setattr(
        SpecDecodingProm,
        "_counter_cls",
        staticmethod(functools.partial(prometheus_client.Counter, registry=registry)),
    )
    spec_config = cast(SpeculativeConfig, SimpleNamespace(num_speculative_tokens=3))
    prom = SpecDecodingProm(
        spec_config, labelnames=["engine"], per_engine_labelvalues={0: ["0"]}
    )
    assert prom.spec_decoding_enabled

    stats = SpecDecodingStats.new(num_spec_tokens=3)
    stats.observe_draft(num_draft_tokens=3, num_accepted_tokens=2)
    prom.observe(stats, engine_idx=0)

    def value(name, **labels):
        return registry.get_sample_value(name, labels)

    assert value("vllm:spec_decode_num_drafts_total", engine="0") == 1
    assert value("vllm:spec_decode_num_draft_tokens_total", engine="0") == 3
    assert value("vllm:spec_decode_num_accepted_tokens_total", engine="0") == 2
    per_pos = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
    assert value(per_pos, engine="0", position="0") == 1
    assert value(per_pos, engine="0", position="1") == 1
    assert value(per_pos, engine="0", position="2") == 0


def test_prom_diffusion_increments_diffusion_counters(monkeypatch):
    registry = prometheus_client.CollectorRegistry()
    monkeypatch.setattr(
        SpecDecodingProm,
        "_counter_cls",
        staticmethod(functools.partial(prometheus_client.Counter, registry=registry)),
    )
    # Diffusion mode is enabled without a speculative config and maps the
    # spec-decode counters onto diffusion-native names.
    prom = SpecDecodingProm(
        speculative_config=None,
        labelnames=["engine"],
        per_engine_labelvalues={0: ["0"]},
        is_diffusion=True,
    )
    assert prom.spec_decoding_enabled

    stats = SpecDecodingStats.new(num_spec_tokens=3)
    stats.observe_draft(num_draft_tokens=3, num_accepted_tokens=2)
    prom.observe(stats, engine_idx=0)

    def value(name, **labels):
        return registry.get_sample_value(name, labels)

    assert value("vllm:diffusion_num_denoising_steps_total", engine="0") == 1
    assert value("vllm:diffusion_num_canvas_positions_total", engine="0") == 3
    assert value("vllm:diffusion_num_committed_tokens_total", engine="0") == 2
