# SPDX-License-Identifier: Apache-2.0
"""Tests for vllm._genesis.config_detect.

Covers:
  - Conservative fallback when no vllm config available
  - Spec-decode detection (none / ngram / mtp)
  - Cudagraph mode detection (NONE vs FULL_AND_PIECEWISE)
  - Upstream PR presence probes (#40798, #40792, #40384, #40074)
  - Recommendation logic for P36 / P40 / P56 / P9 / P37
  - Force-apply env override
  - should_apply() decision flow
"""
from __future__ import annotations

import importlib.util

import pytest

from vllm._genesis import config_detect


# Many tests in this file monkeypatch `vllm.config.get_current_vllm_config`
# directly via `monkeypatch.setattr("vllm.config...", ...)`. pytest's
# setattr resolves the dotted path by importing the parent module, which
# requires a real vllm install. On the CPU-only CI / local-dev env (no
# vllm), skip the whole file. The integration container DOES have vllm
# and runs everything.
def _vllm_config_importable() -> bool:
    try:
        return importlib.util.find_spec("vllm.config") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


pytestmark = pytest.mark.skipif(
    not _vllm_config_importable(),
    reason="vllm.config not importable in this env (CPU-only / no vllm install)",
)


# ════════════════════════════════════════════════════════════════════
# Fakes
# ════════════════════════════════════════════════════════════════════

class FakeSchedulerConfig:
    def __init__(self, max_num_seqs=2, max_num_batched_tokens=4096,
                 enable_chunked_prefill=True, enable_prefix_caching=True):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.enable_chunked_prefill = enable_chunked_prefill
        self.enable_prefix_caching = enable_prefix_caching


class FakeSpeculativeConfig:
    def __init__(self, method="ngram", num_speculative_tokens=3, model=None):
        self.method = method
        self.num_speculative_tokens = num_speculative_tokens
        self.model = model


class FakeCompilationConfig:
    def __init__(self, cudagraph_mode_name="FULL_AND_PIECEWISE", compile_mode_name="VLLM_COMPILE"):
        self.cudagraph_mode = type("CGMode", (), {"name": cudagraph_mode_name})()
        self.mode = type("CMode", (), {"name": compile_mode_name})()


class FakeCacheConfig:
    def __init__(self, kv_cache_dtype="turboquant_k8v4", block_size=128):
        self.kv_cache_dtype = kv_cache_dtype
        self.block_size = block_size


class FakeVllmConfig:
    def __init__(self,
                 scheduler=None, speculative=None, compilation=None, cache=None):
        self.scheduler_config = scheduler if scheduler is not None else FakeSchedulerConfig()
        self.speculative_config = speculative
        self.compilation_config = compilation if compilation is not None else FakeCompilationConfig()
        self.cache_config = cache if cache is not None else FakeCacheConfig()


@pytest.fixture(autouse=True)
def _reset_cache():
    config_detect.clear_for_tests()
    yield
    config_detect.clear_for_tests()


# ════════════════════════════════════════════════════════════════════
# Conservative fallback
# ════════════════════════════════════════════════════════════════════

def test_no_config_returns_resolved_false(monkeypatch):
    def _raise():
        raise RuntimeError("no config")
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", _raise, raising=False,
    )
    profile = config_detect.get_runtime_profile()
    assert profile["resolved"] is False
    assert profile["spec_decode_enabled"] is False
    # cudagraph_capture_active should default conservatively to True
    assert profile["cudagraph_capture_active"] is True


# ════════════════════════════════════════════════════════════════════
# Scheduler probe
# ════════════════════════════════════════════════════════════════════

def test_scheduler_probe_extracts_max_num_seqs(monkeypatch):
    cfg = FakeVllmConfig(scheduler=FakeSchedulerConfig(max_num_seqs=128))
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", lambda: cfg, raising=False,
    )
    profile = config_detect.get_runtime_profile()
    assert profile["max_num_seqs"] == 128
    assert profile["max_num_batched_tokens"] == 4096
    assert profile["enable_chunked_prefill"] is True


# ════════════════════════════════════════════════════════════════════
# Spec-decode probe
# ════════════════════════════════════════════════════════════════════

def test_spec_decode_disabled_when_none(monkeypatch):
    cfg = FakeVllmConfig(speculative=None)
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", lambda: cfg, raising=False,
    )
    profile = config_detect.get_runtime_profile()
    assert profile["spec_decode_enabled"] is False
    assert profile.get("spec_decode_method_kind") is None


def test_spec_decode_ngram_detected(monkeypatch):
    cfg = FakeVllmConfig(
        speculative=FakeSpeculativeConfig(method="ngram", num_speculative_tokens=3)
    )
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", lambda: cfg, raising=False,
    )
    profile = config_detect.get_runtime_profile()
    assert profile["spec_decode_enabled"] is True
    assert profile["spec_decode_method_kind"] == "ngram"
    assert profile["spec_decode_num_speculative_tokens"] == 3


def test_spec_decode_mtp_detected(monkeypatch):
    cfg = FakeVllmConfig(
        speculative=FakeSpeculativeConfig(method="MTP", num_speculative_tokens=4)
    )
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", lambda: cfg, raising=False,
    )
    profile = config_detect.get_runtime_profile()
    assert profile["spec_decode_method_kind"] == "mtp"


# ════════════════════════════════════════════════════════════════════
# Cudagraph mode probe
# ════════════════════════════════════════════════════════════════════

def test_cudagraph_active_default(monkeypatch):
    cfg = FakeVllmConfig()  # default FULL_AND_PIECEWISE
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", lambda: cfg, raising=False,
    )
    profile = config_detect.get_runtime_profile()
    assert profile["cudagraph_capture_active"] is True
    assert "FULL_AND_PIECEWISE" in profile["cudagraph_mode"]


def test_cudagraph_none_detected(monkeypatch):
    cfg = FakeVllmConfig(compilation=FakeCompilationConfig(cudagraph_mode_name="NONE"))
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", lambda: cfg, raising=False,
    )
    profile = config_detect.get_runtime_profile()
    assert profile["cudagraph_capture_active"] is False


# ════════════════════════════════════════════════════════════════════
# Recommendation logic
# ════════════════════════════════════════════════════════════════════

def test_recommend_p36_apply_when_high_concurrency(monkeypatch):
    cfg = FakeVllmConfig(scheduler=FakeSchedulerConfig(max_num_seqs=64))
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", lambda: cfg, raising=False,
    )
    monkeypatch.setattr(config_detect, "_probe_pr40798_active", lambda: False)
    config_detect.clear_for_tests()
    rec, _reason = config_detect.recommend("P36")
    assert rec == "apply"


def test_recommend_p36_skip_when_low_concurrency(monkeypatch):
    cfg = FakeVllmConfig(scheduler=FakeSchedulerConfig(max_num_seqs=2))
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", lambda: cfg, raising=False,
    )
    monkeypatch.setattr(config_detect, "_probe_pr40798_active", lambda: False)
    config_detect.clear_for_tests()
    rec, reason = config_detect.recommend("P36")
    assert rec == "skip"
    # Source text describes WHY: low max_num_seqs makes the memory
    # saving not worth the maintenance cost. Accept either of the
    # two phrasings the source has used over time.
    reason_lower = reason.lower()
    assert (
        "not worth maintaining" in reason_lower
        or "marginal" in reason_lower
    ), f"unexpected skip reason: {reason!r}"


def test_recommend_p36_redundant_when_pr40798_active(monkeypatch):
    cfg = FakeVllmConfig(scheduler=FakeSchedulerConfig(max_num_seqs=1024))
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", lambda: cfg, raising=False,
    )
    monkeypatch.setattr(config_detect, "_probe_pr40798_active", lambda: True)
    config_detect.clear_for_tests()
    rec, reason = config_detect.recommend("P36")
    assert rec == "redundant"
    assert "#40798" in reason


def test_recommend_p40_redundant_when_pr40792_active(monkeypatch):
    cfg = FakeVllmConfig()
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", lambda: cfg, raising=False,
    )
    monkeypatch.setattr(config_detect, "_probe_pr40792_active", lambda: True)
    config_detect.clear_for_tests()
    rec, reason = config_detect.recommend("P40")
    assert rec == "redundant"
    assert "#40792" in reason


def test_recommend_p56_skip_no_spec_decode(monkeypatch):
    cfg = FakeVllmConfig(speculative=None)
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", lambda: cfg, raising=False,
    )
    config_detect.clear_for_tests()
    rec, reason = config_detect.recommend("P56")
    assert rec == "skip"
    assert "speculative_config" in reason


def test_recommend_p56_redundant_when_cudagraph_off(monkeypatch):
    cfg = FakeVllmConfig(
        speculative=FakeSpeculativeConfig(),
        compilation=FakeCompilationConfig(cudagraph_mode_name="NONE"),
    )
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", lambda: cfg, raising=False,
    )
    config_detect.clear_for_tests()
    rec, reason = config_detect.recommend("P56")
    assert rec == "redundant"
    assert "cudagraph_mode=NONE" in reason


def test_recommend_p56_deprecated_when_spec_and_cudagraph(monkeypatch):
    cfg = FakeVllmConfig(speculative=FakeSpeculativeConfig())
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", lambda: cfg, raising=False,
    )
    config_detect.clear_for_tests()
    rec, reason = config_detect.recommend("P56")
    assert rec == "deprecated"
    assert "Probe 4" in reason


def test_recommend_p9_redundant_when_pr40384_active(monkeypatch):
    cfg = FakeVllmConfig()
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", lambda: cfg, raising=False,
    )
    monkeypatch.setattr(config_detect, "_probe_pr40384_active", lambda: True)
    config_detect.clear_for_tests()
    rec, _reason = config_detect.recommend("P9")
    assert rec == "redundant"


# ════════════════════════════════════════════════════════════════════
# Force-apply env override
# ════════════════════════════════════════════════════════════════════

def test_force_apply_env_override(monkeypatch):
    cfg = FakeVllmConfig(scheduler=FakeSchedulerConfig(max_num_seqs=2))
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", lambda: cfg, raising=False,
    )
    monkeypatch.setattr(config_detect, "_probe_pr40798_active", lambda: False)
    monkeypatch.setenv("GENESIS_FORCE_APPLY_P36", "1")
    config_detect.clear_for_tests()
    ok, reason = config_detect.should_apply("P36")
    assert ok is True
    assert "forced" in reason


# ════════════════════════════════════════════════════════════════════
# should_apply integration
# ════════════════════════════════════════════════════════════════════

def test_should_apply_returns_true_for_apply_recommendation(monkeypatch):
    cfg = FakeVllmConfig(scheduler=FakeSchedulerConfig(max_num_seqs=64))
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", lambda: cfg, raising=False,
    )
    monkeypatch.setattr(config_detect, "_probe_pr40798_active", lambda: False)
    config_detect.clear_for_tests()
    ok, _reason = config_detect.should_apply("P36")
    assert ok is True


def test_should_apply_returns_false_for_skip_recommendation(monkeypatch):
    cfg = FakeVllmConfig(scheduler=FakeSchedulerConfig(max_num_seqs=2))
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", lambda: cfg, raising=False,
    )
    monkeypatch.setattr(config_detect, "_probe_pr40798_active", lambda: False)
    config_detect.clear_for_tests()
    ok, reason = config_detect.should_apply("P36")
    assert ok is False
    assert "skip" in reason


# ════════════════════════════════════════════════════════════════════
# Caching
# ════════════════════════════════════════════════════════════════════

def test_profile_cached_across_calls(monkeypatch):
    counter = {"n": 0}

    def _counter_get():
        counter["n"] += 1
        return FakeVllmConfig()

    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", _counter_get, raising=False,
    )
    config_detect.clear_for_tests()
    _ = config_detect.get_runtime_profile()
    _ = config_detect.recommend("P36")
    _ = config_detect.recommend("P40")
    _ = config_detect.should_apply("P56")
    assert counter["n"] == 1
