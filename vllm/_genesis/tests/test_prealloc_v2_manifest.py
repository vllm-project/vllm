# SPDX-License-Identifier: Apache-2.0
"""TDD for prealloc_v2 — manifest schema + cache layer (Level 3 v0.1).

Covers the data layer + serialization that downstream Recorder and
Replay components depend on. Pure CPU, no torch GPU calls.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from vllm._genesis.prealloc_v2 import (
    CallSiteEnvelope,
    PreallocV2Manifest,
    _MANIFEST_VERSION,
    get_cache_dir,
    load_manifest,
    manifest_path,
    recorder_enabled,
    replay_enabled,
    write_manifest,
)


# ─── CallSiteEnvelope ─────────────────────────────────────────────────


class TestCallSiteEnvelope:
    def test_minimal_construction(self):
        c = CallSiteEnvelope(
            file="/path/to/chunk_o.py", lineno=161,
            shape_envelope=[1, 60000, 48, 128],
            dtype="torch.bfloat16",
            device_kind="cuda",
        )
        assert c.n_calls_observed == 0
        assert c.source_hash == ""

    def test_site_key_is_stable(self):
        c1 = CallSiteEnvelope(
            file="/x/chunk_o.py", lineno=161,
            shape_envelope=[1], dtype="bf16", device_kind="cuda",
        )
        c2 = CallSiteEnvelope(
            file="/x/chunk_o.py", lineno=161,
            shape_envelope=[2, 3, 4],  # different shape — same site
            dtype="fp16", device_kind="cuda",
        )
        assert c1.site_key() == c2.site_key()
        assert c1.site_key() == "/x/chunk_o.py:161"


# ─── PreallocV2Manifest ──────────────────────────────────────────────


class TestManifest:
    def test_default_version_matches_constant(self):
        m = PreallocV2Manifest()
        assert m.version == _MANIFEST_VERSION

    def test_cache_key_is_deterministic(self):
        m1 = PreallocV2Manifest(
            model_arch="qwen3_5", max_model_len=180000,
            max_num_batched_tokens=2048, tp_size=1,
            kv_cache_dtype="turboquant_k8v4",
            vllm_pin_sha="01d4d1ad3", genesis_pin_sha="b2924c6",
            scope=["fla", "mamba"],
        )
        m2 = PreallocV2Manifest(
            model_arch="qwen3_5", max_model_len=180000,
            max_num_batched_tokens=2048, tp_size=1,
            kv_cache_dtype="turboquant_k8v4",
            vllm_pin_sha="01d4d1ad3", genesis_pin_sha="b2924c6",
            scope=["mamba", "fla"],   # different order — should normalize
        )
        assert m1.cache_key() == m2.cache_key()

    def test_cache_key_changes_on_max_model_len(self):
        m1 = PreallocV2Manifest(
            model_arch="qwen3_5", max_model_len=180000,
        )
        m2 = PreallocV2Manifest(
            model_arch="qwen3_5", max_model_len=280000,  # changed
        )
        assert m1.cache_key() != m2.cache_key()

    def test_cache_key_changes_on_kv_cache_dtype(self):
        m1 = PreallocV2Manifest(kv_cache_dtype="turboquant_k8v4")
        m2 = PreallocV2Manifest(kv_cache_dtype="fp8_e5m2")
        assert m1.cache_key() != m2.cache_key()

    def test_cache_key_truncated_to_16_hex(self):
        m = PreallocV2Manifest(model_arch="x")
        assert len(m.cache_key()) == 16
        assert all(c in "0123456789abcdef" for c in m.cache_key())


# ─── JSON serialization round-trip ───────────────────────────────────


class TestSerialization:
    def test_empty_round_trip(self):
        m = PreallocV2Manifest(model_arch="qwen3_5", max_model_len=180000)
        raw = m.to_json()
        m2 = PreallocV2Manifest.from_json(raw)
        assert m2.model_arch == "qwen3_5"
        assert m2.max_model_len == 180000
        assert m2.call_sites == {}

    def test_round_trip_with_call_sites(self):
        m = PreallocV2Manifest(
            model_arch="qwen3_5", max_model_len=180000,
            recorded_at="2026-05-05T12:00:00Z",
        )
        m.call_sites["fla.chunk_o:161"] = CallSiteEnvelope(
            file="/usr/.../chunk_o.py", lineno=161,
            shape_envelope=[1, 60000, 48, 128],
            dtype="torch.bfloat16", device_kind="cuda",
            n_calls_observed=168,
        )
        m.call_sites["fla.chunk_delta_h:332"] = CallSiteEnvelope(
            file="/usr/.../chunk_delta_h.py", lineno=332,
            shape_envelope=[1, 938, 48, 128, 128],
            dtype="torch.bfloat16", device_kind="cuda",
            n_calls_observed=168,
        )
        raw = m.to_json()
        m2 = PreallocV2Manifest.from_json(raw)
        assert len(m2.call_sites) == 2
        c = m2.call_sites["fla.chunk_o:161"]
        assert isinstance(c, CallSiteEnvelope)
        assert c.shape_envelope == [1, 60000, 48, 128]
        assert c.n_calls_observed == 168

    def test_version_mismatch_raises(self):
        bad = '{"version": 999, "model_arch": "x"}'
        with pytest.raises(ValueError, match="version mismatch"):
            PreallocV2Manifest.from_json(bad)

    def test_json_is_stable_sorted(self):
        """to_json must be deterministic — same content → same bytes
        (lets cache hashing work across runs)."""
        m1 = PreallocV2Manifest(model_arch="x", max_model_len=100)
        m1.call_sites["a"] = CallSiteEnvelope(
            file="a", lineno=1, shape_envelope=[1], dtype="x",
            device_kind="cuda",
        )
        m1.call_sites["b"] = CallSiteEnvelope(
            file="b", lineno=2, shape_envelope=[2], dtype="y",
            device_kind="cuda",
        )
        m2 = PreallocV2Manifest(model_arch="x", max_model_len=100)
        # Insert in REVERSE order
        m2.call_sites["b"] = CallSiteEnvelope(
            file="b", lineno=2, shape_envelope=[2], dtype="y",
            device_kind="cuda",
        )
        m2.call_sites["a"] = CallSiteEnvelope(
            file="a", lineno=1, shape_envelope=[1], dtype="x",
            device_kind="cuda",
        )
        assert m1.to_json() == m2.to_json()


# ─── Cache directory + write/load ────────────────────────────────────


class TestCacheDirectory:
    def test_default_dir_under_user_home(self, monkeypatch):
        monkeypatch.delenv("GENESIS_PREALLOC_V2_CACHE_DIR", raising=False)
        d = get_cache_dir()
        assert "genesis" in str(d).lower()
        assert "prealloc_v2" in str(d).lower()

    def test_env_override_respected(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GENESIS_PREALLOC_V2_CACHE_DIR", str(tmp_path))
        d = get_cache_dir()
        assert d == tmp_path

    def test_dir_auto_created(self, monkeypatch, tmp_path):
        nested = tmp_path / "deep" / "cache" / "dir"
        monkeypatch.setenv("GENESIS_PREALLOC_V2_CACHE_DIR", str(nested))
        d = get_cache_dir()
        assert d.is_dir()


class TestWriteLoadRoundTrip:
    def test_write_then_load(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GENESIS_PREALLOC_V2_CACHE_DIR", str(tmp_path))
        m = PreallocV2Manifest(
            model_arch="qwen3_5", max_model_len=180000,
            kv_cache_dtype="turboquant_k8v4",
            recorded_at="2026-05-05T12:00:00Z",
        )
        m.call_sites["fla.chunk_o:161"] = CallSiteEnvelope(
            file="/x/chunk_o.py", lineno=161,
            shape_envelope=[1, 60000, 48, 128],
            dtype="torch.bfloat16", device_kind="cuda",
            n_calls_observed=42,
        )

        path = write_manifest(m)
        assert path.is_file()

        loaded = load_manifest(m.cache_key())
        assert loaded is not None
        assert loaded.model_arch == "qwen3_5"
        assert loaded.call_sites["fla.chunk_o:161"].n_calls_observed == 42

    def test_load_missing_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GENESIS_PREALLOC_V2_CACHE_DIR", str(tmp_path))
        assert load_manifest("nonexistent_key") is None

    def test_load_corrupt_returns_none_logs_warn(
        self, monkeypatch, tmp_path, caplog,
    ):
        import logging
        monkeypatch.setenv("GENESIS_PREALLOC_V2_CACHE_DIR", str(tmp_path))
        bad_path = manifest_path("garbage_key")
        bad_path.write_text("{not valid json")
        with caplog.at_level(logging.WARNING, logger="genesis.prealloc_v2"):
            result = load_manifest("garbage_key")
        assert result is None
        # WARN logged so operator knows cache was rejected
        assert any("failed to load manifest" in r.message
                   for r in caplog.records)

    def test_atomic_write_via_tmp_then_replace(
        self, monkeypatch, tmp_path,
    ):
        """write_manifest must use tmp+replace so a crashed write doesn't
        leave a half-written manifest."""
        monkeypatch.setenv("GENESIS_PREALLOC_V2_CACHE_DIR", str(tmp_path))
        m = PreallocV2Manifest(model_arch="x", max_model_len=1)
        path = write_manifest(m)
        # No .tmp left over
        assert not path.with_suffix(".json.tmp").exists()

    def test_overwrite_existing(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GENESIS_PREALLOC_V2_CACHE_DIR", str(tmp_path))
        m1 = PreallocV2Manifest(
            model_arch="x", max_model_len=1, recorded_at="t1",
        )
        write_manifest(m1)
        m2 = PreallocV2Manifest(
            model_arch="x", max_model_len=1, recorded_at="t2",
        )
        write_manifest(m2)
        # Same cache key → overwrite, not duplicate
        loaded = load_manifest(m1.cache_key())
        assert loaded.recorded_at == "t2"


# ─── Env flag helpers ────────────────────────────────────────────────


class TestEnvFlags:
    def test_recorder_default_off(self, monkeypatch):
        monkeypatch.delenv("GENESIS_ENABLE_PREALLOC_V2_RECORDER", raising=False)
        assert recorder_enabled() is False

    def test_recorder_on_via_env(self, monkeypatch):
        monkeypatch.setenv("GENESIS_ENABLE_PREALLOC_V2_RECORDER", "1")
        assert recorder_enabled() is True

    def test_replay_default_off(self, monkeypatch):
        monkeypatch.delenv("GENESIS_ENABLE_PREALLOC_V2_REPLAY", raising=False)
        assert replay_enabled() is False

    def test_replay_on_via_env(self, monkeypatch):
        monkeypatch.setenv("GENESIS_ENABLE_PREALLOC_V2_REPLAY", "true")
        assert replay_enabled() is True

    @pytest.mark.parametrize("val", ["1", "true", "yes", "Y", "ON"])
    def test_truthy_variants(self, monkeypatch, val):
        monkeypatch.setenv("GENESIS_ENABLE_PREALLOC_V2_RECORDER", val)
        assert recorder_enabled() is True

    @pytest.mark.parametrize("val", ["0", "false", "off", ""])
    def test_falsy_variants(self, monkeypatch, val):
        monkeypatch.setenv("GENESIS_ENABLE_PREALLOC_V2_RECORDER", val)
        assert recorder_enabled() is False
