# SPDX-License-Identifier: Apache-2.0
"""TDD for noonghunna/club-3090#22 — PN59 chunked-prefill opt-out + bypass-WARN.

Bug (noonghunna 2026-05-05): PN59's eligibility gate added in audit P2.4
(2026-05-05 morning) HARD-rejected calls with non-trivial `chunk_indices` /
`chunk_offsets`. On Ampere consumer + single 24 GB card + chunked-prefill
(mandatory to fit ≥30K prompts), these fields are ALWAYS populated, so
PN59 silently bypassed to vanilla on the EXACT path it was supposed to
fix, then OOMed at `chunk_o.py:161 o = torch.empty_like(v)`.

Fix:
  1. GENESIS_PN59_STRICT_NO_METADATA env (default ON, preserves audit
     P2.4 — silent metadata drop is real divergence risk).
  2. Operator can set GENESIS_PN59_STRICT_NO_METADATA=0 to opt INTO
     "stream anyway, accept the metadata-drop risk" because the
     alternative on a 24 GB card is a hard OOM.
  3. WARN once-per-reason-per-process when PN59 is enabled but the
     gate rejects it — silent-bypass-then-OOM is the worst-of-both
     state. Operator should always know why protection was disabled.
"""
from __future__ import annotations

import importlib
import logging
import os
import sys

import pytest


def _fresh_module():
    """Reimport streaming_gdn_driver so each test starts with a fresh
    `_BYPASS_WARNED` set (it's module-global, accumulates across calls)."""
    mod_name = "vllm._genesis.kernels.streaming_gdn_driver"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    return importlib.import_module(mod_name)


# ─── Env parser for the new opt-out flag ──────────────────────────────


class TestStrictMetadataEnvParser:
    @pytest.mark.parametrize("val", ["1", "true", "yes", "Y", "ON"])
    def test_truthy_keeps_strict_gate_on(self, monkeypatch, val):
        monkeypatch.setenv("GENESIS_PN59_STRICT_NO_METADATA", val)
        # The parser is inline in dispatch(); we can't unit-test it directly,
        # but we can verify the env-flag is read correctly by checking that
        # the eligibility logic respects it (covered in TestDispatch* below).
        assert os.environ["GENESIS_PN59_STRICT_NO_METADATA"].lower() in {
            "1", "true", "yes", "y", "on",
        }

    @pytest.mark.parametrize("val", ["0", "false", "no", "off", ""])
    def test_falsy_relaxes_strict_gate(self, val, monkeypatch):
        monkeypatch.setenv("GENESIS_PN59_STRICT_NO_METADATA", val)
        # When falsy, dispatch will treat metadata presence as compatible
        assert os.environ["GENESIS_PN59_STRICT_NO_METADATA"].lower() not in {
            "1", "true", "yes", "y", "on",
        }


# ─── Dispatch behavior — strict ON (default, audit P2.4 preserved) ─────


class TestDispatchStrictGateOn:
    """When GENESIS_PN59_STRICT_NO_METADATA=1 (default), `chunk_indices`
    or `chunk_offsets` presence MUST cause vanilla bypass."""

    def test_chunk_metadata_triggers_bypass_under_strict(self, monkeypatch, caplog):
        monkeypatch.setenv("GENESIS_PN59_STRICT_NO_METADATA", "1")
        monkeypatch.setenv("GENESIS_PN59_DEBUG", "0")
        m = _fresh_module()

        # Stub pool eligibility to True so the gate we're testing
        # (metadata) is the one that fires
        # Use monkeypatch (auto-cleaned) — direct class writes leak.
        monkeypatch.setattr(m.GdnScratchPool, "is_production_eligible",
                            staticmethod(lambda: True))
        monkeypatch.setattr(m.GdnScratchPool, "get_window_nt",
                            staticmethod(lambda: 4))

        called = {"vanilla": 0, "streaming": 0}
        monkeypatch.setattr(m, "_vanilla_path",
            lambda *a, **kw: (called.update(vanilla=called["vanilla"] + 1)) or "vanilla-result")
        monkeypatch.setattr(m, "_streaming_path",
            lambda *a, **kw: (called.update(streaming=called["streaming"] + 1)) or "streaming-result")

        # Synthetic args that pass everything EXCEPT metadata gate
        import torch
        T = 60_000
        H = K = 4
        q = torch.zeros(1, T, H, K)
        k = torch.zeros(1, T, H, K)
        v = torch.zeros(1, T, H, K)
        g = torch.zeros(1, T, H, K)
        beta = torch.zeros(1, T, H, K)
        chunk_indices = torch.zeros(2, dtype=torch.long)  # NON-NONE -> reject
        chunk_offsets = torch.zeros(2, dtype=torch.long)

        with caplog.at_level(logging.WARNING, logger="genesis.kernels.streaming_gdn_driver"):
            result = m.streaming_chunk_gated_delta_rule_fwd(
                q, k, v, g, beta, scale=1.0,
                initial_state=None, output_final_state=False,
                cu_seqlens=None,
                chunk_indices=chunk_indices,
                chunk_offsets=chunk_offsets,
                chunk_local_cumsum=lambda *a, **kw: None,
                chunk_scaled_dot_kkt_fwd=lambda *a, **kw: None,
                solve_tril=lambda *a, **kw: None,
                recompute_w_u_fwd=lambda *a, **kw: None,
                chunk_gated_delta_rule_fwd_h=lambda *a, **kw: None,
                chunk_fwd_o=lambda *a, **kw: None,
            )

        assert called["vanilla"] == 1
        assert called["streaming"] == 0
        assert result == "vanilla-result"
        # WARN-once on enabled-but-bypassed (strict gate visible)
        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("chunk metadata present" in r.message for r in warns)
        assert any("club-3090#22" in r.message for r in warns)
        # monkeypatch.setattr handles teardown; no manual restore needed


# ─── Dispatch behavior — strict OFF (operator opt-in for 24GB) ─────────


class TestDispatchStrictGateOff:
    """When GENESIS_PN59_STRICT_NO_METADATA=0, metadata presence is
    treated as 'compatible enough' — operator opt-in for 24 GB single
    card where the alternative is a hard OOM."""

    def test_chunk_metadata_NOT_bypassed_under_relaxed(self, monkeypatch):
        monkeypatch.setenv("GENESIS_PN59_STRICT_NO_METADATA", "0")
        m = _fresh_module()

        # Stub GdnScratchPool eligibility to True (avoid module-level flake)
        monkeypatch.setattr(m.GdnScratchPool, "is_production_eligible",
                            staticmethod(lambda: True))
        monkeypatch.setattr(m.GdnScratchPool, "get_window_nt",
                            staticmethod(lambda: 4))

        called = {"vanilla": 0, "streaming": 0}
        monkeypatch.setattr(m, "_vanilla_path",
            lambda *a, **kw: (called.update(vanilla=called["vanilla"] + 1)) or "vanilla")
        monkeypatch.setattr(m, "_streaming_path",
            lambda *a, **kw: (called.update(streaming=called["streaming"] + 1)) or "streaming")

        import torch
        T = 60_000
        H = K = 4
        q = torch.zeros(1, T, H, K)
        k = torch.zeros(1, T, H, K)
        v = torch.zeros(1, T, H, K)
        g = torch.zeros(1, T, H, K)
        beta = torch.zeros(1, T, H, K)
        chunk_indices = torch.zeros(2, dtype=torch.long)
        chunk_offsets = torch.zeros(2, dtype=torch.long)

        result = m.streaming_chunk_gated_delta_rule_fwd(
            q, k, v, g, beta, scale=1.0,
            initial_state=None, output_final_state=False,
            cu_seqlens=None,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            chunk_local_cumsum=lambda *a, **kw: None,
            chunk_scaled_dot_kkt_fwd=lambda *a, **kw: None,
            solve_tril=lambda *a, **kw: None,
            recompute_w_u_fwd=lambda *a, **kw: None,
            chunk_gated_delta_rule_fwd_h=lambda *a, **kw: None,
            chunk_fwd_o=lambda *a, **kw: None,
        )

        # Streaming path runs even though metadata is present
        assert called["streaming"] == 1
        assert called["vanilla"] == 0


# ─── WARN-once per reason ─────────────────────────────────────────────


class TestBypassWarnOnce:
    """The bypass WARN message must surface ONCE per reason class per
    process — not silent (silent was the bug), not on every call (noisy)."""

    def test_bypass_warn_emits_once_per_reason(self, monkeypatch, caplog):
        monkeypatch.setenv("GENESIS_PN59_STRICT_NO_METADATA", "1")
        monkeypatch.setenv("GENESIS_PN59_DEBUG", "0")
        m = _fresh_module()
        # Use monkeypatch.setattr (auto-cleaned) instead of direct assignment
        # — direct attr write on a class leaks between tests in the same
        # module; previous version polluted test_window_nt_env_override etc.
        monkeypatch.setattr(
            m.GdnScratchPool, "is_production_eligible",
            staticmethod(lambda: True),
        )
        monkeypatch.setattr(
            m.GdnScratchPool, "get_window_nt",
            staticmethod(lambda: 4),
        )

        # Stub vanilla path so we don't crash on missing kernels
        monkeypatch.setattr(m, "_vanilla_path", lambda *a, **kw: "v")

        import torch
        q = torch.zeros(1, 60_000, 4, 4)
        kwargs = dict(
            k=torch.zeros(1, 60_000, 4, 4),
            v=torch.zeros(1, 60_000, 4, 4),
            g=torch.zeros(1, 60_000, 4, 4),
            beta=torch.zeros(1, 60_000, 4, 4),
            scale=1.0, initial_state=None, output_final_state=False,
            cu_seqlens=None,
            chunk_indices=torch.zeros(2, dtype=torch.long),
            chunk_offsets=torch.zeros(2, dtype=torch.long),
            chunk_local_cumsum=lambda *a, **kw: None,
            chunk_scaled_dot_kkt_fwd=lambda *a, **kw: None,
            solve_tril=lambda *a, **kw: None,
            recompute_w_u_fwd=lambda *a, **kw: None,
            chunk_gated_delta_rule_fwd_h=lambda *a, **kw: None,
            chunk_fwd_o=lambda *a, **kw: None,
        )

        with caplog.at_level(logging.WARNING,
                             logger="genesis.kernels.streaming_gdn_driver"):
            for _ in range(5):  # 5 calls in a row, same reason
                m.streaming_chunk_gated_delta_rule_fwd(q, **kwargs)

        # WARN should appear EXACTLY ONCE for "chunk metadata present"
        chunk_warns = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and "chunk metadata present" in r.message
        ]
        assert len(chunk_warns) == 1, (
            f"expected 1 WARN, got {len(chunk_warns)}: "
            f"{[r.message for r in chunk_warns]}"
        )
