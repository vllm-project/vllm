# SPDX-License-Identifier: Apache-2.0
"""TDD for PN59 — streaming-GDN orchestrator (Variant D Phase 2).

Tests cover:
  1. Anchor matches pristine FLA chunk.py orchestrator signature
  2. Replacement keeps full vanilla path AFTER dispatch try/except
  3. Idempotency
  4. Env-flag default OFF + engages on "1"
  5. Registry entry complete + conflicts_with empty (no exclusion)
  6. apply_all wiring registered
  7. Streaming driver bypass: short T → vanilla path
  8. Streaming driver eligibility check honors env flag
"""
from __future__ import annotations

import pytest
import torch


def _wiring():
    from vllm._genesis.wiring.hybrid import patch_N59_streaming_gdn as M
    return M


def _driver():
    from vllm._genesis.kernels import streaming_gdn_driver as D
    return D


# ─── Anchor / replacement tests ─────────────────────────────────────────────


def test_anchor_targets_chunk_orchestrator_signature():
    M = _wiring()
    assert "def chunk_gated_delta_rule_fwd(" in M.ANCHOR_OLD
    assert "scale: float," in M.ANCHOR_OLD
    assert "output_final_state: bool," in M.ANCHOR_OLD
    assert "chunk_offsets: torch.Tensor | None = None," in M.ANCHOR_OLD


def test_replacement_inserts_dispatcher_with_fallback():
    M = _wiring()
    # Marker
    assert "Genesis PN59" in M.ANCHOR_NEW
    # Driver call
    assert "streaming_chunk_gated_delta_rule_fwd" in M.ANCHOR_NEW
    # Strict no-regression: try/except with vanilla fallback
    assert "try:" in M.ANCHOR_NEW
    assert "except Exception:" in M.ANCHOR_NEW
    # Original signature preserved (so vanilla code below remains valid)
    assert "def chunk_gated_delta_rule_fwd(" in M.ANCHOR_NEW


def test_idempotent_on_synthetic(tmp_path):
    from vllm._genesis.wiring.text_patch import (
        TextPatch, TextPatcher, TextPatchResult,
    )
    M = _wiring()
    target = tmp_path / "chunk.py"
    # Synthetic file with anchor + dummy body
    target.write_text(
        "import torch\n\n"
        + M.ANCHOR_OLD
        + "    return None  # vanilla body placeholder\n"
    )
    patcher = TextPatcher(
        patch_name="PN59 test",
        target_file=str(target),
        marker=M.GENESIS_PN59_MARKER,
        sub_patches=[TextPatch(
            name="pn59",
            anchor=M.ANCHOR_OLD,
            replacement=M.ANCHOR_NEW,
            required=True,
        )],
    )
    r1, _ = patcher.apply()
    assert r1 == TextPatchResult.APPLIED
    body = target.read_text()
    assert "Genesis PN59" in body
    assert "streaming_chunk_gated_delta_rule_fwd" in body
    # Idempotency
    r2, _ = patcher.apply()
    assert r2 == TextPatchResult.IDEMPOTENT


# ─── Env-flag gate ──────────────────────────────────────────────────────────


def test_env_flag_default_off(monkeypatch):
    from vllm._genesis.dispatcher import should_apply
    monkeypatch.delenv("GENESIS_ENABLE_PN59_STREAMING_GDN", raising=False)
    decision, _ = should_apply("PN59")
    assert decision is False


def test_env_flag_engages(monkeypatch):
    from vllm._genesis.dispatcher import should_apply
    monkeypatch.setenv("GENESIS_ENABLE_PN59_STREAMING_GDN", "1")
    decision, _ = should_apply("PN59")
    assert decision is True


# ─── Registry / apply_all ───────────────────────────────────────────────────


def test_registry_entry_complete():
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    assert "PN59" in PATCH_REGISTRY
    meta = PATCH_REGISTRY["PN59"]
    assert meta["env_flag"] == "GENESIS_ENABLE_PN59_STREAMING_GDN"
    assert meta["default_on"] is False
    assert meta["category"] == "hybrid"
    assert "model_class" in meta["applies_to"]
    assert "qwen3_5" in meta["applies_to"]["model_class"]


def test_apply_all_registers_pn59():
    from vllm._genesis.patches import apply_all
    assert hasattr(apply_all, "apply_patch_N59_streaming_gdn")


# ─── Driver dispatch logic (CPU-runnable) ───────────────────────────────────


def test_driver_bypasses_when_env_disabled(monkeypatch):
    """Without env flag → vanilla path."""
    monkeypatch.delenv("GENESIS_ENABLE_PN59_STREAMING_GDN", raising=False)
    from vllm._genesis.kernels.streaming_gdn_driver import (
        streaming_chunk_gated_delta_rule_fwd,
    )

    # Mock primitives — just record call args
    calls = []

    def mock_cumsum(g, **kw):
        return g.clone()

    def mock_kkt(**kw):
        return torch.zeros_like(kw["k"])

    def mock_solve(A, **kw):
        return A

    def mock_recompute(k, v, beta, A, g_cumsum, **kw):
        return k.clone(), v.clone()

    def mock_fwd_h(k, w, u, g, initial_state, output_final_state, **kw):
        B, T = k.shape[:2]
        H, V, K = u.shape[-2], u.shape[-1], k.shape[-1]
        BT = 64
        NT = (T + BT - 1) // BT
        h = torch.zeros(B, NT, H, V, K, dtype=k.dtype, device=k.device)
        v_new = u.clone()
        final_state = (torch.zeros(B, H, V, K, dtype=torch.float32, device=k.device)
                       if output_final_state else None)
        calls.append("fwd_h")
        return h, v_new, final_state

    def mock_fwd_o(q, k, v, h, g, scale, **kw):
        calls.append("fwd_o")
        return torch.zeros_like(v)

    B, T, Hg, K, V = 1, 128, 4, 64, 64
    H = 4
    q = torch.randn(B, T, Hg, K)
    k = torch.randn(B, T, Hg, K)
    v = torch.randn(B, T, H, V)
    g = torch.randn(B, T, H)
    beta = torch.randn(B, T, H)

    # Call driver with all primitives
    result = streaming_chunk_gated_delta_rule_fwd(
        q=q, k=k, v=v, g=g, beta=beta, scale=0.1,
        initial_state=None, output_final_state=True,
        cu_seqlens=None, chunk_indices=None, chunk_offsets=None,
        chunk_local_cumsum=mock_cumsum,
        chunk_scaled_dot_kkt_fwd=mock_kkt,
        solve_tril=mock_solve,
        recompute_w_u_fwd=mock_recompute,
        chunk_gated_delta_rule_fwd_h=mock_fwd_h,
        chunk_fwd_o=mock_fwd_o,
        SUPPRESS_LEVEL=0,
    )

    # Vanilla path called fwd_h once + fwd_o once
    assert calls.count("fwd_h") == 1
    assert calls.count("fwd_o") == 1
    # Output tuple shape
    assert len(result) == 7  # (g, o, A, final_state, w_or_None, h_or_None, v_new_or_None)


def test_driver_bypasses_when_short_T(monkeypatch):
    """Short T < threshold → vanilla path even with env flag."""
    monkeypatch.setenv("GENESIS_ENABLE_PN59_STREAMING_GDN", "1")
    from vllm._genesis.kernels.streaming_gdn_driver import (
        streaming_chunk_gated_delta_rule_fwd,
    )

    calls = []
    def _mock_cumsum(g, **kw): return g
    def _mock_kkt(**kw): return torch.zeros_like(kw["k"])
    def _mock_solve(A, **kw): return A
    def _mock_recompute(k, v, beta, A, g_cumsum, **kw): return k.clone(), v.clone()

    def _mock_fwd_h(k, w, u, g, initial_state, output_final_state, **kw):
        B, T = k.shape[:2]
        H, V, K = u.shape[-2], u.shape[-1], k.shape[-1]
        BT = 64
        NT = (T + BT - 1) // BT
        h = torch.zeros(B, NT, H, V, K, dtype=k.dtype, device=k.device)
        v_new = u.clone()
        fs = (torch.zeros(B, H, V, K, dtype=torch.float32) if output_final_state else None)
        calls.append("fwd_h")
        return h, v_new, fs

    def _mock_fwd_o(q, k, v, h, g, scale, **kw):
        calls.append("fwd_o")
        return torch.zeros_like(v)

    # T=128 (small) — below WINDOW_NT(4) * BT(64) * MULT(4) = 1024 threshold
    B, T, Hg, K, V, H = 1, 128, 4, 64, 64, 4
    q = torch.randn(B, T, Hg, K)
    k = torch.randn(B, T, Hg, K)
    v = torch.randn(B, T, H, V)
    g = torch.randn(B, T, H)
    beta = torch.randn(B, T, H)

    streaming_chunk_gated_delta_rule_fwd(
        q=q, k=k, v=v, g=g, beta=beta, scale=0.1,
        initial_state=None, output_final_state=True,
        cu_seqlens=None, chunk_indices=None, chunk_offsets=None,
        chunk_local_cumsum=_mock_cumsum,
        chunk_scaled_dot_kkt_fwd=_mock_kkt,
        solve_tril=_mock_solve,
        recompute_w_u_fwd=_mock_recompute,
        chunk_gated_delta_rule_fwd_h=_mock_fwd_h,
        chunk_fwd_o=_mock_fwd_o,
        SUPPRESS_LEVEL=0,
    )
    # Even with env enabled, short T → still vanilla (single fwd_h+fwd_o)
    assert calls.count("fwd_h") == 1
    assert calls.count("fwd_o") == 1


def test_driver_bypasses_when_multi_seq(monkeypatch):
    """Multi-seq cu_seqlens → vanilla path even with long T."""
    monkeypatch.setenv("GENESIS_ENABLE_PN59_STREAMING_GDN", "1")
    from vllm._genesis.kernels.streaming_gdn_driver import (
        streaming_chunk_gated_delta_rule_fwd,
    )

    calls = []
    def _ck(g, **kw): return g
    def _kkt(**kw): return torch.zeros_like(kw["k"])
    def _st(A, **kw): return A
    def _rwu(k, v, beta, A, g_cumsum, **kw): return k.clone(), v.clone()

    def _fwd_h(k, w, u, g, initial_state, output_final_state, **kw):
        B, T = k.shape[:2]
        H, V, K = u.shape[-2], u.shape[-1], k.shape[-1]
        BT = 64
        NT = (T + BT - 1) // BT
        calls.append("fwd_h")
        return (torch.zeros(B, NT, H, V, K, dtype=k.dtype),
                u.clone(),
                torch.zeros(B, H, V, K, dtype=torch.float32) if output_final_state else None)

    def _fwd_o(q, k, v, h, g, scale, **kw):
        calls.append("fwd_o")
        return torch.zeros_like(v)

    # Long T but multi-seq cu_seqlens (shape != (2,))
    B, T, Hg, K, V, H = 1, 8192, 4, 64, 64, 4
    q = torch.randn(B, T, Hg, K)
    k = torch.randn(B, T, Hg, K)
    v = torch.randn(B, T, H, V)
    g = torch.randn(B, T, H)
    beta = torch.randn(B, T, H)
    cu_seqlens = torch.tensor([0, 4096, 8192])  # 2-seq batch

    streaming_chunk_gated_delta_rule_fwd(
        q=q, k=k, v=v, g=g, beta=beta, scale=0.1,
        initial_state=None, output_final_state=True,
        cu_seqlens=cu_seqlens, chunk_indices=None, chunk_offsets=None,
        chunk_local_cumsum=_ck, chunk_scaled_dot_kkt_fwd=_kkt,
        solve_tril=_st, recompute_w_u_fwd=_rwu,
        chunk_gated_delta_rule_fwd_h=_fwd_h, chunk_fwd_o=_fwd_o,
        SUPPRESS_LEVEL=0,
    )
    # Multi-seq → vanilla regardless of T
    assert calls.count("fwd_h") == 1
    assert calls.count("fwd_o") == 1
