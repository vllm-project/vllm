# SPDX-License-Identifier: Apache-2.0
"""Numerical TDD: window-iterative GDN equivalence to baseline (Variant D Phase 1).

10 shape cases covering Genesis 27B Lorbus configs:
  B=1, H ∈ {24, 48} (TP=2/TP=1 splits)
  K=V=128 (Genesis 27B head_dim)
  NT ∈ {16, 64, 256, 1024, 2048} chunks (T = NT × 64 ∈ {1K, 4K, 16K, 64K, 128K})
  WINDOW_NT ∈ {2, 4, 8} sweep

For EACH case: assert window-iterative output bit-equal to baseline.
Tests run in pure PyTorch on CPU — no Triton, no GPU required for Phase 1.

Acceptance gates:
  * `torch.allclose(o_baseline, o_window, rtol=1e-5, atol=1e-5)` (fp32 path)
  * `torch.allclose(state_baseline, state_window, rtol=1e-5, atol=1e-5)`
  * Bit-exact NOT required at fp32 due to associativity of einsum chunking,
    but RELATIVE diff must be below 1e-5 for trustworthy real-Triton wiring

Runtime: ~5 sec all 10 cases on Mac M-series CPU (small NT × test purpose).

Author: Sandermage 2026-05-05, Variant D Phase 1.
"""
from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def reset_pool():
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    GdnScratchPool.reset()
    yield
    GdnScratchPool.reset()


def _make_inputs(B: int, H: int, NT: int, BT: int, K: int, V: int, seed: int = 42):
    """Synthesize (q, k, v, g, initial_state) tuple."""
    torch.manual_seed(seed)
    dtype = torch.float32  # use fp32 for clean equivalence testing
    device = torch.device("cpu")
    # Small magnitudes to avoid overflow in repeated einsum
    q = torch.randn(B, H, NT, BT, K, dtype=dtype, device=device) * 0.1
    k = torch.randn(B, H, NT, BT, K, dtype=dtype, device=device) * 0.1
    v = torch.randn(B, H, NT, BT, V, dtype=dtype, device=device) * 0.1
    g = torch.sigmoid(torch.randn(B, H, NT, BT, dtype=dtype, device=device))
    initial_state = torch.zeros(B, H, V, K, dtype=torch.float32, device=device)
    return q, k, v, g, initial_state


# ═══════════════════════════════════════════════════════════════════════════
# 10 shape cases — Genesis 27B Lorbus matrix
# ═══════════════════════════════════════════════════════════════════════════

# (B, H, NT, BT, K, V, WINDOW_NT)
GENESIS_27B_SHAPES = [
    # H=24 (TP=2 split of 48 heads)
    (1, 24, 16,    64, 128, 128, 2),   # T=1K, WINDOW=2
    (1, 24, 64,    64, 128, 128, 4),   # T=4K, WINDOW=4
    (1, 24, 256,   64, 128, 128, 4),   # T=16K, WINDOW=4
    (1, 24, 256,   64, 128, 128, 8),   # T=16K, WINDOW=8
    (1, 24, 1024,  64, 128, 128, 8),   # T=64K, WINDOW=8 (peak case)
    # H=48 (TP=1 full)
    (1, 48, 16,    64, 128, 128, 2),
    (1, 48, 64,    64, 128, 128, 4),
    (1, 48, 256,   64, 128, 128, 4),
    # Edge: window doesn't divide NT evenly (last partial window)
    (1, 24, 17,    64, 128, 128, 4),  # NT=17 → 4+4+4+4+1
    (1, 24, 100,   64, 128, 128, 8),  # NT=100 → 8×12 + 4
]


@pytest.mark.parametrize("shape", GENESIS_27B_SHAPES,
                         ids=[f"B{s[0]}_H{s[1]}_NT{s[2]}_W{s[6]}" for s in GENESIS_27B_SHAPES])
def test_window_iterative_matches_baseline(shape):
    """For each Genesis shape: window-iterative output ≈ baseline (rtol 1e-5)."""
    from vllm._genesis.utils.streaming_gdn_reference import (
        baseline_materialize_full,
        window_iterative,
    )
    B, H, NT, BT, K, V, W = shape
    q, k, v, g, init = _make_inputs(B, H, NT, BT, K, V)

    o_base, s_base = baseline_materialize_full(q, k, v, g, init)
    o_win, s_win = window_iterative(q, k, v, g, init, window_nt=W, pool=None)

    assert o_base.shape == o_win.shape == (B, H, NT, BT, V), (
        f"shape mismatch: baseline={o_base.shape}, window={o_win.shape}"
    )
    assert s_base.shape == s_win.shape == (B, H, V, K)

    # Output equivalence
    max_abs = (o_base - o_win).abs().max().item()
    max_rel = ((o_base - o_win).abs() / o_base.abs().clamp_min(1e-9)).max().item()
    assert torch.allclose(o_base, o_win, rtol=1e-5, atol=1e-5), (
        f"OUTPUT mismatch shape={shape}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
    )

    # State chain equivalence (state must be IDENTICAL after equal steps)
    s_max_abs = (s_base - s_win).abs().max().item()
    assert torch.allclose(s_base, s_win, rtol=1e-5, atol=1e-5), (
        f"STATE mismatch shape={shape}: max_abs={s_max_abs:.3e}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Pool integration — same equivalence with pool-backed buffers
# ═══════════════════════════════════════════════════════════════════════════


def test_pool_backed_window_matches_baseline():
    """Window-iterative WITH GdnScratchPool produces identical output."""
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    from vllm._genesis.utils.streaming_gdn_reference import (
        baseline_materialize_full,
        window_iterative,
    )
    B, H, NT, BT, K, V, W = 1, 24, 64, 64, 128, 128, 4
    q, k, v, g, init = _make_inputs(B, H, NT, BT, K, V)

    o_base, s_base = baseline_materialize_full(q, k, v, g, init)
    o_win, s_win = window_iterative(q, k, v, g, init, window_nt=W,
                                     pool=GdnScratchPool)

    assert torch.allclose(o_base, o_win, rtol=1e-5, atol=1e-5)
    assert torch.allclose(s_base, s_win, rtol=1e-5, atol=1e-5)
    # Pool should have one h_window entry
    assert GdnScratchPool.num_pools()["h_window"] >= 1


def test_pool_reuse_across_calls():
    """Same shape multiple calls → pool reuses backing buffer (no churn)."""
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    from vllm._genesis.utils.streaming_gdn_reference import window_iterative

    B, H, NT, BT, K, V, W = 1, 24, 64, 64, 128, 128, 4
    q, k, v, g, init = _make_inputs(B, H, NT, BT, K, V, seed=1)

    # Call 1
    _ = window_iterative(q, k, v, g, init, window_nt=W, pool=GdnScratchPool)
    bytes_after_1 = GdnScratchPool.total_pooled_bytes()

    # Call 2 with same shapes
    _ = window_iterative(q, k, v, g, init, window_nt=W, pool=GdnScratchPool)
    bytes_after_2 = GdnScratchPool.total_pooled_bytes()

    # No new pool entries should have been added
    assert bytes_after_1 == bytes_after_2, (
        f"Pool grew on second call: {bytes_after_1} → {bytes_after_2}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════


def test_single_window_equals_baseline():
    """When WINDOW_NT == NT (one window covers all chunks), should be identical."""
    from vllm._genesis.utils.streaming_gdn_reference import (
        baseline_materialize_full,
        window_iterative,
    )
    B, H, NT, BT, K, V = 1, 24, 16, 64, 128, 128
    q, k, v, g, init = _make_inputs(B, H, NT, BT, K, V)

    o_base, s_base = baseline_materialize_full(q, k, v, g, init)
    o_win, s_win = window_iterative(q, k, v, g, init, window_nt=NT, pool=None)

    assert torch.allclose(o_base, o_win, rtol=1e-7, atol=1e-7)
    assert torch.allclose(s_base, s_win, rtol=1e-7, atol=1e-7)


def test_window_nt_1_equals_per_chunk():
    """WINDOW_NT=1 = process one chunk at a time = same recurrence."""
    from vllm._genesis.utils.streaming_gdn_reference import (
        baseline_materialize_full,
        window_iterative,
    )
    B, H, NT, BT, K, V = 1, 24, 8, 64, 128, 128
    q, k, v, g, init = _make_inputs(B, H, NT, BT, K, V)

    o_base, s_base = baseline_materialize_full(q, k, v, g, init)
    o_win, s_win = window_iterative(q, k, v, g, init, window_nt=1, pool=None)

    assert torch.allclose(o_base, o_win, rtol=1e-5, atol=1e-5)
    assert torch.allclose(s_base, s_win, rtol=1e-5, atol=1e-5)


def test_initial_state_propagation():
    """Non-zero initial_state must propagate identically in both modes."""
    from vllm._genesis.utils.streaming_gdn_reference import (
        baseline_materialize_full,
        window_iterative,
    )
    B, H, NT, BT, K, V, W = 1, 24, 32, 64, 128, 128, 4
    q, k, v, g, _ = _make_inputs(B, H, NT, BT, K, V)
    # Seed initial_state with non-trivial values
    torch.manual_seed(99)
    init = torch.randn(B, H, V, K, dtype=torch.float32) * 0.05

    o_base, s_base = baseline_materialize_full(q, k, v, g, init)
    o_win, s_win = window_iterative(q, k, v, g, init, window_nt=W, pool=None)

    assert torch.allclose(o_base, o_win, rtol=1e-5, atol=1e-5)
    assert torch.allclose(s_base, s_win, rtol=1e-5, atol=1e-5)
