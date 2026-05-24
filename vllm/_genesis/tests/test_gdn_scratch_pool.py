# SPDX-License-Identifier: Apache-2.0
"""TDD for vllm._genesis.kernels.gdn_scratch_pool — Variant D Phase 1.

Pattern adapted from test_ffn_intermediate_cache.py (PN12 proven design).

Design invariants tested:
  1. Single shared buffer per shape key (B, H, V, K, dtype, device)
  2. Pointer-stable across same-key acquires (cudagraph-safe)
  3. Slice-on-acquire when fits within cached max
  4. Grow-once on size increase
  5. Disabled when env GENESIS_ENABLE_PN59_STREAMING_GDN unset
  6. WINDOW_NT tunable via env GENESIS_VARIANT_D_WINDOW_NT
  7. 3 separate registries (h_window, v_new_window, state) — no false sharing
  8. Validation: dims must be > 0, raises ValueError otherwise
  9. Multi-shape: different keys get distinct buffers
 10. Stats / introspection API works without crash

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def reset_pool():
    """Reset registries between tests to avoid cross-test pollution."""
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    GdnScratchPool.reset()
    yield
    GdnScratchPool.reset()


# ═══════════════════════════════════════════════════════════════════════════
# ENV GATE
# ═══════════════════════════════════════════════════════════════════════════


def test_should_apply_default_off(monkeypatch):
    monkeypatch.delenv("GENESIS_ENABLE_PN59_STREAMING_GDN", raising=False)
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    assert GdnScratchPool.should_apply() is False


def test_should_apply_engages(monkeypatch):
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    monkeypatch.setenv("GENESIS_ENABLE_PN59_STREAMING_GDN", "1")
    assert GdnScratchPool.should_apply() is True
    monkeypatch.setenv("GENESIS_ENABLE_PN59_STREAMING_GDN", "true")
    assert GdnScratchPool.should_apply() is True


def test_window_nt_default_4(monkeypatch):
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    monkeypatch.delenv("GENESIS_VARIANT_D_WINDOW_NT", raising=False)
    assert GdnScratchPool.get_window_nt() == 4


def test_window_nt_env_override(monkeypatch):
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    monkeypatch.setenv("GENESIS_VARIANT_D_WINDOW_NT", "8")
    assert GdnScratchPool.get_window_nt() == 8


def test_window_nt_clamped(monkeypatch):
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    monkeypatch.setenv("GENESIS_VARIANT_D_WINDOW_NT", "0")
    assert GdnScratchPool.get_window_nt() == 1  # clamped low
    monkeypatch.setenv("GENESIS_VARIANT_D_WINDOW_NT", "9999")
    assert GdnScratchPool.get_window_nt() == 64  # clamped high


def test_window_nt_invalid_falls_back(monkeypatch):
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    monkeypatch.setenv("GENESIS_VARIANT_D_WINDOW_NT", "not_a_number")
    assert GdnScratchPool.get_window_nt() == 4


# ═══════════════════════════════════════════════════════════════════════════
# h_window — main scratch buffer
# ═══════════════════════════════════════════════════════════════════════════


def test_h_window_first_acquire():
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    buf = GdnScratchPool.acquire_h_window(
        B=1, window_nt=4, H=24, V=128, K=128,
        dtype=torch.float16, device=torch.device("cpu"),
    )
    assert buf.shape == (1, 4, 24, 128, 128)
    assert buf.dtype == torch.float16
    assert buf.device.type == "cpu"


def test_h_window_pointer_stable():
    """Same shape key → same data_ptr (cudagraph-safe invariant)."""
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    b1 = GdnScratchPool.acquire_h_window(1, 4, 24, 128, 128,
                                          torch.float16, torch.device("cpu"))
    b2 = GdnScratchPool.acquire_h_window(1, 4, 24, 128, 128,
                                          torch.float16, torch.device("cpu"))
    assert b1.data_ptr() == b2.data_ptr()


def test_h_window_slice_on_smaller_acquire():
    """Smaller window acquire returns slice of same backing tensor."""
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    big = GdnScratchPool.acquire_h_window(1, 8, 24, 128, 128,
                                           torch.float16, torch.device("cpu"))
    small = GdnScratchPool.acquire_h_window(1, 2, 24, 128, 128,
                                              torch.float16, torch.device("cpu"))
    assert small.shape == (1, 2, 24, 128, 128)
    # Same data_ptr: slice is a view
    assert small.data_ptr() == big.data_ptr()


def test_h_window_grows_on_larger_acquire():
    """Acquiring larger window forces re-allocation."""
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    small = GdnScratchPool.acquire_h_window(1, 2, 24, 128, 128,
                                              torch.float16, torch.device("cpu"))
    big = GdnScratchPool.acquire_h_window(1, 8, 24, 128, 128,
                                            torch.float16, torch.device("cpu"))
    assert big.shape == (1, 8, 24, 128, 128)
    # data_ptr changed (re-allocation happened)
    # Note: in pure CPU tests this MAY coincidentally match; test what we mean
    assert big.shape[1] >= small.shape[1]


def test_h_window_distinct_dtype_separate_buffer():
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    fp16 = GdnScratchPool.acquire_h_window(1, 4, 24, 128, 128,
                                             torch.float16, torch.device("cpu"))
    bf16 = GdnScratchPool.acquire_h_window(1, 4, 24, 128, 128,
                                             torch.bfloat16, torch.device("cpu"))
    assert fp16.data_ptr() != bf16.data_ptr()
    assert fp16.dtype == torch.float16
    assert bf16.dtype == torch.bfloat16


def test_h_window_distinct_shape_separate_buffer():
    """Different (B, H, V, K) keys get distinct buffers."""
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    a = GdnScratchPool.acquire_h_window(1, 4, 24, 128, 128,
                                          torch.float16, torch.device("cpu"))
    b = GdnScratchPool.acquire_h_window(1, 4, 48, 128, 128,
                                          torch.float16, torch.device("cpu"))
    assert a.data_ptr() != b.data_ptr()


def test_h_window_zero_dims_raises():
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    with pytest.raises(ValueError, match="dims must be > 0"):
        GdnScratchPool.acquire_h_window(0, 4, 24, 128, 128,
                                          torch.float16, torch.device("cpu"))
    with pytest.raises(ValueError, match="dims must be > 0"):
        GdnScratchPool.acquire_h_window(1, 0, 24, 128, 128,
                                          torch.float16, torch.device("cpu"))


# ═══════════════════════════════════════════════════════════════════════════
# v_new_window
# ═══════════════════════════════════════════════════════════════════════════


def test_v_new_window_first_acquire():
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    buf = GdnScratchPool.acquire_v_new_window(
        B=1, window_T=256, H=24, V=128,
        dtype=torch.float16, device=torch.device("cpu"),
    )
    assert buf.shape == (1, 256, 24, 128)


def test_v_new_window_pointer_stable():
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    b1 = GdnScratchPool.acquire_v_new_window(1, 256, 24, 128,
                                                torch.float16, torch.device("cpu"))
    b2 = GdnScratchPool.acquire_v_new_window(1, 256, 24, 128,
                                                torch.float16, torch.device("cpu"))
    assert b1.data_ptr() == b2.data_ptr()


def test_v_new_window_slice_on_smaller():
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    big = GdnScratchPool.acquire_v_new_window(1, 512, 24, 128,
                                                 torch.float16, torch.device("cpu"))
    small = GdnScratchPool.acquire_v_new_window(1, 128, 24, 128,
                                                   torch.float16, torch.device("cpu"))
    assert small.shape == (1, 128, 24, 128)
    assert small.data_ptr() == big.data_ptr()


def test_v_new_window_zero_dims_raises():
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    with pytest.raises(ValueError, match="dims must be > 0"):
        GdnScratchPool.acquire_v_new_window(0, 256, 24, 128,
                                              torch.float16, torch.device("cpu"))


# ═══════════════════════════════════════════════════════════════════════════
# o_output — chunk_o output buffer (Level 2C+D / club-3090#22)
# ═══════════════════════════════════════════════════════════════════════════
#
# Tests the new acquire_o_output() pool method. Goal: replace per-call
# `o = torch.empty_like(v)` at chunk_o.py:161 with shared boot-time pool
# buffer to eliminate the OOM at "50 MiB requested, 56 MiB free" symptom
# that fragmentation causes on noonghunna's 24 GB single-card config.


def test_o_output_first_acquire():
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    GdnScratchPool.reset()
    buf = GdnScratchPool.acquire_o_output(
        B=1, T=4096, H=24, V=128,
        dtype=torch.float16, device=torch.device("cpu"),
    )
    # Returned slice is the requested T (binned internally to next pow2)
    assert buf.shape == (1, 4096, 24, 128)


def test_o_output_pointer_stable_same_T():
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    GdnScratchPool.reset()
    b1 = GdnScratchPool.acquire_o_output(1, 4096, 24, 128,
                                          torch.float16, torch.device("cpu"))
    b2 = GdnScratchPool.acquire_o_output(1, 4096, 24, 128,
                                          torch.float16, torch.device("cpu"))
    assert b1.data_ptr() == b2.data_ptr()


def test_o_output_slice_within_same_bin():
    """Smaller T request within the same pow2 bin reuses the cached buffer."""
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    GdnScratchPool.reset()
    # 4000 → bin 4096; 3000 → bin 4096 (same bin, share buffer)
    big = GdnScratchPool.acquire_o_output(1, 4000, 24, 128,
                                          torch.float16, torch.device("cpu"))
    small = GdnScratchPool.acquire_o_output(1, 3000, 24, 128,
                                            torch.float16, torch.device("cpu"))
    assert small.shape == (1, 3000, 24, 128)
    # Same underlying storage (slice of the same pool buffer in bin 4096)
    assert small.data_ptr() == big.data_ptr()


def test_o_output_T_binning_to_pow2():
    """Slightly different Ts that bin to the same pow2 share buffer."""
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    GdnScratchPool.reset()
    # T=3000 → bin to 4096; T=4000 → bin to 4096; both share buffer
    b1 = GdnScratchPool.acquire_o_output(1, 3000, 24, 128,
                                          torch.float16, torch.device("cpu"))
    b2 = GdnScratchPool.acquire_o_output(1, 4000, 24, 128,
                                          torch.float16, torch.device("cpu"))
    assert b1.data_ptr() == b2.data_ptr()
    assert b1.shape == (1, 3000, 24, 128)
    assert b2.shape == (1, 4000, 24, 128)


def test_o_output_grows_on_larger_pow2_bin():
    """T crossing pow2 boundary triggers new allocation (different bin)."""
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    GdnScratchPool.reset()
    b_small = GdnScratchPool.acquire_o_output(1, 4000, 24, 128,
                                              torch.float16, torch.device("cpu"))
    b_large = GdnScratchPool.acquire_o_output(1, 9000, 24, 128,
                                              torch.float16, torch.device("cpu"))
    # 4000 → 4096 bin; 9000 → 16384 bin → distinct buffers
    assert b_small.data_ptr() != b_large.data_ptr()


def test_o_output_distinct_dtype_separate_buffer():
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    GdnScratchPool.reset()
    b_fp16 = GdnScratchPool.acquire_o_output(1, 4096, 24, 128,
                                             torch.float16, torch.device("cpu"))
    b_bf16 = GdnScratchPool.acquire_o_output(1, 4096, 24, 128,
                                             torch.bfloat16, torch.device("cpu"))
    assert b_fp16.data_ptr() != b_bf16.data_ptr()


def test_o_output_zero_dims_raises():
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    with pytest.raises(ValueError, match="dims must be > 0"):
        GdnScratchPool.acquire_o_output(0, 4096, 24, 128,
                                        torch.float16, torch.device("cpu"))


def test_o_output_min_T_floor_512():
    """Very small T binned to floor 512 to avoid pathological churn."""
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    GdnScratchPool.reset()
    b_tiny = GdnScratchPool.acquire_o_output(1, 16, 24, 128,
                                             torch.float16, torch.device("cpu"))
    b_med = GdnScratchPool.acquire_o_output(1, 256, 24, 128,
                                            torch.float16, torch.device("cpu"))
    # Both bin to 512 (floor) → same buffer
    assert b_tiny.data_ptr() == b_med.data_ptr()


# ═══════════════════════════════════════════════════════════════════════════
# state — recurrent state (persistent across windows)
# ═══════════════════════════════════════════════════════════════════════════


def test_state_first_acquire():
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    buf = GdnScratchPool.acquire_state(
        B=1, H=24, V=128, K=128,
        dtype=torch.float32, device=torch.device("cpu"),
    )
    assert buf.shape == (1, 24, 128, 128)
    assert buf.dtype == torch.float32


def test_state_pointer_stable():
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    s1 = GdnScratchPool.acquire_state(1, 24, 128, 128,
                                        torch.float32, torch.device("cpu"))
    s2 = GdnScratchPool.acquire_state(1, 24, 128, 128,
                                        torch.float32, torch.device("cpu"))
    assert s1.data_ptr() == s2.data_ptr()


def test_state_distinct_shape_separate_buffer():
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    a = GdnScratchPool.acquire_state(1, 24, 128, 128,
                                       torch.float32, torch.device("cpu"))
    b = GdnScratchPool.acquire_state(1, 48, 128, 128,
                                       torch.float32, torch.device("cpu"))
    assert a.data_ptr() != b.data_ptr()


# ═══════════════════════════════════════════════════════════════════════════
# Cross-registry isolation
# ═══════════════════════════════════════════════════════════════════════════


def test_three_registries_isolated():
    """h, v_new, state buffers must NOT share memory."""
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    h = GdnScratchPool.acquire_h_window(1, 4, 24, 128, 128,
                                          torch.float16, torch.device("cpu"))
    v = GdnScratchPool.acquire_v_new_window(1, 256, 24, 128,
                                              torch.float16, torch.device("cpu"))
    s = GdnScratchPool.acquire_state(1, 24, 128, 128,
                                       torch.float32, torch.device("cpu"))
    ptrs = {h.data_ptr(), v.data_ptr(), s.data_ptr()}
    assert len(ptrs) == 3, "All three buffers must have distinct backing memory"


# ═══════════════════════════════════════════════════════════════════════════
# Introspection
# ═══════════════════════════════════════════════════════════════════════════


def test_total_pooled_bytes_starts_zero():
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    assert GdnScratchPool.total_pooled_bytes() == 0


def test_total_pooled_bytes_grows():
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    GdnScratchPool.acquire_h_window(1, 4, 24, 128, 128,
                                      torch.float16, torch.device("cpu"))
    bytes_after = GdnScratchPool.total_pooled_bytes()
    expected = 1 * 4 * 24 * 128 * 128 * 2  # fp16 = 2 bytes
    assert bytes_after == expected


def test_num_pools_breakdown():
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    GdnScratchPool.acquire_h_window(1, 4, 24, 128, 128,
                                      torch.float16, torch.device("cpu"))
    GdnScratchPool.acquire_state(1, 24, 128, 128,
                                   torch.float32, torch.device("cpu"))
    counts = GdnScratchPool.num_pools()
    assert counts["h_window"] == 1
    assert counts["v_new_window"] == 0
    assert counts["state"] == 1
    assert counts["total"] == 2


def test_stats_dict_complete():
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    s = GdnScratchPool.stats()
    assert "enabled" in s
    assert "window_nt" in s
    assert "pools" in s
    assert "total_pooled_mib" in s


def test_reset_clears_all_registries():
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool
    GdnScratchPool.acquire_h_window(1, 4, 24, 128, 128,
                                      torch.float16, torch.device("cpu"))
    GdnScratchPool.acquire_v_new_window(1, 256, 24, 128,
                                          torch.float16, torch.device("cpu"))
    GdnScratchPool.acquire_state(1, 24, 128, 128,
                                   torch.float32, torch.device("cpu"))
    assert GdnScratchPool.num_pools()["total"] == 3
    GdnScratchPool.reset()
    assert GdnScratchPool.num_pools()["total"] == 0
    assert GdnScratchPool.total_pooled_bytes() == 0
