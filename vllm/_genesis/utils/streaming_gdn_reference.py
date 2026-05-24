# SPDX-License-Identifier: Apache-2.0
"""Pure-PyTorch reference for streaming-GDN numerical correctness proof.

This module provides BOTH the baseline (materialize-full) AND the
window-iterative variants of the GDN forward in pure PyTorch. Used
ONLY for Phase 1 numerical TDD — proves window-iterative produces
bit-identical output vs baseline BEFORE we touch Triton kernels.

Why this is necessary
---------------------
Triton kernel `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` already
processes chunks recurrently in registers (`b_h1..b_h4`). The window-
iterative driver just slices T into windows of WINDOW_NT chunks and
calls the kernel per-window with state passed forward through
initial_state. If our window-driver is mathematically equivalent (in
pure Python with identical numerics), it WILL be equivalent when wired
to the Triton kernel.

This module deliberately uses ONLY torch ops that are deterministic in
fp32 — no Triton kernels — so the test passes on CPU-only Mac.

Mathematical model (gated delta rule, simplified)
-------------------------------------------------
For each chunk c in 0..NT:
  state = decay(g_c) * state + outer(k_c.T @ v_c)
  o_c = q_c @ state

In materialize-full baseline:
  h = empty(B, NT, H, V, K)
  for c in 0..NT: state_evolve; h[:, c] = state
  for c in 0..NT: o[:, c] = q[:, c] @ h[:, c]      # parallelizable

In window-iterative variant:
  state = initial
  for w_start in 0..NT step W:
    h_window = empty(B, W, H, V, K)
    for c in w_start..w_start+W: state_evolve; h_window[:, c-w_start] = state
    for c in w_start..w_start+W: o[:, c] = q[:, c] @ h_window[:, c-w_start]
    drop h_window
  # state continues from last chunk

Both produce IDENTICAL state trajectory (same recurrence) and
IDENTICAL o per chunk (same matmul). Window-iterative only differs
in WHEN h is materialized — bit-exact gate must hold.

Author: Sandermage 2026-05-05, Variant D Phase 1.
"""
from __future__ import annotations

import torch


def _gdn_chunk_recurrence_step(
    state: torch.Tensor,  # (B, H, V, K) fp32
    k_c: torch.Tensor,    # (B, H, BT, K)
    v_c: torch.Tensor,    # (B, H, BT, V)
    g_c: torch.Tensor,    # (B, H, BT) — gate
) -> torch.Tensor:
    """Simplified GDN state recurrence for one chunk.

    Real FLA does more (delta rule, beta, decay matrix), but for
    numerical equivalence proof we only need the structural property:
    each step takes prior state + chunk inputs → new state.

    Args:
      state: (B, H, V, K) running state, fp32
      k_c, v_c, g_c: chunk inputs

    Returns:
      new_state (B, H, V, K) fp32
    """
    # Apply gate decay (use mean of gate over BT dim for simple test)
    decay = g_c.mean(dim=-1, keepdim=False).unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
    state = state * decay
    # Add chunk contribution: outer-product (k.T @ v) summed over chunk tokens
    # k_c: (B, H, BT, K), v_c: (B, H, BT, V) → kv: (B, H, V, K)
    kv = torch.einsum("bhtk,bhtv->bhvk", k_c, v_c)
    state = state + kv
    return state


def baseline_materialize_full(
    q: torch.Tensor,     # (B, H, NT, BT, K)
    k: torch.Tensor,     # (B, H, NT, BT, K)
    v: torch.Tensor,     # (B, H, NT, BT, V)
    g: torch.Tensor,     # (B, H, NT, BT)
    initial_state: torch.Tensor,  # (B, H, V, K) fp32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Baseline: materialize full h then compute o.

    Returns:
      o: (B, H, NT, BT, V)
      final_state: (B, H, V, K)
    """
    B, H, NT, BT, K = q.shape
    V = v.shape[-1]
    device, dtype = q.device, q.dtype

    # Materialize full h
    h = torch.empty(B, NT, H, V, K, dtype=torch.float32, device=device)
    state = initial_state.clone()
    for c in range(NT):
        state = _gdn_chunk_recurrence_step(state, k[:, :, c], v[:, :, c], g[:, :, c])
        h[:, c] = state

    # Compute o (parallelizable in real implementation)
    o = torch.empty(B, H, NT, BT, V, dtype=dtype, device=device)
    for c in range(NT):
        # q[:, :, c]: (B, H, BT, K), h[:, c].permute(...): (B, H, V, K)
        # o = q @ state.T = (B, H, BT, K) @ (B, H, K, V) → (B, H, BT, V)
        state_chunk = h[:, c]  # (B, H, V, K)
        o[:, :, c] = torch.einsum("bhtk,bhvk->bhtv", q[:, :, c], state_chunk).to(dtype)

    final_state = state
    return o, final_state


def window_iterative(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor,
    window_nt: int,
    pool: object | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Window-iterative variant: process WINDOW_NT chunks at a time.

    Uses GdnScratchPool if `pool` is not None — otherwise vanilla allocation.

    Args:
      q/k/v/g: same as baseline
      initial_state: (B, H, V, K) fp32
      window_nt: chunks per window (typically 4-8)
      pool: optional GdnScratchPool instance for h_window reuse

    Returns:
      o: (B, H, NT, BT, V)
      final_state: (B, H, V, K)
    """
    B, H, NT, BT, K = q.shape
    V = v.shape[-1]
    device, dtype = q.device, q.dtype

    # Output buffer (B, H, NT, BT, V) — same shape as baseline output
    o = torch.empty(B, H, NT, BT, V, dtype=dtype, device=device)

    state = initial_state.clone()

    for w_start in range(0, NT, window_nt):
        w_end = min(w_start + window_nt, NT)
        cur_window = w_end - w_start

        # Acquire window scratch buffer
        if pool is not None:
            h_window = pool.acquire_h_window(
                B=B, window_nt=cur_window, H=H, V=V, K=K,
                dtype=torch.float32, device=device,
            )
        else:
            h_window = torch.empty(B, cur_window, H, V, K,
                                    dtype=torch.float32, device=device)

        # Phase 1 within window: state evolution writes h_window
        for i, c in enumerate(range(w_start, w_end)):
            state = _gdn_chunk_recurrence_step(
                state, k[:, :, c], v[:, :, c], g[:, :, c]
            )
            h_window[:, i] = state

        # Phase 2: consume h_window, write o slice
        for i, c in enumerate(range(w_start, w_end)):
            state_chunk = h_window[:, i]  # (B, H, V, K)
            o[:, :, c] = torch.einsum(
                "bhtk,bhvk->bhtv", q[:, :, c], state_chunk
            ).to(dtype)
        # h_window dropped here (or returned to pool — pool keeps it for reuse)

    final_state = state
    return o, final_state
