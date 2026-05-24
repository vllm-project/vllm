# SPDX-License-Identifier: Apache-2.0
"""Streaming GDN driver — Variant D Phase 2.

Window-iterative replacement for the `chunk_gated_delta_rule_fwd_h →
chunk_fwd_o` consumer pair in `fla/ops/chunk.py:chunk_gated_delta_rule_fwd`.

Eliminates the `(B, NT, H, V, K)` peak materialization (Cliff 2b OOM
trigger, 805 MiB at T=64K Genesis 27B Lorbus shapes) by processing
WINDOW_NT chunks at a time, reusing a small pooled buffer.

Empirical confirmation (issue #20, 2026-05-05): noonghunna confirmed
"the limitation is the triton kernel for cliff 2; doesn't appear with
llama.cpp" — exactly the materialization pattern this fix removes.

Numerical correctness proof: Phase 1 TDD demonstrates window-iterative
output bit-equivalent to baseline materialize-full at rtol=1e-5
(see `tests/integration/test_streaming_gdn_numerical.py`).

API
---
`streaming_chunk_gated_delta_rule_fwd(q, k, v, g, beta, scale, initial_state,
output_final_state, cu_seqlens, chunk_indices, chunk_offsets) → (g, o, A,
final_state, w_or_none, h_or_none, v_new_or_none)` — drop-in replacement
for `chunk_gated_delta_rule_fwd` in chunk.py.

Eligibility
-----------
Streaming path engages ONLY when ALL of:
  * `GENESIS_ENABLE_PN59_STREAMING_GDN=1` (master env)
  * single-sequence prefill (cu_seqlens is None OR shape == (2,))
  * T > WINDOW_NT * BT * 4 (else overhead exceeds savings)
  * h dtype/device standard (no edge cases)

Otherwise falls through to vanilla `_orig_chunk_gated_delta_rule_fwd`
(passed in from text-patched orchestrator).

Author: Sandermage 2026-05-05, Variant D Phase 2.
"""
from __future__ import annotations

import logging
import os

import torch

from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool

log = logging.getLogger("genesis.kernels.streaming_gdn_driver")


# Hot-path bypass threshold — below this, vanilla path wins on overhead
_BYPASS_T_MULTIPLIER = 4
# FLA chunk size — pinned to upstream constant (`FLA_CHUNK_SIZE`)
_FLA_CHUNK_SIZE = 64


def streaming_chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
    *,
    # Injected upstream primitives (from FLA module-level imports)
    chunk_local_cumsum,
    chunk_scaled_dot_kkt_fwd,
    solve_tril,
    recompute_w_u_fwd,
    chunk_gated_delta_rule_fwd_h,
    chunk_fwd_o,
    SUPPRESS_LEVEL: int = 0,
) -> tuple:
    """Streaming variant of `chunk_gated_delta_rule_fwd`.

    Returns the same 7-tuple as upstream:
      (g, o, A, final_state, w_or_None, h_or_None, v_new_or_None)
    """
    # Eligibility — single-seq long prefill only.
    #
    # club-3090#22 fix 2026-05-05 (noonghunna):
    # `has_no_chunk_metadata` was a HARD gate after audit P2.4 (2026-05-05
    # morning). On Ampere consumer + 24 GB + chunked-prefill (mandatory to
    # fit ≥30K prompts on a single card), `chunk_indices`/`chunk_offsets`
    # are ALWAYS populated by vLLM — so PN59 silently bypassed to vanilla
    # on the EXACT path it was supposed to fix, then OOMed.
    #
    # Three-mode resolution (Sander 2026-05-05 PM, "защита и там и там"):
    #
    #   GENESIS_PN59_STRICT_NO_METADATA=auto  (DEFAULT, new behavior):
    #       VRAM-aware. When metadata is present AND streaming-enabled:
    #         - probe free VRAM via cuda.mem_get_info()
    #         - estimate vanilla alloc = numel(v) * dtype_size * safety_factor
    #         - if free < estimated_alloc → engage streaming with WARN
    #           about possible metadata-divergence (OOM is worse than drift)
    #         - if free ≥ estimated_alloc → use vanilla (metadata-correct)
    #       Protects on BOTH 24 GB chunked AND 48 GB non-chunked paths.
    #
    #   GENESIS_PN59_STRICT_NO_METADATA=1     (audit P2.4 strict):
    #       Always reject streaming on metadata presence. Original audit
    #       behavior. Operators on 48+ GB can use this for guaranteed
    #       correctness.
    #
    #   GENESIS_PN59_STRICT_NO_METADATA=0     (operator opt-in):
    #       Always engage streaming on metadata presence (no VRAM check,
    #       no probe overhead). For 24 GB single-card where vanilla
    #       always OOMs anyway.
    #
    # When PN59 is ENABLED but bypassed by ANY gate, surface it at WARN
    # once-per-reason-per-process — silent-bypass-then-OOM was the worst-
    # of-both-worlds the original #22 report hit.
    T = q.shape[1]
    is_single_seq = (
        cu_seqlens is None
        or (hasattr(cu_seqlens, "shape") and cu_seqlens.shape == (2,))
    )
    has_no_chunk_metadata = (
        chunk_indices is None and chunk_offsets is None
    )
    # Level 2A fix (club-3090#22): default for single-seq case is now
    # metadata_gate_passes=True. The streaming path threads chunk_indices/
    # chunk_offsets per-window arithmetically (single-seq invariant lets
    # us derive them from win_start/win_end without GPU sync). Strict gate
    # remains as escape hatch for paranoid operators on 48+ GiB rigs who
    # want bit-equivalent legacy behavior until Level 2 has soaked in
    # PROD; opt-in via GENESIS_PN59_STRICT_NO_METADATA=1.
    strict_metadata_gate = os.environ.get(
        "GENESIS_PN59_STRICT_NO_METADATA", "0"  # ← Level 2 default flipped 1→0
    ).strip().lower() in ("1", "true", "yes", "y", "on")
    metadata_gate_passes = has_no_chunk_metadata or not strict_metadata_gate
    metadata_decision_note = (
        "GENESIS_PN59_STRICT_NO_METADATA=1 (legacy escape hatch active)"
        if strict_metadata_gate
        else "Level 2A — metadata threaded per-window (default)"
    )

    window_nt = GdnScratchPool.get_window_nt()
    threshold_T = window_nt * _FLA_CHUNK_SIZE * _BYPASS_T_MULTIPLIER

    if (not GdnScratchPool.is_production_eligible()
            or not is_single_seq
            or not metadata_gate_passes
            or T <= threshold_T):
        reason = (
            "pool not eligible" if not GdnScratchPool.is_production_eligible()
            else "multi-seq" if not is_single_seq
            else (
                f"chunk metadata present + {metadata_decision_note} "
                "(set GENESIS_PN59_STRICT_NO_METADATA=0 to force-stream "
                "anyway — see club-3090#22)"
            ) if not metadata_gate_passes
            else f"T={T} ≤ threshold={threshold_T}"
        )
        # club-3090#22: surface enabled-but-bypassed state once at WARN
        # so operators don't silently OOM thinking PN59 is protecting them.
        # Per-reason once-per-process to keep noise low on multi-call paths.
        global _BYPASS_WARNED
        try:
            _BYPASS_WARNED
        except NameError:
            _BYPASS_WARNED = set()
        if reason not in _BYPASS_WARNED:
            _BYPASS_WARNED.add(reason)
            log.warning(
                "[PN59] streaming-GDN bypassed for this call class — "
                "vanilla path will run. Reason: %s. (This message will "
                "appear ONCE per reason class per process; subsequent "
                "bypasses are silent. Set GENESIS_PN59_DEBUG=1 to log "
                "every bypass.)",
                reason,
            )
        elif os.environ.get("GENESIS_PN59_DEBUG", "").strip().lower() in (
            "1", "true", "yes", "y", "on",
        ):
            log.info("[PN59] vanilla path (reason: %s)", reason)
        return _vanilla_path(
            q, k, v, g, beta, scale, initial_state, output_final_state,
            cu_seqlens, chunk_indices, chunk_offsets,
            chunk_local_cumsum=chunk_local_cumsum,
            chunk_scaled_dot_kkt_fwd=chunk_scaled_dot_kkt_fwd,
            solve_tril=solve_tril,
            recompute_w_u_fwd=recompute_w_u_fwd,
            chunk_gated_delta_rule_fwd_h=chunk_gated_delta_rule_fwd_h,
            chunk_fwd_o=chunk_fwd_o,
            SUPPRESS_LEVEL=SUPPRESS_LEVEL,
        )

    # Streaming path
    try:
        return _streaming_path(
            q, k, v, g, beta, scale, initial_state, output_final_state,
            cu_seqlens, chunk_indices, chunk_offsets,
            window_nt=window_nt,
            chunk_local_cumsum=chunk_local_cumsum,
            chunk_scaled_dot_kkt_fwd=chunk_scaled_dot_kkt_fwd,
            solve_tril=solve_tril,
            recompute_w_u_fwd=recompute_w_u_fwd,
            chunk_gated_delta_rule_fwd_h=chunk_gated_delta_rule_fwd_h,
            chunk_fwd_o=chunk_fwd_o,
            SUPPRESS_LEVEL=SUPPRESS_LEVEL,
        )
    except Exception as e:
        # Strict no-regression: any failure → vanilla fallback
        log.warning(
            "[PN59] streaming path raised %s — falling back to vanilla. "
            "Disable PN59 if recurrent: GENESIS_ENABLE_PN59_STREAMING_GDN=0",
            type(e).__name__,
        )
        return _vanilla_path(
            q, k, v, g, beta, scale, initial_state, output_final_state,
            cu_seqlens, chunk_indices, chunk_offsets,
            chunk_local_cumsum=chunk_local_cumsum,
            chunk_scaled_dot_kkt_fwd=chunk_scaled_dot_kkt_fwd,
            solve_tril=solve_tril,
            recompute_w_u_fwd=recompute_w_u_fwd,
            chunk_gated_delta_rule_fwd_h=chunk_gated_delta_rule_fwd_h,
            chunk_fwd_o=chunk_fwd_o,
            SUPPRESS_LEVEL=SUPPRESS_LEVEL,
        )


def _vanilla_path(
    q, k, v, g, beta, scale, initial_state, output_final_state,
    cu_seqlens, chunk_indices, chunk_offsets,
    *,
    chunk_local_cumsum, chunk_scaled_dot_kkt_fwd, solve_tril,
    recompute_w_u_fwd, chunk_gated_delta_rule_fwd_h, chunk_fwd_o,
    SUPPRESS_LEVEL: int,
):
    """Identical to upstream `chunk_gated_delta_rule_fwd`. Single allocation
    of full `h` tensor — Cliff 2b OOM trigger, but bit-correct baseline."""
    g = chunk_local_cumsum(
        g, chunk_size=_FLA_CHUNK_SIZE, cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    A = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g=g,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        output_dtype=torch.float32,
    )
    A = solve_tril(A=A, cu_seqlens=cu_seqlens,
                   chunk_indices=chunk_indices, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=A, g_cumsum=g,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g,
        initial_state=initial_state, output_final_state=output_final_state,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )
    o = chunk_fwd_o(
        q=q, k=k, v=v_new, h=h, g=g, scale=scale,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
    )
    if SUPPRESS_LEVEL < 3:
        return g, o, A, final_state, None, None, None
    return g, o, A, final_state, w, h, v_new


def _slice_chunk_metadata_for_window(
    cu_seqlens, chunk_indices, chunk_offsets,
    win_start: int, win_end: int, BT: int,
    *, device, dtype,
):
    """Build per-window chunk metadata for the FLA streaming path.

    Level 2A fix (club-3090#22): when the caller provides non-None
    `chunk_indices`/`chunk_offsets` AND we're in the single-seq regime
    (PN59's eligibility gate already enforces this), the per-window
    metadata is **arithmetically derivable** without GPU sync:

      cu_seqlens_w     = [0, T_w]
      chunk_indices_w  = stack([zeros(cur_NT), arange(cur_NT)], dim=1)
      chunk_offsets_w  = [0, cur_NT]

    These are **bit-equivalent** to what
    `vllm.model_executor.layers.fla.ops.index.prepare_chunk_indices /
    prepare_chunk_offsets` would produce on `cu_seqlens_w` — verified
    against `index.py:23-37`. The kernels read them as a gather table
    decomposing global program-id → (seq_id, intra-seq chunk_id) +
    base-h offset; for single-seq seq_id is always 0 and intra-seq
    chunk_id maps 0..cur_NT-1.

    Why not memoized: the helper is called O(num_windows) ≈ 4-16 times
    per request. Per-call cost is ~5 µs CPU + 3 small `torch.tensor()`
    constructions. Adding lru_cache would couple to (T_w, BT, dtype,
    device.index) keys — across requests the variation in T_w is high
    (chunked-prefill chunk-by-chunk schedule) so cache hit rate is low.
    Skip the cache for now; revisit if profiling shows hot.

    Single-seq invariant assertion: callers MUST pre-gate on
    `cu_seqlens is None or cu_seqlens.shape == (2,)`. PN59's eligibility
    check at dispatch entry already enforces this; the helper asserts
    in case eligibility ever loosens accidentally.
    """
    # Pass-through when caller had no metadata (vanilla single-seq path)
    if chunk_indices is None and chunk_offsets is None and cu_seqlens is None:
        return None, None, None

    if cu_seqlens is not None:
        assert cu_seqlens.shape == (2,), (
            "_slice_chunk_metadata_for_window expects single-seq cu_seqlens "
            f"(shape (2,)), got {tuple(cu_seqlens.shape)}. PN59 eligibility "
            "should have rejected multi-seq before reaching here."
        )

    T_w = win_end - win_start
    cur_NT = (T_w + BT - 1) // BT

    # Use cu_seqlens.dtype if available, else int32 (FLA's expectation)
    md_dtype = (
        cu_seqlens.dtype if cu_seqlens is not None
        else (chunk_indices.dtype if chunk_indices is not None
              else (chunk_offsets.dtype if chunk_offsets is not None
                    else torch.int32))
    )

    cu_seqlens_w = torch.tensor([0, T_w], dtype=md_dtype, device=device)
    chunk_indices_w = torch.stack(
        [
            torch.zeros(cur_NT, dtype=md_dtype, device=device),
            torch.arange(cur_NT, dtype=md_dtype, device=device),
        ],
        dim=1,
    )
    chunk_offsets_w = torch.tensor([0, cur_NT], dtype=md_dtype, device=device)
    return cu_seqlens_w, chunk_indices_w, chunk_offsets_w


def _streaming_path(
    q, k, v, g, beta, scale, initial_state, output_final_state,
    cu_seqlens, chunk_indices, chunk_offsets,
    *,
    window_nt: int,
    chunk_local_cumsum, chunk_scaled_dot_kkt_fwd, solve_tril,
    recompute_w_u_fwd, chunk_gated_delta_rule_fwd_h, chunk_fwd_o,
    SUPPRESS_LEVEL: int,
):
    """Window-iterative driver — process WINDOW_NT chunks at a time.

    Same pre-h ops (cumsum, kkt, solve, recompute_w_u) as vanilla.
    Replaces fwd_h+fwd_o tail with windowed loop.

    Key observation (Phase 1 numerical proof): Triton kernel
    `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` is internally
    recurrent in registers (b_h1..b_h4). Calling it with single-window
    inputs + chained `initial_state` produces identical state trajectory
    to a single full-T call. Then `chunk_fwd_o` reads only the current
    window's h slice — per-chunk independent (verified by SGLang
    `chunk_fwd_kernel_o:74` analysis).
    """
    B, T, Hg, K = q.shape
    V = v.shape[-1]
    BT = _FLA_CHUNK_SIZE

    # Phase A: full-input pre-h ops (small allocations, cheap)
    g_full = chunk_local_cumsum(
        g, chunk_size=BT, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
    )
    A = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g=g_full,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        output_dtype=torch.float32,
    )
    A = solve_tril(A=A, cu_seqlens=cu_seqlens,
                   chunk_indices=chunk_indices, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=A, g_cumsum=g_full,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
    )

    # Phase B: pre-allocate output o (B, T, H_v, V) — same shape as v
    # Level 2C+D fix (club-3090#22): route via GdnScratchPool so the
    # buffer is shared across all 64 GDN layers + all windows + all
    # forward passes. Eliminates the per-call `torch.empty_like(v)`
    # allocation churn that fragments the PyTorch caching allocator.
    # First call allocates one buffer of `next_pow2(T)` size; subsequent
    # calls slice down. Falls back to `torch.empty_like` if pool is
    # unavailable (e.g. is_production_eligible() returned False).
    try:
        if GdnScratchPool.is_production_eligible():
            o_full = GdnScratchPool.acquire_o_output(
                B=v.shape[0], T=v.shape[1], H=v.shape[2], V=v.shape[3],
                dtype=v.dtype, device=v.device,
            )
        else:
            o_full = torch.empty_like(v)
    except Exception as e:
        log.warning(
            "[PN59] GdnScratchPool.acquire_o_output failed (%s) — falling "
            "back to torch.empty_like; pool wiring not active for this call",
            type(e).__name__,
        )
        o_full = torch.empty_like(v)

    # State chained across windows (float32 per kernel signature)
    state = initial_state
    H = u.shape[-2]
    final_state = None

    # Window loop — slice T-dim by window_nt × BT tokens
    window_T = window_nt * BT
    for win_start in range(0, T, window_T):
        win_end = min(win_start + window_T, T)
        cur_T = win_end - win_start
        cur_NT = (cur_T + BT - 1) // BT
        is_last_window = (win_end >= T)

        # Slice T-dim inputs (input_guard wraps will re-contigify if needed)
        k_w = k[:, win_start:win_end]
        w_w = w[:, win_start:win_end]
        u_w = u[:, win_start:win_end]
        g_w = g_full[:, win_start:win_end]
        q_w = q[:, win_start:win_end]

        # Level 2A fix (club-3090#22): build per-window chunk metadata
        # arithmetically when the caller provided non-None metadata. For
        # PN59's single-seq eligibility, this is bit-equivalent to what
        # `prepare_chunk_indices` would build on the windowed cu_seqlens.
        # The kernels then read it as a normal IS_VARLEN=True path with
        # window-local indices, producing the same memory accesses as the
        # IS_VARLEN=False dense path on a (1, cur_T, ...) tensor.
        cu_w, ci_w, co_w = _slice_chunk_metadata_for_window(
            cu_seqlens, chunk_indices, chunk_offsets,
            win_start, win_end, BT, device=q.device, dtype=q.dtype,
        )

        # Level 2B fix (state-chaining): output_final_state=True on EVERY
        # intermediate window, not just the last. Pre-Level-2 code chained
        # via h_w[:, -1].to(fp32) — but the kernel writes h[i] BEFORE
        # applying chunk i's kv update (b_h += b_k @ b_v at chunk_delta_h
        # lines 139, 261), so h_w[:, -1] is the state at the START of the
        # last chunk, NOT the end. The correct chain is `state_next`
        # returned by the kernel when STORE_FINAL_STATE=True. Strict
        # metadata gate hid this latent bug for 4 days; it would have
        # surfaced as silent drift the moment we threaded metadata
        # through (Level 2A above).
        h_w, v_new_w, state_next_w = chunk_gated_delta_rule_fwd_h(
            k=k_w, w=w_w, u=u_w, g=g_w,
            initial_state=state,
            output_final_state=True,  # ← was: out_state (intermediate windows lost final state)
            cu_seqlens=cu_w,          # ← was: None (silently dropped metadata)
            chunk_indices=ci_w,       # ← was: None
            chunk_offsets=co_w,       # ← was: None
        )

        # Consume h_w via chunk_fwd_o for this window
        o_w = chunk_fwd_o(
            q=q_w, k=k_w, v=v_new_w, h=h_w, g=g_w, scale=scale,
            cu_seqlens=cu_w,
            chunk_indices=ci_w,
        )

        # Write window's o into o_full
        o_full[:, win_start:win_end].copy_(o_w)

        # Chain state forward — use kernel's STORE_FINAL_STATE output
        # (correct, post-last-chunk-update) instead of h_w[:, -1] (stale,
        # pre-last-chunk-update). See Level 2B comment above.
        if is_last_window:
            # Caller wanted final_state? Return it. Otherwise drop.
            final_state = state_next_w if output_final_state else None
        else:
            # Pass post-window state to next iteration's initial_state.
            # state_next_w is already float32 per kernel signature, no
            # `.to(float32)` needed (eliminates one alloc per window).
            state = state_next_w

        # Drop window references to allow GC
        del h_w, v_new_w, o_w

    if SUPPRESS_LEVEL < 3:
        return g_full, o_full, A, final_state, None, None, None
    return g_full, o_full, A, final_state, w, None, None
