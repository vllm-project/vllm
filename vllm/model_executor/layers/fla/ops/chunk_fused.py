# SPDX-License-Identifier: Apache-2.0
"""Fused fwd_h + fwd_o kernel for chunk_gated_delta_rule.

The original pipeline materializes the full hidden-state tensor `h` of shape
[B, NT, H, V, K] (~512 MB for B=4 T=4096 H=64 K=V=128, bf16) in HBM in
chunk_gated_delta_rule_fwd_h, then re-reads it in chunk_fwd_o. On MI300X this
exceeds the 256 MB Infinity Cache so the round-trip is the dominant cost.

This kernel keeps the recurrent state `b_h` in registers across the chunk
loop and immediately consumes it for the output computation `o[i_t]`. We
also avoid storing `v_new` because the value-delta `b_v` is consumed in-place
for the `b_A @ b_v` term.

Constraints:
  - Non-varlen path only (cu_seqlens is None). The varlen path keeps the
    unfused implementation.
  - Supports K up to 256 (matches original kernel) by stacking 64-wide
    register tiles. Tested with K=V=128.
  - GVA (Hg < H) supported: Q/K read from i_h // (H // Hg) head, while V
    accumulates per H head, identical to fwd_h / fwd_o.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

from .op import exp
from .utils import FLA_CHUNK_SIZE, use_cuda_graph


def _fused_configs():
    cfgs = [
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [1, 2]
        for BV in [16, 32, 64, 128]
    ]
    # AMD-specific tuning variants: the fused kernel is register-heavy
    # (b_h + b_A + b_o accumulators), so waves_per_eu matters more here.
    for BV in [16, 32, 64]:
        for nonkdim in [16, 32]:
            for waves in [0, 1, 2]:
                for num_warps in [2, 4, 8]:
                    cfgs.append(
                        triton.Config(
                            {
                                "BV": BV,
                                "matrix_instr_nonkdim": nonkdim,
                                "waves_per_eu": waves,
                                "kpack": 2,
                            },
                            num_warps=num_warps,
                            num_stages=1,
                        )
                    )
    return cfgs


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
    }
)
@triton.autotune(
    configs=_fused_configs(),
    key=["H", "K", "V", "BT", "T"],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=["T"])
def chunk_fused_h_o_kernel(
    # state-update inputs (from recompute_w_u)
    k,
    u,                # = "v" param of fwd_h: the recomputed v
    w,
    g,
    # output / final state buffers
    o,
    h0,
    ht,
    # attention input
    q,
    scale,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    # non-varlen only
    bos, eos = i_n * T, i_n * T + T
    NT = tl.cdiv(T, BT)

    # b_h state, stacked 64-wide along K (matches original fwd_h layout).
    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([BV, 64], dtype=tl.float32)

    # Pointer offsets
    # u and v_new use V-head H
    # w uses V-head H
    # k uses Q/K-head Hg = i_h // (H // Hg)
    # q uses Q/K-head Hg
    # o uses V-head H
    i_hg = i_h // (H // Hg)
    u_base = u + ((bos * H + i_h) * V).to(tl.int64)
    w_base = w + ((bos * H + i_h) * K).to(tl.int64)
    k_base = k + ((bos * Hg + i_hg) * K).to(tl.int64)
    q_base = q + ((bos * Hg + i_hg) * K).to(tl.int64)
    o_base = o + ((bos * H + i_h) * V).to(tl.int64)
    if USE_G:
        g_base = g + bos * H + i_h

    stride_v = H * V
    stride_q = Hg * K
    stride_k = Hg * K
    stride_w = H * K

    if USE_INITIAL_STATE:
        h0_p = h0 + i_nh * V * K
    if STORE_FINAL_STATE:
        ht_p = ht + i_nh * V * K

    # Load initial state
    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0_p, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(h0_p, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(h0_p, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(h0_p, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    # Main recurrence: process one chunk per iteration.
    for i_t in range(NT):
        # ---- fwd_h step 1: compute b_v_new = u - w @ b_h^T (pre-gate) ----
        p_w1 = tl.make_block_ptr(
            w_base, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        b_w1 = tl.load(p_w1, boundary_check=(0, 1))
        b_v_delta = tl.dot(b_w1, tl.trans(b_h1).to(b_w1.dtype))
        if K > 64:
            p_w2 = tl.make_block_ptr(
                w_base, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_w2 = tl.load(p_w2, boundary_check=(0, 1))
            b_v_delta += tl.dot(b_w2, tl.trans(b_h2).to(b_w2.dtype))
        if K > 128:
            p_w3 = tl.make_block_ptr(
                w_base, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_w3 = tl.load(p_w3, boundary_check=(0, 1))
            b_v_delta += tl.dot(b_w3, tl.trans(b_h3).to(b_w3.dtype))
        if K > 192:
            p_w4 = tl.make_block_ptr(
                w_base, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_w4 = tl.load(p_w4, boundary_check=(0, 1))
            b_v_delta += tl.dot(b_w4, tl.trans(b_h4).to(b_w4.dtype))

        p_u = tl.make_block_ptr(
            u_base, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_u, boundary_check=(0, 1)) - b_v_delta  # [BT, BV] fp32

        # ---- fwd_o step: compute o[i_t, V-tile] using current b_h ----
        # We also pre-load k-tiles here and keep them in registers: the same
        # data is consumed later by the state-update step (`b_h += k^T @ v_new`).
        # This halves the number of k HBM loads per chunk.
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        b_A = tl.zeros([BT, BT], dtype=tl.float32)

        # K-quadrant 0
        p_q1 = tl.make_block_ptr(
            q_base, (T, K), (stride_q, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        p_k1 = tl.make_block_ptr(
            k_base, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
        )
        b_q1 = tl.load(p_q1, boundary_check=(0, 1))
        b_k1 = tl.load(p_k1, boundary_check=(0, 1))  # [64, BT] — used again below
        # b_o += b_q @ b_h1^T  (b_h1: [BV,64]; trans -> [64,BV]; result [BT,BV])
        b_o += tl.dot(b_q1, tl.trans(b_h1).to(b_q1.dtype))
        b_A += tl.dot(b_q1, b_k1)

        if K > 64:
            p_q2 = tl.make_block_ptr(
                q_base, (T, K), (stride_q, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            p_k2 = tl.make_block_ptr(
                k_base, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_q2 = tl.load(p_q2, boundary_check=(0, 1))
            b_k2 = tl.load(p_k2, boundary_check=(0, 1))
            b_o += tl.dot(b_q2, tl.trans(b_h2).to(b_q2.dtype))
            b_A += tl.dot(b_q2, b_k2)
        if K > 128:
            p_q3 = tl.make_block_ptr(
                q_base, (T, K), (stride_q, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            p_k3 = tl.make_block_ptr(
                k_base, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1)
            )
            b_q3 = tl.load(p_q3, boundary_check=(0, 1))
            b_k3 = tl.load(p_k3, boundary_check=(0, 1))
            b_o += tl.dot(b_q3, tl.trans(b_h3).to(b_q3.dtype))
            b_A += tl.dot(b_q3, b_k3)
        if K > 192:
            p_q4 = tl.make_block_ptr(
                q_base, (T, K), (stride_q, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            p_k4 = tl.make_block_ptr(
                k_base, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1)
            )
            b_q4 = tl.load(p_q4, boundary_check=(0, 1))
            b_k4 = tl.load(p_k4, boundary_check=(0, 1))
            b_o += tl.dot(b_q4, tl.trans(b_h4).to(b_q4.dtype))
            b_A += tl.dot(b_q4, b_k4)

        # Apply per-token gate to o and A (matches fwd_o)
        if USE_G:
            p_g = tl.make_block_ptr(g_base, (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_o = b_o * exp(b_g)[:, None]
            b_A = b_A * exp(b_g[:, None] - b_g[None, :])

        # Causal mask on A
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
        b_A = tl.where(m_A, b_A, 0)

        # Use the just-computed b_v (pre-gate) as v_new for the A @ v term.
        # Cast to k's dtype to match fwd_o behavior.
        b_o = (b_o + tl.dot(b_A.to(k.dtype.element_ty), b_v.to(k.dtype.element_ty))) * scale

        p_o = tl.make_block_ptr(
            o_base, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        # ---- fwd_h state update: apply gate, then b_h += k^T @ v_new ----
        last_idx = min((i_t.to(tl.int64) + 1) * BT, T) - 1
        if USE_G:
            m_t2 = (i_t.to(tl.int64) * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            b_v = b_v * tl.where(m_t2, exp(b_g_last - b_g), 0)[:, None]
            b_g_last_e = exp(b_g_last)
            b_h1 *= b_g_last_e
            if K > 64:
                b_h2 *= b_g_last_e
            if K > 128:
                b_h3 *= b_g_last_e
            if K > 192:
                b_h4 *= b_g_last_e

        b_v = b_v.to(k.dtype.element_ty)

        # Reuse the k-tiles already in registers from the fwd_o step above —
        # no additional HBM loads.
        b_h1 += tl.trans(tl.dot(b_k1, b_v))
        if K > 64:
            b_h2 += tl.trans(tl.dot(b_k2, b_v))
        if K > 128:
            b_h3 += tl.trans(tl.dot(b_k3, b_v))
        if K > 192:
            b_h4 += tl.trans(tl.dot(b_k4, b_v))

    if STORE_FINAL_STATE:
        p_ht1 = tl.make_block_ptr(ht_p, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        tl.store(p_ht1, b_h1.to(p_ht1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht2 = tl.make_block_ptr(ht_p, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            tl.store(p_ht2, b_h2.to(p_ht2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht3 = tl.make_block_ptr(ht_p, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            tl.store(p_ht3, b_h3.to(p_ht3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht4 = tl.make_block_ptr(ht_p, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            tl.store(p_ht4, b_h4.to(p_ht4.dtype.element_ty), boundary_check=(0, 1))


def chunk_fused_fwd_h_o(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    q: torch.Tensor,
    g: torch.Tensor | None,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    chunk_size: int = FLA_CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fused fwd_h + fwd_o for non-varlen path. Returns (o, final_state)."""
    B, T, Hg, K = k.shape
    V = u.shape[-1]
    H = u.shape[-2]
    BT = chunk_size
    N = B
    assert K <= 256, "fused kernel only supports head dim K <= 256."

    o = torch.empty(B, T, H, V, device=q.device, dtype=q.dtype)
    final_state = (
        k.new_empty(N, H, V, K, dtype=torch.float32) if output_final_state else None
    )

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    chunk_fused_h_o_kernel[grid](
        k=k,
        u=u,
        w=w,
        g=g,
        o=o,
        h0=initial_state,
        ht=final_state,
        q=q,
        scale=scale,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
    )
    return o, final_state
