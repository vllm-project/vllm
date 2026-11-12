# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
"""Adaptive key pruning's planner: a score upper bound per row, and the estimate that decides whether a call prunes.

A key can pass the gate only if its score does, so a cheap upper bound ``U >= s`` lets the selection skip every key
whose bound already fails the gate without changing a selected bit: the gate test ``fma(-inv, x, c0) < th + 1`` is
non-increasing in ``x``, so a bound that fails it means a score that fails it. The kept keys are scored in full by the
scan's own arithmetic (the refine kernels), so the candidates, and everything after them, are the scan's.

The bound, after DoSA (arXiv 2610.13990, sections 2.2-2.5), for a row with heads ``q_h`` and weights ``w_h`` and a key
``k`` with kv scale ``s_k``::

    U = s_k * (sum_{j in R} wt_j relu(q_j . k) - relu(a . k) + E ||k||)

* ``R`` holds ``r = 14`` retained heads: the strongest positive ones by ``w_h ||q_h||``, or every positive head and
  then the negative ones by ``|w_h| ||q_h|| min(p_h, 1 - p_h)`` when there are at most ``r`` positive heads (``p_h``:
  the head's activation rate on a key sample).
* An omitted positive head is projected onto the retained head it is most aligned with, ``q_h = alpha_h q_j + u_h``,
  ``alpha_h >= 0``. Then ``relu(q_h . k) <= alpha_h relu(q_j . k) + relu(u_h . k)``: ``w_h alpha_h`` folds into
  ``wt_j``, and the residuals are bounded together by ``sqrt(sum_{h,g} max(0, w_h u_h . w_g u_g)) ||k||`` (or ``sum_h
  w_h ||u_h|| ||k||`` when smaller).
* Omitted negative heads only subtract. Those active on at least half of the sample are aggregated, ``a = sum |w_h|
  q_h``; by subadditivity ``relu(a . k)`` is at most what they subtract. The aggregate goes to the kernels as two rows
  of the operand type, a high part and its rounding error; what is left of that error joins ``E``.
* ``E`` also carries the FP32 rounding of ``wt`` and ``a`` and ``SCREEN_ROUNDING`` of both sides' weight mass: the
  kernels' dot products and the scan's are both within ``(gamma_{H+1} + gamma_128 + 2u) s ||k|| M`` of exact.

Adaptive pruning estimates, before the scan, how much of the KV the refine would score: the refine scores a key for
all ``EST_GROUP`` rows of a row group once any of them keeps it, so the estimate is the share of the in-window sampled
keys whose bound passes the seed gate of some row of their group, pooled over the sampled groups (one in
``EST_GROUPS``, at most ``EST_MAX_GROUPS``). Trained rows keep nearly the same keys as their neighbors, untrained
ones do not, so one row's share says little about its group's: a group's union can be 2-3 times its rows' shares.
The call prunes when the estimate is at or below ``adaptive_threshold``, where pruning's predicted cost
(``PRUNE_COSTS``) meets the scan's, otherwise it scans every key. The GPU decides: the estimate never reaches the
host, both paths launch and the one not taken sees empty windows.
"""

from __future__ import annotations

import functools
import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from .kernels import launch
from .kernels.prologue import EST_GROUP, EST_WORDS

# The pruning kernels' columns per row: the retained heads and the negative aggregate's two rows.
PRUNE_COLUMNS = 16

# Both sides' dot products are within (gamma_{H+1} + gamma_128 + 2u) of exact per unit of absolute weight mass times
# s ||k||; 2^-14 is about five times that at 64 heads.
SCREEN_ROUNDING = 2.0**-14

# FP32 planning (the kernels below) differs from the exact plan by at most ~gamma_33 of the absolute weight mass
# in the folded weights, the residuals and the aggregate; 2^-17 of that mass covers it four times over.
_PLAN_ROUNDING = 2.0**-17
_PLAN_SAMPLES = 256
# One row group (EST_GROUP rows: the refine's, a key is scored for the whole group or not at all) in this many makes
# the estimate, which so plans one row in EST_GROUPS, and at most EST_MAX_GROUPS groups spread over the call (their
# planning is the estimate's cost). An estimate row leaves two 256-bit masks over the sample keys
# (EST_WORDS words): in its window, and past its seed gate.
EST_GROUPS = 16
EST_MAX_GROUPS = 128
assert EST_WORDS == 2 * (_PLAN_SAMPLES // 32)


# estimate and gated are runtime values with no specialization, so the estimate's rows (estimate set), the rest of an
# adaptive plan (gated) and a whole plan run one compiled kernel: on SM90, Triton's code for a constant step and for a
# constant gate planned FP8 aggregate rows and allowances differently (survival estimates up to 5x off, and twice the
# refine work).
@triton.jit
def _operand_row(row, col, LAYOUT: tl.constexpr):
    """Where row `row`, column `col` of the pruning operand goes: SM100 (LAYOUT 1) takes row pairs column by column,
    [pair, column, row of pair]; SM90 (LAYOUT 2) 16-row tiles permuted by prune_sm90.n_of (row 4b + a, column
    2d + e of a tile is operand row 64a + 8d + 2b + e)."""
    if LAYOUT == 1:
        return (row // 2) * 32 + col * 2 + row % 2
    else:
        r = row % 16
        return (row // 16) * 256 + 64 * (r % 4) + 8 * (col // 2) + 2 * (r // 4) + col % 2


@triton.jit
def _estimate_row(row, G: tl.constexpr, stride):
    """Whether `row` is in one of the estimate's row groups (the first of every `stride` groups of G rows)."""
    return (row // G) % stride == 0


@triton.jit(do_not_specialize=["estimate", "gated", "stride"])
def _plan_kernel(
    Q,  # [num_q, H, 128] fp8 or bf16
    W,  # [num_q, H] fp32
    KS,  # [S, 128] fp8 or bf16 sample keys
    KSCALE,  # [S] fp32 their kv scales (ones for bf16)
    KNSC,  # [S] fp32 their scale * ||k|| (_key_norm_kernel)
    COLS,  # [S] int32 their positions
    STARTS,
    ENDS,
    ORIGIN,  # [num_q] the seed gate's bucket map, as the scan reads it
    INV,
    TH,
    QS,  # [padded * (R + 2), 128] fp8 or bf16 out: the pruning kernels' operand (_operand_row)
    WS,  # [num_q, R + 2] fp32 out
    ALLOW,  # [num_q] fp32 out
    EST,  # [estimate rows, EST_WORDS] int32 out: per estimate row, its in-window and past-the-gate sample masks
    FLAG,  # [1] int32, gated only: nonzero when the call prunes
    num_q,
    padded,  # rows of the operand, num_q rounded up to its row group; the rows past num_q are zeroed
    rounding,
    estimate,  # nonzero: the estimate's rows only (program p is row p % G of estimate group p // G); else every row
    gated,  # nonzero: the rest of an adaptive plan, the rows the estimate pass left, when FLAG is set
    stride,  # the estimate's groups: one in this many (estimate_stride)
    H: tl.constexpr,
    R: tl.constexpr,
    S: tl.constexpr,
    G: tl.constexpr,
    BF16: tl.constexpr,  # bf16 operands: the aggregate as two bf16 rows (scale 1)
    LAYOUT: tl.constexpr,  # the operand's layout (_operand_row)
):
    pid = tl.program_id(0).to(tl.int64)
    row = pid
    if estimate != 0:
        row = (pid // G) * (G * stride) + pid % G
    est_row = _estimate_row(row, G, stride)
    d = tl.arange(0, 128)
    if gated != 0:
        if (tl.load(FLAG) == 0) | est_row:
            return
    if (estimate != 0) & (row >= num_q):
        # A group's rows past the call: no samples, so they leave the group's union alone.
        pad_slot = (row // (G * stride)) * G + row % G
        tl.store(EST + pad_slot * (S // 16) + tl.arange(0, S // 16), tl.zeros([S // 16], dtype=tl.int32))
    if row >= num_q:
        if row < padded:
            c = tl.arange(0, R + 2)
            tl.store(QS + _operand_row(row, c, LAYOUT)[:, None] * 128 + d[None, :],
                     tl.zeros([R + 2, 128], dtype=QS.dtype.element_ty))
        return
    h = tl.arange(0, H)
    q8 = tl.load(Q + row * H * 128 + h[:, None] * 128 + d[None, :])
    q = q8.to(tl.float32)
    w = tl.load(W + row * H + h)
    qn = tl.sqrt(tl.sum(q * q, axis=1))

    # Activation rate of each head on the in-window samples.
    s = tl.arange(0, S)
    k8 = tl.load(KS + s[:, None] * 128 + d[None, :])
    cols = tl.load(COLS + s)
    ks = tl.load(STARTS + row)
    ke = tl.load(ENDS + row)
    inwin = (cols >= ks) & (cols < ke)
    act = tl.dot(q8, tl.trans(k8)) > 0
    n_in = tl.sum(inwin.to(tl.float32), axis=0)
    p = tl.sum((act & inwin[None, :]).to(tl.float32), axis=1) / tl.maximum(n_in, 1.0)
    p = tl.where(n_in > 0, p, 0.5)

    # Retained heads: every positive head first (by w |q|), then negative ones by |w| |q| min(p, 1 - p);
    # a head's slot is its rank.
    pos = w > 0
    neg = w < 0
    strength = tl.where(pos, w * qn, -w * qn * tl.minimum(p, 1.0 - p))
    tier = tl.where(pos, 2.0, 0.0) + strength / (tl.max(strength, axis=0) + 1e-30)
    tier = tl.where(pos | neg, tier, -1.0)
    hh = h[None, :]
    ahead = (tier[None, :] > tier[:, None]) | ((tier[None, :] == tier[:, None]) & (hh < h[:, None]))
    rank = tl.sum(ahead.to(tl.int32), axis=1)
    kept = rank < R
    omit_pos = pos & ~kept
    omit_neg = neg & ~kept

    # Each omitted positive head projected onto its best retained anchor; rows that retain every positive head skip
    # this. The bound holds for any alpha >= 0 whose residual v it charges, so the Gram matrix that picks anchors
    # and alphas comes from BF16 tensor-core products (q's rows are exact in BF16), and the anchors' rows are
    # loaded rather than multiplied out. The residuals' Gram matrix, which e_joint must not underestimate, comes
    # from a two-way BF16 split v = v0 + v1 + r (|r| <= 2^-16 |v|) as v0 v0' + v0 v1' + v1 v0': the products are
    # exact, the terms left out are at most 3 * 2^-16 |v_i| |v_j| and a truncating FP32 accumulator adds at most
    # 2^-16 of the same, so the sum over pairs is within 2^-14 e_sep^2 -- a quarter of the 2^-12 e_sep^2 added under
    # the root. (FP32 dot products here ran 4x slower than the rest of the plan on SM90.)
    wt = tl.where(kept, w, 0.0)
    e = tl.sum(tl.zeros([H], dtype=tl.float32), axis=0)
    if tl.sum(omit_pos.to(tl.int32), axis=0) > 0:
        qb = q8.to(tl.bfloat16)
        gram = tl.dot(qb, tl.trans(qb))  # [H, H]
        gain = tl.where(kept[None, :], tl.maximum(gram, 0.0) * tl.maximum(gram, 0.0) / tl.maximum(qn * qn, 1e-30)[None, :],
                        -1.0)
        anchor = tl.argmax(gain, axis=1)
        onehot = (hh == anchor[:, None]).to(tl.float32)  # [h, anchor]
        g_anchor = tl.sum(gram * onehot, axis=1)
        n_anchor = tl.sum((qn * qn)[None, :] * onehot, axis=1)
        alpha = tl.where(omit_pos, tl.maximum(g_anchor, 0.0) / tl.maximum(n_anchor, 1e-30), 0.0)
        wt += tl.sum(onehot * (w * alpha)[:, None], axis=0)
        q_anchor = tl.load(Q + row * H * 128 + anchor[:, None] * 128 + d[None, :]).to(tl.float32)
        v = tl.where(omit_pos[:, None], w[:, None] * (q - alpha[:, None] * q_anchor), 0.0)
        vn = tl.sqrt(tl.sum(v * v, axis=1))
        e_sep = tl.sum(vn, axis=0)
        v0 = v.to(tl.bfloat16)
        v1 = (v - v0.to(tl.float32)).to(tl.bfloat16)
        gv = tl.dot(v0, tl.trans(v0)) + (tl.dot(v0, tl.trans(v1)) + tl.dot(v1, tl.trans(v0)))
        e_joint = tl.sqrt(tl.sum(tl.sum(tl.maximum(gv, 0.0), axis=1), axis=0) + 2.0**-12 * e_sep * e_sep)
        e = tl.minimum(e_sep, e_joint)

    # The negative aggregate as two rows of the operand type: the high part and its rounding error, over
    # power-of-two scales for FP8 (scale 1 for BF16, whose range is FP32's).
    agg = tl.where(omit_neg & (p >= 0.5), -w, 0.0)
    a = tl.sum(agg[:, None] * q, axis=0)
    if BF16:
        scale_hi = tl.sum(tl.zeros([H], dtype=tl.float32), axis=0) + 1.0
        scale_lo = scale_hi
        a8_hi = a.to(tl.bfloat16)
        hi = a8_hi.to(tl.float32)
        a8_lo = (a - hi).to(tl.bfloat16)
        lo = a8_lo.to(tl.float32)
    else:
        scale_hi = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(a), axis=0), 1e-30) / 448.0)))
        a8_hi = (a / scale_hi).to(tl.float8e4nv)
        hi = a8_hi.to(tl.float32) * scale_hi
        scale_lo = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(a - hi), axis=0), 1e-30) / 448.0)))
        a8_lo = ((a - hi) / scale_lo).to(tl.float8e4nv)
        lo = a8_lo.to(tl.float32) * scale_lo
    quant = tl.sqrt(tl.sum((a - hi - lo) * (a - hi - lo), axis=0))
    an = tl.sqrt(tl.sum(hi * hi, axis=0)) + tl.sqrt(tl.sum(lo * lo, axis=0))

    mass = tl.sum(tl.abs(w) * qn, axis=0) + tl.sum(tl.abs(wt) * qn, axis=0) + an + e
    allow = (e + quant) * (1.0 + 2.0**-20) + rounding * mass
    allow = tl.where(allow < float("inf"), allow, float("inf"))

    # Which samples pruning would keep for this row: the bound on the samples against the seed gate (the kernel's
    # loose_raw_gate). An estimate only, for adaptive pruning, so the estimate groups' rows make it; the sample dot
    # products are formed again here rather than kept live through the plan.
    if est_row:
        k8 = tl.load(KS + s[:, None] * 128 + d[None, :])
        dots = tl.dot(q8, tl.trans(k8))  # [H, S]
        reduced = tl.sum(wt[:, None] * tl.maximum(dots, 0.0), axis=0) - tl.maximum(
            tl.sum(agg[:, None] * dots, axis=0), 0.0)
        bound = tl.load(KSCALE + s) * reduced + allow * tl.load(KNSC + s)
        inv = tl.load(INV + row)
        c0 = -tl.load(ORIGIN + row) * inv
        edge = tl.load(TH + row).to(tl.float32) + 1.0
        slack = 2.0**-16 * tl.maximum(tl.maximum(tl.abs(c0), tl.abs(edge)), 1.0)
        gate = tl.where((inv > 0) & (inv <= 3.4e38) & (tl.abs(c0) <= 3.4e38), (c0 - edge - slack) / inv,
                        -float("inf"))
        est_slot = (row // (G * stride)) * G + row % G
        bit = (tl.arange(0, 32).to(tl.int64))[None, :]
        win_words = tl.sum(tl.reshape(inwin.to(tl.int64), [S // 32, 32]) << bit, axis=1)
        pass_words = tl.sum(tl.reshape((inwin & (bound > gate)).to(tl.int64), [S // 32, 32]) << bit, axis=1)
        w8 = tl.arange(0, S // 32)
        tl.store(EST + est_slot * (S // 16) + w8, win_words.to(tl.uint32).to(tl.int32, bitcast=True))
        tl.store(EST + est_slot * (S // 16) + S // 32 + w8, pass_words.to(tl.uint32).to(tl.int32, bitcast=True))

    slot = tl.where(kept, rank, R)
    tl.store(QS + _operand_row(row, slot, LAYOUT)[:, None] * 128 + d[None, :], q8, mask=kept[:, None])
    tl.store(QS + _operand_row(row, R, LAYOUT) * 128 + d, a8_hi)
    tl.store(QS + _operand_row(row, R + 1, LAYOUT) * 128 + d, a8_lo)
    tl.store(WS + row * (R + 2) + slot, wt, mask=kept)
    tl.store(WS + row * (R + 2) + R, scale_hi)
    tl.store(WS + row * (R + 2) + R + 1, scale_lo)
    tl.store(ALLOW + row, allow)


@triton.jit
def _key_norm_kernel(KV, SCALES, OUT, n, n_out, FLAG, BLOCK: tl.constexpr, GATED: tl.constexpr,
                     HAS_SCALES: tl.constexpr):
    """scale * ||k|| per key (||k|| without HAS_SCALES: BF16 keys), rounded up, and zeros to n_out, BLOCK keys at a
    time over the grid; GATED: only when FLAG is set."""
    if GATED:
        if tl.load(FLAG) == 0:
            return
    d = tl.arange(0, 128)
    for start in range(tl.program_id(0) * BLOCK, n_out, tl.num_programs(0) * BLOCK):
        i = start + tl.arange(0, BLOCK)
        valid = i < n
        k = tl.load(KV + i[:, None].to(tl.int64) * 128 + d[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        norm = tl.sqrt(tl.sum(k * k, axis=1))
        if HAS_SCALES:
            norm = norm * tl.load(SCALES + i, mask=valid, other=0.0)
        tl.store(OUT + i, tl.where(valid, norm * (1.0 + 2.0**-20), 0.0), mask=i < n_out)


def key_norms(plan, kv: torch.Tensor, flag: Optional[torch.Tensor] = None) -> None:
    """plan.knsc = scale * ||k|| of every key (only when flag is set, given one); a grid of a few programs an SM, so
    a gated launch the call does not need costs little."""
    n_out = plan.knsc.numel()
    grid = min(triton.cdiv(n_out, 64), 8 * launch.sm_count(kv.device.index))
    _key_norm_kernel[(grid,)](kv, plan.scales_or(kv), plan.knsc, kv.shape[0], n_out,
                              plan.knsc if flag is None else flag, BLOCK=64, GATED=flag is not None,
                              HAS_SCALES=plan.kv_scales is not None)


@functools.lru_cache(maxsize=64)
def _plan_sample_cols(nk: int, device_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """The plan's sample keys, int32 and int64: (2j + 1) nk // (2 S) for j < S."""
    cols = (2 * torch.arange(_PLAN_SAMPLES, device=torch.device("cuda", device_index)) + 1) * nk // (2 * _PLAN_SAMPLES)
    return cols.to(torch.int32), cols


class _Plan:
    """Buffers and sample keys of one call's pruning plan."""

    def __init__(self, q, weights, kv, kv_scales, ks, ke, origin, inv_delta, th_bucket):
        num_q, heads, dim = q.shape
        nk = kv.shape[0]
        dev = q.device
        self.layout = 2 if torch.cuda.get_device_capability(dev)[0] == 9 else 1
        group = 16 if self.layout == 2 else 2
        self.padded = triton.cdiv(num_q, group) * group
        self.q_prune = torch.empty(self.padded * PRUNE_COLUMNS, dim, dtype=q.dtype, device=dev)
        self.w_prune = torch.empty(num_q, PRUNE_COLUMNS, dtype=torch.float32, device=dev)
        self.allowance = torch.empty(num_q, dtype=torch.float32, device=dev)
        self.est_rows = estimate_rows(num_q)
        self.est = torch.empty(self.est_rows, EST_WORDS, dtype=torch.int32, device=dev)
        if th_bucket is None:
            origin = torch.zeros(num_q, dtype=torch.float32, device=dev)
            inv_delta = torch.zeros(num_q, dtype=torch.float32, device=dev)
            th_bucket = torch.zeros(num_q, dtype=torch.int32, device=dev)
        # Padded to 4 floats, as the TMA of every per-key vector reads.
        self.knsc = torch.empty((nk + 3) // 4 * 4, dtype=torch.float32, device=dev)
        self.cols, picked = _plan_sample_cols(nk, dev.index)
        self.k_samples = kv[picked].contiguous()
        self.bf16 = kv_scales is None
        if self.bf16:
            self.s_samples = torch.ones(_PLAN_SAMPLES, dtype=torch.float32, device=dev)
        else:
            self.s_samples = kv_scales.reshape(-1)[picked].float().contiguous()
        # The samples' scale * ||k||, by the kernel and arithmetic of the full table (knsc[picked]). The samples are
        # gathered once here: a plan kernel reading them from their scattered rows took a fifth longer on SM90.
        self.knsc_samples = torch.empty(_PLAN_SAMPLES, dtype=torch.float32, device=dev)
        _key_norm_kernel[(triton.cdiv(_PLAN_SAMPLES, 64),)](
            self.k_samples, self.s_samples, self.knsc_samples, _PLAN_SAMPLES, _PLAN_SAMPLES, self.knsc_samples,
            BLOCK=64, GATED=False, HAS_SCALES=not self.bf16,
        )
        self.kv_scales = kv_scales
        self.args = (q, weights, ks, ke, origin, inv_delta, th_bucket)

    def rows(self, num_q: int, estimate: bool, flag: Optional[torch.Tensor]) -> None:
        """Plan the estimate groups' rows (estimate), else every row; with ``flag``, the rows the estimate pass left,
        if it is set."""
        q, weights, ks, ke, origin, inv_delta, th_bucket = self.args
        _plan_kernel[(self.est_rows if estimate else self.padded,)](
            q, weights, self.k_samples, self.s_samples, self.knsc_samples, self.cols, ks, ke, origin, inv_delta,
            th_bucket, self.q_prune, self.w_prune, self.allowance, self.est, ks if flag is None else flag,
            num_q, self.padded, SCREEN_ROUNDING + _PLAN_ROUNDING, int(estimate), int(flag is not None),
            estimate_stride(num_q), H=q.shape[1], R=PRUNE_COLUMNS - 2, S=_PLAN_SAMPLES, G=EST_GROUP, BF16=self.bf16,
            LAYOUT=self.layout, num_warps=4,
        )

    def scales_or(self, kv: torch.Tensor) -> torch.Tensor:
        """The kv scales for the key-norm kernel; any float tensor for BF16 keys (HAS_SCALES off, never read)."""
        return self.knsc if self.kv_scales is None else self.kv_scales

    def operands(self):
        return self.q_prune, self.w_prune, self.allowance, self.knsc, self.est


def estimate_stride(num_q: int) -> int:
    """The estimate takes the first of every this many row groups: EST_GROUPS, or more past EST_MAX_GROUPS groups."""
    return max(EST_GROUPS, triton.cdiv(triton.cdiv(num_q, EST_GROUP), EST_MAX_GROUPS))


def estimate_rows(num_q: int) -> int:
    """Rows the estimate plans: EST_GROUP for each of its groups (the first of every estimate_stride)."""
    return triton.cdiv(triton.cdiv(num_q, EST_GROUP), estimate_stride(num_q)) * EST_GROUP


def estimate_shares(est: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(union, row)`` from the estimate's masks, both 0-dim FP32 tensors: over the estimate groups' in-window
    samples, the share that passes some row's gate of the group (what the decision uses), and over the estimate
    rows' in-window samples, the share that passes the row's own gate. Without in-window samples, 1."""
    words = est.view(-1, EST_GROUP, EST_WORDS)
    bits = (words.unsqueeze(-1) >> torch.arange(32, device=est.device, dtype=torch.int32)) & 1
    win, passed = bits[:, :, : EST_WORDS // 2], bits[:, :, EST_WORDS // 2:]

    def share(w, p):
        n = w.sum().float()
        return torch.where(n > 0, p.sum().float() / n.clamp(min=1), torch.ones_like(n))

    return share(win.amax(dim=1), passed.amax(dim=1)), share(win, passed)



# Adaptive pruning: from ADAPTIVE_MIN_KEYS keys every call plans the estimate's rows, and the GPU picks pruning or the
# scan from that call's estimate. Below it the scan is ahead whatever the estimate.
ADAPTIVE_MIN_KEYS = 128 * 1024

# Pruning's kernels over the scan's, per call: p * 128K / L (key norms and the plan, per row as the scan is per key)
# + c (the mask, a fixed share, and the refine's fixed part) + b * U (the refine of the kept share U of the row
# groups' windows: estimate_shares' union). Least squares over each stage's measured times on trained and
# initialized indexers and random weights and keys, per device and operand type; a device not listed takes the entry
# of its architecture's ARCH_DEVICE.
PRUNE_COSTS = {
    ("B200", "fp8"): (0.0823, 0.6094, 1.7193),
    ("B200", "bf16"): (0.2130, 0.6015, 1.7920),
    ("H800", "fp8"): (0.0262, 0.5296, 1.4184),
    ("H800", "bf16"): (0.0095, 0.5334, 1.3627),
    ("H20", "fp8"): (0.0272, 0.5098, 1.0018),
    ("H20", "bf16"): (0.0094, 0.5130, 1.0029),
    ("H100", "fp8"): (0.1293, 0.5078, 1.3995),
    ("H100", "bf16"): (0.1089, 0.5054, 1.3802),
    ("H200", "fp8"): (0.1260, 0.5078, 1.3854),
    ("H200", "bf16"): (0.1000, 0.5062, 1.3826),
}
ARCH_DEVICE = {10: "B200", 9: "H800"}


def adaptive_eligible(q: torch.Tensor, seq_len_kv: int) -> bool:
    """Whether the GPU chooses per call (from ADAPTIVE_MIN_KEYS keys); shorter calls scan."""
    return seq_len_kv >= ADAPTIVE_MIN_KEYS


@functools.lru_cache(maxsize=None)
def prune_costs(device_index: int, bf16: bool) -> Tuple[float, float, float]:
    """(p, c, b) of PRUNE_COSTS for this GPU; FLASHTOPK_PRUNE_COSTS="p,c,b" overrides."""
    env = os.environ.get("FLASHTOPK_PRUNE_COSTS")
    if env:
        p, c, b = (float(x) for x in env.split(","))
        return p, c, b
    words = set(torch.cuda.get_device_name(device_index).split())
    kind = "bf16" if bf16 else "fp8"
    for (device, dtype), costs in PRUNE_COSTS.items():
        if device in words and dtype == kind:
            return costs
    return PRUNE_COSTS[(ARCH_DEVICE.get(torch.cuda.get_device_capability(device_index)[0], "H800"), kind)]


@functools.lru_cache(maxsize=4096)
def adaptive_threshold(device_index: int, bf16: bool, seq_len_kv: int) -> float:
    """The group estimate at or below which a call prunes: where pruning's predicted cost meets the scan's,
    p * 128K / L + c + b * U = 1."""
    p, c, b = prune_costs(device_index, bf16)
    return min(max((1.0 - p * 131072 / seq_len_kv - c) / b, 0.0), 1.0)


def adaptive_estimate(q, weights, kv, kv_scales, ks, ke, origin, inv_delta, th_bucket, flag, scan_ks, prune_ks,
                      estimate=None):
    """The plan of the estimate groups' rows, then the GPU's choice: ``flag`` (int32[1]) set when this call's group
    estimate is at or below ``adaptive_threshold``, ``scan_ks`` / ``prune_ks`` the windows' starts for the scan and
    for the pruning kernels (the row's own on the path chosen, past its end on the other); ``estimate`` (FP32[1]),
    when given, receives the estimate. Returns the plan, which ``adaptive_finish`` completes."""
    num_q = q.shape[0]
    plan = _Plan(q, weights, kv, kv_scales, ks, ke, origin, inv_delta, th_bucket)
    plan.rows(num_q, True, None)
    threshold = adaptive_threshold(q.device.index, kv_scales is None, kv.shape[0])
    launch.decide(plan.est, ks, ke, scan_ks, prune_ks, flag, threshold, estimate)
    return plan


def adaptive_finish(plan, kv, flag):
    """The pruning kernels' operands: the key norms and the rows the estimate left, planned only when the call
    prunes (flag set). Enqueued after the scan, whose launch then waits for none of it."""
    key_norms(plan, kv, flag)
    plan.rows(plan.args[0].shape[0], False, flag)
    return plan.operands()


@torch.no_grad()
def plan_operands(q, weights, kv, kv_scales, ks, ke, origin=None, inv_delta=None, th_bucket=None):
    """``(q_prune, w_prune, allowance, knsc, est)`` for every row: the always-prune plan.

    ``q_prune`` (FP8 or BF16): per row the retained heads' rows of ``q``, then the aggregate's high and low rows, laid
    out as the pruning kernel reads them (``_operand_row``); ``w_prune`` ``[num_q, 16]`` FP32: the folded weights,
    then the two rows' scales. The bound is ``s_k * (sum_j w_j relu(q_j . k) - relu(w_14 q_14 . k + w_15 q_15 . k)
    + allowance ||k||)``. ``knsc`` ``[seq_len_kv]``: ``scale * ||k||`` rounded up; ``est``: the estimate rows'
    sample masks (``estimate_shares``).
    """
    num_q = q.shape[0]
    plan = _Plan(q, weights, kv, kv_scales, ks, ke, origin, inv_delta, th_bucket)
    key_norms(plan, kv)
    plan.rows(num_q, False, None)
    return plan.operands()
