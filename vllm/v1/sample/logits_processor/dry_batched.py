# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batched torch implementation of the DRY penalty.

Computes the same per-position match lengths as the sequential
Z-algorithm in ``dry.py``, but vectorized across every DRY request in the
batch. The comparisons that determine the match length ending at
position ``i`` all lie along one fixed diagonal: for offset ``k = n-1-i``
they compare ``window[n-1-j-k] == window[n-1-j]`` for ``j = 0, 1, 2, ...``
until the first mismatch. Evaluating those comparisons as an ``[R, K, J]``
tensor and taking a cumprod-sum along ``j`` yields every match length
with no sequential scan.

Capping ``J`` at ``allowed_length + max_exponent`` is exact: the exponent
clamp maps every longer match to the same penalty. Requests whose cap is
unusable (``max_exponent == 0``, i.e. ``base <= 1.000001``, or a cap
beyond ``_J_BUDGET``) are routed to the sequential reference
implementation instead.

``dry_core`` is the tensor-level entry point, shared by both sampler
stacks: the V1-runner logits processor wraps it with list-of-ints window
building (``apply_dry_batched``); the V2-runner module gathers windows
directly from its GPU-resident token history.
"""

from collections.abc import Sequence
from typing import Protocol

import torch


class _DryParams(Protocol):
    """Per-request DRY parameters (structurally, dry._DryState)."""

    multiplier: float
    base: float
    allowed_length: int
    max_exponent: int
    breakers: frozenset[int]
    _breaker_mask: "torch.Tensor | None"


# Requests needing a longer match cap than this are handled by the
# sequential reference path; beyond it the [R, K, J] tensors stop being
# cheap. base=1.1 needs 930; base=1.05 needs 1818.
_J_BUDGET = 2048

# Byte budget for the per-chunk transients of the match scan. The gather
# W32[:, idx1] materializes [R, chunk, J] int32 (4 B/elem) before the
# comparison reduces it to bool; with the masks, the int8 cumprod and
# the int16 row sums, the marginal transient cost is ~8 B/elem. 24 keeps
# headroom for the fixed R*vocab int64 l_max floor and allocator slack
# (and, measured, slightly better wall time than tighter chunking).
_CHUNK_BYTE_BUDGET = 256 * 1024 * 1024
_CHUNK_PEAK_BYTES_PER_ELEM = 24


def eligible(state: _DryParams) -> bool:
    """Whether the batched path computes this request exactly."""
    return (
        state.max_exponent > 0
        and state.allowed_length + state.max_exponent <= _J_BUDGET
    )


def _breaker_vocab_mask(
    state: _DryParams, vocab_size: int, device: torch.device
) -> torch.Tensor:
    """Cached [vocab] bool mask of this request's breaker token ids."""
    cached = state._breaker_mask
    if cached is not None and cached.shape[0] == vocab_size:
        return cached
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    if state.breakers:
        ids = torch.tensor(sorted(state.breakers), dtype=torch.int64, device=device)
        mask[ids[ids < vocab_size]] = True
    state._breaker_mask = mask
    return mask


def dry_core(
    logits: torch.Tensor,
    row_idx: torch.Tensor,
    W: torch.Tensor,
    n_r: torch.Tensor,
    allowed: torch.Tensor,
    max_exp: torch.Tensor,
    mult: torch.Tensor,
    base: torch.Tensor,
    breaker_masks: list[torch.Tensor | None],
) -> torch.Tensor:
    """Apply DRY penalties in place given batched window tensors.

    Args:
      logits: [B, vocab] float tensor, modified in place.
      row_idx: [R] int64, row of ``logits`` for each DRY request.
      W: [R, N] int64 windows, RIGHT-aligned (window tokens occupy the
        trailing ``n_r`` columns; leading padding is ignored via masks).
      n_r: [R] int64 window lengths.
      allowed / max_exp: [R] int64 per-request parameters.
      mult / base: [R] float32 per-request parameters (already rounded
        through float32, as llama.cpp stores them).
      breaker_masks: per-request [vocab] bool masks (or None for no
        breakers).
    """
    device = logits.device
    vocab = logits.shape[-1]
    R, N = W.shape

    # Per-request rep_limit: distance from the end of the nearest breaker
    # (llama.cpp step 1). A breaker at column c is j = N-1-c tokens from
    # the end.
    rep_limit = n_r.clone()
    any_breakers = any(bm is not None and bool(bm.any()) for bm in breaker_masks)
    if any_breakers:
        Bwin = torch.stack(
            [
                bm[W[r].clamp(min=0)]
                if bm is not None
                else torch.zeros(N, dtype=torch.bool, device=device)
                for r, bm in enumerate(breaker_masks)
            ]
        )
        valid_cols = torch.arange(N, device=device)[None, :] >= (N - n_r)[:, None]
        Bwin &= valid_cols
        has_breaker = Bwin.any(dim=1)
        # max breaker column -> nearest to the end.
        max_col = torch.where(
            has_breaker,
            (Bwin * torch.arange(1, N + 1, device=device)[None, :]).max(dim=1).values
            - 1,
            torch.zeros_like(n_r),
        )
        rep_limit = torch.where(has_breaker, (N - 1) - max_col, n_r)

    # llama.cpp: if rep_limit < allowed_length, the request produces
    # nothing this step.
    active = rep_limit >= allowed

    # Match lengths per offset k, chunked to bound memory. Token ids fit
    # int32 (as does the -1 padding), and gathering int32 instead of int64
    # halves the dominant per-chunk transient.
    W32 = W.to(torch.int32)
    J = int(min(int((allowed + max_exp).max()), N))
    J = max(J, 1)
    idx2 = torch.arange(N - 1, N - 1 - J, -1, device=device)  # [J]
    suffix = W32.gather(1, idx2.expand(R, J))  # [R, J]
    K = N - 1  # offsets 1..N-1
    chunk = max(1, _CHUNK_BYTE_BUDGET // (_CHUNK_PEAK_BYTES_PER_ELEM * max(1, R * J)))

    # Scatter target: per-(row, token) longest charged match.
    l_max = torch.full((R * vocab,), -1, dtype=torch.int64, device=device)

    for k0 in range(1, K + 1, chunk):
        k1 = min(k0 + chunk, K + 1)
        ks = torch.arange(k0, k1, device=device)  # [C]
        C = ks.shape[0]
        # idx1[c, j] = N-1-j-k ; invalid (out of window) entries masked.
        idx1 = idx2[None, :] - ks[:, None]  # [C, J]
        invalid = idx1 < (N - n_r)[:, None, None]  # [R, C, J]
        eq = W32[:, idx1.clamp(min=0)] == suffix[:, None, :]  # [R, C, J]
        eq &= ~invalid
        del invalid
        # Run length of leading True along j = the match length. Explicit
        # dtypes matter: integral cumprod otherwise accumulates in int64,
        # materializing two [R, C, J] int64 copies. Values are 0/1 and the
        # run length is <= J <= _J_BUDGET, so int8/int16 are exact.
        L = eq.cumprod(dim=2, dtype=torch.int8).sum(dim=2, dtype=torch.int16)
        del eq
        L = torch.minimum(L, rep_limit[:, None])

        # Follower token of offset k lives at column N-k; the offset is
        # meaningful only while position i = N-1-k is inside the window
        # (k <= n_r - 1).
        valid_k = ks[None, :] <= (n_r - 1)[:, None]  # [R, C]
        charge = (allowed[:, None] <= L) & valid_k & active[:, None]
        if charge.any():
            followers = W.gather(1, (N - ks).clamp(max=N - 1).expand(R, C))
            rows = torch.arange(R, device=device)[:, None].expand(R, C) * vocab
            flat = (rows + followers.clamp(min=0))[charge]
            l_max.scatter_reduce_(0, flat, L[charge], reduce="amax", include_self=True)

    # Penalties: multiplier * base ** min(L - allowed, max_exp). llama.cpp's
    # std::pow promotes its float base with an int exponent to double, so
    # the penalty is computed in double precision and saturates only when
    # stored into the float32 logit. Mirror that: pow in float64, saturate
    # on the final cast. A float32 pow saturates too early: 0.8 * 2**128
    # must remain a finite logit (-2.72e38), not -inf.
    l_max = l_max.view(R, vocab)
    charged = l_max >= 0
    if not bool(charged.any()):
        return logits
    exponent = torch.minimum(l_max - allowed[:, None], max_exp[:, None])
    penalty = mult[:, None].double() * torch.pow(
        base[:, None].double(), exponent.to(torch.float64)
    )
    penalty = torch.where(charged, penalty, torch.zeros_like(penalty))
    for r, bm in enumerate(breaker_masks):
        if bm is not None and bool(bm.any()):
            penalty[r] = torch.where(bm, torch.zeros_like(penalty[r]), penalty[r])

    logits.index_put_(
        (row_idx,), logits[row_idx] - penalty.to(logits.dtype), accumulate=False
    )
    return logits


def apply_dry_batched(
    logits: torch.Tensor,
    entries: Sequence[tuple[int, _DryParams, list[int]]],
) -> torch.Tensor:
    """Apply DRY for ``entries`` = [(row_index, state, window)] where the
    windows are Python lists (the V1-runner path).

    Every entry must satisfy ``eligible(state)`` and have a non-None
    window. Modifies ``logits`` in place and returns it.
    """
    if not entries:
        return logits
    device = logits.device
    vocab = logits.shape[-1]
    R = len(entries)
    lens = [len(w) for _, _, w in entries]
    N = max(lens)

    # Right-aligned padded windows; -1 never matches a real token because
    # the validity mask excludes padded columns from every comparison.
    W = torch.full((R, N), -1, dtype=torch.int64)
    for r, (_, _, w) in enumerate(entries):
        W[r, N - len(w) :] = torch.tensor(w, dtype=torch.int64)
    W = W.to(device, non_blocking=True)

    return dry_core(
        logits,
        row_idx=torch.tensor(
            [row for row, _, _ in entries], dtype=torch.int64, device=device
        ),
        W=W,
        n_r=torch.tensor(lens, dtype=torch.int64, device=device),
        allowed=torch.tensor(
            [s.allowed_length for _, s, _ in entries],
            dtype=torch.int64,
            device=device,
        ),
        max_exp=torch.tensor(
            [s.max_exponent for _, s, _ in entries],
            dtype=torch.int64,
            device=device,
        ),
        mult=torch.tensor(
            [s.multiplier for _, s, _ in entries],
            dtype=torch.float32,
            device=device,
        ),
        base=torch.tensor(
            [s.base for _, s, _ in entries], dtype=torch.float32, device=device
        ),
        breaker_masks=[_breaker_vocab_mask(s, vocab, device) for _, s, _ in entries],
    )
