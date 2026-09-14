# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Match computation for the DRY (Don't Repeat Yourself) repetition penalty.

Penalizes tokens that would extend a token sequence already present in the
context, with a penalty growing exponentially in the length of the repeated
sequence: ``multiplier * base ** (repeat_length - allowed_length)`` is
subtracted from the logit. ``repetition_penalty`` scales individual tokens
regardless of context; DRY targets verbatim loops.

The matching semantics follow llama.cpp's ``llama_sampler_dry_apply``
(itself ported from the koboldcpp implementation by pi6am; the DRY scheme
was designed by p-e-w for text-generation-webui), so identical settings
produce identical behavior in both runtimes. Parameter names and defaults
also follow llama.cpp.

Two implementations of the same quantity live here. ``_dry_penalties`` is a
direct sequential port of llama.cpp's Z-algorithm, used as the reference and
for requests the vectorized path cannot compute exactly. ``dry_core`` is the
vectorized form: the comparisons that determine the match length ending at
position ``i`` all lie along one fixed diagonal, so evaluating them as an
``[R, K, J]`` tensor and taking a cumprod-sum along ``j`` yields every match
length with no sequential scan. Capping ``J`` at
``allowed_length + max_exponent`` is exact, because the exponent clamp maps
every longer match to the same penalty; requests whose cap is unusable
(``max_exponent == 0``, i.e. ``base <= 1.000001``, or a cap beyond
``_J_BUDGET``) fall back to the sequential form.
"""

import torch


def _dry_penalties(
    window: list[int],
    breakers: frozenset[int],
    multiplier: float,
    base: float,
    allowed_length: int,
    max_exp: int,
) -> dict[int, float]:
    """Compute DRY penalties for one request.

    Port of the scan in ``llama_sampler_dry_apply``: a reverse-direction
    Z-algorithm finds, for each position, the length of the match between
    the window suffix and the sequence ending at that position; each
    match's follower token is charged the penalty for the longest repeat
    it would extend.

    Returns a dict mapping token id -> penalty to subtract from its logit.
    """
    m = len(window)

    def rat(i: int) -> int:  # reverse access: rat(0) == last token
        return window[m - 1 - i]

    # Step 1: the nearest breaker from the end caps the match length.
    rep_limit = m
    for i in range(m):
        if rat(i) in breakers:
            rep_limit = i
            break
    if rep_limit < allowed_length:
        return {}

    # Step 2: reverse-direction Z-algorithm -> per-position repeat counts
    # (forward-indexed into ``window`` via cnt[last - k]).
    cnt = [0] * m
    last = m - 1
    lt = rt = 0
    for k in range(1, m):
        if k > rt:
            # Outside the current Z-box: extend naively.
            z = 0
            while z + k < m and rat(z) == rat(z + k):
                z += 1
            cnt[last - k] = min(z, rep_limit)
            if z > 0:
                lt, rt = k, k + z - 1
        else:
            p = k - lt
            right_part_len = rt - k + 1
            if cnt[last - p] < right_part_len:
                # Fully inside the Z-box: copy.
                cnt[last - k] = min(cnt[last - p], rep_limit)
            else:
                # Touches the right edge: extend past it.
                j = rt + 1
                while j < m and rat(j) == rat(j - k):
                    j += 1
                cnt[last - k] = min(j - k, rep_limit)
                lt, rt = k, j - 1

    # Step 3: map each repeat's follower token to the longest repeat that
    # it would extend.
    max_token_repeat: dict[int, int] = {}
    for i in range(m - 1):
        repeat_len = cnt[i]
        if repeat_len >= allowed_length:
            tok = window[i + 1]
            if max_token_repeat.get(tok, -1) < repeat_len:
                max_token_repeat[tok] = repeat_len

    # Step 4: exponential penalty, exponent clamped for float32 safety.
    # Breaker tokens are never penalized. A value overflowing float32
    # saturates the logit to -inf downstream, as in llama.cpp.
    penalties: dict[int, float] = {}
    for tok, repeat_len in max_token_repeat.items():
        if tok in breakers:
            continue
        exponent = repeat_len - allowed_length
        if max_exp and exponent > max_exp:
            exponent = max_exp
        penalties[tok] = multiplier * (base**exponent)
    return penalties


_J_BUDGET = 2048
# The match-length accumulators are int16 (the cumprod/sum below, and l_max),
# so a run length must stay representable: at 32768 the sum wraps negative and
# at 65535 it aliases l_max's -1 sentinel exactly. Both are far outside any
# useful budget, but the exactness argument lives at the use sites, so pin it
# here where the number is set.
assert _J_BUDGET < 2**15, "_J_BUDGET must fit int16 match-length accumulators"

# Byte budget for the per-chunk transients of the match scan. The gather
# W32[:, idx1] materializes [R, chunk, J] int32 (4 B/elem) before the
# comparison reduces it to bool; with the masks, the int8 cumprod and
# the int16 row sums, the marginal transient cost is ~8 B/elem. 24 keeps
# headroom for the fixed R*vocab int64 l_max floor and allocator slack
# (and, measured, slightly better wall time than tighter chunking).
_CHUNK_BYTE_BUDGET = 256 * 1024 * 1024
_CHUNK_PEAK_BYTES_PER_ELEM = 24


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
    # `bm is not None` alone: _breaker_mask returns None for a request with
    # no breakers, so the extra `bool(bm.any())` was a per-request host sync
    # that could only confirm what the None already said.
    any_breakers = any(bm is not None for bm in breaker_masks)
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

    # Scatter target: per-(row, token) longest charged match. int16, not int64: the
    # value is a match length bounded by J <= _J_BUDGET (2048), so int16 is exact, and
    # this is the one allocation that is unavoidably R*vocab. At R=256 and a 128k vocab
    # that is 65 MB rather than 262 MB.
    l_max = torch.full((R * vocab,), -1, dtype=torch.int16, device=device)

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
            # .to(int16) to match l_max: torch.minimum against the int64 rep_limit above
            # promotes L back to int64, and scatter_reduce_ requires src.dtype ==
            # self.dtype. The cast is exact - L is a run length over J <= _J_BUDGET
            # (2048) columns and minimum only lowers it - and rep_limit itself must NOT
            # be narrowed instead, since it is a window length that legitimately exceeds
            # int16 at long contexts.
            l_max.scatter_reduce_(
                0, flat, L[charge].to(torch.int16), reduce="amax", include_self=True
            )

    # Penalties: multiplier * base ** min(L - allowed, max_exp). llama.cpp's std::pow
    # promotes its float base with an int exponent to double, so the penalty is computed
    # in double precision and saturates only when stored into the float32 logit. Mirror
    # that: pow in float64, saturate on the final cast. A float32 pow saturates too
    # early: 0.8 * 2**128 must remain a finite logit (-2.72e38), not -inf.  COMPUTED ON
    # THE CHARGED ENTRIES ONLY. A dense [R, vocab] formulation is the natural way to
    # write this and it is what this function used to do, but it materializes l_max, the
    # exponent, its float64 cast, the pow result, the product and the where output at
    # full size: measured at 41 bytes per (row, vocab) entry, 1931 MiB at R=384 on a
    # 128k vocab and linear from there, which is an out-of-memory crash whose trigger is
    # how many concurrent requests enabled DRY. The charged set is tiny by comparison -
    # of 32.8M entries at R=256, about 1024 are ever penalized - so everything below
    # works on a [M] vector of charged entries and writes them back with a scatter. Same
    # arithmetic, same dtypes, same order of operations per entry.
    charged_flat = (l_max >= 0).nonzero(as_tuple=True)[0]  # [M] int64
    if charged_flat.numel() == 0:
        return logits
    rows_of = charged_flat.div(vocab, rounding_mode="floor")  # [M]
    cols_of = charged_flat - rows_of * vocab  # [M]
    lm = l_max[charged_flat].to(torch.int64)  # [M]
    exponent = torch.minimum(lm - allowed[rows_of], max_exp[rows_of])
    penalty = mult[rows_of].double() * torch.pow(
        base[rows_of].double(), exponent.to(torch.float64)
    )

    # Breaker zeroing, also sparse. Gathering each row's mask at the charged columns
    # costs M elements rather than a full-vocab where per request, and it drops the
    # per-request ``bool(bm.any())`` device sync the dense version needed
    # (``_breaker_mask`` already returns None when a request has no breakers, so that
    # test was redundant as well as synchronous).
    if any_breakers:
        brk = torch.zeros_like(cols_of, dtype=torch.bool)
        for r, bm in enumerate(breaker_masks):
            if bm is None:
                continue
            brk |= (rows_of == r) & bm[cols_of]
        penalty = torch.where(brk, torch.zeros_like(penalty), penalty)

    # accumulate=True adds, so the value is negated: for one entry this is logit + (-p)
    # against the dense version's logit - p, which is the same IEEE result. Each flat
    # index is unique, so accumulation never doubles a penalty.
    logits.index_put_(
        (row_idx[rows_of], cols_of), (-penalty).to(logits.dtype), accumulate=True
    )
    return logits
