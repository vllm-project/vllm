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
# The run-length sum below is int16, so a run length must stay representable: at 32768
# it wraps negative. (The cumprod feeding it is int8 and holds only 0/1.) That is far
# outside any useful budget, but the exactness argument lives at the use sites, so pin
# it here where the number is set.
assert _J_BUDGET < 2**15, "_J_BUDGET must fit the int16 run-length sum"

# Byte budget for the per-chunk transients of the match scan. The gather
# W32[:, idx1] materializes [R, chunk, J] int32 (4 B/elem) before the
# comparison reduces it to bool; with the masks, the int8 cumprod and
# the int16 row sums, the marginal transient cost is ~8 B/elem. 24 keeps
# headroom for the fixed R*vocab penalty accumulator floor and allocator slack
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
    j_budget: int,
) -> torch.Tensor:
    """Apply DRY penalties in place given batched window tensors.

    Args:
      logits: [B, vocab] float tensor, modified in place.
      row_idx: [R] int64, row of ``logits`` for each DRY request.
      W: [R, N] int64 windows, RIGHT-aligned (window tokens occupy the
        trailing ``n_r`` columns; leading columns are ignored via ``n_r``
        masks, whatever they hold).
      n_r: [R] int64 window lengths.
      allowed / max_exp: [R] int64 per-request parameters.
      mult / base: [R] float32 per-request parameters (already rounded
        through float32, as llama.cpp stores them).
      breaker_masks: per-request [vocab] bool masks (or None for no
        breakers). ``vocab`` must be ``logits.shape[-1]``: the masks are
        applied with ``masked_fill_`` on a row of that width.
      j_budget: max(allowed + max_exp) computed by the caller on the host. It
        sizes a tensor, so it cannot be read off the device without a sync.
    """
    device = logits.device
    vocab = logits.shape[-1]
    R, N = W.shape

    # j_budget sizes the int16 run-length accumulator, and it arrives as an unchecked
    # host int now that the caller computes it. The routing predicate in apply_dry is
    # what actually bounds it - this assert is a second reader's sanity check, and -O
    # erases it - but it is free and it fails loudly rather than wrapping a run length.
    assert j_budget <= _J_BUDGET, f"j_budget {j_budget} over _J_BUDGET {_J_BUDGET}"
    # base >= 1 and mult >= 0 are preconditions too - see the amax comment below - but
    # they live in device tensors here, and reading them back would be the very sync
    # this path exists to avoid. apply_dry checks them on the host instead.

    # Per-request rep_limit: distance from the end of the nearest breaker
    # (llama.cpp step 1). A breaker at column c is j = N-1-c tokens from
    # the end.
    rep_limit = n_r.clone()
    # `bm is not None` alone: _breaker_mask returns None for a request with
    # no breakers, so the extra `bool(bm.any())` was a per-request host sync
    # that could only confirm what the None already said.
    any_breakers = any(bm is not None for bm in breaker_masks)
    bmask = None
    if any_breakers:
        # STACKED ONCE, used twice: here to find each request's nearest breaker, and
        # after the scan to zero the penalty on breaker tokens. The alternative is a
        # per-row index at both sites, 2R kernel launches a step, 768 at R=384.
        # llama.cpp's default breaker set resolves to ~3900 ids, so almost every real
        # DRY request takes this path, and stacking takes about a quarter off the step
        # on it; the figures are in the commit message. For a row that has a mask the
        # stack costs one byte per (row, vocab) entry next to the 4 the penalty
        # accumulator already costs, and copies a mask that is cached and resident
        # anyway (DryState._breaker_masks). A row with no breakers has nothing cached
        # and gets a fresh [vocab] zeros here every step, so a batch mixing the two
        # costs up to two bytes per (row, vocab); that case is unmeasured.
        bmask = torch.stack(
            [
                bm
                if bm is not None
                else torch.zeros(vocab, dtype=torch.bool, device=device)
                for bm in breaker_masks
            ]
        )
        Bwin = bmask.gather(1, W.clamp(min=0))
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

    # Match lengths per offset k, chunked to bound memory. Token ids fit int32, and
    # gathering int32 instead of int64 halves the dominant per-chunk transient.
    W32 = W.to(torch.int32)
    # J COMES FROM THE HOST. It is a tensor SHAPE, so it has to be known host-side, and
    # deriving it here as int((allowed + max_exp).max()) read two device tensors back
    # every step. The caller already holds both as numpy, so it computes J and passes it
    # in; see apply_dry.
    J = max(1, min(j_budget, N))
    idx2 = torch.arange(N - 1, N - 1 - J, -1, device=device)  # [J]
    suffix = W32.gather(1, idx2.expand(R, J))  # [R, J]
    K = N - 1  # offsets 1..N-1
    chunk = max(1, _CHUNK_BYTE_BUDGET // (_CHUNK_PEAK_BYTES_PER_ELEM * max(1, R * J)))

    # THE ACCUMULATOR HOLDS THE PENALTY, not the match length. A token reached from
    # several offsets must be penalized ONCE, for its longest match, and the offsets are
    # walked in chunks whose count varies with batch size and window length. Reducing
    # penalties with amax makes the result independent of that split: the penalty is
    # non-decreasing in the match length (see the preconditions above), so amax over
    # penalties picks the same winner as amax over lengths, and unlike a pass that
    # subtracts from the logits it is idempotent when a token appears in two chunks.
    # One extra element is the trash slot that uncharged entries scatter to, so no
    # boolean indexing and no host-side count is needed.
    pen = torch.zeros(R * vocab + 1, dtype=torch.float32, device=device)

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
        # NO `if charge.any()` AND NO BOOLEAN INDEXING. Both read the device back to the
        # host - `any()` for the branch, `x[charge]` for the output size - once per
        # chunk per step, on the sampler's path. Instead every (row, offset) pair is
        # scattered unconditionally and the uncharged ones are routed to a trash slot
        # one past the end of the accumulator, which is never read. Shapes are fixed,
        # so nothing has to be known host-side.
        followers = W.gather(1, (N - ks).clamp(max=N - 1).expand(R, C))
        rows = torch.arange(R, device=device)[:, None].expand(R, C) * vocab
        flat = rows + followers.clamp(min=0)
        # A Python int, not a device tensor: torch.tensor(x, device=...) here would be a
        # pageable host-to-device copy, which is itself a synchronization.
        flat = torch.where(charge, flat, R * vocab)
        # L comes out of the sum as int16 and out of the minimum as int64, since
        # rep_limit is a window length and legitimately exceeds int16 at long contexts;
        # the explicit cast keeps the exponent int64 if n_r's dtype ever narrows.
        # llama.cpp's std::pow promotes its float base with an int exponent to double,
        # so the penalty is computed in double and narrows to float32 only at the end.
        # Mirror that: pow in float64, saturate on the cast into the accumulator below.
        # A float32 pow saturates too early - 0.8 * 2**128 must stay finite (2.72e38).
        exponent = torch.minimum(L.to(torch.int64) - allowed[:, None], max_exp[:, None])
        p_chunk = mult[:, None].double() * torch.pow(
            base[:, None].double(), exponent.to(torch.float64)
        )
        # NO where() ON p_chunk. Uncharged entries were routed to the trash slot above,
        # so the value they carry is never read, and masking it would cost another
        # [R, C] float64 buffer per chunk in a loop whose whole point is not to.
        pen.scatter_reduce_(
            0,
            flat.reshape(-1),
            p_chunk.to(torch.float32).reshape(-1),
            reduce="amax",
            include_self=True,
        )

    # One subtract, once per (row, token). Only the penalty is held at [R, vocab], 4
    # bytes an entry; the exponent, its float64 cast, the pow result and the product
    # stay at [R, C] inside the loop. Carrying all of them at full width measures 41.1
    # bytes an entry against 5.5 here, 1931 MiB at R=384 on a 128k vocab against 259,
    # and it scales linearly in the batch, so the trigger is how many concurrent
    # requests enabled DRY. That is a measurement of 0edfb48091, an ancestor of this
    # commit, not an estimate: check it out and run the same script.
    # Breakers zero the penalty rather than the logit, so a breaker keeps its value.
    # The penalty narrowed float64 -> float32 in the scan loop above and narrows again
    # into logits.dtype below, where llama.cpp narrows once. Unreachable in-tree: the
    # sampler hands this function float32 logits (apply_sampling_params copies them to
    # float32 first), so the second narrowing is a no-op. It would cost an ulp against
    # llama.cpp on a bf16 caller.
    pen2 = pen[:-1].view(R, vocab)
    if bmask is not None:
        pen2.masked_fill_(bmask, 0.0)
    # index_add_ with the negated accumulator, in place. `logits[row_idx] -= pen2` reads
    # far better and costs a full [R, vocab] copy on the gather, plus another on the
    # cast when the logits are not float32: measured 380.1 MiB against 259.1 MiB at
    # batch 384 on a 128k vocab, float32 logits.
    pen2.neg_()
    logits.index_add_(
        0, row_idx, pen2 if logits.dtype == pen2.dtype else pen2.to(logits.dtype)
    )
    return logits
