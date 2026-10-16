# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""This example implements the DRY (Don't Repeat Yourself) repetition penalty as
a Model Runner V2 custom logits processor, ``DryState``.

A token that would extend a repeat of at least ``allowed_length`` tokens loses
``multiplier * base ** (repeat_length - allowed_length)`` from its logit, with the
exponent clamped as in llama.cpp.
Parameter names and matching semantics follow llama.cpp's
``llama_sampler_dry_apply``. llama.cpp's version is itself ported from pi6am's
koboldcpp implementation of p-e-w's scheme for text-generation-webui.

Pass the class to ``LLM`` and configure it per request through ``extra_args``::

    LLM(..., logits_processors=[DryState])
    SamplingParams(extra_args={"dry_multiplier": 0.8})

The recognized keys are ``dry_multiplier`` (0.0, off), ``dry_base`` (1.75),
``dry_allowed_length`` (2), ``dry_penalty_last_n`` (-1, the whole context)
and ``dry_sequence_breakers``. This processor claims the ``dry_*`` namespace,
so any other ``dry_*`` key fails the request. Speculative decoding is not
supported and is refused at construction.

Run the example with::

    python examples/features/logits_processor/dry.py

It sends one prompt twice in one batch, once without DRY and once with it, and
prints both outputs, yielding an output similar to that shown below:

Generated Outputs:
------------------------------------------------------------
Prompt:    'The future of AI is', without DRY
Output:    ' in the hands of the people.\n\nThe future of AI is in the hands of
             the people.\n\nThe future of AI is in the hands of the
             people.\n\nThe future of AI is in the hands of the people.\n\nThe
             future of AI is in the hands of the people.\n'
------------------------------------------------------------
Prompt:    'The future of AI is', with DRY
Output:    ' in the hands of the people.\n\nThe future of AI is the future of
             the human race.\n\nThe future of AI is a future of the human race,
             and the future of humanity.\n\nThe future of AI is an AI that is
             capable of making decisions that are not based on human
             intelligence.'
------------------------------------------------------------
"""

import math
import weakref
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.sample.logits_processor import (
    LogitsContext,
    LogitsProcessor,
    LogitsProcRequestState,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

_J_BUDGET = 2048
# Run lengths feed an int16 sum, so they must stay below 2**15.
assert _J_BUDGET < 2**15, "_J_BUDGET must fit the int16 run-length sum"

# Byte budget for the per-chunk transients of the match scan. The gather
# W32[:, idx1] materializes [R, chunk, J] int32 (4 B/elem) before the
# comparison reduces it to bool; with the masks, the int8 cumprod and
# the int16 row sums, the marginal transient cost is ~8 B/elem. 24 keeps
# headroom for the fixed R*vocab penalty accumulator floor and allocator slack.
_CHUNK_BYTE_BUDGET = 256 * 1024 * 1024
_CHUNK_PEAK_BYTES_PER_ELEM = 24

# llama.cpp's FLOAT_MAX_LOG (src/llama-sampler.cpp): ln(float32 max).
_FLOAT_MAX_LOG = 88.7228391

# Breakers are truncated to this many code points. llama.cpp cuts at this many
# bytes (llama_sampler_init_dry), which keeps fewer characters of a non-ASCII
# breaker.
_MAX_BREAKER_CHAR_LEN = 40

# Upper bound on distinct breaker sets cached per tokenizer.
_MAX_CACHED_BREAKER_SETS = 64

STR_SPEC_DEC_REJECTS_DRY = (
    "The DRY logits processor is not supported when speculative decoding is enabled."
)

DEFAULT_DRY_SEQUENCE_BREAKERS = ("\n", ":", '"', "*")
"""llama.cpp's default DRY sequence breakers."""

MAX_DRY_SEQUENCE_BREAKERS = 64
"""Upper bound on the dry_sequence_breakers list length. Each multi-character
breaker in an uncached set scans the vocabulary."""

_DRY_INT_MAX = 2**31 - 1
"""Upper bound on the integral DRY parameters, matching llama-server's
INT32_MAX cap. The state below stores them in an int64 numpy array, where
anything at or above 2**63 raises OverflowError inside execute_model and
takes the engine down."""

_MAX_CACHED_BREAKER_MASKS = 64
"""Upper bound on distinct breaker sets holding a device mask, mirroring
_MAX_CACHED_BREAKER_SETS on the host side."""

_WARMUP_WINDOW = 8
"""Window length of the dry_core call in DryState.__init__, long enough to
hold a match at the default dry_allowed_length."""

_DRY_DEFAULTS: dict[str, Any] = {
    "dry_multiplier": 0.0,
    "dry_base": 1.75,
    "dry_allowed_length": 2,
    "dry_penalty_last_n": -1,  # whole context; llama.cpp's default is 64
    "dry_sequence_breakers": DEFAULT_DRY_SEQUENCE_BREAKERS,
}
"""Recognized extra_args keys and their defaults."""


# ---------------------------------------------------------------------------
# The match computation
# ---------------------------------------------------------------------------
# ``_dry_penalties`` is a sequential port of llama.cpp's Z-algorithm, used as the
# reference. ``dry_core`` is the vectorized form: the comparisons that determine
# the match length ending at position ``i`` lie along one diagonal, so a
# cumprod-sum along ``j`` of an ``[R, K, J]`` tensor gives every match length
# without a sequential scan. Capping ``J`` at ``allowed_length + max_exponent``
# is exact, because the exponent clamp maps every longer match to the same
# penalty. Requests whose cap is unusable (``max_exponent == 0``, i.e.
# ``base <= 1.000001``, or a cap beyond ``_J_BUDGET``) use the sequential form.


def max_exponent(base: float) -> int:
    """Exponent clamp, mirroring llama.cpp bit-for-bit.

    llama.cpp computes ``FLOAT_MAX_LOG / std::log(dry_base)`` entirely in
    float32. Computing the quotient in float64 lands on the wrong side of
    the integer truncation for some bases: at ``base=2.0`` float64 gives
    127.99999998 -> 127 while float32 gives exactly 128.0 -> 128.
    """
    if base <= 1.000001:
        return 0
    return int(np.float32(_FLOAT_MAX_LOG) / np.log(np.float32(base)))


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

    The tensor bounds below are not checked, to avoid a sync.

    Args:
      logits: [B, vocab] float tensor, modified in place.
      row_idx: [R] int64, row of ``logits`` for each DRY request.
      W: [R, N] int64 windows, right-aligned (window tokens occupy the
        trailing ``n_r`` columns; leading columns are ignored via ``n_r``
        masks, whatever they hold).
      n_r: [R] int64 window lengths.
      allowed: [R] int64 per-request allowed length.
      max_exp: [R] int64 per-request exponent ceiling, > 0.
      mult: [R] float32 per-request multiplier, >= 0, already rounded
        through float32, as llama.cpp stores it.
      base: [R] float32 per-request base, >= 1, rounded the same way.
      breaker_masks: per-request [vocab] bool masks (or None for no
        breakers). ``vocab`` must equal ``logits.shape[-1]``.
      j_budget: max(allowed + max_exp), <= _J_BUDGET, computed by the caller on
        the host.

    """
    device = logits.device
    vocab = logits.shape[-1]
    R, N = W.shape

    # j_budget arrives as an unchecked host int; this check fails loudly rather
    # than wrapping an int16 run length. The routing predicate in DryState.apply
    # is what actually bounds it.
    if j_budget > _J_BUDGET:
        raise ValueError(f"j_budget {j_budget} over _J_BUDGET {_J_BUDGET}")
    # base >= 1 and mult >= 0 are preconditions too (see the amax comment below).
    # They live in device tensors and reading them back would sync; use_dry()
    # and validate_params enforce them instead.

    # rep_limit: distance from the end of the nearest breaker (llama.cpp step 1).
    # A breaker at column c is j = N-1-c tokens from the end.
    rep_limit = n_r.clone()
    # bool(bm.any()) would be a per-request host sync; the None check says the same.
    any_breakers = any(bm is not None for bm in breaker_masks)
    bmask = None
    if any_breakers:
        # Stacked once, used twice: nearest-breaker search here, penalty zeroing
        # after the scan. The masks are cached and resident (DryState._breaker_masks),
        # so the stack costs one byte per (row, vocab) entry.
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

    # llama.cpp: if rep_limit < allowed_length, the request produces nothing.
    active = rep_limit >= allowed

    # Token ids fit int32, and gathering int32 halves the dominant per-chunk transient.
    W32 = W.to(torch.int32)
    # J is a tensor shape, so it must be known host-side; the caller computes it
    # from numpy (see DryState.apply) to avoid a per-step device readback.
    J = max(1, min(j_budget, N))
    idx2 = torch.arange(N - 1, N - 1 - J, -1, device=device)  # [J]
    suffix = W32.gather(1, idx2.expand(R, J))  # [R, J]
    K = N - 1  # offsets 1..N-1
    chunk = max(1, _CHUNK_BYTE_BUDGET // (_CHUNK_PEAK_BYTES_PER_ELEM * max(1, R * J)))

    # amax keeps one penalty per token, for its longest match, however the offsets
    # are chunked, as long as the penalty does not fall as the match grows
    # (base >= 1, mult >= 0).
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
        # Run length of leading True along j = the match length. Explicit dtypes avoid
        # int64 intermediate copies; values are 0/1 and runs are <= J <= _J_BUDGET,
        # so int8/int16 are exact.
        L = eq.cumprod(dim=2, dtype=torch.int8).sum(dim=2, dtype=torch.int16)
        del eq
        # Do not narrow rep_limit to L's int16; window lengths can exceed it.
        L = torch.minimum(L, rep_limit[:, None])

        # Follower token of offset k lives at column N-k; the offset counts only
        # while position i = N-1-k is inside the window (k <= n_r - 1).
        valid_k = ks[None, :] <= (n_r - 1)[:, None]  # [R, C]
        charge = (allowed[:, None] <= L) & valid_k & active[:, None]
        # No `if charge.any()` and no boolean indexing: both sync the host. Uncharged
        # entries scatter to a trash slot one past the accumulator end, never read.
        followers = W.gather(1, (N - ks).clamp(max=N - 1).expand(R, C))
        rows = torch.arange(R, device=device)[:, None].expand(R, C) * vocab
        flat = rows + followers.clamp(min=0)
        # A Python int, not a device tensor: torch.tensor(x, device=...) here would be a
        # pageable host-to-device copy, which is itself a synchronization.
        flat = torch.where(charge, flat, R * vocab)
        # L takes rep_limit's dtype from the minimum above. The cast keeps the
        # exponent int64 whatever dtype n_r, allowed and max_exp arrive in.
        exponent = torch.minimum(L.to(torch.int64) - allowed[:, None], max_exp[:, None])
        # float64 pow, as llama.cpp's std::pow(float, int); a float32 pow overflows
        # early (0.8 * 2**128 is finite in float32).
        p_chunk = mult[:, None].double() * torch.pow(
            base[:, None].double(), exponent.to(torch.float64)
        )
        # No where() on p_chunk: uncharged entries went to the trash slot and are
        # never read.
        pen.scatter_reduce_(
            0,
            flat.reshape(-1),
            p_chunk.to(torch.float32).reshape(-1),
            reduce="amax",
            include_self=True,
        )

    # Of the penalty terms, only the penalty itself is held at [R, vocab] (4 B/entry);
    # the exponent, its float64 cast, the pow result and the product stay at [R, C]
    # inside the loop.
    # Breakers zero the penalty rather than the logit, so a breaker keeps its value.
    # The narrowing to logits.dtype below is a no-op under vLLM's sampler, which
    # always passes float32 logits.
    pen2 = pen[:-1].view(R, vocab)
    if bmask is not None:
        pen2.masked_fill_(bmask, 0.0)
    # index_add_ avoids the extra [R, vocab] gather+copy of `logits[row_idx] -= pen2`.
    pen2.neg_()
    logits.index_add_(
        0, row_idx, pen2 if logits.dtype == pen2.dtype else pen2.to(logits.dtype)
    )
    return logits


# ---------------------------------------------------------------------------
# Breaker resolution
# ---------------------------------------------------------------------------


class _VocabIndex(NamedTuple):
    """A tokenizer's decoded vocabulary, plus a character index into it.

    ``char_ids`` maps every character in ``texts`` to the ids of the tokens
    whose text contains it.
    """

    texts: list[str]
    char_ids: dict[str, list[int]]


# Per-tokenizer caches, weakly keyed so tokenizers can be collected:
# tokenizer -> its decoded vocabulary and character index, and
# tokenizer -> {breaker string tuple -> resolved breaker ids}.
_BreakerIdsPerTokenizer = dict[tuple[str, ...], list[int]]
_vocab_index_cache: "weakref.WeakKeyDictionary[Any, _VocabIndex]" = (
    weakref.WeakKeyDictionary()
)
_breaker_ids_cache: "weakref.WeakKeyDictionary[Any, _BreakerIdsPerTokenizer]" = (
    weakref.WeakKeyDictionary()
)


def _vocab_index(tokenizer: Any) -> _VocabIndex:
    """Decode every token of ``tokenizer`` once and index its characters."""
    index = _vocab_index_cache.get(tokenizer)
    if index is None:
        # Include added tokens above vocab_size, which llama.cpp's scan also covers.
        n_ids = getattr(tokenizer, "max_token_id", tokenizer.vocab_size - 1) + 1
        texts = tokenizer.batch_decode([[i] for i in range(n_ids)])
        char_ids: dict[str, list[int]] = {}
        for i, text in enumerate(texts):
            for char in set(text):
                char_ids.setdefault(char, []).append(i)
        index = _VocabIndex(texts, char_ids)
        _vocab_index_cache[tokenizer] = index
    return index


def resolve_dry_breakers(tokenizer: Any, breaker_strs: tuple[str, ...]) -> list[int]:
    """Resolve breaker strings to the ids of every token whose text contains one.

    This is llama.cpp's ``get_overlapping_token_sequences`` rule, without its
    multi-token restart sequences. Resolution runs in ``add_request``, inside
    ``execute_model``, so a single-character breaker (llama.cpp's defaults and
    most custom sets) resolves by lookup in the character index instead of a
    vocabulary scan. Results are cached per (tokenizer, breaker set).
    """
    breaker_strs = tuple(s[:_MAX_BREAKER_CHAR_LEN] for s in breaker_strs if s)
    if not breaker_strs:
        return []
    per_tok = _breaker_ids_cache.setdefault(tokenizer, {})
    cached = per_tok.get(breaker_strs)
    if cached is not None:
        return list(cached)

    index = _vocab_index(tokenizer)
    ids: set[int] = set()
    for s in breaker_strs:
        if len(s) == 1:
            ids.update(index.char_ids.get(s, ()))
        else:
            ids.update(i for i, text in enumerate(index.texts) if s in text)
    result = sorted(ids)
    if len(per_tok) >= _MAX_CACHED_BREAKER_SETS:
        # Evict the oldest entry (dict preserves insertion order).
        per_tok.pop(next(iter(per_tok)))
    per_tok[breaker_strs] = result
    return list(result)


# ---------------------------------------------------------------------------
# The logits processor
# ---------------------------------------------------------------------------


def _dry_args(sampling_params: SamplingParams) -> dict[str, Any]:
    """Read the request's DRY arguments, filling in the defaults.

    A key present with value None counts as unset, matching how
    ``SamplingParams`` coerces its own None-valued arguments.
    """
    extra = sampling_params.extra_args or {}
    return {
        name: default if extra.get(name) is None else extra[name]
        for name, default in _DRY_DEFAULTS.items()
    }


def use_dry(multiplier: float, base: float, penalty_last_n: int) -> bool:
    """Whether DRY applies, by llama.cpp's gate in llama_sampler_dry_apply."""
    return bool(multiplier) and base >= 1.0 and penalty_last_n != 0


class DryState(LogitsProcessor):
    def __init__(self, vllm_config: "VllmConfig", req_states: LogitsProcRequestState):
        # Refuse speculative decoding here, since admission cannot see its config.
        if vllm_config.speculative_config is not None:
            raise ValueError(STR_SPEC_DEC_REJECTS_DRY)
        self.req_states = req_states
        max_num_reqs = req_states.max_num_reqs
        self.vocab_size = req_states.vocab_size
        self.device = req_states.device

        # float32, to round multiplier and base as llama.cpp's float members do.
        self.multiplier = np.zeros(max_num_reqs, dtype=np.float32)
        self.base = np.zeros(max_num_reqs, dtype=np.float32)
        self.allowed_length = np.zeros(max_num_reqs, dtype=np.int64)
        self.penalty_last_n = np.zeros(max_num_reqs, dtype=np.int64)
        self.max_exponent = np.zeros(max_num_reqs, dtype=np.int64)
        self.use_dry = np.zeros(max_num_reqs, dtype=bool)

        # req_idx -> its breaker set, and breaker set -> [vocab] bool device
        # mask shared by every request that asked for the same breakers.
        self.breaker_ids: dict[int, frozenset[int]] = {}
        self._breaker_masks: dict[frozenset[int], torch.Tensor] = {}

        self._warned_unresolved = False

        # Deferred import: the tokenizer registry pulls in transformers, and
        # the frontend imports this module only to validate params.
        from vllm.tokenizers import cached_tokenizer_from_config

        # None under skip_tokenizer_init.
        self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        # Resolve the default set here to keep the vocabulary decode out of
        # execute_model, where add_request runs.
        self._default_breaker_ids = self._resolve_breakers(
            DEFAULT_DRY_SEQUENCE_BREAKERS
        )
        self._warm_up_dry_core()

    def _warm_up_dry_core(self) -> None:
        """Load dry_core's CUDA kernels before a request needs them.

        The operands match the dtypes and ranks apply passes, and include a breaker
        mask so that the breaker path's kernels load too.
        """
        vocab = self.vocab_size
        # The breaker takes the last id, which must not be the window's token 0.
        if vocab < 2:
            return
        device = self.device
        allowed = _DRY_DEFAULTS["dry_allowed_length"]
        base = _DRY_DEFAULTS["dry_base"]
        max_exp = max_exponent(base)

        def col(value: float, dtype: torch.dtype) -> torch.Tensor:
            return torch.full((1,), value, dtype=dtype, device=device)

        breakers = torch.zeros(vocab, dtype=torch.bool, device=device)
        breakers.narrow(0, vocab - 1, 1).fill_(True)
        dry_core(
            torch.zeros(1, vocab, dtype=torch.float32, device=device),
            row_idx=col(0, torch.int64),
            W=torch.zeros(1, _WARMUP_WINDOW, dtype=torch.int64, device=device),
            n_r=col(_WARMUP_WINDOW, torch.int64),
            allowed=col(allowed, torch.int64),
            max_exp=col(max_exp, torch.int64),
            mult=col(1.0, torch.float32),
            base=col(base, torch.float32),
            breaker_masks=[breakers],
            j_budget=allowed + max_exp,
        )

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        """Check the ``dry_*`` keys of ``extra_args`` at request admission.

        Raises:
            ValueError: on an unknown or out-of-range DRY argument.

        """
        extra = sampling_params.extra_args or {}
        unknown = sorted(
            key for key in extra if key.startswith("dry_") and key not in _DRY_DEFAULTS
        )
        if unknown:
            # A misspelled key would otherwise be ignored silently.
            raise ValueError(
                f"Unknown dry_* extra_args: {', '.join(unknown)}. "
                f"Supported keys: {', '.join(_DRY_DEFAULTS)}."
            )

        args = _dry_args(sampling_params)
        for name in (
            "dry_multiplier",
            "dry_base",
            "dry_allowed_length",
            "dry_penalty_last_n",
        ):
            value = args[name]
            # JSON true/false are bools, which pass isinstance(value, int).
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"{name} must be a number, got {type(value).__name__}."
                )

        multiplier = args["dry_multiplier"]
        base = args["dry_base"]
        allowed_length = args["dry_allowed_length"]
        penalty_last_n = args["dry_penalty_last_n"]
        breakers = args["dry_sequence_breakers"]

        if not math.isfinite(multiplier) or multiplier < 0.0:
            raise ValueError(
                f"dry_multiplier must be non-negative and finite, got {multiplier}."
            )
        if not math.isfinite(base) or base < 0.0:
            raise ValueError(f"dry_base must be non-negative and finite, got {base}.")
        if (
            not isinstance(allowed_length, int)
            or allowed_length < 0
            or allowed_length > _DRY_INT_MAX
        ):
            raise ValueError(
                f"dry_allowed_length must be an integer in [0, {_DRY_INT_MAX}], "
                f"got {allowed_length}."
            )
        if (
            not isinstance(penalty_last_n, int)
            or penalty_last_n < -1
            or penalty_last_n > _DRY_INT_MAX
        ):
            raise ValueError(
                "dry_penalty_last_n must be an integer: -1 (whole context), "
                f"0 (disable), or in [1, {_DRY_INT_MAX}], got {penalty_last_n}."
            )
        if not isinstance(breakers, (list, tuple)) or any(
            not isinstance(s, str) for s in breakers
        ):
            raise ValueError(
                f"dry_sequence_breakers must be a list of strings, got {breakers!r}."
            )
        if len(breakers) > MAX_DRY_SEQUENCE_BREAKERS:
            raise ValueError(
                f"dry_sequence_breakers supports at most "
                f"{MAX_DRY_SEQUENCE_BREAKERS} entries, got {len(breakers)}."
            )
        if multiplier and 0.0 <= base < 1.0:
            # libllama has the same gate. llama-server resets such a base to its
            # default instead.
            logger.warning(
                "dry_base=%s is below 1.0, which disables DRY entirely "
                "(llama.cpp semantics), even though dry_multiplier=%s was "
                "set. No repetition penalty will be applied.",
                base,
                multiplier,
            )

    def _resolve_breakers(self, breakers: tuple[str, ...]) -> frozenset[int]:
        if self._tokenizer is None or not breakers:
            return frozenset()
        return frozenset(resolve_dry_breakers(self._tokenizer, breakers))

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        args = _dry_args(sampling_params)
        multiplier = args["dry_multiplier"]
        base = args["dry_base"]
        penalty_last_n = args["dry_penalty_last_n"]
        enabled = use_dry(multiplier, base, penalty_last_n)
        self.use_dry[req_idx] = enabled
        self.breaker_ids.pop(req_idx, None)
        if not enabled:
            return False
        self.multiplier[req_idx] = multiplier
        self.base[req_idx] = base
        self.allowed_length[req_idx] = args["dry_allowed_length"]
        self.penalty_last_n[req_idx] = penalty_last_n
        self.max_exponent[req_idx] = max_exponent(float(self.base[req_idx]))

        breakers = tuple(args["dry_sequence_breakers"])
        ids = (
            self._default_breaker_ids
            if breakers == DEFAULT_DRY_SEQUENCE_BREAKERS
            else self._resolve_breakers(breakers)
        )
        if ids:
            self.breaker_ids[req_idx] = ids
        elif breakers and self._tokenizer is None and not self._warned_unresolved:
            logger.warning(
                "DRY sequence breakers were not resolved to token ids: this "
                "engine has no tokenizer. Proceeding without breakers."
            )
            self._warned_unresolved = True
        return True

    def _breaker_mask(self, req_idx: int) -> torch.Tensor | None:
        ids = self.breaker_ids.get(req_idx)
        if not ids:
            return None
        mask = self._breaker_masks.get(ids)
        if mask is None:
            # Built on the host to avoid a sync; once per breaker set.
            ids_np = np.fromiter(ids, dtype=np.int64, count=len(ids))
            ids_np = ids_np[ids_np < self.vocab_size]
            m_np = np.zeros(self.vocab_size, dtype=bool)
            m_np[ids_np] = True
            mask = async_tensor_h2d(m_np, self.device)
            if len(self._breaker_masks) >= _MAX_CACHED_BREAKER_MASKS:
                # Evict the oldest set. A client can send a new set with every request.
                self._breaker_masks.pop(next(iter(self._breaker_masks)))
            self._breaker_masks[ids] = mask
        return mask

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        req_indices = ctx.idx_mapping_np
        active_rows = np.flatnonzero(self.use_dry[req_indices])
        if active_rows.size == 0:
            return logits
        if logits.shape[0] != req_indices.shape[0]:
            raise RuntimeError("DRY received draft-expanded logits")

        # host-side bound; reading positions off GPU would sync per step
        cur_len = ctx.seq_lens_upper_bound_np[active_rows].astype(np.int64)

        reqs = req_indices[active_rows]
        last_n = self.penalty_last_n[reqs]
        window_len = np.where(last_n == -1, cur_len, np.minimum(cur_len, last_n))
        allowed = self.allowed_length[reqs]
        keep = window_len > allowed
        if not np.any(keep):
            return logits
        active_rows = active_rows[keep]
        reqs = reqs[keep]
        cur_len = cur_len[keep]
        window_len = window_len[keep]
        allowed = allowed[keep]
        max_exp = self.max_exponent[reqs]

        # Route degenerate-clamp requests (base <= 1.000001 or oversized
        # cap) through the sequential reference implementation.
        fast = (max_exp > 0) & (allowed + max_exp <= _J_BUDGET)
        all_tokens = self.req_states.all_token_ids.gpu

        if np.any(fast):
            f_rows = active_rows[fast]
            f_reqs = reqs[fast]
            f_len = window_len[fast]
            N = int(f_len.max())
            reqs_t = async_tensor_h2d(f_reqs, self.device)
            cur_t = async_tensor_h2d(cur_len[fast], self.device)
            j = torch.arange(N, device=self.device)
            # Right-aligned gather: column j holds token (cur_len - N + j);
            # out-of-window columns are masked inside dry_core via n_r.
            gather_idx = (cur_t[:, None] - N + j[None, :]).clamp(min=0)
            W = all_tokens[reqs_t[:, None], gather_idx].long()
            dry_core(
                logits,
                row_idx=async_tensor_h2d(f_rows, self.device),
                W=W,
                n_r=async_tensor_h2d(f_len, self.device),
                allowed=async_tensor_h2d(allowed[fast], self.device),
                max_exp=async_tensor_h2d(max_exp[fast], self.device),
                mult=async_tensor_h2d(self.multiplier[f_reqs], self.device),
                base=async_tensor_h2d(self.base[f_reqs], self.device),
                breaker_masks=[self._breaker_mask(r) for r in f_reqs],
                j_budget=int((allowed[fast] + max_exp[fast]).max()),
            )

        # The sequential fallback copies each window to the host, an expected sync.
        slow = ~fast
        if np.any(slow):
            with gpu_sync_allowed():
                rows_list = []
                cols_list = []
                vals_list = []
                for row, req, w_len, cur in zip(
                    active_rows[slow], reqs[slow], window_len[slow], cur_len[slow]
                ):
                    window = (
                        all_tokens[int(req), int(cur) - int(w_len) : int(cur)]
                        .cpu()
                        .tolist()
                    )
                    penalties = _dry_penalties(
                        window,
                        self.breaker_ids.get(int(req), frozenset()),
                        float(self.multiplier[req]),
                        float(self.base[req]),
                        int(self.allowed_length[req]),
                        int(self.max_exponent[req]),
                    )
                    for tok, val in penalties.items():
                        rows_list.append(int(row))
                        cols_list.append(tok)
                        vals_list.append(val)
                if rows_list:
                    logits[
                        torch.tensor(rows_list, dtype=torch.int64, device=self.device),
                        torch.tensor(cols_list, dtype=torch.int64, device=self.device),
                    ] -= torch.tensor(
                        vals_list, dtype=torch.float32, device=self.device
                    )
        return logits


# ---------------------------------------------------------------------------
# The demo
# ---------------------------------------------------------------------------


def main():
    # Imported here so that loading DryState from this module does not import LLM.
    from vllm import LLM

    prompts = ["The future of AI is"] * 2
    sampling_params_list = [
        SamplingParams(temperature=0.0, max_tokens=64),
        SamplingParams(
            temperature=0.0, max_tokens=64, extra_args={"dry_multiplier": 0.8}
        ),
    ]
    llm = LLM(model="facebook/opt-125m", logits_processors=[DryState])
    outputs = llm.generate(prompts, sampling_params_list)
    print("\nGenerated Outputs:\n" + "-" * 60)
    for label, output in zip(("without DRY", "with DRY"), outputs):
        print(f"Prompt:    {output.prompt!r}, {label}")
        print(f"Output:    {output.outputs[0].text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()
