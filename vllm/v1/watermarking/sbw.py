# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stateless Bernoulli Watermarking (SBW) generation and detection primitives.

SBW replaces vocabulary permutation (KGW) or tournament reweighting (SynthID)
with independent per-token Bernoulli trials via a counter-based RNG (Philox
4x32-10).  Per-token greenness is O(1): CBRNG(v, seed_t) < gamma, enabling a
single fused kernel with zero intermediate allocations.

Reference
---------
Ceppi & Sanchez, "Flip, Don't Shuffle: Watermarking LLMs at the Speed of
Inference", EMNLP 2026. https://arxiv.org/abs/2609.03844

Two seeding schemes are supported
----------------------------------
selfhash (default)
    Anchored minhash PRF.  The candidate token v is the anchor; context tokens
    [1..H-1] form the minhash prefix.  Seed per (context, v):
        seed = min(
            min_i( key * h(context[i]) * h(v) ),   # prefix minhash
            key * h(v) * h(v),                       # self-term (anchor)
        )
    More robust to editing attacks.  Requires context_width >= 1; at
    context_width == 1 the prefix is empty and the seed is the self-term only
    (fixed green list per token, equivalent to SBW-1).

lefthash
    Additive PRF.  Seed per context row:
        seed = key * sum(context)
    Faster; green list is the same for all tokens in the same context.
    Equivalent to SBW-1 (paper Table 1) when context_width == 1.
"""

import torch

from vllm.config.watermarking import _SBW_DEFAULT_CONTEXT_WIDTH, SBWScheme
from vllm.v1.watermarking.detector import WatermarkDetector
from vllm.v1.watermarking.watermarker import (
    RandomSampler,
    Watermarker,
    WatermarkSample,
)

# ---------------------------------------------------------------------------
# Philox 4x32-10 constants (SBW convention; different key derivation from
# the Gumbel PhiloxPRF which uses domain-separation constants _CONTEXT_DOMAIN /
# _TOKEN_DOMAIN.  Do NOT unify: the two algorithms use incompatible calling
# conventions.)
# ---------------------------------------------------------------------------
_PHILOX_M0: int = 0xD2511F53
_PHILOX_M1: int = 0xCD9E8D57
_PHILOX_W0: int = 0x9E3779B9
_PHILOX_W1: int = 0xBB67AE85
_MASK32: int = 0xFFFFFFFF
_INT31_MAX: int = 0x7FFFFFFF


# ---------------------------------------------------------------------------
# Bob Jenkins integer hash (GPU-native, avoids permutation table)
# ---------------------------------------------------------------------------


def _sbw_hashint(x: torch.Tensor) -> torch.Tensor:
    """Bob Jenkins integer hash. +1 avoids hash(0) == 0."""
    i = (x + 1).to(torch.int32)
    i = i - (i << 6)
    i = i ^ (i >> 17)
    i = i - (i << 9)
    i = i ^ (i << 4)
    i = i - (i << 3)
    i = i ^ (i << 10)
    i = i ^ (i >> 15)
    return i.to(torch.long)


# ---------------------------------------------------------------------------
# Philox 4x32-10 on int32 tensors (Triton-fusible via torch.compile)
# ---------------------------------------------------------------------------


def _sbw_philox4x32_10_i32(
    c0: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
    c3: torch.Tensor,
    k0: torch.Tensor,
    k1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    for _ in range(10):
        prod0 = (c0.long() & _MASK32) * _PHILOX_M0
        prod2 = (c2.long() & _MASK32) * _PHILOX_M1
        hi0 = (prod0 >> 32).to(torch.int32)
        lo0 = prod0.to(torch.int32)
        hi1 = (prod2 >> 32).to(torch.int32)
        lo1 = prod2.to(torch.int32)
        c0 = hi1 ^ c1 ^ k0
        c1 = lo1
        c2 = hi0 ^ c3 ^ k1
        c3 = lo0
        k0 = (k0.long() + _PHILOX_W0).to(torch.int32)
        k1 = (k1.long() + _PHILOX_W1).to(torch.int32)
    return c0, c1, c2, c3


_sbw_philox_compiled = torch.compile(
    _sbw_philox4x32_10_i32, mode="max-autotune-no-cudagraphs"
)


# ---------------------------------------------------------------------------
# Fused kernel: selfhash
# ---------------------------------------------------------------------------


def _sbw_selfhash_inner(
    context: torch.Tensor,  # (B, context_width) int64
    logits: torch.Tensor,  # (B, V) float, mutated in-place
    gamma_int: torch.Tensor,  # () int64 scalar
    delta: torch.Tensor,  # (B,) float, per-row bias (0.0 for non-watermarked)
    hash_key: torch.Tensor,  # () int64 scalar
) -> torch.Tensor:
    """Selfhash: anchored minhash PRF + Philox + green-list bias (in-place).

    context[:, 1:] is the minhash prefix; the candidate token v provides the
    anchor via h(v)*h(v).  When context_width==1 the prefix is empty and the
    seed degenerates to hash_key * h(v)^2.  The Python-level guard on
    prefix.shape[1] gives torch.compile a static graph per context_width,
    avoiding a zero-size reduction that Inductor cannot lower.
    """
    B, V = logits.shape
    device = logits.device
    positions = torch.arange(V, device=device, dtype=torch.long)
    h_pos = _sbw_hashint(positions)  # (V,)
    cand_val = hash_key * h_pos * h_pos  # (V,)
    prefix = context[:, 1:]  # (B, H-1), may be empty
    if prefix.shape[1] > 0:
        h_prefix = _sbw_hashint(prefix)  # (B, H-1)
        prefix_term = h_prefix.unsqueeze(2) * h_pos.unsqueeze(0).unsqueeze(
            0
        )  # (B, H-1, V)
        prefix_min = (hash_key * prefix_term).min(dim=1).values  # (B, V)
        seeds = torch.minimum(prefix_min, cand_val)  # (B, V)
    else:
        seeds = cand_val.unsqueeze(0).expand(B, -1)  # (B, V)
    call_idx = (positions // 4).to(torch.int32)
    word_idx = (positions % 4).to(torch.int32)
    k0 = (seeds & _MASK32).to(torch.int32)
    k1 = ((seeds >> 32) & _MASK32).to(torch.int32)
    zeros = torch.zeros(B, V, device=device, dtype=torch.int32)
    r0, r1, r2, r3 = _sbw_philox_compiled(
        call_idx.unsqueeze(0).expand(B, -1),
        zeros,
        zeros.clone(),
        zeros.clone(),
        k0,
        k1,
    )
    sel = torch.where(
        word_idx == 0,
        r0,
        torch.where(word_idx == 1, r1, torch.where(word_idx == 2, r2, r3)),
    )
    logits = logits + (
        ((sel & _INT31_MAX) < gamma_int).to(logits.dtype) * delta.unsqueeze(-1)
    )
    return logits


_sbw_selfhash_compiled = torch.compile(
    _sbw_selfhash_inner, mode="max-autotune-no-cudagraphs"
)


# ---------------------------------------------------------------------------
# Fused kernel: lefthash
# ---------------------------------------------------------------------------


def _sbw_lefthash_inner(
    context: torch.Tensor,
    logits: torch.Tensor,
    gamma_int: torch.Tensor,
    delta: torch.Tensor,  # (B,) float, per-row bias (0.0 for non-watermarked)
    hash_key: torch.Tensor,
) -> torch.Tensor:
    """Lefthash: additive PRF (one seed per row) + Philox + green-list bias."""
    B, V = logits.shape
    device = logits.device
    seeds = hash_key * context.sum(dim=1)  # (B,)
    positions = torch.arange(V, device=device, dtype=torch.long)
    call_idx = (positions // 4).to(torch.int32)
    word_idx = (positions % 4).to(torch.int32)
    k0 = (seeds & _MASK32).to(torch.int32).unsqueeze(1).expand(-1, V)
    k1 = ((seeds >> 32) & _MASK32).to(torch.int32).unsqueeze(1).expand(-1, V)
    zeros = torch.zeros(B, V, device=device, dtype=torch.int32)
    r0, r1, r2, r3 = _sbw_philox_compiled(
        call_idx.unsqueeze(0).expand(B, -1),
        zeros,
        zeros.clone(),
        zeros.clone(),
        k0,
        k1,
    )
    sel = torch.where(
        word_idx == 0,
        r0,
        torch.where(word_idx == 1, r1, torch.where(word_idx == 2, r2, r3)),
    )
    logits = logits + (
        ((sel & _INT31_MAX) < gamma_int).to(logits.dtype) * delta.unsqueeze(-1)
    )
    return logits


_sbw_lefthash_compiled = torch.compile(
    _sbw_lefthash_inner, mode="max-autotune-no-cudagraphs"
)


# ---------------------------------------------------------------------------
# SBWWatermarker
# ---------------------------------------------------------------------------


class SBWWatermarker(Watermarker):
    """Stateless Bernoulli Watermarker.

    Adds a logit bias of +delta to green tokens determined by an independent
    Bernoulli trial per token (fraction gamma), seeded by the prior context
    via selfhash (anchored minhash PRF) or lefthash (additive PRF).  Normal
    sampling then runs on the biased logits.

    Args:
        key: 64-bit integer secret key.
        context_width: Number of prior tokens used as PRF context.
            Defaults: selfhash → 4, lefthash → 1 (matching the sbw library).
        scheme: "selfhash" (default) or "lefthash".
        gamma: Green-list fraction in (0, 1).  Default 0.5.
        delta: Logit bias added to green tokens.  Default 2.0.
    """

    def __init__(
        self,
        key: int,
        context_width: int | None = None,
        scheme: SBWScheme = "selfhash",
        gamma: float = 0.5,
        delta: float = 2.0,
    ) -> None:
        if not 0 <= key <= 2**64 - 1:
            raise ValueError("SBW key must fit in 64 bits")
        if scheme not in ("selfhash", "lefthash"):
            raise ValueError(
                f"SBW scheme must be 'selfhash' or 'lefthash', got {scheme!r}"
            )
        if not 0.0 < gamma < 1.0:
            raise ValueError(f"gamma must be in (0, 1), got {gamma}")
        if delta < 0.0:
            raise ValueError(f"delta must be >= 0, got {delta}")
        if context_width is None:
            context_width = _SBW_DEFAULT_CONTEXT_WIDTH[scheme]
        if context_width < 1:
            raise ValueError(f"context_width must be >= 1, got {context_width}")

        self.scheme = scheme
        self.gamma = gamma
        self.delta = delta
        self._context_width = context_width
        # Precompute int31 threshold for Philox comparison.
        self._gamma_int = int(gamma * _INT31_MAX)
        # Store key as a tensor so torch.compile sees a stable input.
        self._key = key  # kept for repr / detection
        # Cached device tensors, created lazily on first sample() call and
        # reused every subsequent step.  torch.tensor(..., device="cuda") on
        # the hot path blocks until the GPU is idle, adding ~14ms per step.
        self._hash_key_t: torch.Tensor | None = None
        self._gamma_int_t: torch.Tensor | None = None

    @property
    def context_width(self) -> int:
        return self._context_width

    @property
    def supports_greedy(self) -> bool:
        return True  # bias-based: +delta shifts the argmax toward green tokens

    def __repr__(self) -> str:
        return (
            f"SBWWatermarker(key=<redacted>, scheme={self.scheme!r}, "
            f"context_width={self._context_width}, "
            f"gamma={self.gamma}, delta={self.delta})"
        )

    def _ensure_key_tensors(self, device: torch.device) -> None:
        """Lazily create cached scalar tensors on the target device."""
        if self._hash_key_t is None or self._hash_key_t.device != device:
            self._hash_key_t = torch.tensor(self._key, dtype=torch.int64, device=device)
            self._gamma_int_t = torch.tensor(
                self._gamma_int, dtype=torch.int64, device=device
            )

    def bias(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        delta_vec: torch.Tensor,
    ) -> torch.Tensor:
        """Apply per-row SBW green-list bias to logits and return the result.

        Args:
            logits: (B, V) logit tensor. May be fp16/bf16; bias is applied
                in the tensor's native dtype.
            contexts: (B, context_width) int64 context tensor.
            delta_vec: (B,) float tensor of per-row bias values. Pass
                self.delta for watermarked rows and 0.0 for non-watermarked
                rows so that non-watermarked requests are unaffected.

        Returns:
            Biased logits tensor (same shape and dtype as input).
        """
        self._ensure_key_tensors(logits.device)
        assert self._gamma_int_t is not None
        assert self._hash_key_t is not None
        delta_vec = delta_vec.to(logits.dtype)
        if self.scheme == "selfhash":
            return _sbw_selfhash_compiled(
                contexts, logits, self._gamma_int_t, delta_vec, self._hash_key_t
            )
        else:
            return _sbw_lefthash_compiled(
                contexts, logits, self._gamma_int_t, delta_vec, self._hash_key_t
            )

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sampler: RandomSampler | None = None,
        skip_mask: torch.Tensor | None = None,
    ) -> WatermarkSample:
        """Bias logits toward the green list then delegate to random_sampler.

        In the GPU serving path, bias is applied upstream (before top-k/top-p)
        via bias(). This method is retained for the spec-decode draft path and
        CPU-only unit tests where top-k/top-p is not applied.
        """
        B = logits.shape[0]
        device = logits.device
        if skip_mask is not None:
            # Per-row delta: 0.0 for rows that should not be watermarked.
            delta_vec = torch.where(
                skip_mask,
                torch.zeros(1, dtype=logits.dtype, device=device),
                torch.full((1,), self.delta, dtype=logits.dtype, device=device),
            )
        else:
            delta_vec = torch.full((B,), self.delta, dtype=logits.dtype, device=device)

        biased = self.bias(logits, contexts, delta_vec)

        if random_sampler is not None:
            token_ids = random_sampler(biased)
        else:
            token_ids = biased.argmax(dim=-1)

        return WatermarkSample(token_ids=token_ids, logits=biased)

    def _sample_watermarked(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> WatermarkSample:
        # SBW overrides sample() directly so this path is never reached via
        # the base class. Implemented only to satisfy the ABC.
        return self.sample(logits, contexts, random_sampler=None, skip_mask=None)


# ---------------------------------------------------------------------------
# SBWWatermarkDetector
# ---------------------------------------------------------------------------


class SBWWatermarkDetector(WatermarkDetector):
    """Detects SBW watermarks using the standard KGW z-score test.

    For each token at position t, checks whether it belongs to the green list
    defined by (context_t, scheme, key, gamma).  Counts green tokens W over T
    scored positions and computes:

        z = (W - T * gamma) / sqrt(T * gamma * (1 - gamma))

    which is N(0,1) under the null (unwatermarked text).

    Args:
        key: Same 64-bit integer key used during generation.
        context_width: Same context_width used during generation.
        scheme: Same scheme ("selfhash" or "lefthash") used during generation.
        gamma: Same gamma used during generation.
        p_value_threshold: Reject H0 (no watermark) when p-value < threshold.
        deduplicate_contexts: Skip tokens whose PRF context was already scored
            (reduces false positives from repeated contexts).
    """

    def __init__(
        self,
        key: int,
        context_width: int | None = None,
        scheme: SBWScheme = "selfhash",
        gamma: float = 0.5,
        p_value_threshold: float = 0.01,
        deduplicate_contexts: bool = True,
    ) -> None:
        if context_width is None:
            context_width = _SBW_DEFAULT_CONTEXT_WIDTH[scheme]
        super().__init__(context_width, p_value_threshold, deduplicate_contexts)
        if scheme not in ("selfhash", "lefthash"):
            raise ValueError(
                f"SBW scheme must be 'selfhash' or 'lefthash', got {scheme!r}"
            )
        if not 0.0 < gamma < 1.0:
            raise ValueError(f"gamma must be in (0, 1), got {gamma}")
        self.scheme = scheme
        self.gamma = gamma
        self._key = key
        self._gamma_int = int(gamma * _INT31_MAX)
        # Cached scalar tensors, same rationale as SBWWatermarker.
        self._hash_key_t: torch.Tensor | None = None
        self._gamma_int_t: torch.Tensor | None = None

    def _get_key_tensors(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._hash_key_t is None or self._hash_key_t.device != device:
            self._hash_key_t = torch.tensor(self._key, dtype=torch.int64, device=device)
            self._gamma_int_t = torch.tensor(
                self._gamma_int, dtype=torch.int64, device=device
            )
        assert self._gamma_int_t is not None
        return self._hash_key_t, self._gamma_int_t

    def _score_tokens(
        self, contexts: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Return 1.0 for each green token, 0.0 for red.

        Exploits the counter-based nature of Philox: evaluates the PRF exactly
        once per (context_i, target_i) pair at counter=target_i, without
        computing the full (B, V) green list.  O(B) Philox evaluations instead
        of O(B * V).
        """
        if len(targets) == 0:
            return torch.zeros(0, dtype=torch.float64)

        device = contexts.device
        hash_key, gamma_int = self._get_key_tensors(device)
        targets = targets.to(device=device, dtype=torch.long)
        B = contexts.shape[0]

        # --- compute one seed per row ---
        if self.scheme == "selfhash":
            h_v = _sbw_hashint(targets)  # (B,)
            cand_val = hash_key * h_v * h_v  # (B,)
            prefix = contexts[:, 1:]  # (B, H-1)
            if prefix.shape[1] > 0:
                h_prefix = _sbw_hashint(prefix)  # (B, H-1)
                prefix_terms = hash_key * h_prefix * h_v.unsqueeze(1)  # (B, H-1)
                prefix_min = prefix_terms.min(dim=1).values  # (B,)
                seeds = torch.minimum(prefix_min, cand_val)  # (B,)
            else:
                seeds = cand_val  # (B,)
        else:
            # lefthash: seed depends only on context, not on the token
            seeds = hash_key * _sbw_hashint(contexts).sum(dim=1)  # (B,)

        # --- evaluate Philox at counter = target_i ---
        call_idx = (targets // 4).to(torch.int32)  # (B,)
        word_idx = targets % 4  # (B,) int64
        k0 = (seeds & _MASK32).to(torch.int32)  # (B,)
        k1 = ((seeds >> 32) & _MASK32).to(torch.int32)  # (B,)
        zeros = torch.zeros(B, dtype=torch.int32, device=device)
        r0, r1, r2, r3 = _sbw_philox_compiled(
            call_idx, zeros, zeros.clone(), zeros.clone(), k0, k1
        )
        # Select the right word for each row based on target % 4
        raw = torch.where(
            word_idx == 0,
            r0,
            torch.where(word_idx == 1, r1, torch.where(word_idx == 2, r2, r3)),
        )  # (B,) int32

        green = ((raw & _INT31_MAX) < gamma_int).to(torch.float64)
        return green

    def _aggregate_scores(self, token_scores: torch.Tensor) -> float:
        return token_scores.sum().item()

    def _get_p_value(self, score: float, num_scored_tokens: int) -> float:
        """One-sided z-test p-value: P(Z >= z) under N(0,1)."""
        import math

        T = num_scored_tokens
        if T == 0:
            return 1.0
        W = score
        z = (W - T * self.gamma) / math.sqrt(T * self.gamma * (1 - self.gamma))
        # P(Z >= z) via complementary error function
        return 0.5 * math.erfc(z / math.sqrt(2))
