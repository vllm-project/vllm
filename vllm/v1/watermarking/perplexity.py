# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Perplexity-constrained watermark generation, adapted from
``PPLSingleFastWatermark`` (``ppl_hard``) and ``PPLWatermark`` (``ppl_soft``) in
the reference implementation of Gloaguen et al. (2026)."""

import math

import torch

from vllm.config.watermarking import WatermarkPRFName
from vllm.v1.watermarking.binomial import (
    DEFAULT_NUM_TRIALS,
    BinomialScoreWatermarker,
    copy_to_device,
)
from vllm.v1.watermarking.prfs import PhiloxPRF, WatermarkPRF
from vllm.v1.watermarking.watermarker import RandomSampler, WatermarkSample

_MONTE_CARLO_KEY = 0x50504C5F4D435F31  # Public: the draws need no secrecy.
# The Monte Carlo draws score this many of the most probable tokens. Less
# probable tokens outscore the top ones only for very flat distributions, and
# sorting just these is much cheaper.
_NUM_RANKED_TOKENS = 4096
# The bisection for beta, as in the reference: 60 steps over [0, 100].
_BETA_MAX = 1e2
_BISECTION_STEPS = 60


def _validate_epsilon(epsilon: float) -> None:
    if not epsilon >= 0:
        raise ValueError("epsilon must be non-negative")


def convex_hull_algorithm_vectorized(
    probs: torch.Tensor, scores: torch.Tensor, epsilon: float, H_p: torch.Tensor
) -> torch.Tensor:
    """Highly optimized vectorized convex hull algorithm.

    Uses Pareto-optimality filtering to reduce the effective problem size
    before running the hull construction loop.

    Adapted for vLLM: ``probs`` and ``scores`` hold only each row's candidate
    tokens (see ``hull_candidates``), so the entropy ``H_p`` of the full
    distribution is passed in, and candidates with zero probability are
    ignored. Nothing is read back to the host: the width stays at the number
    of candidates instead of the widest row's Pareto points, and the hull
    keeps exactly the points the monotone chain keeps (it pops a point when
    the turn through it is left or straight), found for all points at once.
    """
    B, V = probs.shape
    device = probs.device

    # --- 1. Preprocessing ---
    probs_safe = probs.clamp(min=1e-30)
    c = -torch.log(probs_safe)

    # Calculate constraint target C
    C = (H_p + epsilon).unsqueeze(1).to(c.dtype)

    # Candidates without probability are not tokens of the distribution.
    scores = torch.where(probs > 0, scores, -1)

    # Sort by cost (ascending)
    c_sorted, idx_sorted = torch.sort(c, dim=1, stable=True)
    scores_sorted = torch.gather(scores, 1, idx_sorted)

    # --- 2. Optimization: Pareto Filtering ---
    # We only care about points where score increases as cost increases.
    # If score drops, that point is dominated by previous cheaper points.

    # Compute running max score along the sorted sequence
    running_max, _ = torch.cummax(scores_sorted, dim=1)

    # Keep points that equal (or improve) the running max
    is_pareto = (scores_sorted >= running_max) & (scores_sorted >= 0)

    # Compress the tensors: Move valid points to the left
    # We create sort keys: Valid indices keep their value, Invalid ones get pushed to V+
    range_indices = torch.arange(V, device=device).unsqueeze(0).expand(B, V)
    sort_keys = torch.where(is_pareto, range_indices, range_indices + V)

    # Sort to cluster valid points at the start
    _, compressed_order = torch.sort(sort_keys, dim=1)

    num_valid = is_pareto.sum(dim=1)
    # Every candidate column stays, so no row's count is read back.
    V_eff = V

    active_indices = compressed_order[:, :V_eff]

    # Gather the subset of costs/scores to process
    c_active = torch.gather(c_sorted, 1, active_indices)
    s_active = torch.gather(scores_sorted, 1, active_indices).to(c.dtype)

    # We also need to mask out the "garbage" padding at the end of rows with
    # fewer than V_eff points
    point_range = torch.arange(V_eff, device=device)
    valid_mask = point_range.unsqueeze(0) < num_valid.unsqueeze(1)

    # --- 3. Vectorized upper hull (on reduced set) ---
    # The monotone chain pops point k when its turn is left or straight, i.e.
    # when the slope into k is at most the slope out of it. The points it
    # keeps are those whose slope in from every earlier point exceeds their
    # slope out to every later point.
    pairs = (
        valid_mask.unsqueeze(2)
        & valid_mask.unsqueeze(1)
        & (point_range.unsqueeze(1) < point_range.unsqueeze(0))
    )
    # slopes[b, a, k]: slope from point a to point k.
    slopes = (s_active.unsqueeze(1) - s_active.unsqueeze(2)) / (
        c_active.unsqueeze(1) - c_active.unsqueeze(2)
    )
    slope_in = torch.where(pairs, slopes, float("inf")).amin(dim=1)
    slope_out = torch.where(pairs, slopes, float("-inf")).amax(dim=2)
    on_hull = valid_mask & (slope_in > slope_out)

    # The stack holds the hull points in order, indices into the active arrays.
    hull_keys = torch.where(on_hull, point_range, point_range + V_eff)
    _, hull_stack = torch.sort(hull_keys, dim=1)
    stack_ptr = on_hull.sum(dim=1)

    # --- 4. Find Intersection with C ---
    # Gather hull costs
    # We need to handle the "padding" in the hull_stack (indices beyond stack_ptr)
    col_idx = torch.arange(V_eff, device=device).unsqueeze(0).expand(B, V_eff)
    in_hull_mask = col_idx < stack_ptr.unsqueeze(1)

    hull_indices = hull_stack
    hull_costs = torch.gather(c_active, 1, hull_indices)

    # Pad invalid hull entries with infinity for searchsorted
    max_val = torch.finfo(hull_costs.dtype).max
    hull_costs_padded = hull_costs.masked_fill(~in_hull_mask, max_val)

    # Binary search for C
    target_idx = torch.searchsorted(hull_costs_padded, C)

    # Identify segment [i, j]
    idx_j_stack = target_idx.squeeze(1).clamp(max=V_eff - 1)
    idx_i_stack = (idx_j_stack - 1).clamp(min=0)

    # Handle C > max_cost (clamp to last valid point)
    last_valid = (stack_ptr - 1).clamp(min=0)
    use_last = idx_j_stack > last_valid
    idx_j_stack = torch.where(use_last, last_valid, idx_j_stack)
    idx_i_stack = torch.where(use_last, last_valid, idx_i_stack)

    batch_range = torch.arange(B, device=device)

    # Get indices into 'active' array
    idx_active_i = hull_stack[batch_range, idx_i_stack]
    idx_active_j = hull_stack[batch_range, idx_j_stack]

    # Get values
    c_i = c_active[batch_range, idx_active_i]
    c_j = c_active[batch_range, idx_active_j]

    # Calculate Alpha
    C_flat = C.squeeze(1)
    denom = c_j - c_i
    is_same = denom.abs() < 1e-20
    alpha = (c_j - C_flat) / (denom + 1e-30)
    alpha = torch.where(is_same, torch.ones_like(alpha), alpha)
    alpha = alpha.clamp(0.0, 1.0)

    # --- 5. Map back to original indices ---
    # 1. Map active_indices -> sorted_indices
    sorted_idx_i = torch.gather(active_indices, 1, idx_active_i.unsqueeze(1)).squeeze(1)
    sorted_idx_j = torch.gather(active_indices, 1, idx_active_j.unsqueeze(1)).squeeze(1)

    # 2. Map sorted_indices -> original vocabulary indices
    orig_i = idx_sorted[batch_range, sorted_idx_i]
    orig_j = idx_sorted[batch_range, sorted_idx_j]

    # 3. Scatter probabilities
    best_q = torch.zeros_like(probs)
    best_q.scatter_add_(1, orig_i.unsqueeze(1), alpha.unsqueeze(1))
    best_q.scatter_add_(1, orig_j.unsqueeze(1), (1.0 - alpha).unsqueeze(1))

    return best_q


def hull_candidates(
    level_log_probs: torch.Tensor,
    level_token_ids: torch.Tensor,
    least_log_probs: torch.Tensor,
    least_token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the probabilities (float64), scores and token ids of the tokens
    that can be on a row's hull: the most probable token of each score, and
    the least probable token of the highest score, where the hull's flat top
    ends. Every other token is dominated by a more probable one with the same
    score. The extra candidate gets probability zero when it is no less
    probable than the top score's most probable token."""
    num_rows, num_levels = level_log_probs.shape
    levels = torch.arange(num_levels, device=level_log_probs.device).expand(
        num_rows, -1
    )
    top_score = torch.where(level_log_probs > float("-inf"), levels, -1).amax(
        dim=-1, keepdim=True
    )
    top_log_probs = level_log_probs.gather(1, top_score.clamp(min=0))
    least_log_probs = least_log_probs.unsqueeze(1)
    least_log_probs = torch.where(
        least_log_probs < top_log_probs, least_log_probs, float("-inf")
    )
    log_probs = torch.cat([level_log_probs, least_log_probs], dim=1).double()
    scores = torch.cat([levels, top_score], dim=1)
    token_ids = torch.cat([level_token_ids, least_token_ids.unsqueeze(1)], dim=1)
    return log_probs.exp(), scores, token_ids


class PPLHardWatermarker(BinomialScoreWatermarker):
    """Perplexity watermark with a hard constraint (Gloaguen et al., 2026),
    adapted from ``PPLSingleFastWatermark``.

    For every context, samples from the distribution ``q`` with the highest
    expected binomial score whose expected surprisal ``-q . log p`` exceeds the
    model's entropy by at most ``epsilon`` nats, where ``p`` is the model's
    distribution after temperature scaling and top-k/top-p filtering. ``q`` puts
    its mass on one or two tokens. As in the reference, when several tokens
    share the highest score and the budget allows, ``q`` spends the rest of the
    budget on the least probable of them.
    """

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        prf: WatermarkPRF | WatermarkPRFName = "philox",
        epsilon: float = 0.0,
        num_trials: int = DEFAULT_NUM_TRIALS,
    ) -> None:
        _validate_epsilon(epsilon)
        super().__init__(key, context_width, prf, num_trials)

        # Scheme parameters
        self.epsilon = epsilon

    def convex_hull_algorithm(
        self, log_probs: torch.Tensor, contexts: torch.Tensor
    ) -> torch.Tensor:
        """Return each row's watermarked distribution, [num_rows, vocab_size],
        for float32 ``log_probs``."""
        probs = log_probs.exp()
        probs_safe = probs.clamp(min=1e-30)
        H_p = -(probs_safe * torch.log(probs_safe)).sum(dim=1)
        candidate_probs, candidate_scores, candidate_token_ids = hull_candidates(
            *self.level_best_and_least_probable_top(log_probs, contexts)
        )
        best_q = convex_hull_algorithm_vectorized(
            candidate_probs, candidate_scores, self.epsilon, H_p
        )
        return torch.zeros_like(probs).scatter_add_(
            1, candidate_token_ids, best_q.to(probs.dtype)
        )

    def watermark_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        skip_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return float32 logits of the watermarked distribution of every row
        not in ``skip_mask``; they are -inf outside its one or two tokens."""
        if type(self.prf) is PhiloxPRF and logits.device.type == "cuda":
            from vllm.v1.worker.gpu.sample.binomial_watermark import (
                philox_ppl_hard_hull,
            )

            best_q, token_ids = philox_ppl_hard_hull(
                logits.float(),
                contexts,
                self.prf.key,
                self._thresholds_on(logits.device),
                self.epsilon,
            )
            log_q = best_q.float().log()
            if skip_mask is None:
                updated_log_probs = torch.full_like(
                    logits, float("-inf"), dtype=torch.float32
                )
            else:
                skip = skip_mask.unsqueeze(-1)
                updated_log_probs = logits.float().masked_fill(~skip, float("-inf"))
                log_q = log_q.masked_fill(skip, float("-inf"))
            # Empty score levels repeat token 0 with q = 0, hence amax.
            return updated_log_probs.scatter_reduce_(1, token_ids, log_q, "amax")

        probs = self.convex_hull_algorithm(logits.float().log_softmax(dim=-1), contexts)

        # Map this back in logits space
        updated_log_probs = torch.log(probs)
        if skip_mask is not None:
            updated_log_probs = torch.where(
                skip_mask.unsqueeze(-1), logits.float(), updated_log_probs
            )
        return updated_log_probs

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sampler: RandomSampler | None = None,
        skip_mask: torch.Tensor | None = None,
    ) -> WatermarkSample:
        if random_sampler is None:
            raise ValueError("hard perplexity watermarking requires a random sampler")
        skip_mask = self._skip_partial_contexts(contexts, skip_mask)
        if (
            type(self.prf) is PhiloxPRF
            and logits.device.type == "cuda"
            and isinstance(random_sampler, RandomSampler)
            and not random_sampler.is_drafting
            and random_sampler.logits_cache is None
        ):
            from vllm.v1.worker.gpu.sample.binomial_watermark import (
                philox_ppl_hard_sample,
            )

            # The token random_sampler would draw from watermark_logits,
            # without building them.
            token_ids = philox_ppl_hard_sample(
                logits.float(),
                contexts,
                self.prf.key,
                self._thresholds_on(logits.device),
                self.epsilon,
                skip_mask,
                random_sampler.expanded_idx_mapping,
                random_sampler.temperatures,
                random_sampler.seeds,
                random_sampler.positions,
                random_sampler.use_fp64,
            )
        else:
            token_ids = random_sampler(
                self.watermark_logits(logits, contexts, skip_mask)
            )
        # The logits before the watermark, as for Gumbel, so processed logprobs
        # do not reveal the scores.
        return WatermarkSample(token_ids, logits)

    def _sample_watermarked(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> WatermarkSample:
        raise ValueError("hard perplexity watermarking requires a random sampler")


def binomial_level_probs(num_trials: int) -> torch.Tensor:
    """Return the probability of each score under Binomial(num_trials, 1/2),
    in float32 as the Monte Carlo draws use it."""
    return torch.tensor(
        [
            math.comb(num_trials, level) / 2**num_trials
            for level in range(num_trials + 1)
        ],
        dtype=torch.float32,
    )


def first_score_ranks(
    uniforms: torch.Tensor, level_probs: torch.Tensor, width: int
) -> torch.Tensor:
    """Return the rank of the first token with each score in Monte Carlo draws
    of ``G``, or ``width`` when it is not among the first ``width`` tokens:
    int64 [..., num_levels] for ``uniforms`` [..., 2 * num_levels].

    In ``s.argmax`` over ``G + beta * logp``, with tokens in decreasing order
    of probability, only the first token with each score can win, so a draw
    of ``G`` matters only through these ranks. They are drawn directly, with
    the same distribution as from i.i.d. scores: after each new score, the
    number of tokens up to the next new one is geometric in the probability of
    the scores not seen yet, and the next new score is one of those, in
    proportion to its probability. ``uniforms[..., 2 * step]`` draws the gap
    and ``uniforms[..., 2 * step + 1]`` the score of the ``step``-th new score.

    Reference implementation, in float64; the parity tests check the Triton
    kernel against it.
    """
    num_levels = len(level_probs)
    uniforms = uniforms.double()
    shape = uniforms.shape[:-1]
    probs = level_probs.to(uniforms.device, torch.float64)
    levels = torch.arange(num_levels, device=uniforms.device)
    unseen = torch.ones((*shape, num_levels), dtype=torch.bool, device=uniforms.device)
    ranks = torch.full_like(unseen, width, dtype=torch.int64)
    rank = torch.full(shape, -1.0, dtype=torch.float64, device=uniforms.device)
    for step in range(num_levels):
        mass = torch.where(unseen, probs, 0.0)
        remaining = mass.sum(dim=-1)
        skip = torch.log(uniforms[..., 2 * step]) / torch.log1p(-remaining)
        skip = torch.where((skip > 1) & (step > 0), skip.ceil(), 1.0)
        rank = (rank + skip).clamp(max=width)
        threshold = uniforms[..., 2 * step + 1] * remaining
        passed = unseen & (mass.cumsum(dim=-1) > threshold.unsqueeze(-1))
        chosen = torch.where(passed, levels, num_levels).amin(dim=-1)
        # Rounding can leave the threshold past the last unseen score.
        last = torch.where(unseen, levels, -1).amax(dim=-1)
        chosen = torch.where(chosen < num_levels, chosen, last)
        hit = levels == chosen.unsqueeze(-1)
        ranks = torch.where(hit, rank.long().unsqueeze(-1), ranks)
        unseen &= ~hit
    return ranks


def _bisection_logic(
    nodes: torch.Tensor,
    target: torch.Tensor,
    eps: float,
    beta_max: float,
    max_iter: int,
) -> torch.Tensor:
    """Reference implementation; the parity tests check the Triton kernel against it."""
    levels = torch.arange(nodes.shape[-1], dtype=nodes.dtype, device=nodes.device)
    present = nodes > float("-inf")
    beta_low = torch.zeros_like(target)
    beta_high = torch.full_like(target, beta_max)
    for _ in range(max_iter):
        mid = (beta_low + beta_high) / 2.0
        s = torch.where(present, levels + mid[:, None, None] * nodes, float("-inf"))
        idx = s.argmax(dim=-1, keepdim=True)
        val = nodes.gather(-1, idx).squeeze(-1).mean(dim=-1) - target + eps
        beta_high = torch.where(val > 0, mid, beta_high)
        beta_low = torch.where(val < 0, mid, beta_low)
    return (beta_low + beta_high) / 2.0


class PPLSoftWatermarker(BinomialScoreWatermarker):
    """Perplexity watermark with a soft constraint (Gloaguen et al., 2026).

    Selects ``argmax_u g_u + beta * log max(p_u, 1e-12)`` deterministically,
    where ``g`` are the tokens' binomial scores and ``p`` is the model's
    distribution after temperature scaling and top-k/top-p filtering. ``beta``
    makes the selected token's log-probability average ``p . log p - epsilon``
    over random scores, estimated from ``n_mc`` Monte Carlo draws of the
    scores. A public PRF derives the draws from the context, so they are new
    for every new context. Only the 4096 most probable tokens are drawn
    and selected from; the target still uses the whole distribution.
    """

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        prf: WatermarkPRF | WatermarkPRFName = "philox",
        epsilon: float = 0.0,
        num_trials: int = DEFAULT_NUM_TRIALS,
        n_mc: int = 128,
    ) -> None:
        _validate_epsilon(epsilon)
        if n_mc < 1:
            raise ValueError("n_mc must be positive")
        # The bisection kernel tracks the scores drawn in a 64-bit mask.
        if num_trials > 62:
            raise ValueError("num_trials must be at most 62")
        super().__init__(key, context_width, prf, num_trials)

        # Scheme Parameters
        self.epsilon = epsilon
        self.n_mc = n_mc
        self.level_probs = binomial_level_probs(num_trials)
        self._device_level_probs: dict[torch.device, torch.Tensor] = {}
        self._monte_carlo_prf = PhiloxPRF(_MONTE_CARLO_KEY)

    def monte_carlo_uniforms(self, contexts: torch.Tensor) -> torch.Tensor:
        """Uniforms for ``n_mc`` Monte Carlo draws per row, [num_rows, n_mc,
        2 * num_levels] (see ``first_score_ranks``), from a public PRF keyed by
        each row's context."""
        num_uniforms = self.n_mc * 2 * len(self.level_probs)
        token_ids = torch.arange(num_uniforms, device=contexts.device)
        uniforms = self._monte_carlo_prf.uniform(contexts, token_ids)
        return uniforms.view(len(contexts), self.n_mc, -1)

    def find_beta_for_logp_constraint(
        self,
        log_probs: torch.Tensor,
        contexts: torch.Tensor,
        max_iter: int = _BISECTION_STEPS,
        ranked: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Solve for beta so that E_G[log p_{i*(G; beta)}] matches the target.

        target = sum_i p_i log p_i - epsilon, over the tokens with probability.

        Args:
            log_probs: float32 [num_rows, vocab_size] log-probabilities
            contexts: [num_rows, context_width] contexts, keying the MC draws
            max_iter: bisection iterations
            ranked: each row's ``min(vocab_size, 4096)`` largest
                log-probabilities in decreasing order, if already computed

        Returns:
            beta: [num_rows]

        """
        vocab_size = log_probs.shape[-1]
        eps = self.epsilon
        support = log_probs > float("-inf")

        # Ensure numerically safe logs, keep consistency for target computation
        p_safe = log_probs.exp().clamp_min(1e-12)
        target = torch.where(support, p_safe * p_safe.log(), 0.0).sum(dim=-1)

        # The MC draws are new for every new context, as the reference draws G
        # anew at every step. Only the first token with each score can win
        # s.argmax, so each draw is made as those tokens' ranks, and the
        # bisection compares just their log-probabilities.
        width = min(vocab_size, _NUM_RANKED_TOKENS)
        if ranked is None:
            ranked = log_probs.topk(width, dim=-1).values
        if log_probs.device.type == "cuda":
            from vllm.v1.worker.gpu.sample.binomial_watermark import (
                monte_carlo_bisection,
            )

            level_probs = copy_to_device(
                self.level_probs, self._device_level_probs, log_probs.device
            )
            beta, _ = monte_carlo_bisection(
                ranked,
                contexts,
                self._monte_carlo_prf.key,
                self.n_mc,
                level_probs,
                target,
                eps,
                _BETA_MAX,
                max_iter,
            )
            return beta
        logp = torch.where(
            ranked > float("-inf"), ranked.exp().clamp_min(1e-12).log(), float("-inf")
        )
        uniforms = self.monte_carlo_uniforms(contexts)
        ranks = first_score_ranks(uniforms, self.level_probs, width)
        nodes = logp.gather(1, ranks.clamp(max=width - 1).flatten(1))
        nodes = nodes.view(ranks.shape).masked_fill(ranks >= width, float("-inf"))
        return _bisection_logic(nodes, target, eps, _BETA_MAX, max_iter)

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sampler: RandomSampler | None = None,
        skip_mask: torch.Tensor | None = None,
    ) -> WatermarkSample:
        """As ``Watermarker.sample``. Rows without a full context are sampled
        by ``random_sampler``; without one, every row is watermarked."""
        if random_sampler is not None:
            skip_mask = self._skip_partial_contexts(contexts, skip_mask)
        if not (
            type(self.prf) is PhiloxPRF
            and logits.device.type == "cuda"
            and (
                (random_sampler is None and skip_mask is None)
                or (
                    isinstance(random_sampler, RandomSampler)
                    and not random_sampler.is_drafting
                    and random_sampler.logits_cache is None
                )
            )
        ):
            return super().sample(logits, contexts, random_sampler, skip_mask)

        from vllm.v1.worker.gpu.sample.binomial_watermark import (
            philox_ppl_soft_sample,
        )

        # _sample_watermarked in a few kernels, with the plain samples of the
        # rows in skip_mask drawn in the same pass over the vocabulary.
        width = min(logits.shape[-1], _NUM_RANKED_TOKENS)
        logits_f32 = logits.float()
        ranked = logits_f32.topk(width, dim=-1).values
        sampler: dict = {}
        if random_sampler is not None:
            sampler = dict(
                expanded_idx_mapping=random_sampler.expanded_idx_mapping,
                temperatures=random_sampler.temperatures,
                seeds=random_sampler.seeds,
                positions=random_sampler.positions,
                use_fp64=random_sampler.use_fp64,
            )
        token_ids = philox_ppl_soft_sample(
            logits_f32,
            contexts,
            self.prf.key,
            self._thresholds_on(logits.device),
            ranked,
            self._monte_carlo_prf.key,
            self.n_mc,
            copy_to_device(self.level_probs, self._device_level_probs, logits.device),
            self.epsilon,
            _BETA_MAX,
            _BISECTION_STEPS,
            skip_mask=skip_mask,
            **sampler,
        )
        return WatermarkSample(token_ids, logits)

    def _sample_watermarked(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> WatermarkSample:
        log_probs = logits.float().log_softmax(dim=-1)
        width = min(log_probs.shape[-1], _NUM_RANKED_TOKENS)
        ranked = log_probs.topk(width, dim=-1).values
        beta = self.find_beta_for_logp_constraint(log_probs, contexts, ranked=ranked)

        # Select among the tokens the draws score.
        log_probs = log_probs.masked_fill(log_probs < ranked[:, -1:], float("-inf"))

        # argmax_u scores_u + beta * log(max(p_u, 1e-12)) is the most probable
        # token of the score s maximizing s + beta * log(max(p_s, 1e-12)).
        level_log_probs, level_token_ids = self.level_best(log_probs, contexts)
        levels = torch.arange(
            level_log_probs.shape[-1], dtype=torch.float32, device=logits.device
        )
        scores = levels + beta.unsqueeze(-1) * torch.log(
            level_log_probs.exp().clamp_min(1e-12)
        )
        scores = torch.where(level_log_probs > float("-inf"), scores, float("-inf"))
        argmax_idx = torch.argmax(scores, dim=-1, keepdim=True)
        return WatermarkSample(
            level_token_ids.gather(1, argmax_idx).squeeze(-1), logits
        )
