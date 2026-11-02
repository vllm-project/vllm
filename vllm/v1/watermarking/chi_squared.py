# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Chi-squared watermark generation, adapted from ``ChiTwoWatermark`` in the
reference implementation of Gloaguen et al. (2026)."""

import torch

from vllm.config.watermarking import WatermarkPRFName
from vllm.v1.watermarking.binomial import (
    DEFAULT_NUM_TRIALS,
    BinomialScoreWatermarker,
    copy_to_device,
)
from vllm.v1.watermarking.prfs import PhiloxPRF, WatermarkPRF
from vllm.v1.watermarking.watermarker import RandomSampler, WatermarkSample


class ChiSquaredWatermarker(BinomialScoreWatermarker):
    """Chi-squared watermark of Gloaguen et al. (2026).

    The weights ``w_u = max(1 + sigma * (g_u + mu), 1e-30)`` make
    ``q_u = p_u * w_u`` the distribution maximizing the expected score of the
    sampled token minus a chi-squared penalty ``chi2(q, p) / (2 * sigma)``.
    ``g`` are the tokens' binomial scores, ``p`` is the model's distribution
    after temperature scaling and top-k/top-p filtering, ``mu`` normalizes
    ``q``, and ``sigma`` is the configured ``delta``. Tokens are sampled from
    ``q``, as in the paper (Theorem D.6). The first ``context_width`` tokens of
    an output, which lack a full context, are not watermarked.

    Differences from the reference implementation: the reference adds
    ``log w`` to the logits before temperature scaling, so it samples in
    proportion to ``p_u * w_u ** (1 / T)`` rather than from ``q``;
    ``_find_mu`` runs on the total probability of each score level rather
    than on single tokens, which gives the same ``mu`` because tokens with
    equal scores get equal weights; and the reference's shortcut for
    one-trial scores is not needed, since ``_find_mu`` covers that case.
    """

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        prf: WatermarkPRF | WatermarkPRFName = "philox",
        delta: float = 1.0,
        num_trials: int = DEFAULT_NUM_TRIALS,
    ) -> None:
        if not delta > 0:
            raise ValueError("delta must be positive")
        super().__init__(key, context_width, prf, num_trials)

        # Scheme parameters
        self.sigma = delta
        self._sigma_tensor = torch.tensor([delta], dtype=torch.float64)
        self._device_sigma: dict[torch.device, torch.Tensor] = {}

    def _find_mu(
        self, p: torch.Tensor, G: torch.Tensor, tol: float = 1e-12
    ) -> torch.Tensor:
        """Solve ``sum(p * max(0, 1 + sigma * (G + mu))) = 1`` for ``mu`` per row.

        Args:
            p: (B, N) Tensor of probabilities/weights
            G: (B, N) Tensor of values
            tol: Tolerance for numerical stability

        Returns:
            mu: (B,) Tensor of solutions

        """
        B, N = p.shape
        sigma = self.sigma

        tau = -1.0 - sigma * G  # Shape (B, N)

        tau_s, indices = torch.sort(tau, dim=1)
        p_s = torch.gather(p, 1, indices)
        G_s = torch.gather(G, 1, indices)

        P_cum = torch.cumsum(p_s, dim=1)

        term = p_s * (1.0 + sigma * G_s)
        A_cum = torch.cumsum(term, dim=1)

        U_candidates = (1.0 - A_cum) / (P_cum + 1e-18)

        cond_positive = P_cum > 1e-18
        cond_left = U_candidates >= (tau_s - tol)

        infinity = torch.full((B, 1), float("inf"), device=p.device, dtype=p.dtype)
        tau_next = torch.cat([tau_s[:, 1:], infinity], dim=1)
        cond_right = U_candidates <= (tau_next + tol)

        is_valid = cond_positive & cond_left & cond_right

        first_valid_idx = torch.argmax(is_valid.long(), dim=1)  # Shape (B,)
        row_has_solution = is_valid.any(dim=1)

        final_idx = torch.where(row_has_solution, first_valid_idx, N - 1)

        U_selected = torch.gather(U_candidates, 1, final_idx.unsqueeze(1)).squeeze(1)

        return U_selected / sigma

    def adjusted_scores(self, level_mass: torch.Tensor) -> torch.Tensor:
        """Return ``log(max(1 + sigma * (s + mu), 1e-30))`` for each row and
        score level ``s``, given the total probability ``level_mass[row, s]`` of
        the tokens scoring ``s``."""
        probs = level_mass.double()
        scores = torch.arange(
            probs.shape[-1], dtype=probs.dtype, device=probs.device
        ).expand_as(probs)
        mu = self._find_mu(probs, scores)

        min_value = 1e-30
        adjusted_scores = torch.log(
            torch.clamp(1 + self.sigma * (scores + mu.unsqueeze(-1)), min=min_value)
        )
        return adjusted_scores.float()

    def watermark_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        skip_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return float32 logits of the watermarked distribution of every row
        not in ``skip_mask``."""
        if type(self.prf) is PhiloxPRF and logits.device.type == "cuda":
            from vllm.v1.worker.gpu.sample.binomial_watermark import (
                philox_chi_squared_weights,
            )

            adjusted_scores = philox_chi_squared_weights(
                logits.float(),
                contexts,
                self.prf.key,
                self._thresholds_on(logits.device),
                copy_to_device(self._sigma_tensor, self._device_sigma, logits.device),
            )
        else:
            log_probs = logits.float().log_softmax(dim=-1)
            adjusted_scores = self.adjusted_scores(self.level_mass(log_probs, contexts))
        return self.add_level_bias(logits, contexts, adjusted_scores, skip_mask)

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sampler: RandomSampler | None = None,
        skip_mask: torch.Tensor | None = None,
    ) -> WatermarkSample:
        if random_sampler is None:
            raise ValueError("chi-squared watermarking requires a random sampler")
        skip_mask = self._skip_partial_contexts(contexts, skip_mask)
        watermarked = self.watermark_logits(logits, contexts, skip_mask)
        # The logits before the watermark, as for Gumbel, so processed logprobs
        # do not reveal the scores.
        return WatermarkSample(random_sampler(watermarked), logits)

    def _sample_watermarked(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> WatermarkSample:
        raise ValueError("chi-squared watermarking requires a random sampler")
