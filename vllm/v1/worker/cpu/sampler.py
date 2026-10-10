# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU sampler that samples top-k/top-p over a small candidate set.

The GPU path masks the full vocab (a full sort for top-p) and then takes a
Gumbel-max over it. Every token that survives top-k/top-p is among the
highest-logit tokens, so when a row's top `NUM_CANDIDATES` tokens contain its
whole kept set, the Gumbel-max over just those candidates is the same token.
The murmur3 noise is addressed by token id, so it is drawn for the candidates
only and matches the full-vocab draw bit for bit. Rows whose kept set may not
fit fall back to the full path; checking that is free since CPU tensors are
host memory.
"""

import numpy as np
import torch

from vllm.config.model import PROCESSED_LOGPROBS_MODES
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.cpu.kernels.gumbel import gumbel_noise
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.sampler import Sampler

NUM_CANDIDATES = 1024


def _sample_candidates(
    logits: torch.Tensor,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
    temp: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    use_fp64: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the sampled token ids and which rows the candidates cover."""
    num_tokens, vocab_size = logits.shape
    num_candidates = min(NUM_CANDIDATES, vocab_size)
    values, token_ids = logits.topk(num_candidates, dim=-1)

    keep = torch.ones_like(values, dtype=torch.bool)
    has_top_k = torch.zeros(num_tokens, dtype=torch.bool)
    covered = torch.zeros(num_tokens, dtype=torch.bool)
    if top_k is not None:
        has_top_k = top_k < vocab_size
        # Tokens tied with the k-th largest logit are kept too.
        kth_rank = (top_k.long() - 1).clamp(max=num_candidates - 1).unsqueeze(1)
        kth_value = values.gather(1, kth_rank)
        keep &= values >= kth_value
        covered = (top_k <= num_candidates) & (values[:, -1:] < kth_value).squeeze(1)
    if top_p is not None:
        # Top-p normalizes over the top-k survivors, else over the full vocab.
        log_z = torch.where(
            has_top_k,
            torch.where(keep, values, float("-inf")).logsumexp(-1),
            logits.logsumexp(-1),
        )
        probs = torch.where(keep, (values - log_z.unsqueeze(1)).exp(), 0.0)
        cum_probs = probs.cumsum(-1)
        keep &= (cum_probs - probs) < top_p.unsqueeze(1)
        # The first non-candidate is dropped once the candidates hold p.
        covered |= ~has_top_k & (cum_probs[:, -1] >= top_p)

    scores = torch.where(keep, values, float("-inf"))
    is_sampled = (temp != 0.0).unsqueeze(1)
    if use_fp64:
        scores = scores.to(torch.float64)
    noise = gumbel_noise(seed.unsqueeze(1), pos.unsqueeze(1), token_ids, use_fp64)
    scores = torch.where(is_sampled, scores + noise, scores)
    # Break ties toward the lower token id, like argmax over the full vocab.
    is_max = scores == scores.max(-1, keepdim=True).values
    sampled = torch.where(is_max, token_ids, vocab_size).min(-1).values
    return sampled, covered


def sample_top_k_top_p(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    pos: torch.Tensor,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
    use_fp64: bool,
) -> torch.Tensor:
    """Same tokens as `apply_top_k_top_p` followed by `gumbel_sample`."""
    req = expanded_idx_mapping.long()
    sampled, covered = _sample_candidates(
        logits.float(), top_k, top_p, temperature[req], seeds[req], pos, use_fp64
    )
    if not bool(covered.all()):
        rows = (~covered).nonzero().squeeze(1)
        fallback_logits = apply_top_k_top_p(
            logits[rows].float(),
            None if top_k is None else top_k[rows],
            None if top_p is None else top_p[rows],
        )
        sampled[rows] = gumbel_sample(
            fallback_logits,
            expanded_idx_mapping[rows],
            temperature,
            seeds,
            pos[rows],
            apply_temperature=False,
            is_drafting=False,
            use_fp64=use_fp64,
        )
    return sampled


class CPUSampler(Sampler):
    def sample(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        seq_lens_upper_bound_np: np.ndarray,
        return_logprobs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The candidate path never materializes the top-k/top-p masked logits.
        needs_masked_logits = self.return_sampling_mask or (
            return_logprobs and self.logprobs_mode in PROCESSED_LOGPROBS_MODES
        )
        if needs_masked_logits:
            return super().sample(
                logits,
                expanded_idx_mapping,
                idx_mapping,
                idx_mapping_np,
                pos,
                input_ids,
                expanded_local_pos,
                seq_lens_upper_bound_np,
                return_logprobs,
            )

        processed_logits = self.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
            seq_lens_upper_bound_np,
            skip_top_k_top_p=True,
        )
        top_k, top_p = self.sampling_states.get_top_k_top_p(
            expanded_idx_mapping, idx_mapping_np
        )
        if top_k is None and top_p is None:
            return self._sample_random(
                processed_logits,
                expanded_idx_mapping,
                idx_mapping_np,
                pos,
                top_k=None,
                top_p=None,
                use_fused_sampler=False,
            )

        sampled = sample_top_k_top_p(
            processed_logits,
            expanded_idx_mapping,
            self.sampling_states.temperature.gpu,
            self.sampling_states.seeds.gpu,
            pos,
            top_k,
            top_p,
            self.use_fp64_gumbel,
        )
        return sampled, processed_logits
