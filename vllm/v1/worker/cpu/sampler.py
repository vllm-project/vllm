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
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.config.model import PROCESSED_LOGPROBS_MODES
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.cpu.kernels.gumbel import gumbel_argmax, gumbel_noise
from vllm.v1.worker.cpu.kernels.sample import token_logprobs, token_ranks
from vllm.v1.worker.cpu.logits_processors import (
    CPUBadWords,
    CPULogitBias,
    CPUPenalties,
)
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.logits_processor.interface import (
    LogitsProcessor,
    LogitsProcRequestState,
)
from vllm.v1.worker.gpu.sample.logprob import LogprobTokenIdsState
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import SamplingStates

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


def gumbel_sample(
    logits: torch.Tensor,
    req: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    pos: torch.Tensor,
    use_fp64: bool,
) -> torch.Tensor:
    """Full-vocab Gumbel-max with already temperature-scaled logits."""
    return gumbel_argmax(logits, temperature[req], seeds[req], pos, False, use_fp64)[1]


def sample_top_k_top_p(
    logits: torch.Tensor,
    req: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    pos: torch.Tensor,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
    use_fp64: bool,
) -> torch.Tensor:
    """Same tokens as `apply_top_k_top_p` followed by `gumbel_sample`."""
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
            fallback_logits, req[rows], temperature, seeds, pos[rows], use_fp64
        )
    return sampled


class CPUSamplingStates(SamplingStates):
    def apply_temperature(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        temp = self.temperature.gpu[expanded_idx_mapping].unsqueeze(1)
        # Greedy rows (temperature 0) are left unscaled.
        logits.div_(torch.where(temp == 0.0, 1.0, temp))

    def apply_min_p(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        if np.all(self.min_p.np[idx_mapping_np] == 0.0):
            return
        min_p = self.min_p.gpu[expanded_idx_mapping].unsqueeze(1)
        threshold = logits.amax(dim=1, keepdim=True) + min_p.log()
        logits.masked_fill_((min_p != 0.0) & (logits < threshold), float("-inf"))


class CPUSampler(Sampler):
    def __init__(
        self, vllm_config: VllmConfig, max_num_reqs: int, vocab_size: int, **kwargs
    ):
        super().__init__(vllm_config, max_num_reqs, vocab_size, **kwargs)
        self.sampling_states = CPUSamplingStates(max_num_reqs, vocab_size)
        # Swap in the CPU built-in processors, keeping pipeline order and any
        # custom processors.
        cpu_processors: dict[type, type[LogitsProcessor]] = {
            LogitBiasState: CPULogitBias,
            PenaltiesState: CPUPenalties,
            BadWordsState: CPUBadWords,
        }
        lp_req_state = LogitsProcRequestState.from_request_state(self.req_states)
        self.logits_processors = [
            cpu_processors[type(p)](vllm_config, lp_req_state)
            if type(p) in cpu_processors
            else p
            for p in self.logits_processors
        ]
        # The runner reads penalties_state.output_bin_counts, which is None.
        self.penalties_state = next(  # type: ignore[assignment]
            p for p in self.logits_processors if isinstance(p, CPUPenalties)
        )

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
        args = (
            expanded_idx_mapping.long(),
            self.sampling_states.temperature.gpu,
            self.sampling_states.seeds.gpu,
            pos,
        )
        # The candidate path never materializes the top-k/top-p masked logits.
        needs_masked_logits = self.return_sampling_mask or (
            return_logprobs and self.logprobs_mode in PROCESSED_LOGPROBS_MODES
        )
        if top_k is None and top_p is None or needs_masked_logits:
            processed_logits = apply_top_k_top_p(processed_logits, top_k, top_p)
            sampled = gumbel_sample(processed_logits, *args, self.use_fp64_gumbel)
        else:
            sampled = sample_top_k_top_p(
                processed_logits, *args, top_k, top_p, self.use_fp64_gumbel
            )
        return sampled, processed_logits

    @staticmethod
    def compute_topk_scores(
        logits: torch.Tensor,
        num_logprobs: int,
        sampled_token_ids: torch.Tensor,
        cu_num_logits: list[int] | torch.Tensor | None = None,
        logprob_token_ids_state: LogprobTokenIdsState | None = None,
        expanded_idx_mapping: torch.Tensor | None = None,
        max_per_req_token_ids: int = 0,
        logits_mode: bool = False,
    ) -> LogprobsTensors:
        sampled = sampled_token_ids.long()
        topk = torch.topk(logits, num_logprobs, dim=-1).indices
        token_ids = torch.cat((sampled.unsqueeze(1), topk), dim=1)
        valid = None
        if max_per_req_token_ids > 0:
            # Requests with logprob_token_ids get those instead of the top-k.
            assert logprob_token_ids_state is not None
            assert expanded_idx_mapping is not None
            num_cols = max(num_logprobs, max_per_req_token_ids)
            req = expanded_idx_mapping.long()
            num_custom = logprob_token_ids_state.num_token_ids.gpu[req].unsqueeze(1)
            custom = logprob_token_ids_state.token_ids.gpu[req, :num_cols].long()
            custom = F.pad(custom, (0, num_cols - custom.shape[1]))
            topk = F.pad(topk, (0, num_cols - num_logprobs))
            has_custom = num_custom > 0
            valid = torch.arange(num_cols) < torch.where(
                has_custom, num_custom, num_logprobs
            )
            cols = torch.where(valid, torch.where(has_custom, custom, topk), 0)
            token_ids = torch.cat((sampled.unsqueeze(1), cols), dim=1)
            valid = F.pad(valid, (1, 0), value=True)
        if logits_mode:
            scores = logits.gather(1, token_ids).float()
        else:
            scores = token_logprobs(logits, token_ids)
        if valid is not None:
            scores = scores.masked_fill(~valid, float("-inf"))
        is_tensor = isinstance(cu_num_logits, torch.Tensor)
        return LogprobsTensors(
            logprob_token_ids=token_ids.to(sampled_token_ids.dtype),
            logprobs=scores,
            selected_token_ranks=token_ranks(logits, sampled),
            cu_num_generated_tokens=None if is_tensor else cu_num_logits,
            cu_num_generated_tokens_tensor=cu_num_logits if is_tensor else None,
        )
