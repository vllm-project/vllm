# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate sampling: on a step whose sampling needs only the largest logits,
each rank takes the k largest logits of its vocabulary shard and the ranks swap
them (``_xpu_C.mono_shard_topk``) instead of all-gathering the logits; vLLM's
sampler and rejection sampler then run on the [rows, w] candidates, and the
sampled candidate positions are mapped back to token ids.

The candidates hold every request's top-k (k = 1 for greedy requests), so the
result has the stock distribution: greedy picks the same token, top-k / top-p
sampling draws from the same k tokens with the same probabilities. A step
takes the candidates only if every request in it is greedy or samples with a
top_k within the kernel's limit, and none needs anything over the whole
vocabulary: logprobs, penalties, logit bias, bad words, allowed tokens,
min_tokens, min_p, a thinking budget, structured output, an explicit seed
(its draw would differ from the stock one), custom logits processors,
probabilistic drafts. Every other step all-gathers the logits."""

from typing import cast

import numpy as np
import torch
import torch.nn.functional as F

from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
    RejectionSampler,
    gather_draft_sampled,
    get_max_chunk_logits,
)

# rows of the candidate exchange (a decode or verify step of the kernels)
MAX_ROWS = 8
# the narrowest candidate row: vLLM's top-k / top-p kernels run on >= 128
# columns
MIN_WIDTH = 128


def _request_ok(sp, k_max: int) -> bool:
    """Whether a request's sampling needs only its top-k (top_k <= k_max) or
    its argmax."""
    if (
        sp.logprobs is not None
        or sp.presence_penalty != 0.0
        or sp.frequency_penalty != 0.0
        or sp.repetition_penalty != 1.0
        or sp.logit_bias
        or sp.bad_words
        or sp.allowed_token_ids
        or sp.min_tokens
        or sp.min_p != 0.0
        or sp.thinking_token_budget is not None
        or sp.structured_outputs is not None
    ):
        return False
    if sp.temperature == 0.0:
        return True
    return sp.seed is None and 1 <= sp.top_k <= k_max


class Candidates:
    """The per-step decision and the exchange. ``plan`` (host values only, the
    same on every rank) runs in ``compute_sample_logits`` with the batch
    ``ModelState.prepare_inputs`` saw; the samplers read ``idx`` (the
    candidates' token ids) of the step that produced candidates."""

    def __init__(self, vllm_config, region, rank: int, max_num_reqs: int):
        self.region, self.rank = region, rank
        self.k_max = int(torch.ops._xpu_C.mono_shard_topk_max_k())
        self.ok = np.zeros(max_num_reqs, dtype=bool)
        self.k = np.zeros(max_num_reqs, dtype=np.int32)  # 1 for greedy
        self.batch = None  # this step's input batch (prepare_inputs)
        self.idx: torch.Tensor | None = None  # [rows, w] token ids, or None
        spec = vllm_config.speculative_config
        mc = vllm_config.model_config
        self.enabled = (
            self.k_max > 0
            and not mc.logits_processors
            and vllm_config.watermark_config is None
            and not mc.return_sampling_mask
            and not mc.enable_trace_replay
            and (
                spec is None
                or (
                    spec.draft_sample_method == "greedy"
                    and spec.rejection_sample_method == "standard"
                )
            )
        )

    def add_request(self, req_idx: int, sampling_params) -> None:
        ok = _request_ok(sampling_params, self.k_max)
        self.ok[req_idx] = ok
        greedy = sampling_params.temperature == 0.0
        self.k[req_idx] = 1 if greedy else (sampling_params.top_k if ok else 0)

    def plan(self, rows: int) -> int | None:
        """K for this step's candidates, or None for the all-gathered logits."""
        batch, self.batch, self.idx = self.batch, None, None
        if not self.enabled or batch is None or not 1 <= rows <= MAX_ROWS:
            return None
        if batch.has_structured_output_reqs:
            return None
        reqs = batch.idx_mapping_np
        if not np.all(self.ok[reqs]):
            return None
        return int(np.max(self.k[reqs]))

    def logits(self, shard: torch.Tensor, k: int, verify: bool) -> torch.Tensor:
        """[rows, w] fp32 candidate logits (-inf padding) from this rank's
        [rows, vocab / 2] logits shard; ``idx`` gets their token ids. A verify
        step gets one more -inf column, the position of a draft token that is
        not a candidate (target probability 0)."""
        rows = shard.shape[0]
        width = max(2 * k, MIN_WIDTH)
        vals = torch.empty(rows, width, dtype=torch.float32, device=shard.device)
        idx = torch.empty(rows, width, dtype=torch.int32, device=shard.device)
        own, peer = self.region.candidates()
        torch.ops._xpu_C.mono_shard_topk(
            vals, idx, shard.contiguous(), k, self.rank, own, peer
        )
        if verify:
            vals = F.pad(vals, (0, 1), value=float("-inf"))
            idx = F.pad(idx, (0, 1), value=-1)
        self.idx = idx.to(torch.int64)
        return vals

    def take(self, logits: torch.Tensor) -> torch.Tensor | None:
        """The candidates' token ids if ``logits`` are this step's candidates
        (the dummy sampler runs compute their logits elsewhere), once."""
        idx, self.idx = self.idx, None
        if idx is None or idx.shape != logits.shape:
            return None
        return idx


def draft_positions(idx: torch.Tensor, draft_sampled: torch.Tensor) -> torch.Tensor:
    """Draft token ids -> candidate positions. The draft in row j is verified
    by the logits of row j - 1 (a request's first row holds no draft), so it is
    looked up in that row's candidates; a draft that is not a candidate gets the
    last, -inf column, a placeholder (-1) stays -1."""
    miss = idx.shape[1] - 1
    prev = torch.cat([idx[:1], idx[:-1]])
    hit = prev == draft_sampled.unsqueeze(1)
    pos = torch.where(hit.any(dim=1), hit.int().argmax(dim=1), miss)
    return torch.where(draft_sampled < 0, -1, pos).to(draft_sampled.dtype)


class CandidateSampler:
    """vLLM's sampler, on the candidates of a candidate step."""

    def __init__(self, sampler, candidates: Candidates):
        self.sampler, self.candidates = sampler, candidates

    def __getattr__(self, name):
        return getattr(self.sampler, name)

    def add_request(self, req_idx: int, sampling_params) -> None:
        self.sampler.add_request(req_idx, sampling_params)
        self.candidates.add_request(req_idx, sampling_params)

    def __call__(self, logits: torch.Tensor, input_batch) -> SamplerOutput:
        idx = self.candidates.take(logits)
        out = self.sampler(logits, input_batch)
        if idx is not None:  # candidate positions -> token ids
            st = out.sampled_token_ids
            out.sampled_token_ids = idx.gather(1, st.to(torch.int64)).to(st.dtype)
        return out


class CandidateRejectionSampler(RejectionSampler):
    """vLLM's rejection sampler, on the candidates of a candidate verify step:
    the draft tokens are mapped to candidate positions (the extra -inf column
    when a draft is not a candidate), the verified positions back to token
    ids."""

    def __init__(self, sampler: CandidateSampler, spec_config, device):
        super().__init__(cast(Sampler, sampler), spec_config, device)
        self.candidates = sampler.candidates

    def __call__(self, logits, input_batch, draft_logits=None) -> SamplerOutput:
        idx = self.candidates.take(logits)
        if idx is None:
            return super().__call__(logits, input_batch, draft_logits)
        assert draft_logits is None
        draft_sampled, pos = gather_draft_sampled(
            input_batch.input_ids,
            input_batch.positions,
            input_batch.logits_indices,
            input_batch.expanded_idx_mapping,
            input_batch.expanded_local_pos,
            self.sampler.req_states.prefill_len.gpu,
        )
        sampled, num_sampled, _, _ = self._verify_in_chunks(
            logits,
            input_batch,
            None,
            draft_positions(idx, draft_sampled),
            pos,
            get_max_chunk_logits(logits.shape[1]),
            None,
        )
        # [reqs, k + 1] candidate positions -> token ids of the request's rows
        # of idx (entries past num_sampled are unused: clamped only to stay in
        # bounds)
        start = input_batch.cu_num_logits[: input_batch.num_reqs].to(torch.int64)
        rows = start.unsqueeze(1) + torch.arange(sampled.shape[1], device=idx.device)
        rows = rows.clamp_(max=idx.shape[0] - 1)
        cols = sampled.clamp(0, idx.shape[1] - 1).unsqueeze(-1)
        sampled = idx[rows].gather(2, cols).squeeze(-1).to(sampled.dtype)
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            num_sampled,
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.sampler.req_states.prefill_len.gpu,
        )
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            sampling_mask_tensors=None,
        )
