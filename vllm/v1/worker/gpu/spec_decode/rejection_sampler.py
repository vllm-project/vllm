# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Iterator

import numpy as np
import torch

from vllm.config import SpeculativeConfig
from vllm.config.model import PROCESSED_LOGPROBS_MODES
from vllm.distributed.parallel_state import get_tp_group
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    get_num_sampled_and_rejected,
)
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    rejection_sample,
)

# Cap on the FP32 target-logits buffer materialized by apply_sampling_params.
# TODO(mgoin): Chunking is a workaround. The rejection kernels already upcast
# per vocab block on load and apply ops like temperature and gumbel, so folding
# sampling-param application into those kernels would remove this buffer and
# its traffic entirely.
MAX_CHUNK_BYTES = 2**30  # 1GB
_FP32_BYTES = 4


def get_max_chunk_logits(vocab_size: int) -> int:
    """Largest number of logits rows one verification chunk may hold."""
    return max(1, MAX_CHUNK_BYTES // (vocab_size * _FP32_BYTES))


def _iter_request_chunks(
    cu_num_logits: np.ndarray, max_chunk_logits: int
) -> Iterator[tuple[int, int]]:
    """Yield maximally packed request ranges without splitting requests."""
    assert max_chunk_logits > 0
    num_reqs = cu_num_logits.size - 1
    start = 0
    while start < num_reqs:
        max_logit = int(cu_num_logits[start]) + max_chunk_logits
        end = int(np.searchsorted(cu_num_logits, max_logit, side="right") - 1)
        end = min(num_reqs, max(start + 1, end))
        yield start, end
        start = end


@triton.jit
def _flatten_sampled_kernel(
    # [num_logits]
    flat_sampled_ptr,
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    for i in range(num_sampled):
        token_id = tl.load(sampled_ptr + req_idx * sampled_stride + i)
        tl.store(flat_sampled_ptr + start_idx + i, token_id)


@triton.jit(do_not_specialize=["max_spec_len"])
def _compact_rejection_sample_kernel(
    output_token_ids_ptr,
    cu_num_draft_tokens_ptr,
    draft_token_ids_ptr,
    target_draft_probs_ptr,
    bonus_token_ids_ptr,
    recovered_token_ids_ptr,
    uniform_probs_ptr,
    max_spec_len,
):
    """Compact top-k rejection decisions into one output row per request."""
    req_idx = tl.program_id(0)
    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            token_idx = start_idx + pos
            draft_token_id = tl.load(draft_token_ids_ptr + token_idx)
            target_prob = tl.load(target_draft_probs_ptr + token_idx)
            uniform_prob = tl.load(uniform_probs_ptr + token_idx)
            accepted = target_prob >= uniform_prob
            token_id = draft_token_id
            if not accepted:
                rejected = True
                token_id = tl.load(recovered_token_ids_ptr + token_idx)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                token_id,
            )

    if not rejected:
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )


@triton.jit(do_not_specialize=["max_spec_len"])
def _compact_greedy_rejection_sample_kernel(
    output_token_ids_ptr,
    target_token_ids_ptr,
    draft_sampled_ptr,
    cu_num_logits_ptr,
    max_spec_len,
):
    req_idx = tl.program_id(0)
    start_idx = 0 if req_idx == 0 else tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_draft_tokens = end_idx - start_idx - 1

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            target_token_id = tl.load(target_token_ids_ptr + start_idx + pos).to(
                tl.int64
            )
            draft_token_id = tl.load(draft_sampled_ptr + start_idx + pos + 1).to(
                tl.int64
            )
            accepted = target_token_id == draft_token_id
            token_id = draft_token_id
            if not accepted:
                rejected = True
                token_id = target_token_id
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                token_id,
            )

    if not rejected:
        bonus_token_id = tl.load(target_token_ids_ptr + end_idx - 1).to(tl.int64)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )


class RejectionSampler:
    def __init__(
        self,
        sampler: Sampler,
        spec_config: SpeculativeConfig,
        device: torch.device,
    ):
        self.sampler = sampler
        self.num_speculative_steps = spec_config.num_speculative_tokens
        self.enable_adaptive_verification = spec_config.enable_adaptive_verification
        rejection_sample_method = spec_config.rejection_sample_method
        self.use_block_verification: bool = False
        self.synthetic_conditional_rates: torch.Tensor | None = None
        if rejection_sample_method == "synthetic":
            assert spec_config.synthetic_acceptance_rates is not None
            self.synthetic_conditional_rates = torch.tensor(
                unconditional_to_conditional_rates(
                    spec_config.synthetic_acceptance_rates
                ),
                dtype=torch.float32,
                device=device,
            )
        elif rejection_sample_method == "block":
            self.use_block_verification = True

    def _get_logprobs_tensors(
        self,
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        logits: torch.Tensor,
        cu_num_logits: torch.Tensor,
        cu_num_logits_np: np.ndarray,
        max_num_logprobs: int,
    ) -> LogprobsTensors | None:
        if max_num_logprobs == NO_LOGPROBS:
            return None

        num_reqs = cu_num_logits.shape[0] - 1
        num_logits = logits.shape[0]
        flat_sampled = torch.zeros(
            num_logits, dtype=sampled.dtype, device=sampled.device
        )
        _flatten_sampled_kernel[(num_reqs,)](
            flat_sampled,
            sampled,
            sampled.stride(0),
            num_sampled,
            cu_num_logits,
            num_warps=1,
        )
        expanded_logits = num_logits != num_reqs
        cu_num_generated_tokens: list[int] | torch.Tensor | None = None
        if expanded_logits:
            if self.enable_adaptive_verification:
                # Adaptive verification keeps the true per-request boundaries
                # on device only; cu_num_logits_np holds the pre-compacted
                # layout.
                cu_num_generated_tokens = cu_num_logits.clone()
            else:
                cu_num_generated_tokens = cu_num_logits_np.tolist()
        return compute_topk_scores(
            logits,
            max_num_logprobs,
            flat_sampled,
            cu_num_generated_tokens,
            logits_mode=self.sampler.logprobs_mode
            in ("raw_logits", "processed_logits"),
        )

    def _verify(
        self,
        logits: torch.Tensor,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        pos: torch.Tensor,
        cu_num_logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        processed_logits = self.sampler.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping,
            idx_mapping_np,
            pos,
            draft_sampled,
            expanded_local_pos,
        )
        sampled, num_sampled = rejection_sample(
            processed_logits,
            draft_logits,
            draft_sampled,
            cu_num_logits,
            pos,
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            self.sampler.sampling_states.temperature.gpu,
            self.sampler.sampling_states.seeds.gpu,
            self.num_speculative_steps,
            self.synthetic_conditional_rates,
            use_fp64=self.sampler.use_fp64_gumbel,
            use_block_verification=self.use_block_verification,
        )
        return processed_logits, sampled, num_sampled

    def _verify_in_chunks(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        pos: torch.Tensor,
        max_chunk_logits: int,
        max_num_logprobs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, LogprobsTensors | None]:
        cu_num_logits_np = input_batch.cu_num_logits_np
        use_processed_logits = self.sampler.logprobs_mode in PROCESSED_LOGPROBS_MODES
        num_reqs = input_batch.num_reqs

        if logits.shape[0] <= max_chunk_logits:
            # One chunk covers the batch. Adaptive verification compacts the logits
            # without updating cu_num_logits_np (it keeps the pre-compacted layout),
            # so the stale sums must not pick chunk boundaries; its budget cap
            # guarantees the compacted batch always lands here.
            request_chunks: Iterable[tuple[int, int]] = ((0, num_reqs),)
        else:
            assert not self.enable_adaptive_verification
            request_chunks = _iter_request_chunks(cu_num_logits_np, max_chunk_logits)

        sampled_chunks: list[torch.Tensor] = []
        num_sampled_chunks: list[torch.Tensor] = []
        logprobs_chunks: list[LogprobsTensors] = []

        for start, end in request_chunks:
            lo = int(cu_num_logits_np[start])
            hi = int(cu_num_logits_np[end])
            chunk_cu_num_logits_np = cu_num_logits_np[start : end + 1] - lo
            chunk_cu_num_logits = input_batch.cu_num_logits[start : end + 1] - lo
            # draft_logits uses persistent request-state indices and stays global.
            processed_logits, sampled, num_sampled = self._verify(
                logits[lo:hi],
                draft_logits,
                draft_sampled[lo:hi],
                pos[lo:hi],
                chunk_cu_num_logits,
                input_batch.idx_mapping[start:end],
                input_batch.idx_mapping_np[start:end],
                input_batch.expanded_idx_mapping[lo:hi],
                input_batch.expanded_local_pos[lo:hi],
            )
            chunk_logprobs = self._get_logprobs_tensors(
                sampled,
                num_sampled,
                processed_logits if use_processed_logits else logits[lo:hi],
                chunk_cu_num_logits,
                chunk_cu_num_logits_np,
                max_num_logprobs,
            )
            if chunk_logprobs is not None:
                logprobs_chunks.append(chunk_logprobs)
            del processed_logits
            sampled_chunks.append(sampled)
            num_sampled_chunks.append(num_sampled)

        if len(sampled_chunks) == 1:
            logprobs_tensors = logprobs_chunks[0] if logprobs_chunks else None
            return sampled_chunks[0], num_sampled_chunks[0], logprobs_tensors

        logprobs_tensors = None
        if logprobs_chunks:
            expanded_logits = logits.shape[0] != input_batch.num_reqs
            logprobs_tensors = LogprobsTensors.cat(
                logprobs_chunks,
                cu_num_generated_tokens=(
                    cu_num_logits_np.tolist() if expanded_logits else None
                ),
            )

        sampled = torch.cat(sampled_chunks)
        num_sampled = torch.cat(num_sampled_chunks)
        return sampled, num_sampled, logprobs_tensors

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None = None,
    ) -> SamplerOutput:
        # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
        # that num_nans is computed before applying penalties and temperature.
        num_nans = get_num_nans(logits) if self.sampler.compute_nans else None

        draft_sampled = input_batch.input_ids[input_batch.logits_indices]
        pos = input_batch.positions[input_batch.logits_indices]

        max_num_logprobs = self.sampler.sampling_states.max_num_logprobs(
            input_batch.idx_mapping_np
        )
        chunk_logit_limit = get_max_chunk_logits(logits.shape[1])
        sampled, num_sampled, logprobs_tensors = self._verify_in_chunks(
            logits,
            input_batch,
            draft_logits,
            draft_sampled,
            pos,
            chunk_logit_limit,
            max_num_logprobs,
        )

        num_sampled, num_rejected = get_num_sampled_and_rejected(
            num_sampled,
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.sampler.req_states.prefill_len.gpu,
        )

        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )

    @staticmethod
    def _sample_from_candidate_logits(
        candidate_logits: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Sample one token from each compact candidate row."""
        valid = torch.isfinite(candidate_logits)
        noise = torch.empty_like(candidate_logits)
        noise.exponential_()
        scores = (candidate_logits - noise.log()).masked_fill(
            ~valid, -float("inf")
        )
        sample_pos = scores.argmax(dim=-1, keepdim=True)
        return candidate_ids.gather(dim=-1, index=sample_pos).view(-1)

    def _sample_from_topk_candidates_local(
        self,
        candidate_logits: torch.Tensor,
        candidate_ids: torch.Tensor,
        input_batch: InputBatch,
    ) -> SamplerOutput:
        """Verify one-hot drafts using compact target top-k candidates.

        This is deliberately limited to the ordinary probabilistic rejection
        sampler (no synthetic/block verification).  Candidate logits are the
        target distribution restricted to an approximate top-k set; a missing
        draft token therefore has zero mass and is recovered by sampling from
        the remaining candidate set.
        """
        if candidate_logits.ndim != 2:
            raise ValueError("candidate_logits must be rank 2")
        if candidate_ids.shape != candidate_logits.shape:
            raise ValueError("candidate_ids must match candidate_logits")
        if candidate_logits.shape[0] != int(input_batch.cu_num_logits_np[-1]):
            raise ValueError("candidate rows must match the logits layout")
        if input_batch.num_draft_tokens_per_req is None:
            raise ValueError("top-k candidate sampling requires draft tokens")
        if self.synthetic_conditional_rates is not None or self.use_block_verification:
            raise ValueError("compact candidates do not support this rejection mode")

        num_reqs = input_batch.num_reqs
        device = candidate_logits.device
        bonus_indices = input_batch.cu_num_logits[1:].to(torch.int64) - 1
        is_bonus = torch.zeros(
            candidate_logits.shape[0], dtype=torch.bool, device=device
        )
        is_bonus[bonus_indices] = True
        target_indices = torch.arange(
            candidate_logits.shape[0], dtype=torch.int64, device=device
        )[~is_bonus]
        if target_indices.shape[0] != input_batch.num_draft_tokens:
            raise ValueError("candidate rows do not match draft-token layout")

        draft_sampled = input_batch.input_ids[input_batch.logits_indices]
        draft_token_ids = draft_sampled[target_indices + 1]

        bonus_token_ids = self._sample_from_candidate_logits(
            candidate_logits[bonus_indices], candidate_ids[bonus_indices]
        ).to(draft_token_ids.dtype)
        target_logits = candidate_logits[target_indices]
        target_ids = candidate_ids[target_indices]
        draft_ids = draft_token_ids.to(target_ids.dtype).unsqueeze(-1)

        valid = torch.isfinite(target_logits)
        draft_mask = valid & (target_ids == draft_ids)
        safe_logits = target_logits.masked_fill(~valid, -float("inf"))
        max_logits = safe_logits.max(dim=-1, keepdim=True).values
        weights = torch.exp(safe_logits - max_logits)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
        denom = weights.sum(dim=-1)
        draft_weight = torch.where(draft_mask, weights, torch.zeros_like(weights)).sum(
            dim=-1
        )
        target_draft_probs = torch.where(
            denom > 0.0, draft_weight / denom, torch.zeros_like(denom)
        )

        recovered_mask = valid & ~draft_mask
        noise = torch.empty_like(target_logits)
        noise.exponential_()
        recovered_scores = (target_logits - noise.log()).masked_fill(
            ~recovered_mask, -float("inf")
        )
        recovered_pos = recovered_scores.argmax(dim=-1, keepdim=True)
        recovered_token_ids = target_ids.gather(dim=-1, index=recovered_pos).view(-1)
        recovered_token_ids = torch.where(
            recovered_mask.any(dim=-1),
            recovered_token_ids,
            torch.zeros_like(recovered_token_ids),
        ).to(draft_token_ids.dtype)

        uniform_probs = torch.rand(
            (input_batch.num_draft_tokens,),
            dtype=torch.float64,
            device=device,
        )
        cu_num_draft_tokens = input_batch.cu_num_logits[1:] - torch.arange(
            1,
            num_reqs + 1,
            dtype=input_batch.cu_num_logits.dtype,
            device=device,
        )
        sampled = torch.full(
            (num_reqs, self.num_speculative_steps + 1),
            -1,
            dtype=torch.int64,
            device=device,
        )
        if candidate_logits.is_cuda:
            _compact_rejection_sample_kernel[(num_reqs,)](
                sampled,
                cu_num_draft_tokens,
                draft_token_ids,
                target_draft_probs,
                bonus_token_ids,
                recovered_token_ids,
                uniform_probs,
                self.num_speculative_steps,
            )
        else:
            cu_num_draft_tokens_cpu = cu_num_draft_tokens.cpu().tolist()
            sampled_cpu = sampled
            offset = 0
            for req_idx, end in enumerate(cu_num_draft_tokens_cpu):
                rejected = False
                for pos in range(end - offset):
                    token_idx = offset + pos
                    if not rejected:
                        accepted = bool(
                            target_draft_probs[token_idx]
                            >= uniform_probs[token_idx].to(
                                target_draft_probs.dtype
                            )
                        )
                        token_id = draft_token_ids[token_idx]
                        if not accepted:
                            rejected = True
                            token_id = recovered_token_ids[token_idx]
                        sampled_cpu[req_idx, pos] = token_id
                if not rejected:
                    sampled_cpu[req_idx, end - offset] = bonus_token_ids[req_idx]
                offset = end

        num_sampled = (sampled != -1).sum(dim=-1, dtype=torch.int32)
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
        )

    def sample_from_topk_candidates(
        self,
        candidate_logits: torch.Tensor,
        candidate_ids: torch.Tensor,
        input_batch: InputBatch,
    ) -> SamplerOutput:
        """Sample compact MTP candidates once, then synchronize TP ranks.

        ``get_topk_candidates`` all-gathers the candidate pairs, but random
        rejection decisions must still be made by one rank. Otherwise a
        different CUDA RNG state on a TP rank can produce divergent sampled
        tokens even when every rank sees identical candidates.
        """
        if candidate_logits.ndim != 2:
            raise ValueError("candidate_logits must be rank 2")
        if candidate_ids.shape != candidate_logits.shape:
            raise ValueError("candidate_ids must match candidate_logits")
        if candidate_logits.shape[0] != int(input_batch.cu_num_logits_np[-1]):
            raise ValueError("candidate rows must match the logits layout")
        if input_batch.num_draft_tokens_per_req is None:
            raise ValueError("top-k candidate sampling requires draft tokens")
        if self.synthetic_conditional_rates is not None or self.use_block_verification:
            raise ValueError("compact candidates do not support this rejection mode")

        try:
            tp_group = get_tp_group()
        except AssertionError:
            # Keep direct CPU/unit-test use independent of distributed setup.
            tp_group = None
        if tp_group is None or tp_group.world_size == 1:
            return self._sample_from_topk_candidates_local(
                candidate_logits, candidate_ids, input_batch
            )

        if tp_group.rank_in_group == 0:
            output = self._sample_from_topk_candidates_local(
                candidate_logits, candidate_ids, input_batch
            )
            sampled = output.sampled_token_ids
            num_sampled = output.num_sampled
            num_rejected = output.num_rejected
        else:
            sampled = torch.empty(
                (input_batch.num_reqs, self.num_speculative_steps + 1),
                dtype=torch.int64,
                device=candidate_logits.device,
            )
            num_sampled = torch.empty(
                (input_batch.num_reqs,),
                dtype=torch.int32,
                device=candidate_logits.device,
            )
            num_rejected = torch.empty_like(num_sampled)

        tp_group.broadcast(sampled, src=0)
        tp_group.broadcast(num_sampled, src=0)
        tp_group.broadcast(num_rejected, src=0)
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )

    def sample_from_greedy_tokens(
        self,
        target_token_ids: torch.Tensor,
        input_batch: InputBatch,
    ) -> SamplerOutput:
        """Verify greedy drafts using only target token IDs."""
        assert target_token_ids.ndim == 1
        assert target_token_ids.shape[0] == int(input_batch.cu_num_logits_np[-1])
        assert input_batch.num_draft_tokens_per_req is not None
        assert self.synthetic_conditional_rates is None
        assert not self.use_block_verification

        draft_sampled = input_batch.input_ids[input_batch.logits_indices]
        sampled = torch.full(
            (input_batch.num_reqs, self.num_speculative_steps + 1),
            -1,
            dtype=torch.int64,
            device=target_token_ids.device,
        )
        if target_token_ids.is_cuda:
            _compact_greedy_rejection_sample_kernel[(input_batch.num_reqs,)](
                sampled,
                target_token_ids,
                draft_sampled,
                input_batch.cu_num_logits,
                self.num_speculative_steps,
            )
        else:
            cu_num_logits = input_batch.cu_num_logits_np
            for req_idx in range(input_batch.num_reqs):
                start_idx = int(cu_num_logits[req_idx])
                end_idx = int(cu_num_logits[req_idx + 1])
                num_draft_tokens = end_idx - start_idx - 1
                rejected = False
                for pos in range(num_draft_tokens):
                    if rejected:
                        break
                    target_token = int(target_token_ids[start_idx + pos])
                    draft_token = int(draft_sampled[start_idx + pos + 1])
                    sampled[req_idx, pos] = (
                        draft_token if target_token == draft_token else target_token
                    )
                    rejected = target_token != draft_token
                if not rejected:
                    sampled[req_idx, num_draft_tokens] = int(
                        target_token_ids[end_idx - 1]
                    )

        num_sampled = (sampled != -1).sum(dim=-1, dtype=torch.int32)
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
        )
