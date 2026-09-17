# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator


@triton.jit
def _selector_walk_kernel(
    scores_ptr,
    candidate_ptr,
    sample_pos_ptr,
    req_state_ptr,
    temperature_ptr,
    seeds_ptr,
    tokens_ptr,
    realized_scores_ptr,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SAMPLE_PROBABILISTIC: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < top_k
    req_state = tl.load(req_state_ptr + row * num_steps)
    valid = req_state >= 0
    temperature = tl.load(temperature_ptr + req_state, mask=valid, other=0.0)
    seed = tl.load(seeds_ptr + req_state, mask=valid, other=0)
    previous = 0
    for step in range(num_steps):
        flat = row * num_steps + step
        score_base = (flat * top_k + previous) * top_k
        scores = tl.load(
            scores_ptr + score_base + offsets,
            mask=mask & valid,
            other=float("-inf"),
        ).to(tl.float64 if USE_FP64 else tl.float32)
        candidate_base = flat * top_k
        candidates = tl.load(
            candidate_ptr + candidate_base + offsets,
            mask=mask & valid,
            other=0,
        )

        # sample_pos is the predicted token's position P. Sampling keys a draw
        # by the position before the sampled token, P-1.
        sample_pos = tl.load(sample_pos_ptr + flat) - 1
        _, index = gumbel_noised_argmax(
            scores,
            candidates,
            mask & valid,
            seed,
            sample_pos,
            temperature if SAMPLE_PROBABILISTIC else 0.0,
            IS_DRAFTING=True,
            USE_FP64=USE_FP64,
        )

        tl.store(
            realized_scores_ptr + candidate_base + offsets,
            scores,
            mask=mask & valid,
        )
        token = tl.load(candidate_ptr + candidate_base + index, mask=valid, other=0)
        tl.store(tokens_ptr + flat, token, mask=valid)
        previous = index


@triton.jit
def _cache_draft_logits_kernel(
    draft_logits_ptr,
    cached_candidate_ptr,
    candidate_ptr,
    scores_ptr,
    req_state_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    flat = tl.program_id(0)
    req_state = tl.load(req_state_ptr + flat)
    step = flat % num_steps
    offsets = tl.arange(0, BLOCK_K)
    mask = (req_state >= 0) & (offsets < top_k)
    candidate_base = flat * top_k
    cache_base = (req_state * num_steps + step) * top_k
    old_token_ids = tl.load(cached_candidate_ptr + cache_base + offsets, mask=mask)
    logits_base = (
        draft_logits_ptr
        + req_state * draft_logits_stride_0
        + step * draft_logits_stride_1
    )
    tl.store(logits_base + old_token_ids, -float("inf"), mask=mask)
    token_ids = tl.load(candidate_ptr + candidate_base + offsets, mask=mask)
    scores = tl.load(scores_ptr + candidate_base + offsets, mask=mask)
    tl.store(logits_base + token_ids, scores, mask=mask)
    tl.store(cached_candidate_ptr + cache_base + offsets, token_ids, mask=mask)


class CandidateSampler:
    """The shared DFlash2/LiLiCorr candidate walk and realized proposal cache."""

    def __init__(
        self, max_num_reqs: int, num_steps: int, top_k: int, device: torch.device
    ):
        self.num_steps = num_steps
        self.top_k = top_k
        self.scores = torch.empty(
            max_num_reqs, num_steps, top_k, dtype=torch.float32, device=device
        )
        self.cached_candidate_ids = torch.zeros(
            self.scores.shape, dtype=torch.int64, device=device
        )

    def sample(
        self,
        candidate_ids: torch.Tensor,
        scores: torch.Tensor,
        num_reqs: int,
        sample_pos: torch.Tensor,
        idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_logits: torch.Tensor | None,
        use_fp64: bool,
    ) -> None:
        candidate_ids = candidate_ids.contiguous()
        sample_pos = sample_pos.contiguous()
        idx_mapping = idx_mapping.contiguous()
        block_k = triton.next_power_of_2(self.top_k)
        _selector_walk_kernel[(num_reqs,)](
            scores.contiguous(),
            candidate_ids.contiguous(),
            sample_pos,
            idx_mapping,
            temperature,
            seeds,
            draft_tokens,
            self.scores,
            num_steps=self.num_steps,
            top_k=self.top_k,
            BLOCK_K=block_k,
            SAMPLE_PROBABILISTIC=draft_logits is not None,
            USE_FP64=use_fp64,
            num_warps=1,
        )

        if draft_logits is not None:
            self._cache_draft_logits(
                candidate_ids, num_reqs * self.num_steps, idx_mapping, draft_logits
            )

    def _cache_draft_logits(
        self,
        candidate_ids: torch.Tensor,
        num_sample: int,
        idx_mapping: torch.Tensor,
        draft_logits: torch.Tensor,
    ) -> None:
        block_k = triton.next_power_of_2(self.top_k)
        _cache_draft_logits_kernel[(num_sample,)](
            draft_logits,
            self.cached_candidate_ids,
            candidate_ids,
            self.scores,
            idx_mapping,
            draft_logits.stride(0),
            draft_logits.stride(1),
            num_steps=self.num_steps,
            top_k=self.top_k,
            BLOCK_K=block_k,
            num_warps=1,
        )


class DFlash2Speculator(DFlashSpeculator):
    _speculator_name = "DFlash2"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        draft_config = self.draft_model_config.hf_config.dflash_config
        self.selector_top_k = int(draft_config["selector_top_k"])
        self._anchor_indices = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )
        self.candidate_sampler = CandidateSampler(
            self.max_num_reqs, self.num_speculative_steps, self.selector_top_k, device
        )

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        # fp32 so the walk and the rejection that checks it read the same
        # distribution; -inf because the cache kernel writes only the K
        # candidates.
        return torch.float32, -float("inf")

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        last_hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        num_sample = num_reqs * self.num_speculative_steps
        hidden_states = last_hidden_states[self.sample_indices[:num_sample]].view(
            num_reqs, self.num_speculative_steps, -1
        )
        candidate_ids, unary_logits = self.model.compute_candidates(
            hidden_states.flatten(0, 1)
        )
        candidate_ids = candidate_ids.view(
            num_reqs, self.num_speculative_steps, self.selector_top_k
        )
        unary_logits = unary_logits.view_as(candidate_ids)
        anchor_token_ids = self.input_buffers.input_ids[self._anchor_indices[:num_reqs]]
        scores = self.model.model.candidate_selector(
            candidate_ids,
            unary_logits,
            hidden_states,
            anchor_token_ids,
        )
        self.candidate_sampler.sample(
            candidate_ids,
            scores,
            num_reqs,
            self.sample_pos,
            self.sample_idx_mapping,
            self.temperature,
            self.seeds,
            self.draft_tokens,
            self.draft_logits,
            self.use_fp64_gumbel,
        )
        if self.enable_adaptive_verification:
            self._maybe_predict_acceptance(
                self.candidate_sampler.scores[:num_reqs].flatten(0, 1),
                self.sample_idx_mapping[:num_sample],
                self.sample_col[:num_sample],
            )
