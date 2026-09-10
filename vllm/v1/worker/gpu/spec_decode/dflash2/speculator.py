# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    DispatchSpec,
    TritonWarmupTensor,
    triton_kernel,
)
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
        self._selector_scores = torch.empty(
            self.max_num_reqs,
            self.num_speculative_steps,
            self.selector_top_k,
            dtype=torch.float32,
            device=device,
        )
        self._cached_candidate_ids = torch.zeros(
            self._selector_scores.shape, dtype=torch.int64, device=device
        )
        _SELECTOR_WALK_KERNEL.register_warmup(speculator=self)
        if self.draft_logits is not None:
            _CACHE_DRAFT_LOGITS_KERNEL.register_warmup(speculator=self)

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        # fp32 so the walk and the rejection that checks it read the same
        # distribution; -inf because the cache kernel writes only the K
        # candidates.
        return torch.float32, -float("inf")

    def _sample_path(
        self,
        candidate_ids: torch.Tensor,
        scores: torch.Tensor,
        num_reqs: int,
    ) -> None:
        block_k = triton.next_power_of_2(self.selector_top_k)
        _SELECTOR_WALK_KERNEL(
            scores.contiguous(),
            candidate_ids.contiguous(),
            self.sample_pos,
            self.sample_idx_mapping,
            self.temperature,
            self.seeds,
            self.draft_tokens,
            self._selector_scores,
            num_steps=self.num_speculative_steps,
            num_reqs=num_reqs,
            top_k=self.selector_top_k,
            BLOCK_K=block_k,
            SAMPLE_PROBABILISTIC=self.draft_logits is not None,
            USE_FP64=self.use_fp64_gumbel,
            num_warps=1,
        )

    def _cache_draft_logits(self, candidate_ids: torch.Tensor, num_sample: int) -> None:
        draft_logits = self.draft_logits
        assert draft_logits is not None
        block_k = triton.next_power_of_2(self.selector_top_k)
        _CACHE_DRAFT_LOGITS_KERNEL(
            draft_logits,
            self._cached_candidate_ids,
            candidate_ids,
            self._selector_scores,
            self.sample_idx_mapping,
            draft_logits.stride(0),
            draft_logits.stride(1),
            num_sample=num_sample,
            num_steps=self.num_speculative_steps,
            top_k=self.selector_top_k,
            BLOCK_K=block_k,
            num_warps=1,
        )

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
        self._sample_path(candidate_ids, scores, num_reqs)
        if self.draft_logits is not None:
            self._cache_draft_logits(candidate_ids, num_sample)


def _selector_walk_warmup_inputs(*, speculator: DFlash2Speculator):
    top_k = speculator.selector_top_k
    int64 = TritonWarmupTensor(torch.int64)
    return dict(
        scores=TritonWarmupTensor(torch.float32),
        candidate=int64,
        sample_pos=int64,
        req_state=TritonWarmupTensor(torch.int32),
        temperature=TritonWarmupTensor(torch.float32),
        seeds=int64,
        tokens=int64,
        realized_scores=TritonWarmupTensor(torch.float32),
        num_steps=speculator.num_speculative_steps,
        num_reqs=1,
        top_k=top_k,
        BLOCK_K=triton.next_power_of_2(top_k),
        SAMPLE_PROBABILISTIC=speculator.draft_logits is not None,
        USE_FP64=speculator.use_fp64_gumbel,
    )


@triton_kernel(kernel=_selector_walk_kernel, warmup_inputs=_selector_walk_warmup_inputs)
def _SELECTOR_WALK_KERNEL(
    scores: torch.Tensor,
    candidate: torch.Tensor,
    sample_pos: torch.Tensor,
    req_state: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    tokens: torch.Tensor,
    realized_scores: torch.Tensor,
    *,
    num_steps: int,
    num_reqs: int,
    top_k: int,
    BLOCK_K: int,
    SAMPLE_PROBABILISTIC: bool,
    USE_FP64: bool,
) -> DispatchSpec:
    return (num_reqs,), dict(
        num_steps=num_steps,
        top_k=top_k,
        BLOCK_K=BLOCK_K,
        SAMPLE_PROBABILISTIC=SAMPLE_PROBABILISTIC,
        USE_FP64=USE_FP64,
        num_warps=1,
    )


def _cache_draft_logits_warmup_inputs(*, speculator: DFlash2Speculator):
    draft_logits = speculator.draft_logits
    assert draft_logits is not None
    top_k = speculator.selector_top_k
    return dict(
        draft_logits=TritonWarmupTensor(
            torch.float32,
            shape=(1, 1, 1),
            strides=(draft_logits.stride(0), draft_logits.stride(1), 1),
        ),
        cached_candidate=TritonWarmupTensor(torch.int64),
        candidate=TritonWarmupTensor(torch.int64),
        scores=TritonWarmupTensor(torch.float32),
        req_state=TritonWarmupTensor(torch.int32),
        draft_logits_stride_0=draft_logits.stride(0),
        draft_logits_stride_1=draft_logits.stride(1),
        num_sample=1,
        num_steps=speculator.num_speculative_steps,
        top_k=top_k,
        BLOCK_K=triton.next_power_of_2(top_k),
    )


@triton_kernel(
    kernel=_cache_draft_logits_kernel,
    warmup_inputs=_cache_draft_logits_warmup_inputs,
)
def _CACHE_DRAFT_LOGITS_KERNEL(
    draft_logits: torch.Tensor,
    cached_candidate: torch.Tensor,
    candidate: torch.Tensor,
    scores: torch.Tensor,
    req_state: torch.Tensor,
    draft_logits_stride_0: int,
    draft_logits_stride_1: int,
    *,
    num_sample: int,
    num_steps: int,
    top_k: int,
    BLOCK_K: int,
) -> DispatchSpec:
    return (num_sample,), dict(
        num_steps=num_steps,
        top_k=top_k,
        BLOCK_K=BLOCK_K,
        num_warps=1,
    )
