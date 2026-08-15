# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.states import RequestState


@triton.jit
def _ngram_scan_kernel(
    token_ids_ptr,  # *int32  [max_num_reqs, token_ids_stride]
    token_ids_stride,
    idx_mapping_ptr,  # *int64  [B]  batch_idx -> req_state_idx
    total_len_ptr,  # *int32  [max_num_reqs]
    num_sampled_ptr,  # *int32  [B]
    scratch_ptr,  # *int64  [B, scratch_stride]  (output)
    scratch_stride,
    L,  # int64 scalar (= max_model_len)
    MIN_N: tl.constexpr,
    MAX_N: tl.constexpr,
    MAX_N_PO2: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1).to(tl.int64)
    Lp1 = tl.cast(L, tl.int64) + 1

    req_state_idx = tl.load(idx_mapping_ptr + b).to(tl.int64)
    seq_len = tl.load(total_len_ptr + req_state_idx).to(tl.int64)
    num_sampled = tl.load(num_sampled_ptr + b)
    eligible_row = (num_sampled > 0) & (seq_len >= MIN_N)

    scratch_off = b * scratch_stride + blk

    # Ineligible rows, and blocks fully past the last candidate match
    # position, write 0 and exit.
    if not (eligible_row & (blk * BLOCK_L <= seq_len - MIN_N - 1)):
        tl.store(scratch_ptr + scratch_off, tl.zeros((), tl.int64))
        return

    row_off = req_state_idx * token_ids_stride

    # Load the length-MAX_N suffix once into registers.
    suf_iota = tl.arange(0, MAX_N_PO2).to(tl.int64)
    suf_pos = seq_len - MAX_N + suf_iota
    suf_in_range = (suf_iota < MAX_N) & (suf_pos >= 0) & (suf_pos < seq_len)
    suffix = tl.load(
        token_ids_ptr + row_off + suf_pos,
        mask=suf_in_range,
        other=-1,
    ).to(tl.int32)

    pos_iota = tl.arange(0, BLOCK_L).to(tl.int64)
    pos = blk * BLOCK_L + pos_iota  # ascending

    best_score = tl.zeros([BLOCK_L], dtype=tl.int64)

    for n_iter in tl.static_range(MIN_N, MAX_N + 1):
        max_pos_n = seq_len - n_iter - 1
        match = (pos >= 0) & (pos <= max_pos_n)
        for j in tl.static_range(0, n_iter):
            tok = tl.load(
                token_ids_ptr + row_off + (pos + j),
                mask=match,
                other=0,
            ).to(tl.int32)
            suf_idx = (MAX_N - n_iter) + j
            suf_val = tl.sum(tl.where(suf_iota == suf_idx, suffix, 0))
            match = match & (tok == suf_val)

        # Pack (n, pos) so a single max yields longest-n, rightmost-pos.
        cand = n_iter * Lp1 + pos + 1
        best_score = tl.where(match, cand, best_score)

    block_best = tl.max(best_score, axis=0)
    tl.store(scratch_ptr + scratch_off, block_best)


@triton.jit
def _ngram_finalize_kernel(
    token_ids_ptr,  # *int32  [max_num_reqs, token_ids_stride]
    token_ids_stride,
    idx_mapping_ptr,  # *int64  [B]
    total_len_ptr,  # *int32  [max_num_reqs]
    num_sampled_ptr,  # *int32  [B]
    last_sampled_ptr,  # *int64  [max_num_reqs]
    scratch_ptr,  # *int64  [B, scratch_stride]
    scratch_stride,
    drafts_ptr,  # *int64  [B, K]           (output, batch indexed)
    num_valid_ptr,  # *int32  [max_num_reqs]   (output, req-slot indexed)
    L,
    N_BLOCKS,
    K: tl.constexpr,
    K_PO2: tl.constexpr,
    N_BLOCKS_PO2: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    Lp1 = tl.cast(L, tl.int64) + 1
    NB = tl.cast(N_BLOCKS, tl.int64)

    req_state_idx = tl.load(idx_mapping_ptr + b).to(tl.int64)

    nb_iota = tl.arange(0, N_BLOCKS_PO2).to(tl.int64)
    nb_in_range = nb_iota < NB
    block_scores = tl.load(
        scratch_ptr + b * scratch_stride + nb_iota,
        mask=nb_in_range,
        other=0,
    )
    score = tl.max(block_scores, axis=0)

    seq_len = tl.load(total_len_ptr + req_state_idx).to(tl.int64)
    num_sampled = tl.load(num_sampled_ptr + b)
    last_tok = tl.load(last_sampled_ptr + req_state_idx)

    has_match = score > 0
    s1 = score - 1
    best_n = tl.where(has_match, s1 // Lp1, tl.zeros_like(s1))
    best_pos = tl.where(has_match, s1 - best_n * Lp1, tl.zeros_like(s1))
    draft_start = tl.where(has_match, best_pos + best_n, tl.zeros_like(s1))

    tokens_avail = tl.maximum(seq_len - draft_start, 0)
    write_ok = (num_sampled > 0) & has_match
    nv = tl.where(write_ok, tl.minimum(tl.cast(K, tl.int64), tokens_avail), 0)
    tl.store(num_valid_ptr + req_state_idx, nv.to(tl.int32))

    row_off = req_state_idx * token_ids_stride
    k_iota = tl.arange(0, K_PO2).to(tl.int64)
    k_in_range = k_iota < K
    gather_idx = tl.minimum(draft_start + k_iota, tl.cast(L, tl.int64) - 1)
    slot_valid = (k_iota < tokens_avail) & write_ok & k_in_range
    gathered = tl.load(
        token_ids_ptr + row_off + gather_idx,
        mask=slot_valid,
        other=0,
    ).to(tl.int64)
    # Invalid slots fall back to the last sampled token; they are either
    # trimmed from the verification batch on GPU or verified as ordinary
    # (rejectable) drafts, so the fill value only affects efficiency.
    out = tl.where(slot_valid, gathered, last_tok)
    tl.store(drafts_ptr + b * K + k_iota, out, mask=k_in_range)


class NgramGPUSpeculator(BaseSpeculator):
    """V2-compatible GPU n-gram speculator."""

    supports_mm_inputs = False
    draft_logits = None
    # Signals that num_valid_drafts holds per-request valid draft counts
    # for GPU-side verification trimming.
    trims_drafts_on_gpu = True

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        if not HAS_TRITON:
            raise RuntimeError("ngram_gpu speculative decoding requires Triton.")
        spec = vllm_config.speculative_config
        assert spec is not None
        assert spec.prompt_lookup_min is not None, (
            "prompt_lookup_min must be configured for ngram_gpu"
        )
        assert spec.prompt_lookup_max is not None, (
            "prompt_lookup_max must be configured for ngram_gpu"
        )
        assert 1 <= spec.prompt_lookup_min <= spec.prompt_lookup_max

        self.vllm_config = vllm_config
        self.device = device
        self.speculative_config = spec
        self.num_speculative_steps: int = spec.num_speculative_tokens

        self.min_n: int = spec.prompt_lookup_min
        self.max_n: int = spec.prompt_lookup_max

        self.max_num_reqs: int = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len: int = vllm_config.model_config.max_model_len

        L = self.max_model_len
        if L >= 1024:
            self.block_l = 256
        elif L >= 256:
            self.block_l = 128
        elif L >= 64:
            self.block_l = 64
        else:
            self.block_l = max(16, triton.next_power_of_2(max(L, 1)))
        self.n_blocks = triton.cdiv(L, self.block_l)

        self.scratch = torch.zeros(
            (self.max_num_reqs, self.n_blocks), dtype=torch.int64, device=device
        )
        # Per request-slot count of usable drafts from the latest proposal,
        # consumed by the model runner's GPU draft trimmer.
        self.num_valid_drafts = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        # Batch-ordered draft output, scattered into RequestState.draft_tokens
        # by the model runner (same contract as the model-based speculators).
        self.drafts = torch.zeros(
            (self.max_num_reqs, self.num_speculative_steps),
            dtype=torch.int64,
            device=device,
        )

        self.req_states: RequestState | None = None

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: Any,
        slot_mappings: Any,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        if dummy_run:
            # No persistent request state may be touched during dummy runs.
            return self.drafts[:num_reqs]

        req_states = self.req_states
        assert req_states is not None, (
            "NgramGPUSpeculator.req_states was not injected by the model runner."
        )

        token_ids = req_states.all_token_ids.gpu
        idx_mapping = input_batch.idx_mapping

        _ngram_scan_kernel[(num_reqs, self.n_blocks)](
            token_ids,
            token_ids.stride(0),
            idx_mapping,
            req_states.total_len.gpu,
            num_sampled,
            self.scratch,
            self.scratch.stride(0),
            self.max_model_len,
            self.min_n,
            self.max_n,
            max(1, triton.next_power_of_2(self.max_n)),
            self.block_l,
            num_warps=4,
            num_stages=2,
        )

        _ngram_finalize_kernel[(num_reqs,)](
            token_ids,
            token_ids.stride(0),
            idx_mapping,
            req_states.total_len.gpu,
            num_sampled,
            last_sampled.view(-1),
            self.scratch,
            self.scratch.stride(0),
            self.drafts,
            self.num_valid_drafts,
            self.max_model_len,
            self.n_blocks,
            self.num_speculative_steps,
            max(1, triton.next_power_of_2(self.num_speculative_steps)),
            max(1, triton.next_power_of_2(self.n_blocks)),
            num_warps=2,
            num_stages=1,
        )
        return self.drafts[:num_reqs]
