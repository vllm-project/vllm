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
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.dp_utils import DPSyncState
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
    drafts_ptr,  # *int64  [B, K]  (output)
    has_match_ptr,  # *bool  [B]  (output)
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
    # Invalid slots fall back to the last sampled token; they are verified as
    # ordinary (rejectable) drafts, so the fill value only affects efficiency.
    out = tl.where(slot_valid, gathered, last_tok)
    tl.store(drafts_ptr + b * K + k_iota, out, mask=k_in_range)
    # A partial copy pads with rejectable filler, so it does not count as a
    # match for callers that would otherwise replace a full model draft.
    tl.store(has_match_ptr + b, write_ok & (tokens_avail >= K))


@triton.jit
def _one_hot_draft_logits_kernel(
    draft_logits_ptr,  # [max_num_reqs, K, V]
    draft_logits_stride_0,
    draft_logits_stride_1,
    idx_mapping_ptr,  # [B]
    has_match_ptr,  # [B]
    drafts_ptr,  # [B, K]
    drafts_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    if not tl.load(has_match_ptr + b):
        return
    step = tl.program_id(1)
    offs = tl.program_id(2) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    req_state_idx = tl.load(idx_mapping_ptr + b).to(tl.int64)
    token = tl.load(drafts_ptr + b * drafts_stride + step)
    row = (
        draft_logits_ptr
        + req_state_idx * draft_logits_stride_0
        + step * draft_logits_stride_1
    )
    vals = tl.where(offs == token, 0.0, float("-inf"))
    tl.store(row + offs, vals.to(row.dtype.element_ty), mask=offs < vocab_size)


def write_one_hot_draft_logits(
    draft_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    has_match: torch.Tensor,
    drafts: torch.Tensor,
) -> None:
    """Overwrite the cached draft logits of matched requests with one-hot rows.

    Copied drafts are deterministic, so probabilistic verification must see
    q = 1 on the copied token.
    """
    num_reqs, num_steps = drafts.shape
    block_size = 1024
    _one_hot_draft_logits_kernel[
        (num_reqs, num_steps, triton.cdiv(draft_logits.shape[-1], block_size))
    ](
        draft_logits,
        draft_logits.stride(0),
        draft_logits.stride(1),
        idx_mapping,
        has_match,
        drafts,
        drafts.stride(0),
        draft_logits.shape[-1],
        BLOCK_SIZE=block_size,
    )


class NgramLookup:
    """Batched GPU n-gram lookup over the request token history.

    For each request, finds the longest suffix of length in [min_n, max_n]
    that occurs earlier in its context (rightmost occurrence wins) and gathers
    the `num_tokens` tokens that followed it.
    """

    def __init__(
        self,
        min_n: int,
        max_n: int,
        num_tokens: int,
        max_num_reqs: int,
        max_model_len: int,
        device: torch.device,
    ):
        if not HAS_TRITON:
            raise RuntimeError("GPU n-gram lookup requires Triton.")
        assert 1 <= min_n <= max_n
        self.min_n = min_n
        self.max_n = max_n
        self.num_tokens = num_tokens
        self.max_model_len = max_model_len

        L = max_model_len
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
            (max_num_reqs, self.n_blocks), dtype=torch.int64, device=device
        )
        # Batch-ordered outputs.
        self.drafts = torch.zeros(
            (max_num_reqs, num_tokens), dtype=torch.int64, device=device
        )
        self.has_match = torch.zeros(max_num_reqs, dtype=torch.bool, device=device)

    def lookup(
        self,
        req_states: RequestState,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor,
        last_sampled: torch.Tensor,
        num_reqs: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ([num_reqs, num_tokens] drafts, [num_reqs] has_match).

        Requests without a match (or that sampled nothing this step) get their
        last sampled token in every slot and has_match False; so do the
        slots past a partial copy, which also leaves has_match False.
        """
        token_ids = req_states.all_token_ids.gpu
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
            self.has_match,
            self.max_model_len,
            self.n_blocks,
            self.num_tokens,
            max(1, triton.next_power_of_2(self.num_tokens)),
            max(1, triton.next_power_of_2(self.n_blocks)),
            num_warps=2,
            num_stages=1,
        )
        return self.drafts[:num_reqs], self.has_match[:num_reqs]


class NgramGPUSpeculator(BaseSpeculator):
    """V2-compatible GPU n-gram speculator."""

    supports_mm_inputs = False
    draft_logits = None

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        req_states: RequestState,
    ):
        spec = vllm_config.speculative_config
        assert spec is not None
        assert spec.prompt_lookup_min is not None, (
            "prompt_lookup_min must be configured for ngram_gpu"
        )
        assert spec.prompt_lookup_max is not None, (
            "prompt_lookup_max must be configured for ngram_gpu"
        )

        self.vllm_config = vllm_config
        self.device = device
        self.req_states = req_states
        self.speculative_config = spec
        self.num_speculative_steps: int = spec.num_speculative_tokens
        self.min_n: int = spec.prompt_lookup_min
        self.max_n: int = spec.prompt_lookup_max
        self.max_num_reqs: int = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len: int = vllm_config.model_config.max_model_len
        self.lookup = NgramLookup(
            self.min_n,
            self.max_n,
            self.num_speculative_steps,
            self.max_num_reqs,
            self.max_model_len,
            device,
        )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        del cudagraph_mode

    def capture(self) -> None:
        return None

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
        dp_sync_state: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        if dummy_run:
            return self.lookup.drafts[:num_reqs]
        drafts, _ = self.lookup.lookup(
            self.req_states,
            input_batch.idx_mapping,
            num_sampled,
            last_sampled,
            num_reqs,
        )
        return drafts
