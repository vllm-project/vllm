# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.v1.worker.gpu.input_batch import InputBatch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.states import RequestState


@support_torch_compile(
    dynamic_arg_dims={
        "token_ids": 0,
        "seq_lens": 0,
        "valid_mask": 0,
        "last_sampled": 0,
    }
)
class _NgramKernel(nn.Module):
    """GPU-accelerated N-gram proposer using fully async tensor operations."""

    def __init__(self, min_n: int, max_n: int, k: int):
        super().__init__()
        assert 1 <= min_n <= max_n, (
            f"min_n must be in [1, max_n]; got min_n={min_n}, max_n={max_n}"
        )
        assert k >= 1
        self.min_n = min_n
        self.max_n = max_n
        self.k = k
        self.num_sizes = max_n - min_n + 1

    def forward(
        self,
        token_ids: torch.Tensor,  # [B, L] int32
        seq_lens: torch.Tensor,  # [B]   int32  (current total_len per req)
        valid_mask: torch.Tensor,  # [B]   bool (row eligible for n-gram lookup)
        last_sampled: torch.Tensor,  # [B]   int64 (fallback for -1 positions)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """For each row, find the longest n-gram suffix match in its context
        and propose the next k tokens. Fully vectorized; no data-dependent
        control flow so the kernel stays torch.compile / CUDA-graph friendly.
        """
        B, L = token_ids.shape
        device = token_ids.device

        # Phase 1: For each n in [min_n, max_n], find the right-most position
        # of the length-n suffix inside each row (excluding the suffix itself).
        # first_match_pos[b, i] = match position for n=min_n+i, or -1.
        first_match_pos = torch.full(
            (B, self.num_sizes), -1, dtype=torch.long, device=device
        )
        batch_idx = torch.arange(B, device=device)

        for i, n in enumerate(range(self.min_n, self.max_n + 1)):
            # Sliding length-n windows; needle = length-n suffix of each row.
            windows = token_ids.unfold(1, n, 1)
            num_windows = windows.shape[1]
            suffix_start = (seq_lens.long() - n).clamp(min=0)
            offsets = torch.arange(n, device=device)
            suffix_idx = suffix_start.unsqueeze(1) + offsets
            suffix = torch.gather(token_ids, 1, suffix_idx)
            matches = (windows == suffix.unsqueeze(1)).all(dim=-1)

            # Mask out windows that overlap (or live past) the suffix.
            max_valid_pos = seq_lens.long() - n - 1
            window_pos = torch.arange(num_windows, device=device)
            matches = matches & (window_pos.unsqueeze(0) <= max_valid_pos.unsqueeze(1))

            # Right-most match via argmax on (pos if match else -1); re-check
            # the chosen index to distinguish a real match from the fallback.
            matched_indices = torch.where(matches, window_pos.unsqueeze(0), -1)
            idx = matched_indices.argmax(dim=1)
            has_match = matches[batch_idx, idx]
            first_match_pos[:, i] = torch.where(has_match, idx, -1)

        # Phase 2: Pick the largest n that produced a match (right-most True
        # along the n-axis).
        best_i = (first_match_pos >= 0).int().flip(dims=[1]).argmax(dim=1)
        best_i = self.num_sizes - 1 - best_i
        best_pos = first_match_pos[batch_idx, best_i]
        ngram_lens_table = torch.arange(
            self.min_n, self.max_n + 1, device=device, dtype=torch.long
        )
        best_n = ngram_lens_table[best_i]
        has_any = best_pos >= 0

        # Phase 3: Gather the next k tokens after the match. No-match rows
        # use draft_start=0 to keep indices in-bounds; they're masked below.
        draft_start = torch.where(
            has_any,
            best_pos + best_n,
            torch.zeros_like(best_pos),
        )
        k_range = torch.arange(self.k, device=device)
        draft_idx = (draft_start.unsqueeze(1) + k_range).clamp_(0, L - 1)
        drafts = torch.gather(token_ids, 1, draft_idx).to(torch.int64)

        # Phase 4: A slot j is valid iff the row has a match, valid_mask is
        # True, and j < seq_len - draft_start (so we don't read past context).
        # num_valid = length of the leading run of True values per row.
        tokens_available = (seq_lens.long() - draft_start).clamp_(min=0)
        valid_positions = k_range.unsqueeze(0) < tokens_available.unsqueeze(1)
        row_valid = has_any & valid_mask
        leading_valid_mask = valid_positions & row_valid.unsqueeze(1)
        cum_valid = leading_valid_mask.int().cumsum(dim=1)
        positions = torch.arange(1, self.k + 1, device=device)
        num_valid = (cum_valid == positions.unsqueeze(0)).int().sum(dim=1)

        # Phase 5: Replace invalid slots with last_sampled.
        safe_drafts = torch.where(
            leading_valid_mask,
            drafts,
            last_sampled.view(-1, 1).expand(B, self.k),
        )
        return safe_drafts, num_valid.to(torch.int32)


if HAS_TRITON:

    @triton.jit
    def _ngram_scan_kernel(
        token_ids_ptr,  # *int32  [B, L]
        seq_lens_ptr,  # *int32  [B]
        valid_mask_ptr,  # *int8   [B]
        scratch_ptr,  # *int64  [B, N_BLOCKS]   (output)
        L,  # int64 scalar
        L_PLUS_1,  # int64 scalar (= L + 1, used for packing)
        N_BLOCKS,  # int64 scalar (stride of scratch's second dim)
        MIN_N: tl.constexpr,
        MAX_N: tl.constexpr,
        MAX_N_PO2: tl.constexpr,
        BLOCK_L: tl.constexpr,
    ):
        b = tl.program_id(0).to(tl.int64)
        blk = tl.program_id(1).to(tl.int64)
        L_ = tl.cast(L, tl.int64)
        Lp1 = tl.cast(L_PLUS_1, tl.int64)
        NB = tl.cast(N_BLOCKS, tl.int64)

        seq_len = tl.load(seq_lens_ptr + b).to(tl.int64)
        row_valid = tl.load(valid_mask_ptr + b).to(tl.int1)
        eligible_row = row_valid & (seq_len >= MIN_N)

        scratch_off = b * NB + blk

        # Ineligible rows (or blocks past the last valid pos) write 0.
        if not eligible_row:
            tl.store(scratch_ptr + scratch_off, tl.zeros((), tl.int64))
            return

        row_off = b * L_

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

            cand = n_iter * Lp1 + pos + 1
            best_score = tl.where(match, cand, best_score)

        block_best = tl.max(best_score, axis=0)
        tl.store(scratch_ptr + scratch_off, block_best)

    @triton.jit
    def _ngram_finalize_kernel(
        token_ids_ptr,  # *int32  [B, L]
        seq_lens_ptr,  # *int32  [B]
        valid_mask_ptr,  # *int8   [B]
        last_sampled_ptr,  # *int64  [B]
        scratch_ptr,  # *int64  [B, N_BLOCKS]
        drafts_ptr,  # *int64  [B, K]   (output)
        num_valid_ptr,  # *int32  [B]      (output)
        L,
        L_PLUS_1,
        N_BLOCKS,
        K: tl.constexpr,
        K_PO2: tl.constexpr,
        N_BLOCKS_PO2: tl.constexpr,
    ):
        b = tl.program_id(0).to(tl.int64)
        L_ = tl.cast(L, tl.int64)
        Lp1 = tl.cast(L_PLUS_1, tl.int64)
        NB = tl.cast(N_BLOCKS, tl.int64)

        nb_iota = tl.arange(0, N_BLOCKS_PO2).to(tl.int64)
        nb_in_range = nb_iota < NB
        block_scores = tl.load(
            scratch_ptr + b * NB + nb_iota,
            mask=nb_in_range,
            other=0,
        )
        score = tl.max(block_scores, axis=0)

        seq_len = tl.load(seq_lens_ptr + b).to(tl.int64)
        row_valid = tl.load(valid_mask_ptr + b).to(tl.int1)
        last_tok = tl.load(last_sampled_ptr + b)

        has_match = score > 0
        s1 = score - 1
        best_n = tl.where(has_match, s1 // Lp1, tl.zeros_like(s1))
        best_pos = tl.where(has_match, s1 - best_n * Lp1, tl.zeros_like(s1))
        draft_start = tl.where(has_match, best_pos + best_n, tl.zeros_like(s1))

        tokens_avail = tl.maximum(seq_len - draft_start, 0)
        write_ok = row_valid & has_match
        nv = tl.where(write_ok, tl.minimum(tl.cast(K, tl.int64), tokens_avail), 0)
        tl.store(num_valid_ptr + b, nv.to(tl.int32))

        row_off = b * L_
        k_iota = tl.arange(0, K_PO2).to(tl.int64)
        k_in_range = k_iota < K
        gather_idx = tl.minimum(draft_start + k_iota, L_ - 1)
        slot_valid = (k_iota < tokens_avail) & write_ok & k_in_range
        gathered = tl.load(
            token_ids_ptr + row_off + gather_idx,
            mask=slot_valid,
            other=0,
        ).to(tl.int64)
        out = tl.where(slot_valid, gathered, last_tok)
        tl.store(drafts_ptr + b * K + k_iota, out, mask=k_in_range)


_NGRAM_SCRATCH: dict[tuple, torch.Tensor] = {}


def _get_ngram_scratch(B: int, n_blocks: int, device: torch.device) -> torch.Tensor:
    key = (device, B, n_blocks)
    buf = _NGRAM_SCRATCH.get(key)
    if buf is None:
        buf = torch.empty((B, n_blocks), dtype=torch.int64, device=device)
        _NGRAM_SCRATCH[key] = buf
    return buf


def _ngram_propose_triton(
    token_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    valid_mask: torch.Tensor,
    last_sampled: torch.Tensor,
    min_n: int,
    max_n: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each row, find the longest n-gram suffix match in its context
    and propose the next k tokens.
    """
    B, L = token_ids.shape
    device = token_ids.device

    drafts = torch.empty((B, k), dtype=torch.int64, device=device)
    num_valid = torch.empty((B,), dtype=torch.int32, device=device)

    if B == 0:
        return drafts, num_valid

    tok = token_ids.contiguous().to(torch.int32)
    seq = seq_lens.contiguous().to(torch.int32)
    vmask = valid_mask.contiguous().to(torch.bool).to(torch.int8)
    last = last_sampled.contiguous().to(torch.int64).view(-1)

    if L >= 1024:
        BLOCK_L = 256
    elif L >= 256:
        BLOCK_L = 128
    elif L >= 64:
        BLOCK_L = 64
    else:
        BLOCK_L = max(16, triton.next_power_of_2(max(L, 1)))

    K_PO2 = max(1, triton.next_power_of_2(k))
    MAX_N_PO2 = max(1, triton.next_power_of_2(max_n))
    n_blocks = (L + BLOCK_L - 1) // BLOCK_L
    n_blocks_po2 = max(1, triton.next_power_of_2(n_blocks))

    scratch = _get_ngram_scratch(B, n_blocks, device)

    L_plus_1 = L + 1
    _ngram_scan_kernel[(B, n_blocks)](
        tok,
        seq,
        vmask,
        scratch,
        L,
        L_plus_1,
        n_blocks,
        min_n,
        max_n,
        MAX_N_PO2,
        BLOCK_L,
        num_warps=4,
        num_stages=2,
    )

    _ngram_finalize_kernel[(B,)](
        tok,
        seq,
        vmask,
        last,
        scratch,
        drafts,
        num_valid,
        L,
        L_plus_1,
        n_blocks,
        k,
        K_PO2,
        n_blocks_po2,
        num_warps=2,
        num_stages=1,
    )
    return drafts, num_valid


class NgramGPUSpeculator:
    """
    V2-compatible GPU n-gram speculator.
    """

    supports_mm_inputs = False
    draft_logits = None

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
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
        self.speculative_config = spec
        self.num_speculative_steps: int = spec.num_speculative_tokens

        self.min_n: int = spec.prompt_lookup_min
        self.max_n: int = spec.prompt_lookup_max

        self.max_num_reqs: int = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len: int = vllm_config.model_config.max_model_len

        # Triton is the default fast path; the torch.compile kernel is
        # only constructed (and used) when Triton is unavailable.
        self.use_triton: bool = HAS_TRITON
        if self.use_triton:
            self.kernel: _NgramKernel
        else:
            self.kernel = (
                _NgramKernel(
                    min_n=self.min_n,
                    max_n=self.max_n,
                    k=self.num_speculative_steps,
                )
                .to(device)
                .eval()
            )

        self.req_states: RequestState | None = None

    def load_model(self, target_model: nn.Module) -> None:
        """No weights to load — ngram is a data-only proposer."""
        pass

    def set_attn(self, *args: Any, **kwargs: Any) -> None:
        """No attention layers owned by this speculator."""
        pass

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        """N-gram kernels are launched directly; no explicit CG capture."""
        pass

    def capture(self, *args: Any, **kwargs: Any) -> None:
        """No graph capture phase required."""
        pass

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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert self.req_states is not None, (
            "NgramGPUSpeculator.req_states was not injected by the model "
            "runner. Ensure model_runner sets `speculator.req_states = "
            "self.req_states` after RequestState is constructed."
        )

        idx_mapping_long = input_batch.idx_mapping.long()

        active_tokens: torch.Tensor = self.req_states.all_token_ids.gpu[
            idx_mapping_long
        ]
        active_seq_lens: torch.Tensor = self.req_states.total_len.gpu[idx_mapping_long]
        active_last_sampled: torch.Tensor = last_sampled.view(-1)[idx_mapping_long]

        valid_mask = (num_sampled > 0) & (active_seq_lens >= self.min_n)

        if self.use_triton:
            drafts, num_valid = _ngram_propose_triton(
                active_tokens,
                active_seq_lens,
                valid_mask,
                active_last_sampled,
                self.min_n,
                self.max_n,
                self.num_speculative_steps,
            )
        else:
            with set_forward_context(None, self.vllm_config):
                drafts, num_valid = self.kernel(
                    active_tokens,
                    active_seq_lens,
                    valid_mask,
                    active_last_sampled,
                )

        return drafts, num_valid
