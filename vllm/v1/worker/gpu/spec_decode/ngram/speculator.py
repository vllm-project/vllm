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
        B, L = token_ids.shape
        device = token_ids.device

        first_match_pos = torch.full(
            (B, self.num_sizes), -1, dtype=torch.long, device=device
        )
        batch_idx = torch.arange(B, device=device)

        for i, n in enumerate(range(self.min_n, self.max_n + 1)):
            windows = token_ids.unfold(1, n, 1)
            num_windows = windows.shape[1]

            suffix_start = (seq_lens.long() - n).clamp(min=0)
            offsets = torch.arange(n, device=device)
            suffix_idx = suffix_start.unsqueeze(1) + offsets
            suffix = torch.gather(token_ids, 1, suffix_idx)

            matches = (windows == suffix.unsqueeze(1)).all(dim=-1)

            max_valid_pos = seq_lens.long() - n - 1
            window_pos = torch.arange(num_windows, device=device)
            matches = matches & (window_pos.unsqueeze(0) <= max_valid_pos.unsqueeze(1))

            idx = matches.int().argmax(dim=1)
            has_match = matches[batch_idx, idx]
            first_match_pos[:, i] = torch.where(has_match, idx.long(), -1)

        best_i = (first_match_pos >= 0).int().flip(dims=[1]).argmax(dim=1)
        best_i = self.num_sizes - 1 - best_i
        best_pos = first_match_pos[batch_idx, best_i]
        ngram_lens_table = torch.arange(
            self.min_n, self.max_n + 1, device=device, dtype=torch.long
        )
        best_n = ngram_lens_table[best_i]
        has_any = best_pos >= 0

        draft_start = torch.where(
            has_any,
            best_pos + best_n,
            torch.zeros_like(best_pos),
        )
        k_range = torch.arange(self.k, device=device)
        draft_idx = (draft_start.unsqueeze(1) + k_range).clamp_(0, L - 1)
        drafts = torch.gather(token_ids, 1, draft_idx).to(torch.int64)

        tokens_available = (seq_lens.long() - draft_start).clamp_(min=0)
        valid_positions = k_range.unsqueeze(0) < tokens_available.unsqueeze(1)
        row_valid = has_any & valid_mask
        leading_valid_mask = valid_positions & row_valid.unsqueeze(1)

        cum_valid = leading_valid_mask.int().cumsum(dim=1)
        positions = torch.arange(1, self.k + 1, device=device)
        num_valid = (cum_valid == positions.unsqueeze(0)).int().sum(dim=1)

        safe_drafts = torch.where(
            leading_valid_mask,
            drafts,
            last_sampled.view(-1, 1).expand(B, self.k),
        )
        return safe_drafts, num_valid.to(torch.int32)


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

        self.kernel = (
            _NgramKernel(
                min_n=self.min_n,
                max_n=self.max_n,
                k=self.num_speculative_steps,
            )
            .to(device)
            .eval()
        )

        self.num_valid_draft_tokens: torch.Tensor = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )

        self.req_states: RequestState | None = None

    def load_model(self, target_model: nn.Module) -> None:
        """No weights to load — ngram is a data-only proposer."""
        pass

    def set_attn(self, *args: Any, **kwargs: Any) -> None:
        """No attention layers owned by this speculator."""
        pass

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        """N-gram kernel is torch.compile-managed; no explicit CG capture."""
        pass

    def capture_model(self) -> None:
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
    ) -> torch.Tensor:
        assert self.req_states is not None, (
            "NgramGPUSpeculator.req_states was not injected by the model "
            "runner. Ensure model_runner sets `speculator.req_states = "
            "self.req_states` after RequestState is constructed."
        )

        num_reqs = input_batch.num_reqs
        idx_mapping_long = input_batch.idx_mapping.long()

        active_tokens: torch.Tensor = self.req_states.all_token_ids.gpu[
            idx_mapping_long
        ]
        active_seq_lens: torch.Tensor = self.req_states.total_len.gpu[idx_mapping_long]
        active_last_sampled: torch.Tensor = last_sampled.view(-1)[idx_mapping_long]

        valid_mask = (num_sampled > 0) & (active_seq_lens >= self.min_n)

        with set_forward_context(None, self.vllm_config):
            drafts, num_valid = self.kernel(
                active_tokens,
                active_seq_lens,
                valid_mask,
                active_last_sampled,
            )

        self.num_valid_draft_tokens[:num_reqs].copy_(num_valid)
        if num_reqs < self.max_num_reqs:
            self.num_valid_draft_tokens[num_reqs:].zero_()

        return drafts

    def get_num_valid_draft_tokens(self, num_reqs: int) -> torch.Tensor:
        """
        Return the last step's per-request valid draft counts.
        """
        return self.num_valid_draft_tokens[:num_reqs]
