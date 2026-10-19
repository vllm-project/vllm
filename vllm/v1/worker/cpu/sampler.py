# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.watermarking.gpu_sampler import GPUWatermarkSampler
from vllm.v1.watermarking.watermarker import RandomSampler
from vllm.v1.worker.cpu.gumbel import gumbel_sample
from vllm.v1.worker.cpu.logits_processors import (
    CPUBadWords,
    CPULogitBiasState,
    CPUPenalties,
)
from vllm.v1.worker.cpu.logprob import compute_topk_scores as compute_topk_scores_cpu
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.logits_processor import LogitsContext
from vllm.v1.worker.gpu.sample.logprob import LogprobTokenIdsState
from vllm.v1.worker.gpu.sample.output import (
    MAX_COMPACT_SUPPORT,
    SamplingMaskTensors,
)
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import SamplingStates
from vllm.v1.worker.gpu.sample.trace_replay import TraceReplayState
from vllm.v1.worker.gpu.states import RequestState

if TYPE_CHECKING:
    from vllm.config.reasoning import ReasoningConfig

_INT32_MAX = np.iinfo(np.int32).max


class CPUThinkingBudgetState:
    def __init__(
        self,
        req_states: RequestState,
        reasoning_config: "ReasoningConfig | None",
    ) -> None:
        self.req_states = req_states
        self.max_num_reqs = req_states.max_num_reqs
        start_ids = (
            []
            if reasoning_config is None
            else reasoning_config.reasoning_start_token_ids or []
        )
        end_ids = (
            []
            if reasoning_config is None
            else reasoning_config.reasoning_end_token_ids or []
        )
        natural_end_ids = (
            []
            if reasoning_config is None
            else reasoning_config.natural_reasoning_end_token_ids or []
        )
        self.enabled = bool(start_ids and end_ids and natural_end_ids)
        if not self.enabled:
            return

        self.start_ids = tuple(start_ids)
        self.end_ids = tuple(end_ids)
        self.natural_end_ids = tuple(natural_end_ids)
        self.budgets = np.full(self.max_num_reqs, -1, dtype=np.int64)
        self.use_budget = np.zeros(self.max_num_reqs, dtype=bool)
        self.cached_last_start = np.full(self.max_num_reqs, -1, dtype=np.int64)
        self.cached_last_end = np.full(self.max_num_reqs, -1, dtype=np.int64)
        self.cached_scan_pos = np.zeros(self.max_num_reqs, dtype=np.int64)

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        if not self.enabled:
            return False
        budget = sampling_params.thinking_token_budget
        self.use_budget[req_idx] = budget is not None
        self.budgets[req_idx] = -1 if budget is None else min(budget, _INT32_MAX)
        self.cached_last_start[req_idx] = -1
        self.cached_last_end[req_idx] = -1
        self.cached_scan_pos[req_idx] = 0
        return budget is not None

    def apply_staged_writes(self) -> None:
        pass

    @staticmethod
    def _last_match(
        tokens: list[int],
        marker: tuple[int, ...],
        offset: int,
        current: int,
    ) -> int:
        marker_list = list(marker)
        for pos in range(len(tokens) - len(marker) + 1):
            if tokens[pos : pos + len(marker)] == marker_list:
                current = offset + pos
        return current

    def _update_committed_cache(self, req_idx: int) -> int:
        total_len = int(self.req_states.total_len.gpu[req_idx])
        scan_pos = int(self.cached_scan_pos[req_idx])
        if scan_pos > total_len:
            scan_pos = 0
            self.cached_last_start[req_idx] = -1
            self.cached_last_end[req_idx] = -1

        tokens = self.req_states.all_token_ids.gpu[req_idx, scan_pos:total_len].tolist()
        self.cached_last_start[req_idx] = self._last_match(
            tokens,
            self.start_ids,
            scan_pos,
            int(self.cached_last_start[req_idx]),
        )
        self.cached_last_end[req_idx] = self._last_match(
            tokens,
            self.natural_end_ids,
            scan_pos,
            int(self.cached_last_end[req_idx]),
        )
        overlap = max(len(self.start_ids), len(self.natural_end_ids)) - 1
        self.cached_scan_pos[req_idx] = max(0, total_len - overlap)
        return total_len

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> None:
        if not self.enabled or not np.any(self.use_budget[ctx.idx_mapping_np]):
            return

        total_lens = {
            req_idx: self._update_committed_cache(int(req_idx))
            for req_idx in ctx.idx_mapping_np
            if self.use_budget[req_idx]
        }
        for row, req_idx in enumerate(ctx.expanded_idx_mapping.tolist()):
            if req_idx < 0 or not self.use_budget[req_idx]:
                continue

            total_len = total_lens[req_idx]
            local_pos = int(ctx.expanded_local_pos[row])
            first_row = row - local_pos
            overlap = (
                max(
                    len(self.start_ids),
                    len(self.natural_end_ids),
                    len(self.end_ids),
                )
                - 1
            )
            tail_start = max(0, total_len - overlap)
            tokens = self.req_states.all_token_ids.gpu[
                req_idx, tail_start:total_len
            ].tolist()
            tokens.extend(ctx.input_ids[first_row + 1 : row + 1].tolist())

            last_start = self._last_match(
                tokens,
                self.start_ids,
                tail_start,
                int(self.cached_last_start[req_idx]),
            )
            last_end = self._last_match(
                tokens,
                self.natural_end_ids,
                tail_start,
                int(self.cached_last_end[req_idx]),
            )
            if last_start < 0 or last_start <= last_end:
                continue
            effective_len = total_len + local_pos
            if effective_len - last_start - len(self.start_ids) < self.budgets[req_idx]:
                continue

            end_prefix_len = 0
            for prefix_len in range(1, min(len(tokens), len(self.end_ids) - 1) + 1):
                if tokens[-prefix_len:] == list(self.end_ids[:prefix_len]):
                    end_prefix_len = prefix_len
            logits[row, self.end_ids[end_prefix_len]] = 1.0e9


class CPUTraceReplayState(TraceReplayState):
    def apply_trace(self, sampled: torch.Tensor, idx_mapping: torch.Tensor) -> None:
        req_indices = idx_mapping.long()
        valid = req_indices >= 0
        safe_indices = req_indices.clamp_min(0)
        steps = (
            self.req_states.total_len.gpu[safe_indices]
            - self.req_states.prompt_len.gpu[safe_indices]
        ).long()
        trace_lens = self.trace_len.gpu[safe_indices].long()
        active = valid & (steps >= 0) & (steps < trace_lens)
        if torch.any(active):
            sampled[active] = self.trace_token_ids.gpu[
                safe_indices[active], steps[active]
            ].to(sampled.dtype)


class CPURandomSampler(RandomSampler):
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return gumbel_sample(
            logits,
            self.expanded_idx_mapping,
            self.temperatures,
            self.seeds,
            self.positions,
            apply_temperature=False,
            is_drafting=self.is_drafting,
            logits_cache=self.logits_cache,
            logits_cache_col=self.logits_cache_col,
            logits_cache_source=self.logits_cache_source,
            use_fp64=self.use_fp64,
        )


@torch.compile(
    dynamic=True,
    fullgraph=True,
    backend=current_platform.simple_compile_backend,
)
def _apply_min_p(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    min_p_values: torch.Tensor,
) -> None:
    min_p = min_p_values[expanded_idx_mapping].unsqueeze(1)
    threshold = logits.amax(dim=1, keepdim=True) + min_p.log()
    logits.masked_fill_((min_p != 0.0) & (logits < threshold), float("-inf"))


class CPUSamplingStates(SamplingStates):
    def apply_temperature(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        temp_np = self.temperature.np[idx_mapping_np]
        if np.all((temp_np == 0.0) | (temp_np == 1.0)):
            return

        temperature = self.temperature.gpu[expanded_idx_mapping].unsqueeze(1)
        logits.div_(torch.where(temperature == 0.0, 1.0, temperature))

    def apply_min_p(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        if np.all(self.min_p.np[idx_mapping_np] == 0.0):
            return

        _apply_min_p(logits, expanded_idx_mapping, self.min_p.gpu)


class CPUSamplingMaskTensors(SamplingMaskTensors):
    @classmethod
    def from_logits(
        cls,
        logits: torch.Tensor,
        cu_num_logits: torch.Tensor,
        num_sampled_tokens: torch.Tensor,
        max_num_kept: int,
        rows_per_request: int = 1,
    ) -> SamplingMaskTensors:
        num_reqs, vocab_size = num_sampled_tokens.shape[0], logits.shape[1]
        max_num_kept = min(max_num_kept, vocab_size, MAX_COMPACT_SUPPORT)
        num_output_rows = num_reqs * rows_per_request
        token_ids = torch.empty(
            (num_output_rows, max_num_kept),
            dtype=torch.int32,
            device=logits.device,
        )
        packed_mask = torch.zeros(
            (num_output_rows, (vocab_size + 7) // 8),
            dtype=torch.uint8,
            device=logits.device,
        )
        counts = torch.zeros(num_output_rows, dtype=torch.int32, device=logits.device)
        bit_weights = 1 << torch.arange(8, dtype=torch.int16, device=logits.device)

        for output_row in range(num_output_rows):
            req_idx, slot_idx = divmod(output_row, rows_per_request)
            if slot_idx >= int(num_sampled_tokens[req_idx]):
                continue
            source_row = int(cu_num_logits[req_idx]) + slot_idx
            support = torch.isfinite(logits[source_row])
            kept_ids = torch.nonzero(support).flatten()
            counts[output_row] = kept_ids.numel()
            num_compact = min(kept_ids.numel(), max_num_kept)
            token_ids[output_row, :num_compact] = kept_ids[:num_compact]

            padded = torch.zeros(
                packed_mask.shape[1] * 8,
                dtype=torch.bool,
                device=logits.device,
            )
            padded[:vocab_size] = support
            packed_mask[output_row] = (
                (padded.view(-1, 8).to(torch.int16) * bit_weights)
                .sum(dim=1)
                .to(torch.uint8)
            )

        return cls(
            token_ids,
            packed_mask,
            counts,
            vocab_size,
            rows_per_request,
        )


class CPUSampler(Sampler):
    """CPU extension point for Model Runner V2 sampling."""

    sampling_states_cls = CPUSamplingStates
    sampling_mask_tensors_cls = CPUSamplingMaskTensors
    logit_bias_state_cls = CPULogitBiasState
    penalties_state_cls = CPUPenalties
    bad_words_state_cls = CPUBadWords
    thinking_budget_state_cls = CPUThinkingBudgetState
    trace_replay_state_cls = CPUTraceReplayState

    def _compute_topk_scores(
        self,
        logits: torch.Tensor,
        num_logprobs: int,
        sampled_token_ids: torch.Tensor,
        cu_num_logits: list[int] | torch.Tensor | None,
        logprob_token_ids_state: LogprobTokenIdsState,
        expanded_idx_mapping: torch.Tensor,
        max_per_req_token_ids: int,
        logits_mode: bool,
    ) -> LogprobsTensors:
        return compute_topk_scores_cpu(
            logits,
            num_logprobs,
            sampled_token_ids,
            cu_num_logits,
            logprob_token_ids_state=logprob_token_ids_state,
            expanded_idx_mapping=expanded_idx_mapping,
            max_per_req_token_ids=max_per_req_token_ids,
            logits_mode=logits_mode,
        )

    @staticmethod
    def _get_num_nans(logits: torch.Tensor) -> torch.Tensor:
        return torch.isnan(logits).sum(dim=1, dtype=torch.int32)

    def _get_num_sampled_and_rejected(
        self, input_batch: InputBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_reqs = input_batch.num_reqs
        num_sampled = input_batch.seq_lens.new_ones(num_reqs)
        num_rejected = torch.empty_like(num_sampled)
        is_chunked_prefilling = (
            input_batch.seq_lens[:num_reqs]
            < self.req_states.prefill_len.gpu[input_batch.idx_mapping]
        )
        num_sampled.masked_fill_(is_chunked_prefilling, 0)
        num_rejected.copy_(
            input_batch.cu_num_logits[1 : num_reqs + 1]
            - input_batch.cu_num_logits[:num_reqs]
        )
        num_rejected.sub_(num_sampled)
        num_rejected.masked_fill_(is_chunked_prefilling, 0)
        return num_sampled, num_rejected

    def _sample_random(
        self,
        processed_logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        top_k: torch.Tensor | None,
        top_p: torch.Tensor | None,
        use_fused_sampler: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert not use_fused_sampler

        # TODO (fadara01): Add C++ fused ops for top-p / top-k sampling.
        processed_logits = apply_top_k_top_p(processed_logits, top_k, top_p)

        sampled = gumbel_sample(
            processed_logits,
            expanded_idx_mapping,
            self.sampling_states.temperature.gpu,
            self.sampling_states.seeds.gpu,
            pos,
            apply_temperature=False,
            is_drafting=False,
            use_fp64=self.use_fp64_gumbel,
        )
        return sampled, processed_logits


class CPUWatermarkSampler(GPUWatermarkSampler, CPUSampler):
    random_sampler_cls = CPURandomSampler
