# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.config.model import PROCESSED_LOGPROBS_MODES, LogprobsMode
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.v1.outputs import LogprobsTensors, SamplingMaskLists
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.watermarking.watermarker import RandomSampler, Watermarker
from vllm.v1.worker.cpu.gumbel import gumbel_sample
from vllm.v1.worker.cpu.input_batch import InputBatch, get_num_sampled_and_rejected
from vllm.v1.worker.cpu.logits_processors import (
    CPUBadWords,
    CPULogitBiasState,
    CPUPenalties,
    LogitsContext,
    LogitsProcessor,
    LogitsProcRequestState,
)
from vllm.v1.worker.cpu.logprob import LogprobTokenIdsState, compute_topk_scores
from vllm.v1.worker.cpu.states import CPURequestState

if TYPE_CHECKING:
    from vllm.config.reasoning import ReasoningConfig

logger = init_logger(__name__)

_INT32_MAX = np.iinfo(np.int32).max
_NP_INT64_MIN = np.iinfo(np.int64).min
_NP_INT64_MAX = np.iinfo(np.int64).max
NO_LOGPROBS = -1
MAX_COMPACT_SUPPORT = 2048


def get_num_nans(logits: torch.Tensor) -> torch.Tensor:
    return torch.isnan(logits).sum(dim=1, dtype=torch.int32)


class CPUThinkingBudgetState:
    """Model Runner V2 state for per-request thinking token budgets."""

    def __init__(
        self,
        req_states: CPURequestState,
        reasoning_config: ReasoningConfig | None,
    ) -> None:
        self.req_states = req_states
        self.max_num_reqs = req_states.max_num_reqs
        self.device = req_states.device
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

        self.thinking_token_budget = np.full(self.max_num_reqs, -1, dtype=np.int32)
        self.use_thinking_budget = np.zeros(self.max_num_reqs, dtype=bool)
        self.cached_last_start = np.full(self.max_num_reqs, -1, dtype=np.int32)
        self.cached_last_end = np.full(self.max_num_reqs, -1, dtype=np.int32)
        self.cached_scan_pos = np.zeros(self.max_num_reqs, dtype=np.int32)
        self.reasoning_start_token_ids = tuple(start_ids)
        self.natural_reasoning_end_token_ids = tuple(natural_end_ids)
        self.reasoning_end_token_ids = tuple(end_ids)

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        if not self.enabled:
            return False
        budget = sampling_params.thinking_token_budget
        use_thinking_budget = budget is not None
        self.use_thinking_budget[req_idx] = use_thinking_budget
        if budget is None:
            budget = -1
        else:
            budget = min(budget, _INT32_MAX)
            self.cached_last_start[req_idx] = -1
            self.cached_last_end[req_idx] = -1
            self.cached_scan_pos[req_idx] = 0
        self.thinking_token_budget[req_idx] = budget
        return use_thinking_budget

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
        total_len = int(self.req_states.total_len[req_idx])
        scan_pos = int(self.cached_scan_pos[req_idx])
        if scan_pos > total_len:
            scan_pos = 0
            self.cached_last_start[req_idx] = -1
            self.cached_last_end[req_idx] = -1

        tokens = self.req_states.all_token_ids[req_idx, scan_pos:total_len].tolist()
        self.cached_last_start[req_idx] = self._last_match(
            tokens,
            self.reasoning_start_token_ids,
            scan_pos,
            int(self.cached_last_start[req_idx]),
        )
        self.cached_last_end[req_idx] = self._last_match(
            tokens,
            self.natural_reasoning_end_token_ids,
            scan_pos,
            int(self.cached_last_end[req_idx]),
        )
        overlap = (
            max(
                len(self.reasoning_start_token_ids),
                len(self.natural_reasoning_end_token_ids),
            )
            - 1
        )
        self.cached_scan_pos[req_idx] = max(0, total_len - overlap)
        return total_len

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> None:
        if not self.enabled or not np.any(self.use_thinking_budget[ctx.idx_mapping_np]):
            return

        total_lens = {
            req_idx: self._update_committed_cache(int(req_idx))
            for req_idx in ctx.idx_mapping_np
            if self.use_thinking_budget[req_idx]
        }
        for row, req_idx in enumerate(ctx.expanded_idx_mapping.tolist()):
            if req_idx < 0 or not self.use_thinking_budget[req_idx]:
                continue

            total_len = total_lens[req_idx]
            local_pos = int(ctx.expanded_local_pos[row])
            first_row = row - local_pos
            overlap = (
                max(
                    len(self.reasoning_start_token_ids),
                    len(self.natural_reasoning_end_token_ids),
                    len(self.reasoning_end_token_ids),
                )
                - 1
            )
            tail_start = max(0, total_len - overlap)
            tokens = self.req_states.all_token_ids[
                req_idx, tail_start:total_len
            ].tolist()
            tokens.extend(ctx.input_ids[first_row + 1 : row + 1].tolist())

            last_start = self._last_match(
                tokens,
                self.reasoning_start_token_ids,
                tail_start,
                int(self.cached_last_start[req_idx]),
            )
            last_end = self._last_match(
                tokens,
                self.natural_reasoning_end_token_ids,
                tail_start,
                int(self.cached_last_end[req_idx]),
            )
            if last_start < 0 or last_start <= last_end:
                continue
            effective_len = total_len + local_pos
            if (
                effective_len - last_start - len(self.reasoning_start_token_ids)
                < self.thinking_token_budget[req_idx]
            ):
                continue

            end_prefix_len = 0
            for prefix_len in range(
                1, min(len(tokens), len(self.reasoning_end_token_ids) - 1) + 1
            ):
                if tokens[-prefix_len:] == list(
                    self.reasoning_end_token_ids[:prefix_len]
                ):
                    end_prefix_len = prefix_len
            logits[row, self.reasoning_end_token_ids[end_prefix_len]] = 1.0e9


class CPUTraceReplayState:
    """Per-request state for inference trace-replay.

    When a request carries ``SamplingParams.trace_decode_token_ids``, the
    sampler overwrites the sampled token at each decode step with the
    predetermined trace token, while real logprobs and ranks are still computed
    from the unmodified logit distribution.
    """

    def __init__(self, req_states: CPURequestState):
        self.max_num_reqs = req_states.max_num_reqs
        self.device = req_states.device
        self.req_states = req_states
        self.trace_token_ids = torch.zeros(
            (self.max_num_reqs, req_states.max_model_len),
            dtype=torch.int32,
            device=self.device,
        )
        self.trace_len = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        trace = sampling_params.trace_decode_token_ids
        if trace is not None:
            self.trace_len[req_idx] = len(trace)
            self.trace_token_ids[req_idx, : len(trace)] = torch.tensor(
                trace, dtype=torch.int32, device=self.trace_token_ids.device
            )
        else:
            self.trace_len[req_idx] = 0

    def apply_trace(self, sampled: torch.Tensor, idx_mapping: torch.Tensor) -> None:
        req_indices = idx_mapping.long()
        valid = req_indices >= 0
        safe_indices = req_indices.clamp_min(0)
        steps = (
            self.req_states.total_len[safe_indices]
            - self.req_states.prompt_len[safe_indices]
        ).long()
        trace_lens = self.trace_len[safe_indices].long()
        active = valid & (steps >= 0) & (steps < trace_lens)
        if torch.any(active):
            sampled[active] = self.trace_token_ids[
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


class CPUSamplingStates:
    def __init__(self, max_num_reqs: int, vocab_size: int, device: torch.device):
        self.max_num_reqs = max_num_reqs
        self.vocab_size = vocab_size

        self.temperature = torch.zeros(max_num_reqs, dtype=torch.float32, device=device)
        self.temperature_np = self.temperature.numpy()
        self.top_k = torch.full(
            (max_num_reqs,), vocab_size, dtype=torch.int32, device=device
        )
        self.top_k_np = self.top_k.numpy()
        self.top_p = torch.ones(max_num_reqs, dtype=torch.float32, device=device)
        self.top_p_np = self.top_p.numpy()
        self.min_p = torch.zeros(max_num_reqs, dtype=torch.float32, device=device)
        self.min_p_np = self.min_p.numpy()
        self.seeds = torch.zeros(max_num_reqs, dtype=torch.int64, device=device)
        self.seeds_np = self.seeds.numpy()
        # Tracks whether `seed` was set explicitly by the user, so callers
        # can fall back from RNG paths that don't honor per-request seeds.
        self.seeds_set = np.zeros(max_num_reqs, dtype=bool)

        self.num_logprobs = np.empty(self.max_num_reqs, dtype=np.int32)
        # -1 means no logprobs are requested.
        self.num_logprobs.fill(NO_LOGPROBS)

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        temperature = sampling_params.temperature
        top_k = sampling_params.top_k
        top_p = sampling_params.top_p
        min_p = sampling_params.min_p

        self.temperature_np[req_idx] = temperature
        self.top_p_np[req_idx] = top_p
        if top_k <= 0 or top_k > self.vocab_size:
            top_k = self.vocab_size
        self.top_k_np[req_idx] = top_k
        self.min_p_np[req_idx] = min_p

        seed = sampling_params.seed
        self.seeds_set[req_idx] = seed is not None
        if seed is None:
            seed = np.random.randint(_NP_INT64_MIN, _NP_INT64_MAX)
        self.seeds_np[req_idx] = seed

        num_logprobs = sampling_params.logprobs
        if num_logprobs is None:
            num_logprobs = NO_LOGPROBS
        elif num_logprobs == -1:
            num_logprobs = self.vocab_size
        self.num_logprobs[req_idx] = num_logprobs

        return temperature != 0.0 and (
            temperature != 1.0
            or min_p != 0.0
            or top_k != self.vocab_size
            or top_p != 1.0
        )

    def apply_temperature(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        temp_np = self.temperature_np[idx_mapping_np]
        if np.all((temp_np == 0.0) | (temp_np == 1.0)):
            # No request requires temperature.
            return

        temperature = self.temperature[expanded_idx_mapping].unsqueeze(1)
        logits.div_(torch.where(temperature == 0.0, 1.0, temperature))

    def apply_min_p(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        if np.all(self.min_p_np[idx_mapping_np] == 0.0):
            # No request uses min_p.
            return

        _apply_min_p(logits, expanded_idx_mapping, self.min_p)

    def get_top_k_top_p(
        self,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        vocab_size: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, bool]:
        """Returns (top_k, top_p, has_topp_only_rows)."""
        top_k_np = self.top_k_np[idx_mapping_np]
        top_p_np = self.top_p_np[idx_mapping_np]
        do_top_k = np.any(top_k_np != self.vocab_size)
        do_top_p = np.any(top_p_np != 1.0)
        top_k = self.top_k[expanded_idx_mapping] if do_top_k else None
        top_p = self.top_p[expanded_idx_mapping] if do_top_p else None
        has_topp_only_rows = bool(
            do_top_p and np.any((top_k_np >= vocab_size) & (top_p_np < 1.0))
        )
        return top_k, top_p, has_topp_only_rows

    def apply_top_k_top_p(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> torch.Tensor:
        top_k, top_p, has_topp_only_rows = self.get_top_k_top_p(
            expanded_idx_mapping, idx_mapping_np, logits.shape[-1]
        )
        if top_k is None and top_p is None:
            return logits
        return apply_top_k_top_p(
            logits, top_k, top_p, has_topp_only_rows=has_topp_only_rows
        )

    def any_greedy(self, idx_mapping_np: np.ndarray) -> bool:
        return bool(np.any(self.temperature_np[idx_mapping_np] == 0.0))

    def any_explicit_seed(self, idx_mapping_np: np.ndarray) -> bool:
        return bool(np.any(self.seeds_set[idx_mapping_np]))

    def max_num_logprobs(self, idx_mapping_np: np.ndarray) -> int:
        return int(np.max(self.num_logprobs[idx_mapping_np]))


@dataclass
class SamplerOutput:
    sampled_token_ids: torch.Tensor
    logprobs_tensors: LogprobsTensors | None
    num_nans: torch.Tensor | None
    num_sampled: torch.Tensor
    num_rejected: torch.Tensor
    sampling_mask_tensors: CPUSamplingMaskTensors | None = None


class CPUSamplingMaskTensors(NamedTuple):
    token_ids: torch.Tensor
    packed_mask: torch.Tensor
    counts: torch.Tensor
    vocab_size: int
    rows_per_request: int = 1

    @classmethod
    def from_logits(
        cls,
        logits: torch.Tensor,
        cu_num_logits: torch.Tensor,
        num_sampled_tokens: torch.Tensor,
        max_num_kept: int,
        rows_per_request: int = 1,
    ) -> CPUSamplingMaskTensors:
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

    @classmethod
    def cat(cls, chunks: Sequence[CPUSamplingMaskTensors]) -> CPUSamplingMaskTensors:
        first = chunks[0]
        assert len({(c.vocab_size, c.rows_per_request) for c in chunks}) == 1
        return cls(
            torch.cat([chunk.token_ids for chunk in chunks]),
            torch.cat([chunk.packed_mask for chunk in chunks]),
            torch.cat([chunk.counts for chunk in chunks]),
            first.vocab_size,
            first.rows_per_request,
        )

    def to_cpu_nonblocking(self) -> CPUSamplingMaskTensors:
        return self

    def tolists(
        self, num_sampled_tokens: np.ndarray | None = None
    ) -> SamplingMaskLists:
        counts = self.counts.numpy()
        token_ids = self.token_ids.numpy()
        packed_mask = self.packed_mask.numpy()
        width = token_ids.shape[1]

        cu_num_generated_tokens = None
        if self.rows_per_request == 1:
            sampled_rows = np.arange(len(counts))
        else:
            assert num_sampled_tokens is not None
            active_slots = (
                np.arange(self.rows_per_request)[None, :] < num_sampled_tokens[:, None]
            )
            sampled_rows = np.flatnonzero(active_slots)
            cu_num_generated_tokens = np.cumsum(
                np.concatenate(([0], num_sampled_tokens))
            ).tolist()

        def support(row: int) -> np.ndarray:
            if counts[row] <= width:
                return token_ids[row, : counts[row]]
            bits = np.unpackbits(
                packed_mask[row], count=self.vocab_size, bitorder="little"
            )
            return np.flatnonzero(bits).astype(np.int32, copy=False)

        supports = [support(row) for row in sampled_rows]
        offsets = np.zeros(len(supports) + 1, dtype=np.int64)
        np.cumsum([len(s) for s in supports], out=offsets[1:])
        flat = np.concatenate([np.empty(0, dtype=np.int32), *supports])
        return SamplingMaskLists(flat, offsets, cu_num_generated_tokens)


class CPUSampler:
    def __init__(
        self,
        vllm_config: VllmConfig,
        max_num_reqs: int,
        vocab_size: int,
        device: torch.device,
        req_states: CPURequestState,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        use_fp64_gumbel: bool = False,
        enable_trace_replay: bool = False,
        return_sampling_mask: bool = False,
    ):
        self.logprobs_mode = logprobs_mode
        self.compute_nans = envs.VLLM_COMPUTE_NANS_IN_LOGITS  # False by default.
        self.use_fp64_gumbel = use_fp64_gumbel

        self.req_states = req_states
        self.sampling_states = CPUSamplingStates(max_num_reqs, vocab_size, device)

        lp_req_state = LogitsProcRequestState.from_request_state(req_states)
        self.penalties_state = CPUPenalties(vllm_config, lp_req_state)
        logit_bias_state = CPULogitBiasState(vllm_config, lp_req_state)
        bad_words_state = CPUBadWords(vllm_config, lp_req_state)

        # List order is pipeline order: bias adds, penalties scale, so the
        # two do not commute.
        self.logits_processors: list[LogitsProcessor] = [
            logit_bias_state,
            self.penalties_state,
            bad_words_state,
        ]

        self.logprob_token_ids_state = LogprobTokenIdsState(max_num_reqs, device)
        self.thinking_budget_state = CPUThinkingBudgetState(
            req_states, vllm_config.reasoning_config
        )
        self.trace_replay_state = (
            CPUTraceReplayState(req_states) if enable_trace_replay else None
        )
        self.needs_logits_processing = np.zeros(max_num_reqs, dtype=bool)
        self.return_sampling_mask = return_sampling_mask

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        needs_processing = self.sampling_states.add_request(req_idx, sampling_params)
        needs_processing |= self.thinking_budget_state.add_request(
            req_idx, sampling_params
        )
        for processor in self.logits_processors:
            needs_processing |= processor.add_request(req_idx, sampling_params)
        self.needs_logits_processing[req_idx] = needs_processing

        self.logprob_token_ids_state.add_request(req_idx, sampling_params)
        if self.trace_replay_state is not None:
            self.trace_replay_state.add_request(req_idx, sampling_params)

    def get_logprobs_dims(
        self, idx_mapping_np: np.ndarray, include_token_ids: bool = True
    ) -> tuple[int, int] | None:
        """(num_logprobs, max_per_req_token_ids) for the given requests, or
        None when none of them want logprobs."""
        max_num_logprobs = self.sampling_states.max_num_logprobs(idx_mapping_np)
        max_token_ids = (
            self.logprob_token_ids_state.max_num_token_ids(idx_mapping_np)
            if include_token_ids
            else 0
        )
        if max_num_logprobs == NO_LOGPROBS and max_token_ids == 0:
            return None
        num_logprobs = max_num_logprobs if max_num_logprobs != NO_LOGPROBS else 0
        return num_logprobs, max_token_ids

    def __call__(self, logits: torch.Tensor, input_batch: InputBatch) -> SamplerOutput:
        expanded_idx_mapping = input_batch.expanded_idx_mapping
        idx_mapping = input_batch.idx_mapping
        idx_mapping_np = input_batch.idx_mapping_np
        cu_num_logits_np = input_batch.cu_num_logits_np
        expanded_local_pos = input_batch.expanded_local_pos
        pos = input_batch.positions[input_batch.logits_indices]
        input_ids = input_batch.input_ids[input_batch.logits_indices]
        seq_lens_upper_bound_np = input_batch.seq_lens_cpu_upper_bound.numpy()

        # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
        # that num_nans is computed before applying penalties and temperature.
        num_nans = get_num_nans(logits) if self.compute_nans else None

        logprobs_dims = self.get_logprobs_dims(idx_mapping_np)

        sampled, processed_logits = self.sample(
            logits,
            expanded_idx_mapping,
            idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
            seq_lens_upper_bound_np,
            return_logprobs=logprobs_dims is not None,
        )

        if self.trace_replay_state is not None:
            self.trace_replay_state.apply_trace(sampled, idx_mapping)

        if logprobs_dims is not None:
            num_logprobs, max_per_req_token_ids = logprobs_dims
            if self.logprobs_mode in PROCESSED_LOGPROBS_MODES:
                logits = processed_logits
            expanded_logits = logits.shape[0] != idx_mapping_np.shape[0]
            cu_num_logits = cu_num_logits_np.tolist() if expanded_logits else None
            logprobs_tensors = compute_topk_scores(
                logits,
                num_logprobs,
                sampled,
                cu_num_logits,
                num_per_req_token_ids=self.logprob_token_ids_state.num_token_ids,
                per_req_token_ids=self.logprob_token_ids_state.token_ids,
                expanded_idx_mapping=input_batch.expanded_idx_mapping,
                max_per_req_token_ids=max_per_req_token_ids,
                logits_mode=self.logprobs_mode in ("raw_logits", "processed_logits"),
            )
        else:
            logprobs_tensors = None

        # 1 sampled token per request, except chunked-prefill requests
        # (seq_len < prefill_len) which aren't done prefilling and produce no
        # output token. num_rejected is always 0 here (one logit per request).
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            input_batch.seq_lens.new_ones(input_batch.num_reqs),
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.req_states.prefill_len,
        )

        sampling_mask_tensors = None
        if self.return_sampling_mask:
            # Size by the validated top_k batch max; wider supports use the bitmask.
            max_num_kept = int(np.max(self.sampling_states.top_k_np[idx_mapping_np]))
            sampling_mask_tensors = CPUSamplingMaskTensors.from_logits(
                processed_logits,
                input_batch.cu_num_logits,
                num_sampled,
                max_num_kept,
            )

        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampled.view(-1, 1),
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            sampling_mask_tensors=sampling_mask_tensors,
        )
        return sampler_output

    def apply_sampling_params(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        seq_lens_upper_bound_np: np.ndarray,
        skip_top_k_top_p: bool = False,
    ) -> torch.Tensor:
        if not np.any(self.needs_logits_processing[idx_mapping_np]):
            return logits

        # Copy logits to a new FP32 tensor.
        logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)
        ctx = LogitsContext(
            expanded_idx_mapping=expanded_idx_mapping,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_local_pos=expanded_local_pos,
            input_ids=input_ids,
            pos=pos,
            seq_lens_upper_bound_np=seq_lens_upper_bound_np,
        )

        # Apply logits processors (native + any custom).
        for processor in self.logits_processors:
            logits = processor.apply(logits, ctx)

        # Forcing runs last so no stage can overwrite the forced end marker
        # or weaken it by scaling.
        self.thinking_budget_state.apply(logits, ctx)

        # Apply temperature in place.
        self.sampling_states.apply_temperature(
            logits, expanded_idx_mapping, idx_mapping_np
        )

        # Apply min_p in place.
        self.sampling_states.apply_min_p(logits, expanded_idx_mapping, idx_mapping_np)

        if skip_top_k_top_p:
            return logits
        return self.sampling_states.apply_top_k_top_p(
            logits, expanded_idx_mapping, idx_mapping_np
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
        del return_logprobs
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
        top_k, top_p, has_topp_only_rows = self.sampling_states.get_top_k_top_p(
            expanded_idx_mapping, idx_mapping_np, logits.shape[-1]
        )
        return self._sample_random(
            processed_logits,
            expanded_idx_mapping,
            idx_mapping_np,
            pos,
            top_k,
            top_p,
            has_topp_only_rows,
            use_fused_sampler=False,
        )

    def _sample_random(
        self,
        processed_logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        top_k: torch.Tensor | None,
        top_p: torch.Tensor | None,
        has_topp_only_rows: bool,
        use_fused_sampler: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert not use_fused_sampler

        # TODO (fadara01): Add C++ fused ops for top-p / top-k sampling.
        processed_logits = apply_top_k_top_p(
            processed_logits,
            top_k,
            top_p,
            has_topp_only_rows=has_topp_only_rows,
        )

        sampled = gumbel_sample(
            processed_logits,
            expanded_idx_mapping,
            self.sampling_states.temperature,
            self.sampling_states.seeds,
            pos,
            apply_temperature=False,
            is_drafting=False,
            use_fp64=self.use_fp64_gumbel,
        )
        return sampled, processed_logits


def _repeated_context_mask(
    all_token_ids: torch.Tensor,
    req_indices: torch.Tensor,
    prompt_lens: torch.Tensor,
    total_lens: torch.Tensor,
    contexts: torch.Tensor,
    max_history: int | None = None,
    include_prompt: bool = False,
    skip_partial_context: bool = False,
) -> torch.Tensor:
    repeated = torch.zeros(len(req_indices), dtype=torch.bool)
    for row, req_idx_tensor in enumerate(req_indices):
        req_idx = int(req_idx_tensor)
        if req_idx < 0:
            continue
        prompt_len = int(prompt_lens[req_idx])
        total_len = int(total_lens[req_idx])
        if skip_partial_context and total_len - prompt_len < contexts.shape[-1]:
            repeated[row] = True
            continue
        sequence_start = 0 if include_prompt else prompt_len
        history_tokens = all_token_ids[req_idx, sequence_start:total_len].tolist()
        prefix = [-1] * contexts.shape[-1]
        current_context = tuple(contexts[row].tolist())
        history_start = (
            max(0, len(history_tokens) - max(0, max_history))
            if max_history is not None
            else 0
        )
        for history_pos, token_id in enumerate(history_tokens):
            if history_pos >= history_start and (
                tuple(prefix[-contexts.shape[-1] :]) == current_context
            ):
                repeated[row] = True
                break
            prefix.append(token_id)
    return repeated


class CPUWatermarkSampler(CPUSampler):
    def __init__(
        self,
        watermarker: Watermarker,
        *args,
        deduplicate_contexts: Literal["none", "single_turn", "all"] = "single_turn",
        deduplicate_contexts_max_history: int | None = 8192,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.watermarker = watermarker
        self.deduplicate_contexts = deduplicate_contexts
        self.deduplicate_contexts_max_history = deduplicate_contexts_max_history
        self.watermarking = torch.ones(
            self.sampling_states.max_num_reqs,
            dtype=torch.bool,
            device=self.req_states.device,
        )
        self.watermarking_np = self.watermarking.numpy()

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        super().add_request(req_idx, sampling_params)
        self.watermarking_np[req_idx] = sampling_params.watermarking
        if sampling_params.watermarking and sampling_params.temperature == 0:
            logger.warning_once(
                "Watermarking is enabled, but greedy decoding "
                "(temperature=0) cannot be watermarked. This request will use "
                "ordinary greedy sampling."
            )

    def _sample_random(
        self,
        processed_logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        top_k: torch.Tensor | None,
        top_p: torch.Tensor | None,
        has_topp_only_rows: bool,
        use_fused_sampler: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        enabled = self.watermarking_np[idx_mapping_np] & (
            self.sampling_states.temperature_np[idx_mapping_np] != 0
        )
        if not np.any(enabled):
            return super()._sample_random(
                processed_logits,
                expanded_idx_mapping,
                idx_mapping_np,
                pos,
                top_k,
                top_p,
                has_topp_only_rows,
                use_fused_sampler,
            )

        processed_logits = apply_top_k_top_p(
            processed_logits,
            top_k,
            top_p,
            has_topp_only_rows=has_topp_only_rows,
        )
        contexts = self._get_contexts(expanded_idx_mapping)
        repeated_contexts = None
        if self.deduplicate_contexts != "none":
            repeated_contexts = self._get_repeated_contexts(
                expanded_idx_mapping, contexts
            )

        temperatures = self.sampling_states.temperature[expanded_idx_mapping]
        needs_mixed_sampling = repeated_contexts is not None or not np.all(enabled)
        skip_mask = None
        if needs_mixed_sampling:
            watermarking = self.watermarking[expanded_idx_mapping] & (temperatures != 0)
            if repeated_contexts is not None:
                watermarking &= ~repeated_contexts
            skip_mask = ~watermarking

        random_sampler = CPURandomSampler(
            expanded_idx_mapping=expanded_idx_mapping,
            temperatures=self.sampling_states.temperature,
            seeds=self.sampling_states.seeds,
            positions=pos,
            use_fp64=self.use_fp64_gumbel,
        )
        output = self.watermarker.sample(
            processed_logits,
            contexts,
            random_sampler,
            skip_mask=skip_mask,
        )
        sampled = output.token_ids
        if self.sampling_states.any_greedy(idx_mapping_np):
            sampled = torch.where(
                temperatures == 0,
                processed_logits.argmax(dim=-1),
                sampled,
            )
        return sampled, output.logits

    def _get_repeated_contexts(
        self,
        expanded_idx_mapping: torch.Tensor,
        contexts: torch.Tensor,
    ) -> torch.Tensor:
        return _repeated_context_mask(
            self.req_states.all_token_ids,
            expanded_idx_mapping,
            self.req_states.prompt_len,
            self.req_states.total_len,
            contexts,
            self.deduplicate_contexts_max_history,
            include_prompt=self.deduplicate_contexts == "all",
            skip_partial_context=self.deduplicate_contexts == "all",
        )

    def _get_contexts(self, expanded_idx_mapping: torch.Tensor) -> torch.Tensor:
        context_width = self.watermarker.context_width
        req_indices = expanded_idx_mapping.to(torch.int64)
        valid_reqs = req_indices >= 0
        safe_req_indices = req_indices.clamp_min(0)
        total_lens = self.req_states.total_len[safe_req_indices].to(torch.int64)
        prompt_lens = self.req_states.prompt_len[safe_req_indices].to(torch.int64)
        offsets = torch.arange(
            -context_width, 0, dtype=torch.int64, device=req_indices.device
        )
        positions = total_lens.unsqueeze(-1) + offsets
        valid = (
            valid_reqs.unsqueeze(-1)
            & (positions >= prompt_lens.unsqueeze(-1))
            & (positions < total_lens.unsqueeze(-1))
        )
        contexts = self.req_states.all_token_ids[
            safe_req_indices.unsqueeze(-1),
            positions.clamp(0, self.req_states.all_token_ids.shape[1] - 1),
        ]
        return torch.where(valid, contexts, -1)
