# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import numpy as np
import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.logits_processor import (
    LogitsContext,
    LogitsProcessor,
    LogitsProcRequestState,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class CPULogitBiasState(LogitBiasState):
    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        if not np.any(self.use_logit_bias[ctx.idx_mapping_np]):
            return logits

        ops.apply_logit_bias_cpu(
            logits,
            ctx.expanded_idx_mapping,
            ctx.pos,
            self.num_allowed_token_ids.gpu,
            self.allowed_token_ids.gpu,
            self.num_logit_bias.gpu,
            self.logit_bias_token_ids.gpu,
            self.logit_bias.gpu,
            self.min_lens.gpu,
            self.num_stop_token_ids.gpu,
            self.restore_when_all_masked.gpu,
            self.stop_token_ids.gpu,
            bool(np.any(self.restore_when_all_masked.np[ctx.idx_mapping_np])),
        )
        return logits


class _CPULogitsProcessor(LogitsProcessor):
    def __init__(self, vllm_config: "VllmConfig", req_states: LogitsProcRequestState):
        self.req_states = req_states

    def _request_rows(self, ctx: LogitsContext):
        for row, req_idx in enumerate(ctx.expanded_idx_mapping.tolist()):
            if req_idx >= 0:
                yield row, req_idx

    def _output_tokens(
        self, req_idx: int, row: int, ctx: LogitsContext
    ) -> torch.Tensor:
        prompt_len = int(self.req_states.prompt_len.np[req_idx])
        total_len = int(self.req_states.total_len.gpu[req_idx])
        first_row = row - int(ctx.expanded_local_pos[row])
        return torch.cat(
            (
                self.req_states.all_token_ids.gpu[req_idx, prompt_len:total_len],
                ctx.input_ids[first_row + 1 : row + 1].to(torch.int32),
            )
        )


class CPUPenalties(_CPULogitsProcessor):
    """Apply penalties only to token IDs present in the request history."""

    output_bin_counts = None

    def __init__(self, vllm_config: "VllmConfig", req_states: LogitsProcRequestState):
        super().__init__(vllm_config, req_states)
        self.repetition_penalty = torch.ones(
            req_states.max_num_reqs, dtype=torch.float32, device=req_states.device
        )
        self.frequency_penalty = torch.zeros_like(self.repetition_penalty)
        self.presence_penalty = torch.zeros_like(self.repetition_penalty)

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        repetition = sampling_params.repetition_penalty
        frequency = sampling_params.frequency_penalty
        presence = sampling_params.presence_penalty
        self.repetition_penalty[req_idx] = repetition
        self.frequency_penalty[req_idx] = frequency
        self.presence_penalty[req_idx] = presence
        return repetition != 1.0 or frequency != 0.0 or presence != 0.0

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        req_indices = ctx.expanded_idx_mapping.long()
        repetition_by_row = self.repetition_penalty[req_indices]
        frequency_by_row = self.frequency_penalty[req_indices]
        presence_by_row = self.presence_penalty[req_indices]
        active_rows = (
            (repetition_by_row != 1.0)
            | (frequency_by_row != 0.0)
            | (presence_by_row != 0.0)
        ).nonzero(as_tuple=True)[0]
        if active_rows.numel() == 0:
            return logits

        vocab_size = logits.shape[1]
        prompt_ids = []
        output_ids = []
        req_indices_list = req_indices.tolist()
        for row in active_rows.tolist():
            req_idx = req_indices_list[row]
            offset = row * vocab_size
            prompt_len = int(self.req_states.prompt_len.np[req_idx])
            prompt_ids.append(
                self.req_states.all_token_ids.gpu[req_idx, :prompt_len] + offset
            )
            output_ids.append(self._output_tokens(req_idx, row, ctx) + offset)

        flat_logits = logits.view(-1)
        output_ids_flat, counts = torch.unique(
            torch.cat(output_ids).long(), return_counts=True
        )
        seen_ids = torch.unique(torch.cat((*prompt_ids, output_ids_flat)).long())

        repetition = repetition_by_row[seen_ids // vocab_size]
        seen_logits = flat_logits[seen_ids]
        flat_logits[seen_ids] = torch.where(
            seen_logits > 0,
            seen_logits / repetition,
            seen_logits * repetition,
        )
        output_rows = output_ids_flat // vocab_size
        frequency = frequency_by_row[output_rows]
        presence = presence_by_row[output_rows]
        flat_logits[output_ids_flat] -= frequency * counts + presence

        return logits


class CPUBadWords(_CPULogitsProcessor):
    def __init__(self, vllm_config: "VllmConfig", req_states: LogitsProcRequestState):
        super().__init__(vllm_config, req_states)
        self.bad_words: list[tuple[tuple[int, ...], ...] | None] = [
            None
        ] * req_states.max_num_reqs

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        bad_words = sampling_params.bad_words_token_ids
        if not bad_words:
            self.bad_words[req_idx] = None
            return False

        if len(bad_words) > envs.VLLM_MAX_NUM_BAD_WORDS:
            raise ValueError(
                f"Too many bad words: {len(bad_words)}. "
                f"The max number is {envs.VLLM_MAX_NUM_BAD_WORDS}."
            )
        num_tokens = sum(map(len, bad_words))
        if num_tokens > envs.VLLM_MAX_BAD_WORDS_TOTAL_TOKENS:
            raise ValueError(
                f"Too many total bad word tokens: {num_tokens}. "
                f"The max is {envs.VLLM_MAX_BAD_WORDS_TOTAL_TOKENS}."
            )

        self.bad_words[req_idx] = tuple(tuple(word) for word in bad_words)
        return True

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        for row, req_idx in self._request_rows(ctx):
            bad_words = self.bad_words[req_idx]
            if bad_words is None:
                continue
            history = self._output_tokens(req_idx, row, ctx).tolist()
            for bad_word in bad_words:
                prefix = bad_word[:-1]
                if len(prefix) <= len(history) and (
                    not prefix or history[-len(prefix) :] == list(prefix)
                ):
                    logits[row, bad_word[-1]] = -torch.inf
        return logits
