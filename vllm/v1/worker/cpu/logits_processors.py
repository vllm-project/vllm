# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.sampling_params import SamplingParams

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.worker.cpu.states import CPURequestState

MAX_NUM_ALLOWED_TOKEN_IDS = 1024
MAX_NUM_LOGIT_BIAS_TOKENS = 1024
MAX_NUM_STOP_TOKEN_IDS = 128


@dataclass(frozen=True)
class LogitsProcRequestState:
    device: torch.device
    max_num_reqs: int
    vocab_size: int
    all_token_ids: torch.Tensor
    prompt_len: torch.Tensor
    prompt_len_np: np.ndarray
    prefill_len: torch.Tensor
    prefill_len_np: np.ndarray
    total_len: torch.Tensor

    @classmethod
    def from_request_state(cls, req_states: CPURequestState) -> LogitsProcRequestState:
        return cls(
            device=req_states.device,
            max_num_reqs=req_states.max_num_reqs,
            vocab_size=req_states.vocab_size,
            all_token_ids=req_states.all_token_ids,
            prompt_len=req_states.prompt_len,
            prompt_len_np=req_states.prompt_len_np,
            prefill_len=req_states.prefill_len,
            prefill_len_np=req_states.prefill_len_np,
            total_len=req_states.total_len,
        )


@dataclass(frozen=True)
class LogitsContext:
    expanded_idx_mapping: torch.Tensor
    idx_mapping: torch.Tensor
    idx_mapping_np: np.ndarray
    expanded_local_pos: torch.Tensor
    input_ids: torch.Tensor
    pos: torch.Tensor
    seq_lens_upper_bound_np: np.ndarray


class LogitsProcessor(ABC):
    def __init__(  # noqa: B027
        self, vllm_config: VllmConfig, req_states: LogitsProcRequestState
    ) -> None:
        pass

    @classmethod  # noqa: B027
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        pass

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        return True

    @abstractmethod
    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        raise NotImplementedError


class CPULogitBiasState(LogitsProcessor):
    def __init__(self, vllm_config: VllmConfig, req_states: LogitsProcRequestState):
        max_num_reqs = req_states.max_num_reqs
        device = req_states.device
        self.prompt_len_np = req_states.prompt_len_np

        self.num_allowed_token_ids = torch.zeros(
            max_num_reqs, dtype=torch.int32, device=device
        )
        self.num_allowed_token_ids_np = self.num_allowed_token_ids.numpy()
        self.allowed_token_ids = torch.zeros(
            (max_num_reqs, MAX_NUM_ALLOWED_TOKEN_IDS),
            dtype=torch.int32,
            device=device,
        )
        self.num_logit_bias = torch.zeros(
            max_num_reqs, dtype=torch.int32, device=device
        )
        self.num_logit_bias_np = self.num_logit_bias.numpy()
        self.logit_bias_token_ids = torch.zeros(
            (max_num_reqs, MAX_NUM_LOGIT_BIAS_TOKENS),
            dtype=torch.int32,
            device=device,
        )
        self.logit_bias = torch.zeros(
            (max_num_reqs, MAX_NUM_LOGIT_BIAS_TOKENS),
            dtype=torch.float32,
            device=device,
        )
        self.min_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.min_lens_np = self.min_lens.numpy()
        self.num_stop_token_ids = torch.zeros(
            max_num_reqs, dtype=torch.int32, device=device
        )
        self.num_stop_token_ids_np = self.num_stop_token_ids.numpy()
        self.restore_when_all_masked = torch.zeros(
            max_num_reqs, dtype=torch.int32, device=device
        )
        self.restore_when_all_masked_np = self.restore_when_all_masked.numpy()
        self.stop_token_ids = torch.zeros(
            (max_num_reqs, MAX_NUM_STOP_TOKEN_IDS),
            dtype=torch.int32,
            device=device,
        )
        self.use_logit_bias = np.zeros(max_num_reqs, dtype=bool)

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        prompt_len = int(self.prompt_len_np[req_idx])
        use_logit_bias = False

        allowed_token_ids = sampling_params.allowed_token_ids
        if allowed_token_ids:
            num_allowed_token_ids = len(allowed_token_ids)
            if num_allowed_token_ids > MAX_NUM_ALLOWED_TOKEN_IDS:
                raise ValueError(
                    f"Too many allowed token IDs: {num_allowed_token_ids}. "
                    f"The max size is {MAX_NUM_ALLOWED_TOKEN_IDS}."
                )
            self.num_allowed_token_ids_np[req_idx] = num_allowed_token_ids
            self.allowed_token_ids[req_idx, :num_allowed_token_ids] = torch.tensor(
                allowed_token_ids,
                dtype=torch.int32,
                device=self.allowed_token_ids.device,
            )
            use_logit_bias = True
        else:
            self.num_allowed_token_ids_np[req_idx] = 0

        logit_bias = sampling_params.logit_bias
        if logit_bias:
            num_logit_bias = len(logit_bias)
            if num_logit_bias > MAX_NUM_LOGIT_BIAS_TOKENS:
                raise ValueError(
                    f"Too many logit bias tokens: {num_logit_bias}. "
                    f"The max size is {MAX_NUM_LOGIT_BIAS_TOKENS}."
                )
            self.num_logit_bias_np[req_idx] = num_logit_bias
            self.logit_bias_token_ids[req_idx, :num_logit_bias] = torch.tensor(
                list(logit_bias),
                dtype=torch.int32,
                device=self.logit_bias_token_ids.device,
            )
            self.logit_bias[req_idx, :num_logit_bias] = torch.tensor(
                list(logit_bias.values()),
                dtype=torch.float32,
                device=self.logit_bias.device,
            )
            use_logit_bias = True
        else:
            self.num_logit_bias_np[req_idx] = 0

        min_tokens = sampling_params.min_tokens
        min_len = prompt_len + min_tokens
        self.min_lens_np[req_idx] = min_len
        stop_token_ids = sampling_params.all_stop_token_ids
        if min_tokens > 0 and stop_token_ids:
            num_stop_token_ids = len(stop_token_ids)
            if num_stop_token_ids > MAX_NUM_STOP_TOKEN_IDS:
                raise ValueError(
                    f"Too many stop tokens: {num_stop_token_ids}. "
                    f"The max size is {MAX_NUM_STOP_TOKEN_IDS}."
                )
            self.num_stop_token_ids_np[req_idx] = num_stop_token_ids
            self.stop_token_ids[req_idx, :num_stop_token_ids] = torch.tensor(
                list(stop_token_ids),
                dtype=torch.int32,
                device=self.stop_token_ids.device,
            )
            self.restore_when_all_masked_np[req_idx] = int(
                sampling_params.structured_outputs is not None
            )
            use_logit_bias = True
        else:
            self.num_stop_token_ids_np[req_idx] = 0
            self.restore_when_all_masked_np[req_idx] = 0

        self.use_logit_bias[req_idx] = use_logit_bias
        return use_logit_bias

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        if not np.any(self.use_logit_bias[ctx.idx_mapping_np]):
            return logits

        enable_stop_token_restore = bool(
            np.any(self.restore_when_all_masked_np[ctx.idx_mapping_np])
        )
        ops.apply_logit_bias_cpu(
            logits,
            ctx.expanded_idx_mapping,
            ctx.pos,
            self.num_allowed_token_ids,
            self.allowed_token_ids,
            self.num_logit_bias,
            self.logit_bias_token_ids,
            self.logit_bias,
            self.min_lens,
            self.num_stop_token_ids,
            self.restore_when_all_masked,
            self.stop_token_ids,
            enable_stop_token_restore,
        )
        return logits


class _CPULogitsProcessor(LogitsProcessor):
    def __init__(self, vllm_config: VllmConfig, req_states: LogitsProcRequestState):
        self.req_states = req_states

    def _request_rows(self, ctx: LogitsContext):
        for row, req_idx in enumerate(ctx.expanded_idx_mapping.tolist()):
            if req_idx >= 0:
                yield row, req_idx

    def _output_tokens(
        self, req_idx: int, row: int, ctx: LogitsContext
    ) -> torch.Tensor:
        prompt_len = int(self.req_states.prompt_len_np[req_idx])
        total_len = int(self.req_states.total_len[req_idx])
        first_row = row - int(ctx.expanded_local_pos[row])
        return torch.cat(
            (
                self.req_states.all_token_ids[req_idx, prompt_len:total_len],
                ctx.input_ids[first_row + 1 : row + 1].to(torch.int32),
            )
        )


class CPUPenalties(_CPULogitsProcessor):
    output_bin_counts = None

    def __init__(self, vllm_config: VllmConfig, req_states: LogitsProcRequestState):
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
            prompt_len = int(self.req_states.prompt_len_np[req_idx])
            prompt = self.req_states.all_token_ids[req_idx, :prompt_len]
            prompt = prompt[(prompt >= 0) & (prompt < vocab_size)]
            prompt_ids.append(prompt + offset)
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
    def __init__(self, vllm_config: VllmConfig, req_states: LogitsProcRequestState):
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
