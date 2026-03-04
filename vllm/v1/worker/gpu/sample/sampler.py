# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import numpy as np
import torch

import vllm.envs as envs
from vllm.config.model import LogprobsMode
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.logprob import compute_topk_logprobs
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS, SamplingStates
from vllm.v1.worker.gpu.states import RequestState


@dataclass(frozen=True, slots=True)
class _LogitsProcessingPlan:
    logit_bias: bool
    penalties: bool
    bad_words: bool
    temperature: bool
    min_p: bool
    top_k: bool
    top_p: bool
    max_num_bad_words: int

    @property
    def requires_processing(self) -> bool:
        return (
            self.logit_bias
            or self.penalties
            or self.bad_words
            or self.temperature
            or self.min_p
            or self.top_k
            or self.top_p
        )

    @property
    def top_k_top_p(self) -> bool:
        return self.top_k or self.top_p


class Sampler:
    def __init__(
        self,
        max_num_reqs: int,
        vocab_size: int,
        device: torch.device,
        req_states: RequestState,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        num_speculative_tokens: int = 1,
    ):
        if logprobs_mode not in ("processed_logprobs", "raw_logprobs"):
            raise NotImplementedError(f"Unsupported logprobs_mode: {logprobs_mode}")
        self.logprobs_mode = logprobs_mode
        self.compute_nans = envs.VLLM_COMPUTE_NANS_IN_LOGITS  # False by default.

        self.sampling_states = SamplingStates(max_num_reqs, vocab_size)
        self.penalties_state = PenaltiesState(req_states)
        self.logit_bias_state = LogitBiasState(max_num_reqs, device)
        self.bad_words_state = BadWordsState(req_states)
        self.num_speculative_tokens = num_speculative_tokens

    def add_request(
        self, req_idx: int, prompt_len: int, sampling_params: SamplingParams
    ) -> None:
        self.sampling_states.add_request(req_idx, sampling_params)
        self.penalties_state.add_request(req_idx, sampling_params)
        self.logit_bias_state.add_request(req_idx, prompt_len, sampling_params)
        self.bad_words_state.add_request(req_idx, sampling_params)

    def apply_staged_writes(self) -> None:
        self.sampling_states.apply_staged_writes()
        self.penalties_state.apply_staged_writes()
        self.logit_bias_state.apply_staged_writes()
        self.bad_words_state.apply_staged_writes()

    def _build_logits_processing_plan(
        self, idx_mapping_np: np.ndarray
    ) -> _LogitsProcessingPlan:
        max_num_bad_words = int(
            self.bad_words_state.num_bad_words.np[idx_mapping_np].max()
        )
        temperature = self.sampling_states.temperature.np[idx_mapping_np]
        return _LogitsProcessingPlan(
            logit_bias=np.any(self.logit_bias_state.use_logit_bias[idx_mapping_np]),
            penalties=np.any(self.penalties_state.use_penalty[idx_mapping_np]),
            bad_words=max_num_bad_words > 0,
            temperature=not np.all((temperature == 0.0) | (temperature == 1.0)),
            min_p=np.any(self.sampling_states.min_p.np[idx_mapping_np] != 0.0),
            top_k=np.any(
                self.sampling_states.top_k.np[idx_mapping_np]
                != self.sampling_states.vocab_size
            ),
            top_p=np.any(self.sampling_states.top_p.np[idx_mapping_np] != 1.0),
            max_num_bad_words=max_num_bad_words,
        )

    def _apply_logits_processing(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        plan: _LogitsProcessingPlan,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> torch.Tensor:
        # Apply logit bias (e.g., allowed_token_ids, min_tokens) in place.
        if plan.logit_bias:
            self.logit_bias_state.apply_logit_bias(
                logits,
                expanded_idx_mapping,
                pos,
            )

        # Apply penalties in place.
        if plan.penalties:
            self.penalties_state.apply_penalties(
                logits,
                expanded_idx_mapping,
                input_ids,
                expanded_local_pos,
                self.num_speculative_tokens,
            )

        # Apply bad words masking in place.
        if plan.bad_words:
            self.bad_words_state.apply_bad_words(
                logits,
                expanded_idx_mapping,
                input_ids,
                expanded_local_pos,
                plan.max_num_bad_words,
            )

        # Apply temperature in place.
        if plan.temperature:
            self.sampling_states.apply_temperature(logits, expanded_idx_mapping)

        # Apply min_p in place.
        if plan.min_p:
            self.sampling_states.apply_min_p(logits, expanded_idx_mapping)

        # Apply top_k and/or top_p. This might or might not return a new tensor.
        if plan.top_k_top_p:
            logits = self.sampling_states.apply_top_k_top_p(
                logits,
                expanded_idx_mapping,
                plan.top_k,
                plan.top_p,
            )

        return logits

    def __call__(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        cu_num_logits_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> SamplerOutput:
        # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
        # that num_nans is computed before applying penalties and temperature.
        num_nans = get_num_nans(logits) if self.compute_nans else None
        sampled, processed_logits = self.sample(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
        )

        max_num_logprobs = self.sampling_states.max_num_logprobs(idx_mapping_np)
        if max_num_logprobs != NO_LOGPROBS:
            if self.logprobs_mode == "processed_logprobs":
                logits = processed_logits
            expanded_logits = logits.shape[0] != idx_mapping_np.shape[0]
            cu_num_logits = cu_num_logits_np.tolist() if expanded_logits else None
            logprobs_tensors = compute_topk_logprobs(
                logits, max_num_logprobs, sampled, cu_num_logits
            )
        else:
            logprobs_tensors = None

        # These are GPU tensors.
        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampled.view(-1, 1),
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
        )
        return sampler_output

    def sample(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        processing_plan = self._build_logits_processing_plan(idx_mapping_np)
        if processing_plan.requires_processing:
            # Copy logits to a new FP32 tensor.
            logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)
            logits = self._apply_logits_processing(
                logits,
                expanded_idx_mapping,
                processing_plan,
                pos,
                input_ids,
                expanded_local_pos,
            )

        # Sample the next token.
        sampled = gumbel_sample(
            logits,
            expanded_idx_mapping,
            self.sampling_states.temperature.gpu,
            self.sampling_states.seeds.gpu,
            pos,
            apply_temperature=False,
        )
        return sampled, logits
