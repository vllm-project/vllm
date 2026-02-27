# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
from vllm.v1.worker.gpu.spec_decode.rejection_sample import (
    rejection_sample as rejection_sample_functional,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sample import (
    sample_recovered_and_bonus_tokens,
)
from vllm.v1.worker.gpu.states import RequestState


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

    def sample(
        self,
        logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        cu_num_logits_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> SamplerOutput:
        # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
        # that num_nans is computed before applying penalties and temperature.
        num_nans = get_num_nans(logits) if self.compute_nans else None
        processed_logits = self._process_logits(
            logits,
            idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
        )
        sampled = gumbel_sample(
            logits,
            idx_mapping,
            self.sampling_states.temperature.gpu,
            self.sampling_states.seeds.gpu,
            pos,
            apply_temperature=False,
        )

        max_num_logprobs = self.sampling_states.max_num_logprobs(idx_mapping_np)
        if max_num_logprobs != NO_LOGPROBS:
            if self.logprobs_mode == "processed_logprobs":
                logits = processed_logits
            expanded_logits = logits.shape[0] != idx_mapping_np.shape[0]
            cu_num_logits = cu_num_logits_np.tolist() if expanded_logits else None
            # TODO: Check if compute_topk_logprobs can handle 2d sampled
            logprobs_tensors = compute_topk_logprobs(
                logits, max_num_logprobs, sampled, cu_num_logits
            )
        else:
            logprobs_tensors = None

        # No draft tokens (common case).
        num_reqs = len(logits)
        num_sampled = torch.ones(num_reqs, dtype=torch.int32, device=logits.device)

        # These are GPU tensors.
        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampled.view(-1, 1),
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            num_sampled=num_sampled,
        )
        return sampler_output

    def rejection_sample(
        self,
        logits: torch.Tensor,  # [num_draft_tokens + num_reqs, vocab_size]
        draft_logits: torch.Tensor,  # [num_draft_tokens + num_reqs]
        cu_num_logits: torch.Tensor,  # [num_reqs + 1]
        cu_num_logits_np: np.ndarray,  # [num_reqs + 1]
        idx_mapping: torch.Tensor,  # [max_num_reqs]
        expanded_idx_mapping: torch.Tensor,  # [num_draft_tokens + num_reqs]
        idx_mapping_np: np.ndarray,  # [num_draft_tokens + num_reqs]
        pos: torch.Tensor,  # [num_draft_tokens + num_reqs]
        input_ids: torch.Tensor,  # [num_draft_tokens + num_reqs]
        expanded_local_pos: torch.Tensor,  # [num_draft_tokens + num_reqs]
        num_speculative_steps: int,
    ) -> SamplerOutput:
        # TODO: Check whether functions expect expanded idx_mapping or not
        processed_logits = self._process_logits(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
        )
        processed_probs = torch.softmax(processed_logits, dim=-1)
        draft_probs = torch.softmax(draft_logits, dim=-1)
        recovered_ids = sample_recovered_and_bonus_tokens(
            processed_probs,
            draft_probs,
            cu_num_logits,
            expanded_idx_mapping,
            self.sampling_states.temperature.gpu,
            self.sampling_states.seeds.gpu,
            pos,
        )
        sampled, num_sampled = rejection_sample_functional(
            input_ids,
            recovered_ids,
            processed_probs,
            draft_probs,
            cu_num_logits,
            self.sampling_states.seeds.gpu,
            pos,
            self.sampling_states.temperature.gpu,
            num_speculative_steps,
        )
        max_num_logprobs = self.sampling_states.max_num_logprobs(idx_mapping_np)
        if max_num_logprobs != NO_LOGPROBS:
            expanded_logits = logits.shape[0] != idx_mapping_np.shape[0]
            cu_num_logits_list = cu_num_logits_np.tolist() if expanded_logits else None
            logprobs_tensors = compute_topk_logprobs(
                processed_logits, max_num_logprobs, sampled, cu_num_logits_list
            )
        else:
            logprobs_tensors = None

        # These are GPU tensors.
        sampler_output = SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs_tensors,
            num_nans=None,
            num_sampled=num_sampled,
        )
        return sampler_output

    def _process_logits(
        self,
        logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> torch.Tensor:
        # Copy logits to a new FP32 tensor.
        logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)

        # Apply logit bias (e.g., allowed_token_ids, min_tokens) in place.
        self.logit_bias_state.apply_logit_bias(logits, idx_mapping, idx_mapping_np, pos)

        # Apply penalties in place.
        self.penalties_state.apply_penalties(
            logits,
            idx_mapping,
            idx_mapping_np,
            input_ids,
            expanded_local_pos,
            self.num_speculative_tokens,
        )

        # Apply bad words masking in place.
        self.bad_words_state.apply_bad_words(
            logits,
            idx_mapping,
            idx_mapping_np,
            input_ids,
            expanded_local_pos,
        )

        # Apply temperature in place.
        self.sampling_states.apply_temperature(logits, idx_mapping, idx_mapping_np)

        # Apply min_p in place.
        self.sampling_states.apply_min_p(logits, idx_mapping, idx_mapping_np)

        # Apply top_k and/or top_p. This might or might not return a new tensor.
        logits = self.sampling_states.apply_top_k_top_p(
            logits, idx_mapping, idx_mapping_np
        )

        return logits
