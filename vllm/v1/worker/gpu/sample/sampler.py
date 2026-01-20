# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

import vllm.envs as envs
from vllm.config.model import LogprobsMode
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.gumbel import apply_temperature, gumbel_sample
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.logprob import compute_topk_logprobs
from vllm.v1.worker.gpu.sample.min_p import apply_min_p
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS, SamplingStates


class Sampler:
    def __init__(
        self,
        max_num_reqs: int,
        vocab_size: int,
        device: torch.device,
        logprobs_mode: LogprobsMode = "raw_logprobs",
    ):
        if logprobs_mode not in ["processed_logprobs", "raw_logprobs"]:
            raise NotImplementedError(f"Unsupported logprobs_mode: {logprobs_mode}")
        self.logprobs_mode = logprobs_mode
        self.compute_nans = envs.VLLM_COMPUTE_NANS_IN_LOGITS  # False by default.

        self.sampling_states = SamplingStates(max_num_reqs, vocab_size)
        self.penalties_state = PenaltiesState(max_num_reqs, vocab_size, device)
        self.logit_bias_state = LogitBiasState(max_num_reqs, device)

    def add_request(
        self,
        req_idx: int,
        prompt_len: int,
        sampling_params: SamplingParams,
    ) -> None:
        self.sampling_states.add_request(req_idx, sampling_params)
        self.penalties_state.add_request(req_idx, sampling_params)
        self.logit_bias_state.add_request(req_idx, prompt_len, sampling_params)

    def apply_staged_writes(
        self,
        prefill_token_ids: torch.Tensor,
        prefill_lens: np.ndarray,
        prompt_lens: np.ndarray,
    ) -> None:
        self.sampling_states.apply_staged_writes()
        self.penalties_state.apply_staged_writes(
            prefill_token_ids, prefill_lens, prompt_lens
        )
        self.logit_bias_state.apply_staged_writes()

    def __call__(
        self,
        logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
    ) -> SamplerOutput:
        # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
        # that num_nans is computed before applying penalties and temperature.
        num_nans = get_num_nans(logits) if self.compute_nans else None
        sampled, processed_logits = self.sample(
            logits, idx_mapping, idx_mapping_np, pos
        )

        max_num_logprobs = self.sampling_states.max_num_logprobs(idx_mapping_np)
        if max_num_logprobs != NO_LOGPROBS:
            logits = (
                processed_logits
                if self.logprobs_mode == "processed_logprobs"
                else logits
            )
            logprobs_tensors = compute_topk_logprobs(logits, max_num_logprobs, sampled)
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
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Copy logits to a new FP32 tensor.
        logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)

        # Apply logit bias (e.g., allowed_token_ids, min_tokens) in place.
        self.logit_bias_state.apply_logit_bias(logits, idx_mapping, idx_mapping_np, pos)

        # Apply penalties in place.
        self.penalties_state.apply_penalties(logits, idx_mapping, idx_mapping_np)

        # Apply temperature in place.
        apply_temperature(logits, idx_mapping, self.sampling_states.temperature.gpu)

        # Apply min_p in place if any request has a non-zero min_p.
        do_min_p = self.sampling_states.do_min_p(idx_mapping_np)
        if do_min_p:
            apply_min_p(logits, idx_mapping, self.sampling_states.min_p.gpu)

        # Apply top_k and/or top_p. This might return a new tensor.
        do_top_k = self.sampling_states.do_top_k(idx_mapping_np)
        top_k = self.sampling_states.top_k.gpu[idx_mapping] if do_top_k else None
        do_top_p = self.sampling_states.do_top_p(idx_mapping_np)
        top_p = self.sampling_states.top_p.gpu[idx_mapping] if do_top_p else None
        if do_top_k or do_top_p:
            logits = apply_top_k_top_p(logits, top_k, top_p)

        # Sample the next token.
        sampled = gumbel_sample(
            logits,
            idx_mapping,
            self.sampling_states.temperature.gpu,
            self.sampling_states.seeds.gpu,
            pos,
            apply_temperature=False,
        )
        return sampled, logits
