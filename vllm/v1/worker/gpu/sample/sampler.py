# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config.model import LogprobsMode
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.logprob import compute_topk_logprobs
from vllm.v1.worker.gpu.sample.metadata import SamplingMetadata
from vllm.v1.worker.gpu.sample.min_p import apply_min_p
from vllm.v1.worker.gpu.sample.penalties import apply_penalties_and_temperature


class Sampler:
    def __init__(
        self,
        logprobs_mode: LogprobsMode = "raw_logprobs",
    ):
        if logprobs_mode not in ["processed_logprobs", "raw_logprobs"]:
            raise NotImplementedError(f"Unsupported logprobs_mode: {logprobs_mode}")
        self.logprobs_mode = logprobs_mode

    def __call__(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        sampled, processed_logits = self.sample(logits, sampling_metadata)
        if sampling_metadata.max_num_logprobs is not None:
            logits = (
                processed_logits
                if self.logprobs_mode == "processed_logprobs"
                else logits
            )
            logprobs_tensors = compute_topk_logprobs(
                logits,
                sampling_metadata.max_num_logprobs,
                sampled,
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
        )
        return sampler_output

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Copy logits to a new FP32 tensor.
        logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)

        # Apply penalties and temperature in place.
        apply_penalties_and_temperature(logits, sampling_metadata)
        # Apply min_p in place.
        apply_min_p(logits, sampling_metadata.min_p)
        # Apply top_k and/or top_p. This might return a new tensor.
        logits = apply_top_k_top_p(
            logits, sampling_metadata.top_k, sampling_metadata.top_p
        )

        sampled = gumbel_sample(
            logits,
            sampling_metadata.temperature,
            sampling_metadata.seeds,
            sampling_metadata.pos,
            apply_temperature=False,
        )
        return sampled, logits
