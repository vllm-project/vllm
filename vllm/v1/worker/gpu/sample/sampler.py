# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config.model import LogprobsMode
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.logprob import compute_topk_logprobs
from vllm.v1.worker.gpu.sample.metadata import SamplingMetadata
from vllm.v1.worker.gpu.sample.penalties import apply_penalties


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
        if sampling_metadata.max_num_logprobs is not None:
            if self.logprobs_mode == "processed_logprobs":
                sampled, logits = self.sample(
                    logits, sampling_metadata, return_logits=True
                )
            else:
                assert self.logprobs_mode == "raw_logprobs"
                sampled, _ = self.sample(logits, sampling_metadata, return_logits=False)

            logprobs_tensors = compute_topk_logprobs(
                logits,
                sampling_metadata.max_num_logprobs,
                sampled,
            )
        else:
            sampled, _ = self.sample(logits, sampling_metadata, return_logits=False)
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
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        is_greedy = sampling_metadata.temperature == 0
        temp = torch.where(is_greedy, 1.0, sampling_metadata.temperature)
        logits = logits / temp.view(-1, 1)
        logits = apply_top_k_top_p(
            logits, sampling_metadata.top_k, sampling_metadata.top_p
        )
        # Apply penalties in place.
        apply_penalties(logits, sampling_metadata)

        sampled = gumbel_sample(
            logits,
            sampling_metadata.temperature,
            sampling_metadata.seeds,
            sampling_metadata.pos,
            apply_temperature=False,
        )
        return sampled, logits if return_logits else None
