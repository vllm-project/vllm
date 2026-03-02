# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FSM-constrained sampler for bonus token sampling."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.v1.sample.metadata import SamplingMetadata

from vllm.custom_fsm import CustomFSM
from vllm.v1.sample.sampler import LogprobsMode, Sampler


class FSMSampler(Sampler):
    """Sampler that constrains token selection to FSM-valid candidates.

    Inherits from Sampler but overrides sample() to enforce FSM constraints.
    """

    def __init__(self, fsm_path: str, logprobs_mode: LogprobsMode = "raw_logprobs"):
        super().__init__(logprobs_mode=logprobs_mode)
        self.fsm = CustomFSM.from_prebuilt(fsm_path)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: "SamplingMetadata",
        logprobs_mode_override: LogprobsMode | None = None,
        predict_bonus_token: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Sample tokens constrained by FSM."""
        logprobs_mode = logprobs_mode_override or self.logprobs_mode

        output_token_ids = sampling_metadata.output_token_ids
        spec_token_ids = sampling_metadata.spec_token_ids

        # Combine output tokens with spec tokens (accepted drafts)
        if spec_token_ids:
            combined_tokens = [
                out + spec for out, spec in zip(output_token_ids, spec_token_ids)
            ]
        else:
            combined_tokens = output_token_ids

        batch_size = logits.shape[0]
        num_reqs = len(combined_tokens)
        sampled = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

        for req_idx in range(num_reqs):
            tokens = combined_tokens[req_idx]
            candidate_tokens = self.fsm.get_next_tokens(tokens)

            if not candidate_tokens:
                sampled[req_idx] = logits[req_idx].argmax()
            elif len(candidate_tokens) == 1:
                sampled[req_idx] = candidate_tokens[0]
            else:
                candidate_logits = logits[req_idx, candidate_tokens]
                candidate_probs = torch.softmax(candidate_logits, dim=-1)
                sampled_idx = torch.multinomial(candidate_probs, num_samples=1).item()
                sampled[req_idx] = candidate_tokens[sampled_idx]

        processed_logprobs = None
        if sampling_metadata.max_num_logprobs is not None:
            if logprobs_mode == "processed_logits":
                processed_logprobs = logits
            elif logprobs_mode == "processed_logprobs":
                processed_logprobs = self.compute_logprobs(logits)

        return sampled, processed_logprobs
