# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FSM-specific rejection sampler that accepts all deterministic drafts."""

from dataclasses import replace

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import SamplerOutput
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

PLACEHOLDER_TOKEN_ID = -1


class FSMRejectionSampler(RejectionSampler):
    """Rejection sampler optimized for FSM deterministic drafts.

    Since FSM only proposes valid tokens, we skip verification and accept all drafts.
    Inherits from RejectionSampler but overrides forward() to bypass rejection sampling.
    """

    def forward(
        self,
        metadata: SpecDecodeMetadata,
        draft_probs: torch.Tensor | None,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        """Accept all FSM draft tokens and sample bonus token."""
        # Sample bonus token using parent's sampler
        bonus_logits_indices = metadata.bonus_logits_indices
        bonus_logits = logits[bonus_logits_indices]
        bonus_sampler_output = self.sampler(
            logits=bonus_logits,
            sampling_metadata=replace(sampling_metadata, max_num_logprobs=-1),
            predict_bonus_token=True,
            logprobs_mode_override="processed_logits"
            if self.is_processed_logprobs_mode
            else "raw_logits",
        )
        bonus_token_ids = bonus_sampler_output.sampled_token_ids

        # Accept all deterministic drafts (no verification needed)
        output_token_ids = _accept_all_drafts(
            metadata.draft_token_ids,
            metadata.num_draft_tokens,
            metadata.max_spec_len,
            metadata.cu_num_draft_tokens,
            bonus_token_ids,
        )

        # Handle logprobs if needed
        logprobs_tensors = None
        if sampling_metadata.max_num_logprobs is not None:
            target_logits = logits[metadata.target_logits_indices].to(torch.float32)
            logprobs_tensors = self._get_logprobs_tensors(
                sampling_metadata.max_num_logprobs,
                metadata,
                logits,
                target_logits,
                bonus_sampler_output.logprobs_tensors.logprobs,
                output_token_ids,
            )

        return SamplerOutput(
            sampled_token_ids=output_token_ids,
            logprobs_tensors=logprobs_tensors,
        )


def _accept_all_drafts(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    bonus_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Accept all draft tokens without verification."""
    batch_size = len(num_draft_tokens)
    device = draft_token_ids.device

    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=device,
    )

    _accept_drafts_kernel[(batch_size,)](
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        bonus_token_ids,
        max_spec_len,
    )

    return output_token_ids


@triton.jit
def _accept_drafts_kernel(
    output_token_ids_ptr,
    cu_num_draft_tokens_ptr,
    draft_token_ids_ptr,
    bonus_token_ids_ptr,
    max_spec_len,
):
    """Kernel to accept all draft tokens."""
    req_idx = tl.program_id(0)

    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    for pos in range(num_draft_tokens):
        draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
            draft_token_id,
        )

    bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
    tl.store(
        output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
        bonus_token_id,
    )
