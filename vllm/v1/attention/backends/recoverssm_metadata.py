# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import abc
from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton


@triton.heuristics(
    {
        "HAS_REQUEST_INDICES": lambda args: args["request_indices_ptr"] is not None,
        "HAS_RESET_MASK": lambda args: args["reset_mask_ptr"] is not None,
    }
)
@triton.jit
def postprocess_recoverssm_align_kernel(
    idx_mapping_ptr,
    num_sampled_ptr,
    request_indices_ptr,
    reset_mask_ptr,
    num_computed_ptr,
    state_idx_ptr,
    num_accepted_ptr,
    HAS_REQUEST_INDICES: tl.constexpr,
    HAS_RESET_MASK: tl.constexpr,
    MAMBA_BLOCK_SIZE: tl.constexpr,
    BLOCK_TABLE_WIDTH: tl.constexpr,
):
    """Restore align bookkeeping after accepted-state materialization."""
    spec_idx = tl.program_id(0)
    if HAS_RESET_MASK and not tl.load(reset_mask_ptr + spec_idx):
        return
    batch_idx = spec_idx
    if HAS_REQUEST_INDICES:
        batch_idx = tl.load(request_indices_ptr + spec_idx)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    if req_state_idx < 0:
        return
    # The metadata retains the forward's batch-ordered, pre-step snapshot.
    # post_update advances the separate persistent request-state tensor, so
    # reconstruct the accepted post-step count here from this batch row.
    num_computed = tl.load(num_computed_ptr + batch_idx)
    num_sampled = tl.load(num_sampled_ptr + batch_idx)
    total_computed = num_computed + num_sampled
    # Match compute_aligned_state_indices(): an exact block boundary owns the
    # completed page, not the following page.
    state_idx = (tl.maximum(total_computed, 1) - 1) // MAMBA_BLOCK_SIZE
    tl.store(
        state_idx_ptr + req_state_idx,
        tl.minimum(state_idx, BLOCK_TABLE_WIDTH - 1),
    )
    tl.store(num_accepted_ptr + req_state_idx, 1)


@dataclass(frozen=True)
class RecoverSSMPostprocessMetadata:
    """Metadata used during postprocessing for align-mode prefix caching."""

    num_spec_decodes: int
    request_indices: torch.Tensor | None
    block_table: torch.Tensor
    num_computed_tokens: torch.Tensor
    block_size: int
    reset_mask: torch.Tensor | None = None


class RecoverSSMMetadata(abc.ABC):
    @abc.abstractmethod
    def commit_recoverssm_state(
        self, num_accepted_tokens: torch.Tensor
    ) -> RecoverSSMPostprocessMetadata | None:
        raise NotImplementedError
