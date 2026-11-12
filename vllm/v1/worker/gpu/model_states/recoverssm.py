# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMMetadata,
    RecoverSSMMetadataBuilder,
)
from vllm.v1.worker.utils import AttentionGroup


class RecoverSSMState:
    """Coordinates RecoverSSM metadata between attention and postprocessing.

    Builders that defer the commit after sampling to the next verify keep, per
    request slot, the accepted tokens of its last verify not yet in its checkpoint
    (`pending`); the next verify replays them from that slot's records. A pending
    replay lives only while its request is in the runner: when the request leaves
    (finished, preempted, or re-added for its next streaming input chunk), it is
    folded into the checkpoint before the slot or the state is used again."""

    def __init__(self, max_num_reqs: int, device: torch.device) -> None:
        self._step: tuple[RecoverSSMMetadata, ...] | None = None
        self.pending = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.slot_ids = torch.arange(max_num_reqs, dtype=torch.int32, device=device)
        # The step's request slots, None under capture.
        self.request_slots: torch.Tensor | None = None
        self.builders: list[RecoverSSMMetadataBuilder] = []
        self.slots: dict[str, int] = {}

    def add_request(self, req_id: str, req_index: int) -> None:
        self.slots[req_id] = req_index
        # A new or resumed request starts from a prefilled checkpoint.
        self.pending[req_index].fill_(0)

    def remove_request(self, req_id: str) -> None:
        """Fold the leaving request's pending replay into its checkpoint now,
        before its slot is reused and before the step writes any state (the
        scheduler may hand a finished request's blocks to another, which zeroes
        them first)."""
        slot = self.slots.pop(req_id, None)
        if slot is None:
            return
        for builder in self.builders:
            builder.fold_pending(self.slot_ids[slot : slot + 1], self.pending)

    def bind(
        self,
        attn_groups: list[list[AttentionGroup]],
        request_slots: torch.Tensor | None,
    ) -> None:
        """Before the metadata build: the step's request slots, and this state on
        the RecoverSSM builders."""
        self.request_slots = request_slots
        self.builders = []
        for group_list in attn_groups:
            for group in group_list:
                builder = group.get_metadata_builder(0)
                if (
                    isinstance(builder, RecoverSSMMetadataBuilder)
                    and builder.use_recoverssm
                ):
                    builder.recoverssm_state = self
                    self.builders.append(builder)

    def record_step(
        self,
        attn_metadata: dict[str, Any],
        attn_groups: list[list[AttentionGroup]],
        *,
        for_capture: bool,
    ) -> None:
        if for_capture:
            self._step = None
            return

        step: list[RecoverSSMMetadata] = []
        for group_list in attn_groups:
            for group in group_list:
                metadata = attn_metadata[group.layer_names[0]]
                if isinstance(metadata, RecoverSSMMetadata):
                    step.append(metadata)
        self._step = tuple(step)

    def commit_step(
        self,
        num_sampled: torch.Tensor | int,
        idx_mapping: torch.Tensor,
        *,
        state_indices: torch.Tensor | None,
        num_accepted_tokens: torch.Tensor,
    ) -> None:
        step = self._step
        self._step = None
        if isinstance(num_sampled, int) or step is None:
            return

        for metadata in step:
            postprocess_meta = metadata.commit_recoverssm_state(num_sampled)
            if postprocess_meta is None:
                continue
            assert state_indices is not None
            # RecoverSSM already restored the accepted state. Update its running
            # column and reset the next-step copy bias to the neutral value.
            _postprocess_recoverssm_align_kernel[(postprocess_meta.num_spec_decodes,)](
                idx_mapping,
                num_sampled,
                postprocess_meta.request_indices,
                postprocess_meta.num_computed_tokens,
                state_indices,
                num_accepted_tokens,
                MAMBA_BLOCK_SIZE=postprocess_meta.block_size,
                BLOCK_TABLE_WIDTH=postprocess_meta.block_table.shape[1],
            )


@triton.heuristics(
    {"HAS_REQUEST_INDICES": lambda args: args["request_indices_ptr"] is not None}
)
@triton.jit
def _postprocess_recoverssm_align_kernel(
    idx_mapping_ptr,
    num_sampled_ptr,
    request_indices_ptr,
    num_computed_ptr,
    state_idx_ptr,
    num_accepted_ptr,
    HAS_REQUEST_INDICES: tl.constexpr,
    MAMBA_BLOCK_SIZE: tl.constexpr,
    BLOCK_TABLE_WIDTH: tl.constexpr,
):
    spec_idx = tl.program_id(0)
    batch_idx = spec_idx
    if HAS_REQUEST_INDICES:
        batch_idx = tl.load(request_indices_ptr + spec_idx)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    if req_state_idx < 0:
        return
    num_sampled = tl.load(num_sampled_ptr + batch_idx)
    num_computed = tl.load(num_computed_ptr + batch_idx)
    tl.store(
        state_idx_ptr + req_state_idx,
        tl.minimum(
            tl.maximum(num_computed + num_sampled - 1, 0) // MAMBA_BLOCK_SIZE,
            BLOCK_TABLE_WIDTH - 1,
        ),
    )
    tl.store(num_accepted_ptr + req_state_idx, 1)
