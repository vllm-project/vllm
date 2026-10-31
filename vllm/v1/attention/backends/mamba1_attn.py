# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, replace
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.mamba.checkpoint import (
    MambaPrefillCheckpointBuilder,
    MambaPrefillCheckpointMetadata,
)
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import AttentionBackend, CommonAttentionMetadata
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import MambaSpec


class Mamba1AttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "MAMBA1_ATTN"

    @staticmethod
    def get_builder_cls() -> type["Mamba1AttentionMetadataBuilder"]:
        return Mamba1AttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True


@dataclass
class Mamba1CheckpointSegments:
    """Prefill rows cut at their internal checkpoint into a head and a tail.

    The selective scan stores only each sequence's final state, so the mixer
    scans the heads (and uncut rows), copies each head's state into its
    checkpoint block, then scans the tails from there. Each pass hides the
    other pass's segments behind ``NULL_BLOCK_ID``.
    """

    query_start_loc: torch.Tensor
    has_initial_states: torch.Tensor
    head_state_indices: torch.Tensor
    tail_state_indices: torch.Tensor
    # Each segment's prefill row, and whether it is a tail.
    rows: torch.Tensor
    is_tail: torch.Tensor


def _segment_state_indices(
    rows: torch.Tensor, is_tail: torch.Tensor, state_indices_p: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    state_indices = state_indices_p[rows]
    return (
        torch.where(is_tail, NULL_BLOCK_ID, state_indices),
        torch.where(is_tail, state_indices, NULL_BLOCK_ID),
    )


@dataclass
class Mamba1AttentionMetadata(BaseMambaAttentionMetadata):
    checkpoint_meta: MambaPrefillCheckpointMetadata | None = None
    checkpoint_segments: Mamba1CheckpointSegments | None = None


class Mamba1AttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[Mamba1AttentionMetadata]
):
    metadata_cls = Mamba1AttentionMetadata

    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.checkpoint_builder = MambaPrefillCheckpointBuilder(
            vllm_config, kv_cache_spec
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs: Any,
    ) -> Mamba1AttentionMetadata:
        common = self._compute_common_metadata(common_attn_metadata)
        if common.num_prefills == 0:
            return common
        first = common.num_reqs - common.num_prefills
        checkpoint = self.checkpoint_builder.build(
            common_attn_metadata, list(range(first, common.num_reqs))
        )
        if checkpoint is None:
            return common
        assert checkpoint.offsets is not None and checkpoint.query_lens is not None
        assert common.has_initial_states_p is not None
        assert common.state_indices_tensor_p is not None

        bounds, rows, is_tail = [0], [], []
        for row, (offset, query_len) in enumerate(
            zip(checkpoint.offsets, checkpoint.query_lens)
        ):
            if offset:
                bounds.append(bounds[-1] + offset)
                rows.append(row)
                is_tail.append(False)
            bounds.append(bounds[-1] + query_len - offset)
            rows.append(row)
            is_tail.append(offset > 0)
        device = common_attn_metadata.query_start_loc.device
        rows_t = async_tensor_h2d(rows, device, torch.int64)
        is_tail_t = async_tensor_h2d(is_tail, device, torch.bool)
        head_state_indices, tail_state_indices = _segment_state_indices(
            rows_t, is_tail_t, common.state_indices_tensor_p
        )
        segments = Mamba1CheckpointSegments(
            query_start_loc=async_tensor_h2d(bounds, device, torch.int32),
            has_initial_states=common.has_initial_states_p[rows_t] | is_tail_t,
            head_state_indices=head_state_indices,
            tail_state_indices=tail_state_indices,
            rows=rows_t,
            is_tail=is_tail_t,
        )
        return replace(common, checkpoint_meta=checkpoint, checkpoint_segments=segments)

    def update_block_table(
        self,
        metadata: Mamba1AttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> Mamba1AttentionMetadata:
        new_metadata = super().update_block_table(metadata, blk_table, slot_mapping)
        segments = metadata.checkpoint_segments
        if segments is None:
            return new_metadata
        assert metadata.checkpoint_meta is not None
        assert new_metadata.state_indices_tensor_p is not None
        head_state_indices, tail_state_indices = _segment_state_indices(
            segments.rows, segments.is_tail, new_metadata.state_indices_tensor_p
        )
        return replace(
            new_metadata,
            checkpoint_meta=metadata.checkpoint_meta.regather_state_indices(blk_table),
            checkpoint_segments=replace(
                segments,
                head_state_indices=head_state_indices,
                tail_state_indices=tail_state_indices,
            ),
        )
