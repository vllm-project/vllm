# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cohere Eagle proposer for speculative decoding.

The cohere eagle models closely follow vLLM's EagleProposer, but handle the
case where the draft layers are grouped into multiple KV cache groups.
"""

from copy import copy

import torch

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.utils import PADDING_SLOT_ID
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


class CohereEagleProposer(EagleProposer):
    """Cohere Eagle proposer with draft layers in multiple KV cache groups."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        """Extend the base proposer with per-group KV bookkeeping.

        The base class assumes a single KV cache group and keeps one
        block table / slot-mapping buffer. Cohere Eagle's layers span
        several groups, each with its own block table and (since the draft
        writes its own KV) its own slot mapping, so we add per-group state
        that the runner and the overridden methods below populate.
        """
        super().__init__(vllm_config, device, runner)

        # Per-group block tables/slot-mapping for multi-group KV cache models.
        # Populated by gpu_model_runner during _prepare_inputs.
        self._per_group_block_tables: dict[int, torch.Tensor] = {}
        self._per_group_slot_mappings: dict[int, torch.Tensor] = {}

        # Slot-mapping buffers for non-primary KV cache groups (the primary
        # group reuses self._slot_mapping_buffer from the base class).
        self._per_group_slot_mapping_buffers: dict[int, torch.Tensor] = {}

    def set_per_group_attn_metadata(
        self,
        gid: int,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Register a group's block table and slot mapping from the runner.

        The base proposer has no such hook because it is single-group. The
        model runner calls this once per KV cache group during input prep
        so the drafter can later build per-group attention metadata.
        """
        self._per_group_block_tables[gid] = block_table
        self._per_group_slot_mappings[gid] = slot_mapping

    def _slot_mapping_buffer_for(self, gid: int) -> torch.Tensor:
        """Return the persistent slot-mapping buffer for a KV cache group.

        The base class owns a single ``self._slot_mapping_buffer`` for the
        primary group; non-primary groups need their own buffers (allocated
        lazily) so their per-step slot mappings don't clobber each other.
        """
        if gid == self.kv_cache_gid:
            return self._slot_mapping_buffer
        buf = self._per_group_slot_mapping_buffers.get(gid)
        if buf is None:
            buf = torch.zeros(self.max_positions, dtype=torch.int64, device=self.device)
            self._per_group_slot_mapping_buffers[gid] = buf
        return buf

    def _get_slot_mapping(
        self,
        num_tokens: int,
        slot_mapping: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build a per-layer slot_mapping keyed by each layer's KV group.

        The base implementation returns a single slot mapping shared by all
        draft layers. Here, we emit one slot mapping per group (from that
        group's own buffer) and map every layer to its group's view, so each
        group writes into different physical blocks.
        """
        per_layer: dict[str, torch.Tensor] = {}
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            buf = self._slot_mapping_buffer_for(gid)
            source = self._per_group_slot_mappings.get(gid, slot_mapping)
            if source is not None and buf.data_ptr() != source.data_ptr():
                n = source.shape[0]
                buf[:n].copy_(source)
                if num_tokens > n:
                    buf[n:num_tokens].fill_(PADDING_SLOT_ID)
            view = buf[:num_tokens]
            for layer_name in attn_group.layer_names:
                per_layer[layer_name] = view
        return per_layer

    def _update_positions_dependent_metadata(
        self,
        positions: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        batch_size: int,
        input_batch_size: int,
        block_size: int,
    ) -> torch.Tensor:
        """Advance positions and recompute slot mappings for every group.

        Called each draft step because positions advance. The base
        method only advances positions and recomputes the slot mapping for
        the primary group. Here the non-primary groups have distinct block
        tables, so we additionally recompute their slot mappings from the new
        positions and each group's own block table.
        """
        old_positions_1d = positions[0] if self.uses_mrope else positions
        positions = super()._update_positions_dependent_metadata(
            positions,
            common_attn_metadata,
            batch_size,
            input_batch_size,
            block_size,
        )
        # Parent already produced slot_mapping for the primary gid.
        self._per_group_slot_mappings[self.kv_cache_gid] = (
            common_attn_metadata.slot_mapping
        )
        # Recompute slot_mapping for the remaining gids using their own block tables.
        new_positions_1d = positions[0] if self.uses_mrope else positions
        exceeds = old_positions_1d + 1 >= self.max_model_len
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            if gid == self.kv_cache_gid:
                continue
            block_table = self._per_group_block_tables.get(gid)
            if block_table is None:
                continue
            n_blocks = block_table.shape[1]
            bn = (new_positions_1d // block_size).clamp(max=n_blocks - 1).to(torch.long)
            block_ids = block_table[:batch_size].gather(1, bn.unsqueeze(1)).squeeze(1)
            sm = block_ids * block_size + (new_positions_1d % block_size)
            sm.masked_fill_(exceeds, PADDING_SLOT_ID)
            buf = self._slot_mapping_buffer_for(gid)
            buf[:batch_size].copy_(sm)
            if input_batch_size > batch_size:
                buf[batch_size:input_batch_size].fill_(PADDING_SLOT_ID)
            self._per_group_slot_mappings[gid] = buf[:batch_size]
        return positions

    def build_per_group_and_layer_attn_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int = 0,
    ) -> tuple[list[object], dict[str, object]]:
        """Build attention metadata using each group's own block table.

        The base class receives one ``common_attn_metadata`` whose block
        table and slot mapping belong to a single group. Because our draft
        layers span multiple groups with different block tables and different
        slot mappings, we swap in the correct per-group tensors before
        building metadata for each draft attention group.
        """
        per_group_attn_metadata: list[object] = []
        per_layer_attn_metadata: dict[str, object] = {}
        # The proposer always works in unpadded shape. Per-group block tables
        # registered via set_per_group_attn_metadata are stored at the model
        # runner's padded shape; slice them to match cm's num_reqs.
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            if gid in self._per_group_block_tables:
                cm = copy(common_attn_metadata)
                cm.block_table_tensor = self._per_group_block_tables[gid][:num_reqs]
                if gid in self._per_group_slot_mappings:
                    sm = self._per_group_slot_mappings[gid]
                    if sm.shape[0] >= num_actual_tokens:
                        sm = sm[:num_actual_tokens]
                    cm.slot_mapping = sm
            else:
                cm = common_attn_metadata
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=cm,
                draft_index=draft_index,
            )
            per_group_attn_metadata.append(attn_metadata)
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
        return per_group_attn_metadata, per_layer_attn_metadata

    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        """Skip the base single-group assertion.

        The base class asserts all draft layers live in one KV cache group.
        Cohere Eagle can span multiple groups, so this is a no-op.
        """
        return

    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        """Create one AttentionGroup per (backend, KV cache group).

        The base class builds a single attention group under the assumption
        that all draft layers share one KV cache spec/backend. Cohere Eagle's
        draft layers are split across KV cache groups, so we group layers by
        (backend, gid) and give each its own metadata builder, then record
        the primary group's gid and block size for the draft loop.
        """
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )

        layer_to_gid: dict[str, int] = {}
        layer_to_spec: dict[str, KVCacheSpec] = {}
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            group_spec = group.kv_cache_spec
            for layer_name in group.layer_names:
                layer_to_gid[layer_name] = gid
                if isinstance(group_spec, UniformTypeKVCacheSpecs):
                    if layer_name in group_spec.kv_cache_specs:
                        layer_to_spec[layer_name] = group_spec.kv_cache_specs[
                            layer_name
                        ]
                    else:
                        target_layer_name = getattr(
                            all_attn_layers.get(layer_name),
                            "kv_sharing_target_layer_name",
                            None,
                        )
                        if (
                            target_layer_name
                            and target_layer_name in group_spec.kv_cache_specs
                        ):
                            layer_to_spec[layer_name] = group_spec.kv_cache_specs[
                                target_layer_name
                            ]
                        else:
                            layer_to_spec[layer_name] = group_spec
                else:
                    layer_to_spec[layer_name] = group_spec

        attention_groups: dict[tuple[tuple[str, str], int], AttentionGroup] = {}
        for layer_name in sorted(self._draft_attn_layer_names):
            if layer_name not in layer_to_spec:
                continue
            attn_layer = all_attn_layers[layer_name]
            attn_backend = attn_layer.get_attn_backend()
            spec = layer_to_spec[layer_name]
            gid = layer_to_gid[layer_name]
            group_key = (attn_backend.full_cls_name(), gid)

            if group_key not in attention_groups:
                kernel_block_size = (
                    kernel_block_sizes[gid]
                    if kernel_block_sizes is not None and gid < len(kernel_block_sizes)
                    else None
                )
                attn_group = AttentionGroup(
                    backend=attn_backend,
                    layer_names=[layer_name],
                    kv_cache_spec=spec,
                    kv_cache_group_id=gid,
                )
                attn_group.create_metadata_builders(
                    self.vllm_config,
                    self.device,
                    kernel_block_size=kernel_block_size,
                )
                attention_groups[group_key] = attn_group
            else:
                attention_groups[group_key].layer_names.append(layer_name)

        self.draft_attn_groups = list(attention_groups.values())
        if self.draft_attn_groups:
            self.kv_cache_gid = self.draft_attn_groups[0].kv_cache_group_id
            self.block_size = (
                self.draft_attn_groups[0]
                .get_metadata_builder()
                .kv_cache_spec.block_size
            )
        else:
            self.kv_cache_gid = 0
            self.block_size = kv_cache_config.kv_cache_groups[
                0
            ].kv_cache_spec.block_size
        logger.debug("Using block size %d for drafting layers", self.block_size)
