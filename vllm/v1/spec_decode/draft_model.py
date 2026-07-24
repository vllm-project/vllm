# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import copy

import torch
import torch.nn as nn
from typing_extensions import override

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.utils import replace
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model
from vllm.tokenizers.registry import get_tokenizer
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.spec_decode.utils import (
    PADDING_SLOT_ID,
    compute_new_slot_mapping,
    eagle_step_update_slot_mapping_and_metadata,
)
from vllm.v1.spec_decode.vocab_mapping import VocabMapping
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


class DraftModelProposer(SpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=False,
            runner=runner,
        )
        self._per_group_block_tables: dict[int, torch.Tensor] = {}
        self._per_group_slot_mappings: dict[int, torch.Tensor] = {}
        self._per_group_slot_mapping_buffers: dict[int, torch.Tensor] = {}
        self._block_size_by_gid: dict[int, int] = {}
        self._is_multi_group = False
        self._raise_if_draft_tp_mismatch()

        self.use_heterogeneous_vocab = self.speculative_config.use_heterogeneous_vocab

        spec = self.speculative_config
        if self.use_heterogeneous_vocab:
            # Heterogeneous vocabularies: build a VocabMapping to translate
            # token IDs between the two tokenizers and constrain draft logits
            # to the intersection so rejection sampling stays lossless.
            target_tokenizer = get_tokenizer(
                spec.target_model_config.tokenizer,
                trust_remote_code=spec.target_model_config.trust_remote_code,
            )
            draft_tokenizer = get_tokenizer(
                spec.draft_model_config.model,
                trust_remote_code=spec.draft_model_config.trust_remote_code,
            )
            self.vocab_mapping: VocabMapping | None = VocabMapping(
                target_tokenizer=target_tokenizer,
                draft_tokenizer=draft_tokenizer,
                target_vocab_size=spec.target_model_config.get_vocab_size(),
                draft_vocab_size=spec.draft_model_config.get_vocab_size(),
                device=device,
            )
        else:
            self._raise_if_vocab_size_mismatch()
            self.vocab_mapping = None

    def _raise_if_vocab_size_mismatch(self):
        self.speculative_config.verify_equal_vocab_size_if_draft_model()

    def _raise_if_draft_tp_mismatch(self):
        # Note(Tomas Ruiz) If we run the target model with TP > 1 and
        # the draft model with TP = 1, then the different TP ranks collide.
        # Specifically when all ranks compile the draft model on rank 0
        # (because TP=1), then the torch compile cache is overwritten and corrupted.
        # We need a mechanism like this: https://github.com/vllm-project/vllm/pull/5414
        # To prevent this error, we assert that both TP sizes must be the same.
        spec_cfg = self.speculative_config
        tgt_tp = spec_cfg.target_parallel_config.tensor_parallel_size
        draft_tp = spec_cfg.draft_parallel_config.tensor_parallel_size
        if draft_tp != tgt_tp:
            raise ValueError(
                f"Currently, 'draft_tensor_parallel_size' and 'tensor_parallel_size' "
                f"must be the same. Got {draft_tp} and {tgt_tp}. "
                "Please pass 'draft_tensor_parallel_size' in the speculative_config."
            )

    @override
    def _create_draft_vllm_config(self) -> VllmConfig:
        base = super()._create_draft_vllm_config()
        spec = self.speculative_config

        return replace(
            base,
            quant_config=None,
            parallel_config=replace(
                spec.draft_parallel_config,
                rank=self.vllm_config.parallel_config.rank,
            ),
            model_config=spec.draft_model_config,
        )

    @override
    def _get_model(self) -> nn.Module:
        from vllm.compilation.backends import set_model_tag

        draft_vllm_config = self._create_draft_vllm_config()
        with set_model_tag("draft_model"):
            model = get_model(
                vllm_config=draft_vllm_config,
                prefix="draft_model",
            )
        return model

    @override
    def _maybe_share_embeddings(self, target_language_model: nn.Module) -> None:
        # Draft models don't share embeddings with the target model
        pass

    @override
    def _maybe_share_lm_head(self, target_language_model: nn.Module) -> None:
        # Draft models don't share lm_head with the target model
        pass

    @override
    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        """Draft models may span multiple KV cache groups."""
        return

    @override
    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        """Initialize AttentionGroups for draft layers using kv_cache_config.

        Supports both single-group and multi-group draft models where attention
        layers belong to different KV cache groups.
        """
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )

        layer_to_gid = self._resolve_draft_layer_kv_cache_groups(kv_cache_config)
        self.kv_cache_gid = min(layer_to_gid.values())

        attention_groups: dict[tuple[object, int, object, object], AttentionGroup] = {}
        for layer_name in self._draft_attn_layer_names:
            gid = layer_to_gid[layer_name]
            group = kv_cache_config.kv_cache_groups[gid]
            kv_cache_spec = group.kv_cache_spec

            attn_backend = all_attn_layers[layer_name].get_attn_backend()
            backend_key = attn_backend.full_cls_name()
            layer_kv_cache_spec = kv_cache_spec
            if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]
            num_heads_q = getattr(all_attn_layers[layer_name], "num_heads", 0)
            group_key = (backend_key, gid, layer_kv_cache_spec, num_heads_q)

            if group_key not in attention_groups:
                kernel_block_size = (
                    kernel_block_sizes[gid]
                    if kernel_block_sizes is not None and gid < len(kernel_block_sizes)
                    else None
                )
                attn_group = AttentionGroup(
                    backend=attn_backend,
                    layer_names=[layer_name],
                    kv_cache_spec=layer_kv_cache_spec,
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
        self._block_size_by_gid = {}
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            block_size = attn_group.get_metadata_builder().kv_cache_spec.block_size
            existing = self._block_size_by_gid.setdefault(gid, block_size)
            if existing != block_size:
                raise ValueError(
                    "Draft KV cache group has multiple block sizes: "
                    f"gid={gid}, {existing} and {block_size}."
                )
        self._is_multi_group = len(self._block_size_by_gid) > 1
        self.block_size = self._block_size_by_gid[self.kv_cache_gid]
        logger.debug("Using block size %d for drafting layers", self.block_size)

    def _resolve_draft_layer_kv_cache_groups(
        self,
        kv_cache_config: KVCacheConfig,
    ) -> dict[str, int]:
        """Map each draft attention layer to its KV cache group id."""
        layer_to_candidate_gids: dict[str, list[int]] = {
            layer_name: [] for layer_name in self._draft_attn_layer_names
        }
        for gid, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            group_layer_names = set(kv_cache_group.layer_names)
            for layer_name in self._draft_attn_layer_names:
                if layer_name in group_layer_names:
                    layer_to_candidate_gids[layer_name].append(gid)

        missing_layers = [
            layer_name
            for layer_name, gids in layer_to_candidate_gids.items()
            if not gids
        ]
        if missing_layers:
            raise ValueError(
                f"Failed to resolve KV cache groups for draft layers: {missing_layers}"
            )

        common_gids = set(layer_to_candidate_gids[next(iter(layer_to_candidate_gids))])
        for gids in layer_to_candidate_gids.values():
            common_gids &= set(gids)

        if common_gids:
            selected_gid = min(common_gids)
            return {
                layer_name: selected_gid for layer_name in self._draft_attn_layer_names
            }

        layer_to_gid: dict[str, int] = {}
        for layer_name, gids in layer_to_candidate_gids.items():
            layer_to_gid[layer_name] = min(gids)

        unique_gids = sorted(set(layer_to_gid.values()))
        logger.info(
            "Draft layers span multiple KV cache groups %s. "
            "Per-group block tables will be used for each group.",
            unique_gids,
        )
        return layer_to_gid

    def _slot_mapping_buffer_for(self, gid: int) -> torch.Tensor:
        if gid == self.kv_cache_gid:
            return self._slot_mapping_buffer
        buf = self._per_group_slot_mapping_buffers.get(gid)
        if buf is None:
            buf = torch.zeros(self.max_positions, dtype=torch.int64, device=self.device)
            self._per_group_slot_mapping_buffers[gid] = buf
        return buf

    def set_per_group_attn_metadata(
        self,
        gid: int,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        self._per_group_block_tables[gid] = block_table
        self._per_group_slot_mappings[gid] = slot_mapping

    @override
    def build_per_group_and_layer_attn_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int = 0,
    ) -> tuple[list[object], dict[str, object]]:
        per_group_attn_metadata: list[object] = []
        per_layer_attn_metadata: dict[str, object] = {}
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            cm = copy(common_attn_metadata)
            cm.block_table_tensor = self._per_group_block_tables.get(
                gid, cm.block_table_tensor
            )[:num_reqs]
            cm.slot_mapping = self._per_group_slot_mappings.get(gid, cm.slot_mapping)[
                :num_actual_tokens
            ]
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=cm,
                draft_index=draft_index,
            )
            per_group_attn_metadata.append(attn_metadata)
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
        return per_group_attn_metadata, per_layer_attn_metadata

    def _store_per_group_slot_mapping(
        self,
        gid: int,
        slot_mapping: torch.Tensor,
        num_tokens: int | None = None,
    ) -> torch.Tensor:
        buf = self._slot_mapping_buffer_for(gid)
        num_actual = slot_mapping.shape[0]
        if buf.data_ptr() != slot_mapping.data_ptr():
            buf[:num_actual].copy_(slot_mapping)
        if num_tokens is not None and num_tokens > num_actual:
            buf[num_actual:num_tokens].fill_(PADDING_SLOT_ID)
        self._per_group_slot_mappings[gid] = buf[:num_actual]
        return buf

    def _get_group_block_table(
        self,
        gid: int,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> torch.Tensor:
        block_table = self._per_group_block_tables.get(
            gid, common_attn_metadata.block_table_tensor
        )
        return block_table[: common_attn_metadata.num_reqs]

    def _get_group_block_size(self, gid: int, attn_group: AttentionGroup) -> int:
        block_size = self._block_size_by_gid.get(gid)
        if block_size is not None:
            return block_size
        return attn_group.get_metadata_builder().kv_cache_spec.block_size

    def _compute_expanded_slot_mapping_for_group(
        self,
        gid: int,
        attn_group: AttentionGroup,
        cad: CommonAttentionMetadata,
        new_positions: torch.Tensor,
        is_rejected_token_mask: torch.Tensor,
        num_new_tokens: int,
    ) -> torch.Tensor:
        group_cad = copy(cad)
        group_cad.block_table_tensor = self._get_group_block_table(gid, cad)
        return compute_new_slot_mapping(
            cad=group_cad,
            new_positions=new_positions,
            is_rejected_token_mask=is_rejected_token_mask,
            block_size=self._get_group_block_size(gid, attn_group),
            num_new_tokens=num_new_tokens,
            max_model_len=self.max_model_len,
        )

    def _update_per_group_slot_mappings_for_expanded_inputs(
        self,
        cad: CommonAttentionMetadata,
        new_positions: torch.Tensor,
        is_rejected_token_mask: torch.Tensor,
        num_new_tokens: int,
    ) -> torch.Tensor | None:
        if not self._is_multi_group:
            return None

        primary_slot_mapping = None
        seen_gids: set[int] = set()
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            if gid in seen_gids:
                continue
            seen_gids.add(gid)
            slot_mapping = self._compute_expanded_slot_mapping_for_group(
                gid,
                attn_group,
                cad,
                new_positions,
                is_rejected_token_mask,
                num_new_tokens,
            )
            self._store_per_group_slot_mapping(gid, slot_mapping)
            if gid == self.kv_cache_gid:
                primary_slot_mapping = slot_mapping
        return primary_slot_mapping

    @override
    def _get_slot_mapping(
        self,
        num_tokens: int,
        slot_mapping: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        per_layer: dict[str, torch.Tensor] = {}
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            buf = self._slot_mapping_buffer_for(gid)
            source = self._per_group_slot_mappings.get(gid, slot_mapping)
            if source is not None:
                n = source.shape[0]
                if buf.data_ptr() != source.data_ptr():
                    buf[:n].copy_(source)
                if num_tokens > n:
                    buf[n:num_tokens].fill_(PADDING_SLOT_ID)
                self._per_group_slot_mappings[gid] = buf[:n]
            view = buf[:num_tokens]
            for layer_name in attn_group.layer_names:
                per_layer[layer_name] = view
        return per_layer

    @override
    def _update_positions_dependent_metadata(
        self,
        positions: torch.Tensor,
        common_attn_metadata,
        batch_size: int,
        input_batch_size: int,
        block_size: int,
    ) -> torch.Tensor:
        positions_1d = positions[0] if self.uses_mrope else positions
        if self.uses_mrope:
            out_pos = self.mrope_positions[0, :batch_size]
        elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            out_pos = self.xdrope_positions[0, :batch_size]
        else:
            out_pos = self.positions[:batch_size]

        if self._is_multi_group:
            primary_gid = self.kv_cache_gid
            primary_group = next(
                (
                    attn_group
                    for attn_group in self.draft_attn_groups
                    if attn_group.kv_cache_group_id == primary_gid
                ),
                self.draft_attn_groups[0],
            )
            primary_slot_buf = self._slot_mapping_buffer_for(primary_gid)
            eagle_step_update_slot_mapping_and_metadata(
                positions_1d=positions_1d,
                block_table_tensor=self._get_group_block_table(
                    primary_gid, common_attn_metadata
                ),
                seq_lens=common_attn_metadata.seq_lens,
                block_size=self._get_group_block_size(primary_gid, primary_group),
                max_model_len=self.max_model_len,
                out_clamped_positions=out_pos,
                out_slot_mapping=primary_slot_buf[:input_batch_size],
                input_batch_size=input_batch_size,
            )
            common_attn_metadata.slot_mapping = primary_slot_buf[:batch_size]
            self._per_group_slot_mappings[primary_gid] = primary_slot_buf[:batch_size]

            new_positions_1d = out_pos[:batch_size]
            exceeds_max = positions_1d + 1 >= self.max_model_len
            seen_gids = {primary_gid}
            for attn_group in self.draft_attn_groups:
                gid = attn_group.kv_cache_group_id
                if gid in seen_gids:
                    continue
                seen_gids.add(gid)
                block_table = self._get_group_block_table(gid, common_attn_metadata)
                group_block_size = self._get_group_block_size(gid, attn_group)
                block_numbers = (new_positions_1d // group_block_size).clamp(
                    max=block_table.shape[1] - 1
                )
                block_ids = block_table[:batch_size].gather(
                    1, block_numbers.to(torch.long).unsqueeze(1)
                )
                slot_mapping = block_ids.squeeze(1).to(torch.int64) * group_block_size
                slot_mapping += new_positions_1d % group_block_size
                slot_mapping.masked_fill_(exceeds_max, PADDING_SLOT_ID)
                buf = self._store_per_group_slot_mapping(
                    gid, slot_mapping, input_batch_size
                )
                if input_batch_size > batch_size:
                    buf[batch_size:input_batch_size].fill_(PADDING_SLOT_ID)
        else:
            eagle_step_update_slot_mapping_and_metadata(
                positions_1d=positions_1d,
                block_table_tensor=common_attn_metadata.block_table_tensor,
                seq_lens=common_attn_metadata.seq_lens,
                block_size=block_size,
                max_model_len=self.max_model_len,
                out_clamped_positions=out_pos,
                out_slot_mapping=self._slot_mapping_buffer[:input_batch_size],
                input_batch_size=input_batch_size,
            )
            common_attn_metadata.slot_mapping = self._slot_mapping_buffer[:batch_size]
            if self.kv_cache_gid in self._per_group_slot_mappings:
                self._per_group_slot_mappings[self.kv_cache_gid] = (
                    common_attn_metadata.slot_mapping
                )
        if self.uses_mrope:
            self.mrope_positions[1:, :batch_size] = self.mrope_positions[0, :batch_size]
            positions = self.mrope_positions[:, :batch_size]
        elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            self.xdrope_positions[1:, :batch_size] = self.xdrope_positions[
                0, :batch_size
            ]
            positions = self.xdrope_positions[0, :batch_size]
        else:
            positions = self.positions[:batch_size]
        common_attn_metadata.max_seq_len = min(
            common_attn_metadata.max_seq_len + 1,
            self.max_model_len,
        )

        if common_attn_metadata._seq_lens_cpu is not None:
            common_attn_metadata._seq_lens_cpu += 1
        if common_attn_metadata._num_computed_tokens_cpu is not None:
            common_attn_metadata._num_computed_tokens_cpu += 1
        if common_attn_metadata.seq_lens_cpu_upper_bound is not None:
            common_attn_metadata.seq_lens_cpu_upper_bound += 1

        return positions
