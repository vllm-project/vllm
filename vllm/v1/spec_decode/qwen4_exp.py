# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import copy

import torch

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.worker.utils import AttentionGroup


class Qwen4ExpMTPProposer(EagleProposer):
    """Speculative decoding proposer for Qwen4Exp MTP."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ) -> None:
        super().__init__(vllm_config, device, runner)
        self._per_group_block_tables: dict[int, torch.Tensor] = {}

    def _get_hidden_size(self) -> int:
        """Return the multi-stream feedback width consumed by Qwen MTP."""
        return int(
            self.draft_model_config.get_hidden_size()
            * self.draft_model_config.hf_config.hc_mult
        )

    def model_returns_tuple(self) -> bool:
        """Qwen MTP returns separate logits and feedback hidden states."""
        return True

    def set_per_group_block_table(
        self,
        gid: int,
        block_table: torch.Tensor,
    ) -> None:
        """Stage one scheduler group's block table for drafting."""
        self._per_group_block_tables[gid] = block_table

    def build_per_group_and_layer_attn_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int = 0,
    ) -> tuple[list[object], dict[str, object]]:
        """Build each Qwen cache owner's metadata from its scheduler group."""
        per_group_attn_metadata: list[object] = []
        per_layer_attn_metadata: dict[str, object] = {}
        common_by_gid: dict[int, CommonAttentionMetadata] = {}
        num_reqs = common_attn_metadata.num_reqs

        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            group_common = common_by_gid.get(gid)
            if group_common is None:
                if gid == self.kv_cache_gid:
                    group_common = common_attn_metadata
                else:
                    block_table = self._per_group_block_tables.get(gid)
                    assert block_table is not None, (
                        f"Missing Qwen draft block table for KV cache group {gid}"
                    )
                    group_common = copy(common_attn_metadata)
                    group_common.block_table_tensor = block_table[:num_reqs]
                common_by_gid[gid] = group_common

            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=group_common,
                draft_index=draft_index,
            )
            per_group_attn_metadata.append(attn_metadata)
            # QSA cache updates consume each builder's mapping from this metadata.
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

        return per_group_attn_metadata, per_layer_attn_metadata

    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        """Initialize Qwen main, compressed, and circular cache owners."""
        num_mtp_layers = self.draft_model_config.hf_text_config.mtp_num_hidden_layers
        if num_mtp_layers != 1:
            raise NotImplementedError(
                "Qwen4Exp MTP proposer only supports one MTP layer"
            )
        assert kernel_block_sizes is not None, (
            "Qwen MTP requires resolved kernel block sizes"
        )
        assert len(kernel_block_sizes) == len(kv_cache_config.kv_cache_groups), (
            "Qwen MTP requires one kernel block size per KV cache group"
        )

        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        layer_to_gid, layer_to_spec = self._map_draft_layers_to_groups(kv_cache_config)
        main_layers = [
            name
            for name, spec in layer_to_spec.items()
            if type(spec) is FullAttentionSpec
        ]
        assert len(main_layers) == 1, (
            "Qwen4Exp MTP requires exactly one main cache owner"
        )
        self.kv_cache_gid = layer_to_gid[main_layers[0]]

        attention_groups: list[AttentionGroup] = []
        for layer_name in sorted(self._draft_attn_layer_names):
            attn_layer = all_attn_layers[layer_name]
            gid = layer_to_gid[layer_name]
            attn_group = AttentionGroup(
                backend=attn_layer.get_attn_backend(),
                layer_names=[layer_name],
                kv_cache_spec=layer_to_spec[layer_name],
                kv_cache_group_id=gid,
            )
            attn_group.create_metadata_builders(
                self.vllm_config,
                self.device,
                kernel_block_size=kernel_block_sizes[gid],
            )
            attention_groups.append(attn_group)

        self.draft_attn_groups = sorted(
            attention_groups,
            key=lambda group: (
                group.kv_cache_group_id != self.kv_cache_gid,
                group.kv_cache_group_id,
                group.backend.full_cls_name(),
                group.layer_names[0],
            ),
        )
        self.block_size = kernel_block_sizes[self.kv_cache_gid]

    def _map_draft_layers_to_groups(
        self,
        kv_cache_config: KVCacheConfig,
    ) -> tuple[dict[str, int], dict[str, KVCacheSpec]]:
        """Map Qwen draft cache owners to scheduler groups and concrete specs."""
        layer_to_gid: dict[str, int] = {}
        layer_to_spec: dict[str, KVCacheSpec] = {}
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            group_spec = group.kv_cache_spec
            for layer_name in group.layer_names:
                if layer_name not in self._draft_attn_layer_names:
                    continue
                assert isinstance(group_spec, UniformTypeKVCacheSpecs), (
                    "Qwen draft cache owners require packed KV cache groups"
                )
                spec = group_spec.kv_cache_specs.get(layer_name)
                assert spec is not None, (
                    f"Qwen draft cache group {gid} has no spec for {layer_name}"
                )
                layer_to_gid[layer_name] = gid
                layer_to_spec[layer_name] = spec

        assert layer_to_spec.keys() == self._draft_attn_layer_names, (
            "Qwen draft KV cache configuration is missing layers: "
            f"{sorted(self._draft_attn_layer_names - layer_to_spec.keys())}"
        )
        return layer_to_gid, layer_to_spec


__all__ = ["Qwen4ExpMTPProposer"]
