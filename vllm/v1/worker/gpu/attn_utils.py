# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
)
from vllm.v1.worker.utils import bind_kv_cache


def get_kv_cache_spec(vllm_config: VllmConfig) -> dict[str, KVCacheSpec]:
    kv_cache_spec: dict[str, KVCacheSpec] = {}
    layer_type = cast(type[Any], AttentionLayerBase)
    attn_layers = get_layers_from_vllm_config(vllm_config, layer_type)
    for layer_name, attn_module in attn_layers.items():
        # Skip modules that don't need KV cache (eg encoder-only attention)
        if spec := attn_module.get_kv_cache_spec(vllm_config):
            kv_cache_spec[layer_name] = spec
    return kv_cache_spec


def init_attn_backend(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
    device: torch.device,
):
    attn_backends: dict[str, type[AttentionBackend]] = {}
    attn_metadata_builders: list[AttentionMetadataBuilder] = []
    flashinfer_workspace: torch.Tensor | None = None
    for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
        layer_names = kv_cache_group_spec.layer_names
        any_layer_name = next(iter(layer_names))

        layer_type = cast(type[Any], AttentionLayerBase)
        attn_layers = get_layers_from_vllm_config(vllm_config, layer_type, layer_names)
        attn_backend = attn_layers[any_layer_name].get_attn_backend()
        for layer_name in layer_names:
            attn_backends[layer_name] = attn_backend

        attn_metadata_builder = attn_backend.get_builder_cls()(
            kv_cache_group_spec.kv_cache_spec,
            layer_names,
            vllm_config,
            device,
        )
        attn_metadata_builders.append(attn_metadata_builder)  # type: ignore

        if "FLASHINFER" in attn_backend.get_name():
            if flashinfer_workspace is None:
                flashinfer_workspace = attn_metadata_builder._get_workspace_buffer()
            else:
                attn_metadata_builder.set_workspace_buffer(flashinfer_workspace)
    return attn_backends, attn_metadata_builders


def _allocate_kv_cache(
    kv_cache_config: KVCacheConfig,
    device: torch.device,
):
    kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=device)
        for layer_name in kv_cache_tensor.shared_by:
            kv_cache_raw_tensors[layer_name] = tensor

    layer_names = set()
    for group in kv_cache_config.kv_cache_groups:
        for layer_name in group.layer_names:
            layer_names.add(layer_name)
    assert layer_names == set(kv_cache_raw_tensors.keys()), (
        "Some layers are not correctly initialized"
    )
    return kv_cache_raw_tensors


def _reshape_kv_cache(
    kv_cache_config: KVCacheConfig,
    kv_cache_raw_tensors: dict[str, torch.Tensor],
    attn_backends: dict[str, AttentionBackend],
) -> dict[str, torch.Tensor]:
    kv_caches: dict[str, torch.Tensor] = {}
    for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
        kv_cache_spec = kv_cache_group_spec.kv_cache_spec
        assert isinstance(kv_cache_spec, AttentionSpec)
        for layer_name in kv_cache_group_spec.layer_names:
            raw_tensor = kv_cache_raw_tensors[layer_name]
            assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
            num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes

            attn_backend = attn_backends[layer_name]
            kv_cache_shape = attn_backend.get_kv_cache_shape(
                num_blocks,
                kv_cache_spec.block_size,
                kv_cache_spec.num_kv_heads,
                kv_cache_spec.head_size,
            )

            # FIXME(woosuk): Add kv_cache_stride_order to all attention backends.
            try:
                kv_cache_stride_order = attn_backend.get_kv_cache_stride_order()
                assert len(kv_cache_stride_order) == len(kv_cache_shape)
            except (AttributeError, NotImplementedError):
                kv_cache_stride_order = tuple(range(len(kv_cache_shape)))

            kv_cache_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)
            inv_order = [
                kv_cache_stride_order.index(i)
                for i in range(len(kv_cache_stride_order))
            ]

            dtype = kv_cache_spec.dtype
            raw_tensor = raw_tensor.view(dtype)
            raw_tensor = raw_tensor.view(kv_cache_shape)
            kv_caches[layer_name] = raw_tensor.permute(*inv_order)
    return kv_caches


def init_kv_cache(
    runner_kv_caches: list[torch.Tensor],
    forward_context: dict[str, Any],
    kv_cache_config: KVCacheConfig,
    attn_backends: dict[str, AttentionBackend],
    device: torch.device,
) -> None:
    kv_cache_raw_tensors = _allocate_kv_cache(kv_cache_config, device)
    kv_caches = _reshape_kv_cache(kv_cache_config, kv_cache_raw_tensors, attn_backends)
    bind_kv_cache(kv_caches, forward_context, runner_kv_caches)


def build_attn_metadata(
    attn_metadata_builders: list[AttentionMetadataBuilder],
    num_reqs: int,
    num_tokens: int,
    query_start_loc_gpu: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_np: np.ndarray,
    num_computed_tokens_cpu: torch.Tensor | None,
    block_tables: Sequence[torch.Tensor],
    slot_mappings: torch.Tensor,
    kv_cache_config: KVCacheConfig,
) -> dict[str, Any]:
    max_query_len = int(query_start_loc_cpu.max())
    seq_lens = seq_lens[:num_reqs]
    seq_lens_cpu = torch.from_numpy(seq_lens_np)
    max_seq_len = int(seq_lens_np.max())

    attn_metadata: dict[str, Any] = {}
    kv_cache_groups = kv_cache_config.kv_cache_groups
    for i, kv_cache_spec in enumerate(kv_cache_groups):
        block_table = block_tables[i]
        slot_mapping = slot_mappings[i]

        common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=query_start_loc_gpu,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            max_seq_len=max_seq_len,
            num_computed_tokens_cpu=num_computed_tokens_cpu,
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            max_query_len=max_query_len,
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
            causal=True,
        )

        attn_metadata_builder = attn_metadata_builders[i]
        metadata = attn_metadata_builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
        )
        for layer_name in kv_cache_spec.layer_names:
            attn_metadata[layer_name] = metadata
    return attn_metadata
