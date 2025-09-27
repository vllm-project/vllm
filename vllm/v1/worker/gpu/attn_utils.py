# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch

from vllm.attention.backends.abstract import AttentionBackend, AttentionType
from vllm.attention.layer import Attention
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.v1.attention.backends.utils import AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec, SlidingWindowSpec)
from vllm.v1.worker.utils import bind_kv_cache


def get_kv_cache_spec(
    vllm_config: VllmConfig,
    kv_cache_dtype: torch.dtype,
) -> dict[str, KVCacheSpec]:
    block_size = vllm_config.cache_config.block_size
    use_mla = vllm_config.model_config.use_mla

    kv_cache_spec: dict[str, KVCacheSpec] = {}
    attn_layers = get_layers_from_vllm_config(vllm_config, Attention)
    for layer_name, attn_module in attn_layers.items():
        assert attn_module.attn_type == AttentionType.DECODER
        if attn_module.sliding_window is not None:
            kv_cache_spec[layer_name] = SlidingWindowSpec(
                block_size=block_size,
                num_kv_heads=attn_module.num_kv_heads,
                head_size=attn_module.head_size,
                dtype=kv_cache_dtype,
                sliding_window=attn_module.sliding_window,
                use_mla=use_mla,
            )
        else:
            kv_cache_spec[layer_name] = FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=attn_module.num_kv_heads,
                head_size=attn_module.head_size,
                dtype=kv_cache_dtype,
                use_mla=use_mla,
            )
    return kv_cache_spec


def init_attn_backend(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
    device: torch.device,
):
    attn_backends: dict[str, AttentionBackend] = {}
    attn_metadata_builders: list[AttentionMetadataBuilder] = []

    attn_layers = get_layers_from_vllm_config(vllm_config, Attention)
    for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
        layer_names = kv_cache_group_spec.layer_names
        any_layer_name = next(iter(layer_names))

        attn_backend = attn_layers[any_layer_name].get_attn_backend()
        for layer_name in layer_names:
            attn_backends[layer_name] = attn_backend

        attn_metadata_builder = attn_backend.get_builder_cls()(
            kv_cache_group_spec.kv_cache_spec,
            layer_names,
            vllm_config,
            device,
        )
        attn_metadata_builders.append(attn_metadata_builder)
    return attn_backends, attn_metadata_builders


def _allocate_kv_cache(
    kv_cache_config: KVCacheConfig,
    device: torch.device,
):
    kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        tensor = torch.zeros(kv_cache_tensor.size,
                             dtype=torch.int8,
                             device=device)
        for layer_name in kv_cache_tensor.shared_by:
            kv_cache_raw_tensors[layer_name] = tensor

    layer_names = set()
    for group in kv_cache_config.kv_cache_groups:
        for layer_name in group.layer_names:
            layer_names.add(layer_name)
    assert layer_names == set(kv_cache_raw_tensors.keys()
                              ), "Some layers are not correctly initialized"
    return kv_cache_raw_tensors


def _reshape_kv_cache(
    kv_cache_config: KVCacheConfig,
    kv_cache_raw_tensors: dict[str, torch.Tensor],
    attn_backends: dict[str, AttentionBackend],
) -> dict[str, torch.Tensor]:
    kv_caches: dict[str, torch.Tensor] = {}
    for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
        kv_cache_spec = kv_cache_group_spec.kv_cache_spec
        for layer_name in kv_cache_group_spec.layer_names:
            raw_tensor = kv_cache_raw_tensors[layer_name]
            assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
            num_blocks = (raw_tensor.numel() // kv_cache_spec.page_size_bytes)

            attn_backend = attn_backends[layer_name]
            kv_cache_shape = attn_backend.get_kv_cache_shape(
                num_blocks, kv_cache_spec.block_size,
                kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)

            dtype = kv_cache_spec.dtype
            kv_cache_stride_order = attn_backend.get_kv_cache_stride_order()
            kv_cache_shape = tuple(kv_cache_shape[i]
                                   for i in kv_cache_stride_order)

            inv_order = [
                kv_cache_stride_order.index(i)
                for i in range(len(kv_cache_stride_order))
            ]

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
):
    kv_cache_raw_tensors = _allocate_kv_cache(kv_cache_config, device)
    kv_caches = _reshape_kv_cache(kv_cache_config, kv_cache_raw_tensors,
                                  attn_backends)
    bind_kv_cache(kv_caches, forward_context, runner_kv_caches)
