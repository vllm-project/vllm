# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layer import Attention
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheSpec,
                                        SlidingWindowSpec)


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


def init_attn_backend(vllm_config: VllmConfig):
    attn_layers = get_layers_from_vllm_config(vllm_config, Attention)


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
):
    pass


def init_kv_cache(
    kv_cache_config: KVCacheConfig,
    device: torch.device,
):
    kv_cache_raw_tensors = _allocate_kv_cache(kv_cache_config, device)
    kv_caches = _reshape_kv_cache(kv_cache_config, kv_cache_raw_tensors)
