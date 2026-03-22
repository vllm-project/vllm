# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""注意力工具函数模块。

本模块提供注意力机制相关的辅助函数，负责：
- 获取 KV 缓存规范
- 初始化注意力后端
- 分配和重塑 KV 缓存
- 构建 slot mappings 和注意力元数据

主要函数：
- get_kv_cache_spec: 获取 KV 缓存规范
- init_attn_backend: 初始化注意力后端
- init_kv_cache: 初始化 KV 缓存
- build_slot_mappings_by_layer: 按层构建 slot mappings
- build_attn_metadata: 构建注意力元数据
"""
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import torch

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend, CommonAttentionMetadata
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.utils import AttentionGroup, bind_kv_cache


def get_kv_cache_spec(vllm_config: VllmConfig) -> dict[str, KVCacheSpec]:
    """从 vLLM 配置获取 KV 缓存规范。

    遍历所有注意力层，收集需要 KV 缓存的模块的规范。

    Args:
        vllm_config: vLLM 配置

    Returns:
        层名到 KV 缓存规范的映射字典
    """
    kv_cache_spec: dict[str, KVCacheSpec] = {}
    layer_type = cast(type[Any], AttentionLayerBase)
    attn_layers = get_layers_from_vllm_config(vllm_config, layer_type)
    for layer_name, attn_module in attn_layers.items():
        # 跳过不需要 KV 缓存的模块（例如仅编码器注意力）
        if spec := attn_module.get_kv_cache_spec(vllm_config):
            kv_cache_spec[layer_name] = spec
    return kv_cache_spec


def init_attn_backend(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
    device: torch.device,
    active_layer_names: set[str] | None = None,
) -> tuple[dict[str, type[AttentionBackend]], list[list[AttentionGroup]]]:
    """初始化注意力后端和注意力组。

    此函数执行以下操作：
    1. 为每个 KV 缓存组创建注意力后端
    2. 将具有相同后端和 KV 缓存规范的层分组到 AttentionGroup
    3. 为每个组创建元数据构建器
    4. 设置共享的工作区缓冲区

    Args:
        kv_cache_config: KV 缓存配置
        vllm_config: vLLM 配置
        device: 设备类型
        active_layer_names: 活动层名集合（可选）

    Returns:
        (注意力后端字典，注意力组列表) 元组
    """
    attn_backends: dict[str, type[AttentionBackend]] = {}
    attn_groups: list[list[AttentionGroup]] = []
    attn_backend_workspace: torch.Tensor | None = None
    for kv_cache_group_id, kv_cache_group_spec in enumerate(
        kv_cache_config.kv_cache_groups
    ):
        layer_names = kv_cache_group_spec.layer_names
        if active_layer_names is not None:
            layer_names = list(active_layer_names.intersection(layer_names))

        layer_type = cast(type[Any], AttentionLayerBase)
        attn_layers = get_layers_from_vllm_config(vllm_config, layer_type, layer_names)

        group_map: dict[tuple[tuple[str, str], KVCacheSpec], AttentionGroup] = {}
        group_order: list[tuple[tuple[str, str], KVCacheSpec]] = []

        for layer_name in layer_names:
            attn_backend = attn_layers[layer_name].get_attn_backend()
            attn_backends[layer_name] = attn_backend

            layer_kv_cache_spec: KVCacheSpec = kv_cache_group_spec.kv_cache_spec
            if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]

            key = (attn_backend.full_cls_name(), layer_kv_cache_spec)
            if key not in group_map:
                group_map[key] = AttentionGroup(
                    attn_backend,
                    [layer_name],
                    layer_kv_cache_spec,
                    kv_cache_group_id,
                )
                group_order.append(key)
            else:
                group_map[key].layer_names.append(layer_name)

        groups = [group_map[key] for key in group_order]
        for group in groups:
            group.create_metadata_builders(
                vllm_config=vllm_config,
                device=device,
                kernel_block_size=None,
                num_metadata_builders=1,
            )
            builder = group.get_metadata_builder(0)
            if attn_backend_workspace is None:
                if hasattr(builder, "_get_workspace_buffer"):
                    attn_backend_workspace = builder._get_workspace_buffer()
            else:
                if hasattr(builder, "set_workspace_buffer"):
                    builder.set_workspace_buffer(attn_backend_workspace)
        attn_groups.append(groups)
    return attn_backends, attn_groups


def _allocate_kv_cache(kv_cache_config: KVCacheConfig, device: torch.device) -> dict[str, torch.Tensor]:
    """分配 KV 缓存原始张量。

    为每个 KV 缓存张量分配内存，并处理共享同一张量的层。

    Args:
        kv_cache_config: KV 缓存配置
        device: 设备类型

    Returns:
        层名到原始 KV 缓存张量的映射
    """
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
        "有些层未正确初始化"
    )
    return kv_cache_raw_tensors


def _reshape_kv_cache(
    kv_cache_config: KVCacheConfig,
    kv_cache_raw_tensors: dict[str, torch.Tensor],
    attn_backends: dict[str, AttentionBackend],
    cache_dtype: str,
) -> dict[str, torch.Tensor]:
    """重塑 KV 缓存张量为正确的形状。

    根据注意力后端的 KV 缓存形状和步幅顺序，将原始张量重塑
    为适合注意力计算的格式。

    Args:
        kv_cache_config: KV 缓存配置
        kv_cache_raw_tensors: 原始 KV 缓存张量
        attn_backends: 注意力后端
        cache_dtype: 缓存数据类型

    Returns:
        层名到重塑后的 KV 缓存张量的映射
    """
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
                cache_dtype,
            )

            # FIXME(woosuk): 将 kv_cache_stride_order 添加到所有注意力后端
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
    cache_dtype: str,
) -> dict[str, torch.Tensor]:
    """初始化 KV 缓存。

    分配原始 KV 缓存张量，重塑为正确的形状，并绑定到前向上下文。

    Args:
        runner_kv_caches: 运行器 KV 缓存列表
        forward_context: 前向上下文
        kv_cache_config: KV 缓存配置
        attn_backends: 注意力后端
        device: 设备类型
        cache_dtype: 缓存数据类型

    Returns:
        层名到 KV 缓存张量的映射
    """
    kv_cache_raw_tensors = _allocate_kv_cache(kv_cache_config, device)
    kv_caches = _reshape_kv_cache(
        kv_cache_config, kv_cache_raw_tensors, attn_backends, cache_dtype
    )
    bind_kv_cache(kv_caches, forward_context, runner_kv_caches)
    return kv_caches


def build_slot_mappings_by_layer(
    slot_mappings: torch.Tensor, kv_cache_config: KVCacheConfig
) -> dict[str, torch.Tensor]:
    """按层构建 slot mappings。

    将 slot mappings 映射到每个注意力层。

    Args:
        slot_mappings: slot mappings 张量
        kv_cache_config: KV 缓存配置

    Returns:
        层名到 slot mapping 张量的映射
    """
    slot_mappings_by_layer: dict[str, torch.Tensor] = {}
    kv_cache_groups = kv_cache_config.kv_cache_groups
    for slot_mapping, kv_cache_group in zip(slot_mappings, kv_cache_groups):
        for layer_name in kv_cache_group.layer_names:
            slot_mappings_by_layer[layer_name] = slot_mapping
    return slot_mappings_by_layer


def build_attn_metadata(
    attn_groups: list[list[AttentionGroup]],
    num_reqs: int,
    num_tokens: int,
    query_start_loc_gpu: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    max_query_len: int,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    block_tables: Sequence[torch.Tensor],
    slot_mappings: torch.Tensor,
    kv_cache_config: KVCacheConfig,
    dcp_local_seq_lens: torch.Tensor | None = None,
    encoder_seq_lens: dict[int, tuple[torch.Tensor, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """构建注意力元数据。

    为每个注意力组构建注意力元数据，包含查询位置、序列长度、
    块表、slot mapping 等信息。

    Args:
        attn_groups: 注意力组列表
        num_reqs: 请求数量
        num_tokens: token 数量
        query_start_loc_gpu: GPU 上的查询起始位置
        query_start_loc_cpu: CPU 上的查询起始位置
        max_query_len: 最大查询长度
        seq_lens: 序列长度
        max_seq_len: 最大序列长度
        block_tables: 块表
        slot_mappings: slot mappings
        kv_cache_config: KV 缓存配置
        dcp_local_seq_lens: DCP 本地序列长度（可选）
        encoder_seq_lens: 编码器序列长度（可选）

    Returns:
        层名到注意力元数据的映射字典
    """
    seq_lens = seq_lens[:num_reqs]
    if dcp_local_seq_lens is not None:
        dcp_local_seq_lens = dcp_local_seq_lens[:num_reqs]

    attn_metadata: dict[str, Any] = {}
    num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
    for i in range(num_kv_cache_groups):
        block_table = block_tables[i]
        slot_mapping = slot_mappings[i]

        common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=query_start_loc_gpu,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            max_query_len=max_query_len,
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
            causal=True,
            dcp_local_seq_lens=dcp_local_seq_lens,
        )
        if encoder_seq_lens and i in encoder_seq_lens:
            encoder_seq_lens_gpu, encoder_seq_lens_cpu = encoder_seq_lens[i]
            common_attn_metadata.encoder_seq_lens = encoder_seq_lens_gpu
            common_attn_metadata.encoder_seq_lens_cpu = encoder_seq_lens_cpu

        for attn_group in attn_groups[i]:
            attn_metadata_builder = attn_group.get_metadata_builder(0)
            metadata = attn_metadata_builder.build(
                common_prefix_len=0, common_attn_metadata=common_attn_metadata
            )
            for layer_name in attn_group.layer_names:
                attn_metadata[layer_name] = metadata
    return attn_metadata
