# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PagedAttention 操作模块。

本模块实现了 PagedAttention 的核心操作，负责：
- 分割 KV 缓存为 Key 缓存和 Value 缓存
- 将 Key 和 Value 写入分页缓存

主要类：
- PagedAttention: PagedAttention 操作类
"""


import torch

from vllm.platforms import current_platform

if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops
elif current_platform.is_xpu():
    from vllm._xpu_ops import xpu_ops as ops  # type: ignore[no-redef]


class PagedAttention:
    """PagedAttention 操作类。

    提供 KV 缓存的分割和写入操作。
    """
    @staticmethod
    def split_kv_cache(
        kv_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """分割 KV 缓存为 Key 缓存和 Value 缓存。

        Args:
            kv_cache: KV 缓存张量
            num_kv_heads: KV 头数量
            head_size: 头大小

        Returns:
            (key_cache, value_cache) 元组
        """
        x = 16 // kv_cache.element_size()
        num_blocks = kv_cache.shape[1]

        key_cache = kv_cache[0]
        key_cache = key_cache.view(num_blocks, num_kv_heads, head_size // x, -1, x)
        value_cache = kv_cache[1]
        value_cache = value_cache.view(num_blocks, num_kv_heads, head_size, -1)
        return key_cache, value_cache

    @staticmethod
    def write_to_paged_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> None:
        """将 Key 和 Value 写入分页缓存。

        Args:
            key: Key 张量
            value: Value 张量
            key_cache: Key 缓存
            value_cache: Value 缓存
            slot_mapping: 槽位映射
            kv_cache_dtype: KV 缓存数据类型
            k_scale: K 缩放因子
            v_scale: V 缩放因子
        """
        ops.reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping.flatten(),
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
