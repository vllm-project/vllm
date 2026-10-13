# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch

from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.utils import KVBlockZeroer

if TYPE_CHECKING:
    from vllm.v1.worker.utils import AttentionGroup


class CPUKVBlockZeroer(KVBlockZeroer):
    """Zero newly allocated logical KV blocks through their tensor views."""

    def __init__(
        self,
        device: torch.device,
        attn_groups_iter: Iterable["AttentionGroup"],
        kernel_block_sizes: list[int],
        static_forward_context: dict[str, Any],
        num_blocks: int,
        runner_only_attn_layers: set[str] | None = None,
    ) -> None:
        self.device = device
        self._cache_views: list[tuple[torch.Tensor, int]] = []

        if runner_only_attn_layers is None:
            runner_only_attn_layers = set()
        for group in attn_groups_iter:
            spec = group.kv_cache_spec
            if not isinstance(spec, AttentionSpec):
                continue
            if group.kv_cache_group_id >= len(kernel_block_sizes):
                continue
            kernel_block_size = kernel_block_sizes[group.kv_cache_group_id]
            assert spec.block_size % kernel_block_size == 0
            for layer_name in group.layer_names:
                if layer_name in runner_only_attn_layers:
                    continue
                kv_cache = static_forward_context[layer_name].kv_cache
                if not isinstance(kv_cache, torch.Tensor):
                    continue
                if kv_cache.device.type != self.device.type:
                    continue
                assert kv_cache.shape[0] % num_blocks == 0, (
                    f"{layer_name}: {kv_cache.shape[0]} kernel blocks is not a "
                    f"multiple of {num_blocks} logical blocks"
                )
                self._cache_views.append((kv_cache, kv_cache.shape[0] // num_blocks))

    def zero_block_ids(self, block_ids: list[int]) -> None:
        if not block_ids or not self._cache_views:
            return

        logical_block_ids = torch.tensor(block_ids, dtype=torch.int64)
        indices_by_ratio = {1: logical_block_ids}
        for kv_cache, blocks_per_logical_block in self._cache_views:
            kernel_block_ids = indices_by_ratio.get(blocks_per_logical_block)
            if kernel_block_ids is None:
                offsets = torch.arange(blocks_per_logical_block, dtype=torch.int64)
                kernel_block_ids = (
                    logical_block_ids[:, None] * blocks_per_logical_block + offsets
                ).flatten()
                indices_by_ratio[blocks_per_logical_block] = kernel_block_ids
            kv_cache[kernel_block_ids] = 0
