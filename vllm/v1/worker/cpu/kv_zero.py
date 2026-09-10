# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch equivalent of the KV block zeroing kernel in ``v1/worker/utils.py``.

Patched at method level, like the block tables: the kernel addresses segments
absolutely, and a torch implementation cannot resolve an address back to the
tensor it points into. Holding the block views instead makes the segment walk
unnecessary, since zeroing a block's rows covers every dimension inside it.
"""

from collections.abc import Iterable
from typing import Any

import torch

from vllm.v1.kv_cache_interface import AttentionSpec


def init(
    self,
    device: torch.device,
    attn_groups_iter: Iterable[Any],
    kernel_block_sizes: list[int],
    static_forward_context: dict[str, Any],
    num_blocks: int,
    runner_only_attn_layers: set[str] | None = None,
) -> None:
    self.device = device
    # One entry per distinct buffer, with the number of kernel-block rows a
    # logical block owns there under virtual block splitting.
    self._views: list[tuple[torch.Tensor, int]] = []
    seen: set[int] = set()

    for group in attn_groups_iter:
        if not isinstance(group.kv_cache_spec, AttentionSpec):
            continue
        if group.kv_cache_group_id >= len(kernel_block_sizes):
            continue
        for layer_name in group.layer_names:
            if runner_only_attn_layers and layer_name in runner_only_attn_layers:
                continue
            kv = static_forward_context[layer_name].kv_cache
            if not isinstance(kv, torch.Tensor) or kv.data_ptr() in seen:
                continue
            seen.add(kv.data_ptr())
            self._views.append((kv, kv.shape[0] // num_blocks))


def zero_block_ids(self, block_ids: list[int]) -> None:
    if not block_ids or not self._views:
        return
    ids = torch.as_tensor(block_ids, dtype=torch.long)
    for kv, ratio in self._views:
        rows = ids
        if ratio > 1:
            rows = ((ids * ratio).unsqueeze(1) + torch.arange(ratio)).flatten()
        kv[rows] = 0
