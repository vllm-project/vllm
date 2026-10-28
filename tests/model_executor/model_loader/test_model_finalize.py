# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RL weight updates call ``load_weights`` once per bucket: the model-level
finish step runs once after the last call."""

from types import SimpleNamespace

import torch
from torch import nn

from vllm.model_executor.model_loader.reload import finalize_layerwise_reload


class _HookCounter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def process_weights_after_loading(self) -> None:
        self.calls += 1


def test_reload_finish_runs_model_hook_once():
    model = _HookCounter()
    finalize_layerwise_reload(model, None)
    assert model.calls == 1


def test_llama4_dp_vision_qkv_shards_split_across_calls():
    """Llama4 DP vision: q/k/v land in their rows in any order, one per call."""
    from vllm.model_executor.models.mllama4 import Llama4VisionAttention

    attn = SimpleNamespace(q_size=4, kv_size=4)
    q, k, v = (torch.full((4, 3), float(i)) for i in range(3))
    param = torch.empty(12, 3)
    for shard_id, tensor in (("v", v), ("k", k), ("q", q)):
        Llama4VisionAttention._load_qkv_shard(attn, param, tensor, shard_id)
    torch.testing.assert_close(param, torch.cat([q, k, v]))


def test_llama4_dp_vision_qkv_block_scales_split_across_calls():
    """Block scales of the q/k/v parts land in their rows of the fused scale."""
    from vllm.model_executor.models.mllama4 import Llama4VisionAttention

    # 12 output rows in blocks of 2: q, k and v scales have 2 rows each.
    attn = SimpleNamespace(q_size=4, kv_size=4)
    q, k, v = (torch.full((2, 3), float(i)) for i in range(3))
    param = torch.empty(6, 3)
    for shard_id, tensor in (("k", k), ("v", v), ("q", q)):
        Llama4VisionAttention._load_qkv_shard(attn, param, tensor, shard_id)
    torch.testing.assert_close(param, torch.cat([q, k, v]))
