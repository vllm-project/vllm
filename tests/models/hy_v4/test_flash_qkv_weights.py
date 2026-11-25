# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.models.hy_v4_flash.nvidia.model import HYV4FlashForCausalLM


def test_flash_qkv_checkpoint_shards():
    """Checkpoint order must not determine which half receives Q and KV."""
    prefix = "model.layers.0.self_attn."
    mapper = HYV4FlashForCausalLM.hf_to_vllm_mapper
    weights = [
        (prefix + "kv_a_proj_with_mqa.weight", torch.empty(1, 4)),
        (prefix + "q_a_proj.weight", torch.empty(2, 4)),
    ]
    mapped = [(n, getattr(w, "shard_id", None)) for n, w in mapper.apply(weights)]
    fused = prefix + "fused_qkv_a_proj.weight"
    assert mapped == [(fused, 1), (fused, 0)]
