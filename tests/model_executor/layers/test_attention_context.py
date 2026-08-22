# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention.attention import get_attention_context


def test_get_attention_context_uses_base_slot_mapping_from_list():
    layer_name = "model.layers.0.self_attn.attn"
    kv_cache = torch.empty(0)
    attn_layer = SimpleNamespace(kv_cache=kv_cache)
    base_attn_metadata = object()
    draft_attn_metadata = object()
    base_slot_mapping = torch.tensor([0, 1])
    draft_slot_mapping = torch.tensor([2, 3])

    vllm_config = VllmConfig()
    vllm_config.compilation_config.static_forward_context = {
        layer_name: attn_layer,
    }

    with set_forward_context(
        attn_metadata=[
            {layer_name: base_attn_metadata},
            {layer_name: draft_attn_metadata},
        ],
        vllm_config=vllm_config,
        slot_mapping=[
            {layer_name: base_slot_mapping},
            {layer_name: draft_slot_mapping},
        ],
    ):
        attn_metadata, returned_attn_layer, returned_kv_cache, layer_slot_mapping = (
            get_attention_context(layer_name)
        )

    assert attn_metadata is base_attn_metadata
    assert returned_attn_layer is attn_layer
    assert returned_kv_cache is kv_cache
    assert layer_slot_mapping is base_slot_mapping
