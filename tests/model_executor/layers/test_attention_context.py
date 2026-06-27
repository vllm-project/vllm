# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for vllm.model_executor.layers.attention.attention."""

from types import SimpleNamespace

import torch

from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers.attention.attention import get_attention_context


def _make_context(slot_mapping):
    layer_name = "layer.0"
    attn_layer = SimpleNamespace(kv_cache=torch.empty(0))
    return ForwardContext(
        no_compile_layers={layer_name: attn_layer},
        attn_metadata={layer_name: None},
        slot_mapping=slot_mapping,
    )


def test_get_attention_context_slot_mapping_list():
    # Speculative decoding ubatch passes slot_mapping as list[dict]; [0] is the
    # base-model dict. See https://github.com/vllm-project/vllm/issues/46830.
    layer_name = "layer.0"
    slots = torch.arange(4)
    ctx = _make_context([{layer_name: slots}])
    with override_forward_context(ctx):
        _, _, _, layer_slot_mapping = get_attention_context(layer_name)
    assert torch.equal(layer_slot_mapping, slots)
