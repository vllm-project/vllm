# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import types

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata


@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("strided_output", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mixed_spec_output_preserves_order_and_padding(
    monkeypatch, prefill, strided_output, dtype
):
    """Kernel outputs must reach the caller's storage in original token order."""
    spec_indices = torch.tensor([1, 2, 5, 6])
    non_spec_indices = torch.tensor([0, 3, 4])
    meta = GDNAttentionMetadata(
        num_prefills=int(prefill),
        num_prefill_tokens=2 if prefill else 0,
        num_decodes=1 if prefill else 3,
        num_decode_tokens=1 if prefill else 3,
        num_spec_decodes=2,
        num_spec_decode_tokens=4,
        num_actual_tokens=7,
        spec_sequence_masks=torch.tensor(
            [False, True, False, True] if prefill else [False, True, False, False, True]
        ),
        spec_token_indx=spec_indices,
        non_spec_token_indx=non_spec_indices,
        spec_query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        non_spec_query_start_loc=torch.tensor(
            [0, 1, 3] if prefill else [0, 1, 2, 3], dtype=torch.int32
        ),
        spec_state_indices_tensor=torch.tensor([[1, 2], [3, 4]]),
        non_spec_state_indices_tensor=torch.tensor([5, 6, 7]),
        prefill_state_indices=torch.tensor([5, 6]),
        prefill_has_initial_state=torch.tensor([True, True]),
    )
    layer = types.SimpleNamespace(
        prefix="test",
        enable_packed_recurrent_decode=False,
        kv_cache=(torch.zeros(8, 1, 1), torch.zeros(8, 2, 3, 2)),
        conv1d=types.SimpleNamespace(weight=torch.zeros(1, 1, 1), bias=None),
        activation="silu",
        A_log=torch.zeros(2),
        dt_bias=torch.zeros(2),
        num_k_heads=2,
        tp_size=1,
        head_k_dim=2,
        head_v_dim=3,
        key_dim=4,
        value_dim=6,
    )
    layer.rearrange_mixed_qkv = types.MethodType(
        gdn.QwenGatedDeltaNetAttention.rearrange_mixed_qkv, layer
    )

    # Keep the real partitioning and assembly, with deterministic kernel outputs.
    def attention(**kwargs):
        return kwargs["v"], kwargs["initial_state"]

    def post_conv(**kwargs):
        q, k, v = layer.rearrange_mixed_qkv(kwargs["conv_output"])
        gates = torch.zeros(v.shape[1:3], dtype=dtype)
        return q.squeeze(0), k.squeeze(0), v.squeeze(0), gates, gates

    layer.chunk_gated_delta_rule = attention
    monkeypatch.setattr(
        gdn,
        "get_forward_context",
        lambda: types.SimpleNamespace(attn_metadata={"test": meta}),
    )
    monkeypatch.setattr(gdn, "causal_conv1d_update", lambda x, *args, **kwargs: x)
    monkeypatch.setattr(gdn, "causal_conv1d_fn", lambda x, *args, **kwargs: x)
    monkeypatch.setattr(gdn, "fused_sigmoid_gating_delta_rule_update", attention)
    monkeypatch.setattr(gdn, "fused_post_conv_prep", post_conv)
    mixed_qkv = torch.arange(7 * 14, dtype=dtype).view(7, 14)
    backing = torch.full((10, 2, 6 if strided_output else 3), -1, dtype=dtype)
    output = backing[1:9, :, ::2] if strided_output else backing[1:9]
    alias = output.view_as(output)
    gates = torch.zeros(7, 2, dtype=dtype)

    gdn.QwenGatedDeltaNetAttention._forward_core(layer, mixed_qkv, gates, gates, output)

    torch.testing.assert_close(alias[:7], mixed_qkv[:, 8:].reshape(7, 2, 3))
    assert torch.all(alias[7:] == -1)
    assert torch.all(backing[[0, 9]] == -1)
    if strided_output:
        assert torch.all(backing[:, :, 1::2] == -1)
