# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only checks for the GDN profile warmup input sizing."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)


def test_profile_warmup_reuses_projected_tokens():
    num_tokens = 7
    num_k_heads = 2
    num_v_heads = 3
    head_k_dim = 4
    head_v_dim = 5
    qkv_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
    z_dim = num_v_heads * head_v_dim
    projected = torch.empty(num_tokens, qkv_dim + z_dim, dtype=torch.bfloat16)
    prep_calls = []
    chunk_calls = []

    layer = SimpleNamespace(
        _prefill_kernels_warmed_up=False,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        tp_size=1,
        A_log=torch.zeros(num_v_heads),
        dt_bias=torch.zeros(num_v_heads),
        prefix="test.gdn",
        gdn_prefill_backend="triton",
        get_state_dtype=lambda: (None, torch.float32),
    )

    def fake_prep(**kwargs):
        prep_calls.append(kwargs)
        assert kwargs["conv_output"].data_ptr() == projected.data_ptr()
        assert kwargs["conv_output"].shape == (num_tokens, qkv_dim)
        return (
            torch.empty(num_tokens, num_k_heads, head_k_dim),
            torch.empty(num_tokens, num_k_heads, head_k_dim),
            torch.empty(num_tokens, num_v_heads, head_v_dim),
            torch.empty(num_tokens, num_v_heads),
            torch.empty(num_tokens, num_v_heads),
        )

    def fake_chunk(**kwargs):
        chunk_calls.append(kwargs)
        return kwargs["q"], kwargs["initial_state"]

    layer.chunk_gated_delta_rule = fake_chunk
    with patch.object(
        qwen_gdn_linear_attn, "fused_post_conv_prep", side_effect=fake_prep
    ):
        QwenGatedDeltaNetAttention._warmup_prefill_kernels(
            layer,  # type: ignore[arg-type]
            projected,
            z_dim,
        )

    assert len(prep_calls) == 1
    assert len(chunk_calls) == 1
    assert chunk_calls[0]["q"].shape[1] == num_tokens
    assert chunk_calls[0]["cu_seqlens"].tolist() == [0, num_tokens]
