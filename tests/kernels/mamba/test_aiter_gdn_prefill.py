# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    compute_causal_conv1d_metadata,
)


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="AITER GDN prefill requires ROCm",
)
@pytest.mark.parametrize("lengths", [[128, 193], [8192]])
def test_aiter_gdn_prefill_split_qkv(lengths: list[int]) -> None:
    """Compare AITER channel-last split-QKV output with vLLM's Triton path."""
    pytest.importorskip("aiter")
    try:
        from aiter.ops.causal_conv1d_fwd_split_qkv import (
            causal_conv1d_split_qkv_hip_fn,
        )
    except ImportError:
        pytest.skip("Installed AITER lacks the GDN prefill split-QKV op")

    torch.manual_seed(0)
    device = torch.device("cuda")
    num_k_heads, num_v_heads = 4, 8
    head_k_dim = head_v_dim = 128
    k_dim = num_k_heads * head_k_dim
    v_dim = num_v_heads * head_v_dim
    conv_dim = 2 * k_dim + v_dim
    total_tokens = sum(lengths)

    mixed_qkv = (
        torch.randn(total_tokens, conv_dim, device=device, dtype=torch.bfloat16) * 0.1
    )
    channel_last = mixed_qkv.transpose(0, 1)
    assert channel_last.stride() == (1, conv_dim)

    weight = torch.randn(conv_dim, 4, device=device, dtype=torch.bfloat16) * 0.1
    bias = torch.randn(conv_dim, device=device, dtype=torch.bfloat16) * 0.1
    query_start_loc_cpu = torch.tensor(
        [0, *itertools.accumulate(lengths)], dtype=torch.int32
    )
    query_start_loc = query_start_loc_cpu.to(device)
    cache_indices = torch.arange(1, len(lengths) + 1, device=device, dtype=torch.int32)
    has_initial_state = torch.tensor(
        [i % 2 == 0 for i in range(len(lengths))],
        device=device,
        dtype=torch.bool,
    )

    conv_state = (
        torch.randn(
            len(lengths) + 1,
            conv_dim,
            3,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    triton_state = conv_state.clone()
    aiter_state = conv_state.clone()

    nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
        query_start_loc_cpu,
        device=device,
        block_sizes=(8, 64),
    )
    metadata = SimpleNamespace(
        nums_dict=nums_dict,
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=token_chunk_offset_ptr,
    )

    triton_output = causal_conv1d_fn(
        channel_last,
        weight,
        bias,
        activation="silu",
        conv_states=triton_state,
        has_initial_state=has_initial_state,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
        metadata=metadata,
    ).transpose(0, 1)
    q, k, v = causal_conv1d_split_qkv_hip_fn(
        x=channel_last,
        weight=weight,
        bias=bias,
        conv_states=aiter_state,
        query_start_loc=query_start_loc,
        k_dim=k_dim,
        v_dim=v_dim,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation="silu",
        pad_slot_id=NULL_BLOCK_ID,
        block_m=64,
        metadata=metadata,
    )

    torch.testing.assert_close(
        torch.cat((q, k, v), dim=-1),
        triton_output,
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(aiter_state, triton_state, atol=0, rtol=0)
