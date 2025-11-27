#! SPDX-License-Identifier: Apache-2.0
#! SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os

import torch
import torch.nn as nn

from vllm.v1.attention.backends.hybrid_attn import HybridAttentionImpl


class _ToyLayer(nn.Module):
    """Minimal layer exposing an SSM adapter attribute.

    This is used to exercise HybridAttentionImpl end-to-end without relying
    on the full model stack.
    """

    def __init__(self, ssm_adapter: nn.Module) -> None:
        super().__init__()
        self.ssm_adapter = ssm_adapter


class _PrefixSumAdapter(nn.Module):
    """Adapter that mirrors HybridSSMAdapter's prefix-sum behavior.

    We keep this test-local to avoid depending on vLLM config plumbing while
    still validating the synthetic long-range effect.
    """

    def forward_history_branch_decode(
        self, hidden_states: torch.Tensor, attn_metadata=None
    ) -> torch.Tensor:
        num_tokens = getattr(attn_metadata, "num_actual_tokens", hidden_states.shape[0])
        prefix = torch.cumsum(hidden_states[:num_tokens], dim=0)
        out = torch.zeros_like(hidden_states)
        out[:num_tokens] = prefix
        return out


def test_hybrid_attention_prefix_sum_synthetic_task():
    """Synthetic long-range task for the hybrid attention prefix-sum rule.

    We construct a sequence of per-token \"values\" v_t and ask the hybrid
    backend to return the prefix sum at each step. A pure-attention backend
    stubbed to zero cannot solve this task, but the hybrid path with the
    prefix-sum adapter can, demonstrating that the SSM branch can carry
    history-dependent information beyond the immediate token.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_heads = 2
    head_size = 3
    num_kv_heads = 2
    num_tokens = 5

    impl = HybridAttentionImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=1.0,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )

    # Replace Triton impl with a stub that simply copies query into output.
    class _CopyTritonImpl:
        def forward(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale=None,
            output_block_scale=None,
        ):
            output.copy_(query)

    impl._triton_impl = _CopyTritonImpl()  # type: ignore[attr-defined]

    # Hybrid adapter that performs a prefix sum over the token dimension.
    adapter = _PrefixSumAdapter()
    layer = _ToyLayer(adapter).to(device)

    # Simple per-token scalar values broadcast across heads and head_size.
    base = torch.arange(1, num_tokens + 1, dtype=torch.float32, device=device)
    query = base.view(num_tokens, 1, 1).expand(num_tokens, num_heads, head_size)

    key = torch.zeros_like(query)
    value = torch.zeros_like(query)
    kv_cache = torch.empty(0, device=device)
    output = torch.empty_like(query)

    class _Meta:
        def __init__(self, num_actual_tokens: int):
            self.num_actual_tokens = num_actual_tokens

    attn_metadata = _Meta(num_tokens)

    out = impl.forward(
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=output,
    )

    # Expect prefix sums along the token dimension, broadcast across heads.
    expected_prefix = torch.cumsum(base, dim=0).view(num_tokens, 1, 1).expand_as(out)
    assert torch.allclose(out, expected_prefix)


