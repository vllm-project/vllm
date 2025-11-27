#! SPDX-License-Identifier: Apache-2.0
#! SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.config import set_current_vllm_config
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
    model_parallel_is_initialized,
)
from vllm.model_executor.layers.hybrid_ssm_adapter import HybridSSMAdapter
from vllm.v1.attention.backends.hybrid_attn import HybridAttentionImpl


def test_hybrid_ssm_adapter_state_shape_and_dtype():
    """HybridSSMAdapter should match Mamba1 state shape and dtype helpers."""
    vllm_config = create_vllm_config(add_mock_model_methods=False)
    cache_config = vllm_config.cache_config
    model_config = vllm_config.model_config

    # Provide reasonable defaults for Mamba state layout.
    cache_config.mamba_block_size = 16

    hidden_size = 64
    ssm_state_size = 8
    conv_kernel_size = 3
    intermediate_size = 4 * hidden_size

    with set_current_vllm_config(vllm_config):
        adapter = HybridSSMAdapter(
            hidden_size=hidden_size,
            ssm_state_size=ssm_state_size,
            conv_kernel_size=conv_kernel_size,
            intermediate_size=intermediate_size,
            model_config=model_config,
            cache_config=cache_config,
            prefix="hybrid.ssm",
        )

    # State shape and dtype should be consistent with the calculators.
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateDtypeCalculator,
        MambaStateShapeCalculator,
    )

    if model_parallel_is_initialized():
        tp_world_size = get_tensor_model_parallel_world_size()
    else:
        # In single-process unit tests, model parallel is often not initialized.
        # Fall back to the tensor-parallel size from the vLLM config, which is
        # 1 by default in these tests.
        tp_world_size = vllm_config.parallel_config.tensor_parallel_size
    expected_shapes = MambaStateShapeCalculator.mamba1_state_shape(
        tp_world_size=tp_world_size,
        intermediate_size=intermediate_size,
        state_size=ssm_state_size,
        conv_kernel=conv_kernel_size,
    )
    expected_dtypes = MambaStateDtypeCalculator.mamba1_state_dtype(
        model_config.dtype,
        cache_config.mamba_cache_dtype,
        cache_config.mamba_ssm_cache_dtype,
    )

    assert tuple(adapter.get_state_shape()) == expected_shapes
    assert adapter.get_state_dtype() == expected_dtypes

    # Prefill / decode history branches should preserve shape.
    hidden_states = torch.randn(5, hidden_size)

    class PrefillMeta:
        num_prefill_tokens = hidden_states.shape[0]

    class DecodeMeta:
        num_decode_tokens = hidden_states.shape[0]

    prefill_out = adapter.forward_history_branch_prefill(
        hidden_states, attn_metadata=PrefillMeta()
    )
    decode_out = adapter.forward_history_branch_decode(
        hidden_states, attn_metadata=DecodeMeta()
    )

    assert prefill_out.shape == hidden_states.shape
    assert decode_out.shape == hidden_states.shape


def test_hybrid_attention_impl_fuses_ssm_output():
    """HybridAttentionImpl should add SSM contribution on top of attention."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_heads = 4
    head_size = 8
    num_kv_heads = 2
    num_tokens = 6

    impl = HybridAttentionImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=1.0,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )

    # Replace the internal Triton impl with a stub that writes zeros to output.
    class _StubTritonImpl:
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
            output.zero_()

    impl._triton_impl = _StubTritonImpl()  # type: ignore[attr-defined]

    class _StubSSMAdapter(torch.nn.Module):
        def forward_history_branch_decode(self, hidden_states, attn_metadata=None):
            # Return a constant tensor so we can check fusion.
            return torch.full_like(hidden_states, 2.0)

    class _StubLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ssm_adapter = _StubSSMAdapter()

    layer = _StubLayer().to(device)

    query = torch.zeros(num_tokens, num_heads, head_size, device=device)
    key = torch.zeros_like(query)
    value = torch.zeros_like(query)
    kv_cache = torch.empty(0, device=device)
    output = torch.empty_like(query)

    class _StubMetadata:
        pass

    attn_metadata = _StubMetadata()
    attn_metadata.num_actual_tokens = num_tokens

    out = impl.forward(
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=output,
    )

    assert torch.allclose(out, torch.full_like(query, 2.0))


