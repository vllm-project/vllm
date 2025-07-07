# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for v1 attention backends without GPUModelRunner dependency."""

import pytest
import torch

from tests.v1.attention.utils import (BatchSpec, create_common_attn_metadata,
                                      create_standard_kv_cache_spec,
                                      create_vllm_config,
                                      get_attention_backend)
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import FullAttentionSpec


def _convert_dtype_to_torch(dtype):
    """Convert ModelDType to torch.dtype."""
    if isinstance(dtype, str):
        if dtype == "auto":
            return torch.float16  # Default dtype for testing
        elif dtype in STR_DTYPE_TO_TORCH_DTYPE:
            return STR_DTYPE_TO_TORCH_DTYPE[dtype]
        else:
            raise ValueError(f"Unknown dtype: {dtype}")
    elif isinstance(dtype, torch.dtype):
        return dtype
    else:
        raise ValueError(f"Unknown dtype: {dtype}")


# Define common batch configurations
BATCH_SPECS = {
    "small_decode":
    BatchSpec(batch_size=2, seq_lens=[32, 40], query_lens=[1, 1]),
    "small_prefill":
    BatchSpec(batch_size=2, seq_lens=[32, 40], query_lens=[8, 8]),
    "mixed_small":
    BatchSpec(batch_size=4, seq_lens=[32, 40, 48, 56], query_lens=[1, 1, 5,
                                                                   5]),
    "medium_decode":
    BatchSpec(batch_size=8,
              seq_lens=[128, 256, 512, 1024, 128, 256, 512, 1024],
              query_lens=[1, 1, 1, 1, 1, 1, 1, 1]),
    "medium_prefill":
    BatchSpec(batch_size=4,
              seq_lens=[256, 512, 1024, 2048],
              query_lens=[16, 16, 16, 16]),
    "mixed_medium":
    BatchSpec(batch_size=6,
              seq_lens=[512, 1024, 2048, 512, 1024, 2048],
              query_lens=[1, 1, 1, 7, 7, 7]),
    "large_decode":
    BatchSpec(batch_size=32, seq_lens=[2048] * 32, query_lens=[1] * 32),
    "large_prefill":
    BatchSpec(batch_size=8, seq_lens=[4096] * 8, query_lens=[32] * 8),
    "single_decode":
    BatchSpec(batch_size=1, seq_lens=[1024], query_lens=[1]),
    "single_prefill":
    BatchSpec(batch_size=1, seq_lens=[1024], query_lens=[64]),
}


def create_dummy_kv_cache(kv_cache_spec: FullAttentionSpec,
                          device: torch.device) -> torch.Tensor:
    """Create a dummy KV cache tensor for testing."""
    # Create a reasonably sized KV cache for testing
    num_blocks = 100
    kv_cache = torch.randn(
        2,  # K and V
        num_blocks,
        kv_cache_spec.block_size,
        kv_cache_spec.num_kv_heads,
        kv_cache_spec.head_size,
        dtype=_convert_dtype_to_torch(kv_cache_spec.dtype),
        device=device,
    )
    return kv_cache


class MockAttentionLayer:
    """A mock attention layer for testing."""

    def __init__(self):
        self._q_scale = torch.tensor(1.0)
        self._k_scale = torch.tensor(1.0)
        self._v_scale = torch.tensor(1.0)


def run_attention_backend(backend_name: str, kv_cache_spec: FullAttentionSpec,
                          vllm_config, device: torch.device,
                          common_attn_metadata: CommonAttentionMetadata,
                          query: torch.Tensor, key: torch.Tensor,
                          value: torch.Tensor,
                          kv_cache: torch.Tensor) -> torch.Tensor:
    """Run attention computation using the specified backend's AttentionImpl."""

    builder_cls, impl_cls = get_attention_backend(backend_name)

    # Build metadata
    builder = builder_cls(kv_cache_spec, vllm_config, device)
    attn_metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_attn_metadata,
    )

    # Instantiate implementation
    num_heads = kv_cache_spec.num_kv_heads
    head_size = kv_cache_spec.head_size
    scale = 1.0 / (head_size**0.5)
    impl = impl_cls(
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        num_kv_heads=num_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )

    # Create mock layer and output buffer
    mock_layer = MockAttentionLayer()
    output = torch.empty_like(query)

    # Run forward pass
    # NOTE: The query, key, and value are already shaped correctly
    # in the calling test function.
    output = impl.forward(mock_layer,
                          query,
                          key,
                          value,
                          kv_cache,
                          attn_metadata,
                          output=output)

    return output


@pytest.mark.parametrize("batch_spec_name", [
    "small_decode", "small_prefill", "mixed_small", "medium_decode",
    "medium_prefill", "mixed_medium"
])
@pytest.mark.parametrize("model", ["meta-llama/Meta-Llama-3-8B"])
def test_backend_correctness(batch_spec_name: str, model: str):
    """
    Test that all backends produce similar outputs to a reference implementation
    using torch.nn.functional.scaled_dot_product_attention.

    This test works by:
    1. Generating a batch of sequences with specified context and query lengths.
    2. Computing a ground-truth attention output using torch.sdpa on
       contiguous Q, K, and V tensors.
    3. Simulating vLLM's paged KV cache: It takes the context portion of the
       K/V tensors and manually places them into a paged buffer according to
       the test's (randomly generated) block table.
    4. Running each vLLM attention backend with the new queries and the
       simulated paged KV cache.
    5. Comparing the vLLM backend's output to the ground-truth SDPA output.
    """
    batch_spec = BATCH_SPECS[batch_spec_name]
    vllm_config = create_vllm_config(model_name=model)
    device = torch.device("cuda:0")

    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)
    common_attn_metadata = create_common_attn_metadata(
        batch_spec, vllm_config.cache_config.block_size, device)

    # 1. Setup
    batch_size = batch_spec.batch_size
    seq_lens = batch_spec.seq_lens
    query_lens = batch_spec.query_lens
    num_q_heads = vllm_config.model_config.get_num_attention_heads(
        vllm_config.parallel_config)
    num_kv_heads = vllm_config.model_config.get_num_kv_heads(
        vllm_config.parallel_config)
    head_size = vllm_config.model_config.get_head_size()
    dtype = _convert_dtype_to_torch(vllm_config.model_config.dtype)
    block_size = vllm_config.cache_config.block_size
    scale = 1.0 / (head_size**0.5)

    # 2. Generate data and compute SDPA reference output
    all_q_vllm, all_k_vllm, all_v_vllm = [], [], []
    all_sdpa_outputs = []
    all_k_context, all_v_context = [], []

    for i in range(batch_size):
        s_len = seq_lens[i]
        q_len = query_lens[i]
        context_len = s_len - q_len

        # Generate Q, K, V for the whole sequence to be used in SDPA
        q_for_sdpa = torch.randn(q_len,
                                 num_q_heads,
                                 head_size,
                                 dtype=dtype,
                                 device=device)
        k_full = torch.randn(s_len,
                             num_kv_heads,
                             head_size,
                             dtype=dtype,
                             device=device)
        v_full = torch.randn(s_len,
                             num_kv_heads,
                             head_size,
                             dtype=dtype,
                             device=device)

        # SDPA expects (N, H, L, D), so unsqueeze batch and permute
        q_sdpa_in = q_for_sdpa.unsqueeze(0).transpose(1, 2)
        k_sdpa_in = k_full.unsqueeze(0).transpose(1, 2)
        v_sdpa_in = v_full.unsqueeze(0).transpose(1, 2)

        # Create a causal mask that reflects that the query tokens are at the
        # end of the full sequence.
        attn_mask = torch.ones(q_len, s_len, dtype=torch.bool,
                               device=device).tril(diagonal=context_len)

        sdpa_out_i = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa_in, k_sdpa_in, v_sdpa_in, attn_mask=attn_mask, scale=scale)
        # Convert back to (L, H, D)
        all_sdpa_outputs.append(sdpa_out_i.transpose(1, 2).squeeze(0))

        # Inputs for vLLM backends are just the new tokens
        all_q_vllm.append(q_for_sdpa)
        all_k_vllm.append(k_full[context_len:])
        all_v_vllm.append(v_full[context_len:])

        # Contextual K/V data used to populate the paged cache
        all_k_context.append(k_full[:context_len])
        all_v_context.append(v_full[:context_len])

    query_vllm = torch.cat(all_q_vllm, dim=0)
    key_vllm = torch.cat(all_k_vllm, dim=0)
    value_vllm = torch.cat(all_v_vllm, dim=0)
    sdpa_output = torch.cat(all_sdpa_outputs, dim=0)

    # 3. Simulate Paged KV Cache and a realistic slot_mapping
    block_table = common_attn_metadata.block_table_tensor
    num_blocks = int(block_table.max().item()) + 1
    kv_cache = torch.zeros(2,
                           num_blocks,
                           block_size,
                           num_kv_heads,
                           head_size,
                           dtype=dtype,
                           device=device)

    # Create a realistic slot mapping that corresponds to the block table
    slot_mapping_list = []
    query_start_locs = common_attn_metadata.query_start_loc_cpu.tolist()

    for i in range(batch_size):
        context_len = seq_lens[i] - query_lens[i]
        start_idx = query_start_locs[i]
        end_idx = query_start_locs[i + 1]

        for token_idx_in_query in range(end_idx - start_idx):
            token_seq_idx = context_len + token_idx_in_query
            logical_block_idx = token_seq_idx // block_size
            offset_in_block = token_seq_idx % block_size
            physical_block_num = int(block_table[i, logical_block_idx].item())
            slot = physical_block_num * block_size + offset_in_block
            slot_mapping_list.append(slot)

    common_attn_metadata.slot_mapping = torch.tensor(slot_mapping_list,
                                                     dtype=torch.long,
                                                     device=device)

    # Populate the cache with the context tokens
    for i in range(batch_size):
        k_context, v_context = all_k_context[i], all_v_context[i]
        context_len = k_context.shape[0]

        for token_idx in range(context_len):
            logical_block_idx = token_idx // block_size
            offset_in_block = token_idx % block_size
            phys_block_num = int(block_table[i, logical_block_idx].item())

            kv_cache[0, phys_block_num, offset_in_block] = k_context[token_idx]
            kv_cache[1, phys_block_num, offset_in_block] = v_context[token_idx]

    # 4. Run vLLM backends and compare
    backends_to_test = ["flash_attn", "flex_attention"]
    for backend_name in backends_to_test:
        try:
            backend_output = run_attention_backend(backend_name, kv_cache_spec,
                                                   vllm_config, device,
                                                   common_attn_metadata,
                                                   query_vllm, key_vllm,
                                                   value_vllm, kv_cache)

            # Check shape and dtype consistency
            assert backend_output.shape == sdpa_output.shape, (
                f"[{backend_name}] shape {backend_output.shape} != "
                f"SDPA shape {sdpa_output.shape}")
            assert backend_output.dtype == sdpa_output.dtype, (
                f"[{backend_name}] dtype {backend_output.dtype} != "
                f"SDPA dtype {sdpa_output.dtype}")

            assert torch.isfinite(backend_output).all(), (
                f"[{backend_name}] produced non-finite values")

            # Check numerical similarity
            rtol = 1e-5 if backend_output.dtype == torch.float32 else 1e-2
            atol = 1e-4 if backend_output.dtype == torch.float32 else 1e-3

            max_diff = torch.max(torch.abs(backend_output -
                                           sdpa_output)).item()
            assert torch.allclose(
                backend_output, sdpa_output, rtol=rtol, atol=atol), (
                    f"[{backend_name}] output differs from SDPA baseline. "
                    f"Max diff: {max_diff:.6f}")

        except Exception as e:
            if "not available" in str(e) or "not supported" in str(e).lower():
                pytest.skip(f"{backend_name} not available/supported: {e}")
            else:
                pytest.fail(f"[{backend_name}] failed: {e}")
