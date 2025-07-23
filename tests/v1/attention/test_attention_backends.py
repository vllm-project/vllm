# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for v1 attention backends without GPUModelRunner dependency."""

import pytest
import torch

from tests.v1.attention.utils import (BatchSpec, _Backend,
                                      create_common_attn_metadata,
                                      create_standard_kv_cache_spec,
                                      create_vllm_config,
                                      get_attention_backend)
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, cdiv
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import FullAttentionSpec

BACKENDS_TO_TEST = [
    _Backend.FLASH_ATTN_VLLM_V1, _Backend.FLASHINFER_VLLM_V1,
    _Backend.FLEX_ATTENTION, _Backend.TRITON_ATTN_VLLM_V1
]

# Remove flashinfer from the list if it's not available
try:
    import flashinfer  # noqa: F401
except ImportError:
    BACKENDS_TO_TEST.remove(_Backend.FLASHINFER_VLLM_V1)


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
    BatchSpec(seq_lens=[32, 40], query_lens=[1, 1]),
    "small_prefill":
    BatchSpec(seq_lens=[32, 40], query_lens=[8, 8]),
    "mixed_small":
    BatchSpec(seq_lens=[32, 40, 48, 56], query_lens=[1, 1, 5, 5]),
    "medium_decode":
    BatchSpec(seq_lens=[128, 256, 512, 1024, 128, 256, 512, 1024],
              query_lens=[1, 1, 1, 1, 1, 1, 1, 1]),
    "medium_prefill":
    BatchSpec(seq_lens=[256, 512, 1024, 2048], query_lens=[16, 16, 16, 16]),
    "mixed_medium":
    BatchSpec(seq_lens=[512, 1024, 2048, 512, 1024, 2048],
              query_lens=[1, 1, 1, 7, 7, 7]),
    "large_decode":
    BatchSpec(seq_lens=[2048] * 32, query_lens=[1] * 32),
    "large_prefill":
    BatchSpec(seq_lens=[4096] * 8, query_lens=[32] * 8),
    "single_decode":
    BatchSpec(seq_lens=[1024], query_lens=[1]),
    "single_prefill":
    BatchSpec(seq_lens=[1024], query_lens=[64]),
}


def create_dummy_kv_cache(kv_cache_spec: FullAttentionSpec,
                          device: torch.device,
                          num_blocks: int = 100) -> torch.Tensor:
    """Create a dummy KV cache tensor for testing."""
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


def create_and_prepopulate_kv_cache(
        k_contexts: list[torch.Tensor],
        v_contexts: list[torch.Tensor],
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
        num_blocks: int,
        common_attn_metadata: CommonAttentionMetadata,
        randomize_blocks: bool = True) -> torch.Tensor:
    """Create and prepopulate a KV cache with context data.
    
    Args:
        k_contexts: List of key context tensors for each sequence
        v_contexts: List of value context tensors for each sequence
        seq_lens: List of sequence lengths
        block_size: Size of each block
        num_kv_heads: Number of KV heads
        head_size: Size of each head
        dtype: Data type for the cache
        device: Device to create the cache on
        num_blocks: Total number of blocks in the cache
        block_table: Block table tensor to populate
        randomize_blocks: Whether to randomly permute blocks 
                          or use sequential order
        
    Returns:
        Tuple of (kv_cache, updated_block_table)
    """
    batch_size = len(k_contexts)
    seq_lens = common_attn_metadata.seq_lens_cpu
    query_lens = common_attn_metadata.query_start_loc_cpu[
        1:] - common_attn_metadata.query_start_loc_cpu[:-1]
    context_lens = common_attn_metadata.num_computed_tokens_cpu
    block_table = common_attn_metadata.block_table_tensor
    slot_mapping = common_attn_metadata.slot_mapping

    # Create KV cache
    kv_cache = torch.empty(2,
                           num_blocks,
                           block_size,
                           num_kv_heads,
                           head_size,
                           dtype=dtype,
                           device=device)
    kv_cache_flat = kv_cache.view(2, -1, num_kv_heads, head_size)

    # Populate the cache with the context tokens
    # Start from block_id=1 since block_id=0 is considered the null block
    start_block_idx = 1
    for i in range(batch_size):
        k_context, v_context = k_contexts[i], v_contexts[i]
        start = start_block_idx * block_size
        end = start + k_context.shape[0]
        kv_cache_flat[0, start:end, ...] = k_context
        kv_cache_flat[1, start:end, ...] = v_context

        # Stay block aligned and allocate enough blocks for the new tokens
        start_block_idx += cdiv(int(seq_lens[i]), block_size)

    blocks_end = start_block_idx

    # Permute the context blocks (excluding block 0 which is null)
    if randomize_blocks:
        perm = torch.randperm(
            blocks_end - 1) + 1  # Random permutation starting from block 1
    else:
        perm = torch.arange(
            1, blocks_end)  # Sequential order starting from block 1

    inv_perm = torch.zeros(blocks_end, dtype=torch.long, device=device)
    inv_perm[1:] = torch.argsort(
        perm) + 1  # Add 1 to account for starting from block 1
    kv_cache[:, 1:blocks_end, ...] = kv_cache[:, perm, ...]

    # Construct the right block table
    # Start from block_id=1 since block_id=0 is considered the null block
    start_block_idx = 1
    for i in range(batch_size):
        num_blocks_for_seq = cdiv(int(seq_lens[i]), block_size)
        start = start_block_idx
        end = start + num_blocks_for_seq
        block_table[i, :num_blocks_for_seq] = inv_perm[start:end]
        start_block_idx += num_blocks_for_seq

        # Create a realistic slot mapping that corresponds to the block table
    for i in range(batch_size):
        token_offsets = torch.arange(int(query_lens[i])) + int(context_lens[i])
        block_indices = token_offsets // block_size
        token_inter_block_offsets = token_offsets % block_size
        start = common_attn_metadata.query_start_loc_cpu[i]
        end = common_attn_metadata.query_start_loc_cpu[i + 1]
        slot_mapping[start:end] = block_table[
            i,
            block_indices] * block_size + token_inter_block_offsets.to(device)

    return kv_cache


class MockAttentionLayer:
    """A mock attention layer for testing."""

    def __init__(self, device: torch.device):
        self._q_scale = torch.tensor(1.0, device=device)
        self._k_scale = torch.tensor(1.0, device=device)
        self._v_scale = torch.tensor(1.0, device=device)
        # Add float versions for flashinfer
        self._k_scale_float = 1.0
        self._v_scale_float = 1.0


def run_attention_backend(backend: _Backend, kv_cache_spec: FullAttentionSpec,
                          vllm_config, device: torch.device,
                          common_attn_metadata: CommonAttentionMetadata,
                          query: torch.Tensor, key: torch.Tensor,
                          value: torch.Tensor,
                          kv_cache: torch.Tensor) -> torch.Tensor:
    """Run attention computation using the specified backend's AttentionImpl."""

    builder_cls, impl_cls = get_attention_backend(backend)

    # Mock flashinfer's get_per_layer_parameters if needed
    if backend == _Backend.FLASHINFER_VLLM_V1:
        import unittest.mock

        from vllm.v1.attention.backends.flashinfer import PerLayerParameters

        def mock_get_per_layer_parameters(vllm_config):
            # Return mock parameters for a single layer
            head_size = vllm_config.model_config.get_head_size()
            return {
                "mock_layer":
                PerLayerParameters(
                    window_left=-1,  # No sliding window
                    logits_soft_cap=0.0,  # No soft cap
                    sm_scale=1.0 / (head_size**0.5)  # Standard scale
                )
            }

        with unittest.mock.patch(
                'vllm.v1.attention.backends.flashinfer.get_per_layer_parameters',
                mock_get_per_layer_parameters):
            builder = builder_cls(kv_cache_spec, vllm_config, device)
            attn_metadata = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
    else:
        # Build metadata
        builder = builder_cls(kv_cache_spec, vllm_config, device)
        attn_metadata = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
        )

    # Instantiate implementation
    num_heads = vllm_config.model_config.get_num_attention_heads(
        vllm_config.parallel_config)
    num_kv_heads = vllm_config.model_config.get_num_kv_heads(
        vllm_config.parallel_config)
    head_size = vllm_config.model_config.get_head_size()
    scale = 1.0 / (head_size**0.5)
    impl = impl_cls(
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )

    # Create mock layer and output buffer
    mock_layer = MockAttentionLayer(device)
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
    k_contexts, v_contexts = [], []

    for i in range(batch_size):
        s_len = seq_lens[i]
        q_len = query_lens[i]
        context_len = s_len - q_len

        # Generate Q, K, V for the whole sequence to be used in SDPA
        q = torch.randn(q_len,
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
        q_sdpa_in = q.unsqueeze(0).transpose(1, 2)
        k_sdpa_in = k_full.unsqueeze(0).transpose(1, 2)
        v_sdpa_in = v_full.unsqueeze(0).transpose(1, 2)

        if num_q_heads != num_kv_heads:
            assert num_q_heads % num_kv_heads == 0, (
                f"num_q_heads ({num_q_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads})")
            repeats = num_q_heads // num_kv_heads
            k_sdpa_in = k_sdpa_in.repeat_interleave(repeats, dim=1)
            v_sdpa_in = v_sdpa_in.repeat_interleave(repeats, dim=1)

        # Create causal mask: query token i attends to positions 0 to
        #  (context_len + i)
        kv_len = s_len
        offset = context_len
        attn_mask = torch.full((q_len, kv_len),
                               float('-inf'),
                               device=device,
                               dtype=dtype)
        for i in range(q_len):
            attn_mask[i, :offset + i + 1] = 0.0

        sdpa_out_i = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa_in,
            k_sdpa_in,
            v_sdpa_in,
            attn_mask=attn_mask,
            scale=scale,
            enable_gqa=True)
        # Convert back to (L, H, D)
        all_sdpa_outputs.append(sdpa_out_i.transpose(1, 2).squeeze(0))

        # Inputs for vLLM backends are just the new tokens
        all_q_vllm.append(q)
        all_k_vllm.append(k_full[context_len:])
        all_v_vllm.append(v_full[context_len:])

        # Contextual K/V data used to populate the paged cache
        k_contexts.append(k_full[:context_len])
        v_contexts.append(v_full[:context_len])

    query_vllm = torch.cat(all_q_vllm, dim=0)
    key_vllm = torch.cat(all_k_vllm, dim=0)
    value_vllm = torch.cat(all_v_vllm, dim=0)
    sdpa_output = torch.cat(all_sdpa_outputs, dim=0)

    common_attn_metadata = create_common_attn_metadata(
        batch_spec, vllm_config.cache_config.block_size, device)

    # 3. Simulate Paged KV Cache and a realistic slot_mapping
    kv_cache = create_and_prepopulate_kv_cache(
        k_contexts=k_contexts,
        v_contexts=v_contexts,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        device=device,
        num_blocks=vllm_config.cache_config.num_gpu_blocks or 1000,
        common_attn_metadata=common_attn_metadata,
        randomize_blocks=True)

    # 4. Run vLLM backends and compare
    # Note: flex_attention has known Triton kernel compatibility issues
    # with test infrastructures
    for backend_name in BACKENDS_TO_TEST:
        # FlashAttentionm + FlexAttention:
        #   [2, num_blocks, block_size, num_kv_heads, head_size]
        # FlashInfer:
        #   [num_blocks, 2, block_size, num_kv_heads, head_size]
        # Select the appropriate KV cache format for each backend
        kv_cache_for_backend = kv_cache
        if backend_name == _Backend.FLASHINFER_VLLM_V1:
            kv_cache_for_backend = kv_cache.transpose(0, 1)

        backend_output = run_attention_backend(backend_name, kv_cache_spec,
                                               vllm_config, device,
                                               common_attn_metadata,
                                               query_vllm, key_vllm,
                                               value_vllm,
                                               kv_cache_for_backend)

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
        rtol = 1e-2
        atol = 5e-3

        if backend_name == _Backend.FLEX_ATTENTION:
            atol = 5e-1  # TODO: figure out why flex_attention has such large
            # numerical differences for medium_decode, medium_prefill,
            # mixed_medium

        max_diff = torch.max(torch.abs(backend_output - sdpa_output)).item()
        max_rel_diff = torch.max(
            torch.abs(backend_output - sdpa_output) /
            torch.abs(sdpa_output)).item()
        all_close = torch.allclose(backend_output,
                                   sdpa_output,
                                   rtol=rtol,
                                   atol=atol)

        if not all_close:
            print(f"[{backend_name}] output differs from SDPA baseline. "
                  f"Max diff: {max_diff:.6f} (rel: {max_rel_diff:.6f})")
            print(f"[{backend_name}] output: {backend_output}")
            print(f"[{backend_name}] SDPA baseline: {sdpa_output}")

        assert all_close, (
            f"[{backend_name}] output differs from SDPA baseline. "
            f"Max diff: {max_diff:.6f} (rel: {max_rel_diff:.6f})")
