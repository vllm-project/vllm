# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for v1 MLA backends without GPUModelRunner dependency."""

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
    _Backend.CUTLASS_MLA, _Backend.FLASHMLA_VLLM_V1, _Backend.FLASH_ATTN_MLA,
    _Backend.TRITON_MLA_VLLM_V1
]

# Remove CUTLASS_MLA from the list if not using sm100
if not torch.cuda.is_available() or torch.cuda.get_device_properties(
        0).major < 10:
    BACKENDS_TO_TEST.remove(_Backend.CUTLASS_MLA)

torch.manual_seed(42)


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


def create_and_prepopulate_kv_cache(
        kv_c_contexts: list[torch.Tensor],
        k_pe_contexts: list[torch.Tensor],
        block_size: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
        num_blocks: int,
        common_attn_metadata: CommonAttentionMetadata,
        randomize_blocks: bool = True) -> torch.Tensor:
    """Create and prepopulate an MLA KV cache with context data.
    
    Args:
        kv_c_contexts: List of latent KV context tensors for each sequence
        k_pe_contexts: List of key positional embedding context tensors
                       for each sequence
        block_size: Size of each block
        head_size: Size of each head (latent dimension)
        dtype: Data type for the cache
        device: Device to create the cache on
        num_blocks: Total number of blocks in the cache
        common_attn_metadata: Common attention metadata
        randomize_blocks: Whether to randomly permute blocks 
                          or use sequential order
        
    Returns:
        MLA KV cache tensor
    """
    batch_size = len(kv_c_contexts)
    seq_lens = common_attn_metadata.seq_lens_cpu
    query_lens = common_attn_metadata.query_start_loc_cpu[
        1:] - common_attn_metadata.query_start_loc_cpu[:-1]
    context_lens = common_attn_metadata.num_computed_tokens_cpu
    block_table = common_attn_metadata.block_table_tensor
    slot_mapping = common_attn_metadata.slot_mapping

    # Create MLA KV cache: (num_blocks, block_size, head_size)
    kv_cache = torch.empty(num_blocks,
                           block_size,
                           head_size,
                           dtype=dtype,
                           device=device)
    kv_cache_flat = kv_cache.view(-1, head_size)

    # Populate the cache with the context tokens
    # Start from block_id=1 since block_id=0 is considered the null block
    start_block_idx = 1
    for i in range(batch_size):
        kv_c_context, k_pe_context = kv_c_contexts[i], k_pe_contexts[i]
        kv_context = torch.cat([kv_c_context, k_pe_context.squeeze(1)], dim=-1)
        start = start_block_idx * block_size
        end = start + kv_context.shape[0]
        kv_cache_flat[start:end, ...] = kv_context

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
    kv_cache[1:blocks_end, ...] = kv_cache[perm, ...]

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


def run_attention_backend(backend: _Backend, kv_cache_spec: FullAttentionSpec,
                          layer_names: list[str], vllm_config,
                          device: torch.device,
                          common_attn_metadata: CommonAttentionMetadata,
                          query: torch.Tensor, kv_c: torch.Tensor,
                          k_pe: torch.Tensor, kv_cache: torch.Tensor,
                          kv_lora_rank: int, qk_nope_head_dim: int,
                          qk_rope_head_dim: int, v_head_dim: int,
                          mock_kv_b_proj) -> torch.Tensor:
    """Run attention computation using the specified backend's AttentionImpl."""

    builder_cls, impl_cls = get_attention_backend(backend)

    # Build metadata
    builder = builder_cls(kv_cache_spec, layer_names, vllm_config, device)
    attn_metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_attn_metadata,
    )

    # Instantiate MLA implementation
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
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        q_lora_rank=None,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        qk_head_dim=qk_nope_head_dim + qk_rope_head_dim,
        v_head_dim=v_head_dim,
        kv_b_proj=mock_kv_b_proj,
    )

    # Process weights to create W_UK_T and W_UV attributes needed by MLA
    act_dtype = _convert_dtype_to_torch(vllm_config.model_config.dtype)
    impl.process_weights_after_loading(act_dtype)

    # Create mock layer and output buffer
    mock_layer = MockAttentionLayer(device)
    num_tokens = query.shape[0]
    output = torch.empty(num_tokens,
                         num_heads * v_head_dim,
                         dtype=query.dtype,
                         device=query.device)

    # Run forward pass
    # NOTE: The query, key, and value are already shaped correctly
    # in the calling test function.
    output = impl.forward(mock_layer,
                          query,
                          kv_c,
                          k_pe,
                          kv_cache,
                          attn_metadata,
                          output=output)

    return output


@pytest.mark.parametrize("batch_spec_name", [
    "small_decode", "small_prefill", "mixed_small", "medium_decode",
    "medium_prefill", "mixed_medium", "large_decode", "large_prefill",
    "single_decode", "single_prefill"
])
@pytest.mark.parametrize("model", ["deepseek-ai/DeepSeek-V2-Lite-Chat"])
def test_backend_correctness(dist_init, batch_spec_name: str, model: str):
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
    vllm_config = create_vllm_config(model_name=model,
                                     max_model_len=max(batch_spec.seq_lens),
                                     num_gpu_blocks=2048)
    device = torch.device("cuda:0")

    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)

    # 1. Setup
    batch_size = batch_spec.batch_size
    seq_lens = batch_spec.seq_lens
    query_lens = batch_spec.query_lens
    num_q_heads = vllm_config.model_config.get_num_attention_heads(
        vllm_config.parallel_config)
    head_size = vllm_config.model_config.get_head_size()
    dtype = _convert_dtype_to_torch(vllm_config.model_config.dtype)
    block_size = vllm_config.cache_config.block_size
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    qk_nope_head_dim = 128
    v_head_dim = 128
    total_head_size = kv_lora_rank + qk_rope_head_dim
    assert kv_lora_rank + qk_rope_head_dim == head_size, \
        f"MLA dimensions don't match: {total_head_size} != {head_size}"
    scale = 1.0 / (total_head_size**0.5)

    # 2. Generate data and compute SDPA reference output for MLA
    all_q_vllm, all_kv_c_vllm, all_k_pe_vllm = [], [], []
    all_sdpa_outputs: list[list[torch.Tensor]] = []
    kv_c_contexts, k_pe_contexts = [], []

    # Create shared MLA weight matrices for consistency across all sequences
    W_UK = torch.randn(kv_lora_rank,
                       num_q_heads,
                       qk_nope_head_dim,
                       dtype=dtype,
                       device=device)
    W_UV = torch.randn(kv_lora_rank,
                       num_q_heads,
                       v_head_dim,
                       dtype=dtype,
                       device=device)
    kv_b_proj_weight = torch.cat([W_UK, W_UV], dim=-1)

    for i, backend in enumerate(BACKENDS_TO_TEST):
        all_sdpa_outputs.append([])

    for i in range(batch_size):
        s_len = seq_lens[i]
        q_len = query_lens[i]
        context_len = s_len - q_len

        # Generate MLA tensors
        # Q has both nope and rope components:
        # [q_len, num_heads, qk_nope_head_dim + qk_rope_head_dim]
        q_c = torch.randn(q_len,
                          num_q_heads,
                          qk_nope_head_dim + qk_rope_head_dim,
                          dtype=dtype,
                          device=device)

        # KV_C (latent K/V): [s_len, kv_lora_rank]
        kv_c_full = torch.randn(s_len,
                                kv_lora_rank,
                                dtype=dtype,
                                device=device)

        # K_PE (rope component): [s_len, 1, qk_rope_head_dim]
        k_pe_full = torch.randn(s_len,
                                1,
                                qk_rope_head_dim,
                                dtype=dtype,
                                device=device)

        # Determine if this is decode or prefill
        is_decode = []
        for i, backend in enumerate(BACKENDS_TO_TEST):
            builder_cls, _ = get_attention_backend(backend)
            is_decode.append(q_len <= builder_cls.reorder_batch_threshold)

        # Split q into nope and rope components
        q_nope, q_pe = q_c.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)

        #######################################################
        # Decode path: MQA-style attention in latent space
        # Transform q_nope to latent space: q_nope @ W_UK
        # q_nope: [1, num_heads, qk_nope_head_dim]
        # W_UK: [kv_lora_rank, num_heads, qk_nope_head_dim]
        ql_nope = torch.einsum("qnh,lnh->qnl", q_nope,
                               W_UK)  # [1, num_heads, kv_lora_rank]

        # Build MQA attention inputs
        # Q: [1, num_heads, kv_lora_rank + qk_rope_head_dim]
        q_mqa = torch.cat([ql_nope, q_pe], dim=-1)
        # K: [s_len, kv_lora_rank + qk_rope_head_dim]
        # (broadcasted to all heads)
        k_mqa = torch.cat([kv_c_full, k_pe_full.squeeze(1)], dim=-1)
        k_mqa = k_mqa.unsqueeze(1).expand(-1, num_q_heads, -1)
        # V: [s_len, kv_lora_rank] (broadcasted to all heads)
        v_mqa = kv_c_full.unsqueeze(1).expand(-1, num_q_heads, -1)

        # Create custom attention mask for decode path:
        # - Query tokens can attend to all context tokens
        # - Query tokens can only attend to query tokens up to their position
        attn_mask = torch.ones(q_len, s_len, dtype=torch.bool, device=device)
        # Apply causal mask only to the query portion (context_len onwards)
        causal_mask = torch.tril(torch.ones(q_len, q_len, device=device))
        attn_mask[:, context_len:] = causal_mask

        # SDPA expects (N, H, L, D)
        q_sdpa_in = q_mqa.unsqueeze(0).transpose(1, 2)
        k_sdpa_in = k_mqa.unsqueeze(0).transpose(1, 2)
        v_sdpa_in = v_mqa.unsqueeze(0).transpose(1, 2)

        sdpa_out_i_decode = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa_in, k_sdpa_in, v_sdpa_in, attn_mask=attn_mask, scale=scale)
        sdpa_out_i_decode = sdpa_out_i_decode.transpose(1, 2).squeeze(
            0)  # [1, num_heads, kv_lora_rank]

        # Project back to output space: sdpa_out @ W_UV
        sdpa_out_i_decode = torch.einsum("qnl,lnv->qnv", sdpa_out_i_decode,
                                         W_UV)
        sdpa_out_i_decode = sdpa_out_i_decode.flatten(start_dim=-2)

        #######################################################
        # Prefill path: MHA-style attention with full sequence
        # Apply kv_b_proj to the full kv_c tensor
        kv_nope_full = torch.einsum("sl,lnh->snh", kv_c_full, kv_b_proj_weight)
        k_nope_full, v_full = kv_nope_full.split(
            [qk_nope_head_dim, v_head_dim], dim=-1)

        # Build attention inputs for full sequence
        q_mha = torch.cat([q_nope, q_pe],
                          dim=-1)  # [q_len, num_heads, total_dim]
        k_pe_full_expanded = k_pe_full.expand(-1, num_q_heads, -1)
        k_full = torch.cat([k_nope_full, k_pe_full_expanded], dim=-1)

        # Create custom attention mask:
        # - Query tokens can attend to all context tokens
        # - Query tokens can only attend to query tokens up to their pos
        attn_mask = torch.ones(q_len, s_len, dtype=torch.bool, device=device)
        # Apply causal mask only to the query portion (context_len onwards)
        causal_mask = torch.tril(torch.ones(q_len, q_len, device=device))
        attn_mask[:, context_len:] = causal_mask

        # SDPA expects (N, H, L, D)
        q_sdpa_in = q_mha.unsqueeze(0).transpose(1, 2)
        k_sdpa_in = k_full.unsqueeze(0).transpose(1, 2)
        v_sdpa_in = v_full.unsqueeze(0).transpose(1, 2)

        # Single attention call with custom mask
        sdpa_out_i_prefill = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa_in, k_sdpa_in, v_sdpa_in, attn_mask=attn_mask, scale=scale)
        sdpa_out_i_prefill = sdpa_out_i_prefill.transpose(1, 2).squeeze(0)
        sdpa_out_i_prefill = sdpa_out_i_prefill.flatten(start_dim=-2)

        for i, backend in enumerate(BACKENDS_TO_TEST):
            if is_decode[i]:
                all_sdpa_outputs[i].append(sdpa_out_i_decode)
            else:
                all_sdpa_outputs[i].append(sdpa_out_i_prefill)

        # Inputs for vLLM MLA backends are just the new tokens
        all_q_vllm.append(q_c)
        all_kv_c_vllm.append(kv_c_full[context_len:])  # New kv_c tokens
        all_k_pe_vllm.append(k_pe_full[context_len:])  # New k_pe tokens

        # Contextual K/V data used to populate the paged cache (MLA format)
        kv_c_contexts.append(kv_c_full[:context_len])
        k_pe_contexts.append(k_pe_full[:context_len])

    # Concatenate all sequences (no reordering needed)
    query_vllm = torch.cat(all_q_vllm, dim=0)
    kv_c_vllm = torch.cat(all_kv_c_vllm, dim=0)
    k_pe_vllm = torch.cat(all_k_pe_vllm, dim=0)
    sdpa_outputs = []
    for i, backend in enumerate(BACKENDS_TO_TEST):
        sdpa_outputs.append(torch.cat(all_sdpa_outputs[i], dim=0))

    # Create mock kv_b_proj using the same weights as reference implementation
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    mock_kv_b_proj = ColumnParallelLinear(input_size=kv_lora_rank,
                                          output_size=num_q_heads *
                                          (qk_nope_head_dim + v_head_dim),
                                          bias=False).to(device=device,
                                                         dtype=dtype)

    # Set the mock weights to match our reference implementation
    # Reshape W_UK and W_UV to match the expected kv_b_proj format
    # [kv_lora_rank, num_heads, qk_nope_head_dim + v_head_dim]
    kv_b_proj_weight = kv_b_proj_weight.view(
        kv_lora_rank, num_q_heads * (qk_nope_head_dim + v_head_dim))
    mock_kv_b_proj.weight = torch.nn.Parameter(kv_b_proj_weight.T)

    # Create metadata using original batch spec
    common_attn_metadata = create_common_attn_metadata(
        batch_spec, vllm_config.cache_config.block_size, device)

    # 3. Simulate Paged KV Cache and a realistic slot_mapping
    kv_cache = create_and_prepopulate_kv_cache(
        kv_c_contexts=kv_c_contexts,
        k_pe_contexts=k_pe_contexts,
        block_size=block_size,
        head_size=head_size,
        dtype=dtype,
        device=device,
        num_blocks=vllm_config.cache_config.num_gpu_blocks,
        common_attn_metadata=common_attn_metadata,
        randomize_blocks=True)

    # 4. Run vLLM backends and compare
    for i, backend_name in enumerate(BACKENDS_TO_TEST):
        backend_output = run_attention_backend(
            backend_name, kv_cache_spec, ["placeholder"], vllm_config, device,
            common_attn_metadata, query_vllm, kv_c_vllm, k_pe_vllm, kv_cache,
            kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim,
            mock_kv_b_proj)

        # Check shape and dtype consistency
        assert backend_output.shape == sdpa_outputs[i].shape, (
            f"[{backend_name}] shape {backend_output.shape} != "
            f"SDPA shape {sdpa_outputs[i].shape}")
        assert backend_output.dtype == sdpa_outputs[i].dtype, (
            f"[{backend_name}] dtype {backend_output.dtype} != "
            f"SDPA dtype {sdpa_outputs[i].dtype}")

        assert torch.isfinite(backend_output).all(), (
            f"[{backend_name}] produced non-finite values")

        # Check numerical similarity
        rtol = 1e-2
        atol = 5e-1

        max_diff = torch.max(torch.abs(backend_output -
                                       sdpa_outputs[i])).item()
        max_rel_diff = torch.max(
            torch.abs(backend_output - sdpa_outputs[i]) /
            torch.abs(sdpa_outputs[i])).item()
        all_close = torch.allclose(backend_output,
                                   sdpa_outputs[i],
                                   rtol=rtol,
                                   atol=atol)

        assert all_close, (
            f"[{backend_name}] output differs from SDPA baseline. "
            f"Max diff: {max_diff:.6f}, max rel diff: {max_rel_diff:.6f})")
