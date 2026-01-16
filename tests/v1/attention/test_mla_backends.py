# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for v1 MLA backends without GPUModelRunner dependency.

Known Issues:
- FLASH_ATTN_MLA backend occasionally produces NaN values in
  test_backend_correctness[mixed_small] when run after
  test_backend_correctness[small_prefill], but passes when run alone.
"""

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
    try_get_attention_backend,
)
from vllm import _custom_ops as ops
from vllm.config.vllm import set_current_vllm_config
from vllm.model_executor.layers.attention.mla_attention import QueryLenSupport
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.fa_utils import flash_attn_supports_mla
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.ops.flashmla import is_flashmla_dense_supported
from vllm.v1.kv_cache_interface import FullAttentionSpec

BACKENDS_TO_TEST = [
    AttentionBackendEnum.CUTLASS_MLA,
    AttentionBackendEnum.FLASHMLA,
    AttentionBackendEnum.FLASH_ATTN_MLA,
    AttentionBackendEnum.FLASHINFER_MLA,
    AttentionBackendEnum.TRITON_MLA,
]

# Remove sm100 backends from the list if not using sm100
if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).major < 10:
    BACKENDS_TO_TEST.remove(AttentionBackendEnum.CUTLASS_MLA)
    BACKENDS_TO_TEST.remove(AttentionBackendEnum.FLASHINFER_MLA)

# Remove FLASH_ATTN_MLA from the list if not supported
if not flash_attn_supports_mla():
    BACKENDS_TO_TEST.remove(AttentionBackendEnum.FLASH_ATTN_MLA)

# Remove FLASHMLA from the list if not supported
if not is_flashmla_dense_supported()[0]:
    BACKENDS_TO_TEST.remove(AttentionBackendEnum.FLASHMLA)

SPEC_DECODE_BACKENDS = []
for backend in BACKENDS_TO_TEST:
    builder_cls, _ = try_get_attention_backend(backend)
    query_len_support = getattr(
        builder_cls, "query_len_support", QueryLenSupport.SINGLE_ONLY
    )
    if query_len_support != QueryLenSupport.SINGLE_ONLY:
        SPEC_DECODE_BACKENDS.append(backend)

BACKEND_BLOCK_SIZES = {}
for backend in BACKENDS_TO_TEST:
    supported_sizes = backend.get_class().get_supported_kernel_block_sizes()
    if supported_sizes:
        default_size = supported_sizes[0]
        block_size = (
            default_size if isinstance(default_size, int) else default_size.base
        )
    else:
        block_size = 16
    BACKEND_BLOCK_SIZES[backend] = block_size

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
    "small_decode": BatchSpec(seq_lens=[32, 40], query_lens=[1, 1]),
    "small_prefill": BatchSpec(seq_lens=[32, 40], query_lens=[8, 8]),
    "mixed_small": BatchSpec(seq_lens=[32, 40, 48, 56], query_lens=[1, 1, 5, 5]),
    "medium_decode": BatchSpec(
        seq_lens=[128, 256, 512, 1024, 128, 256, 512, 1024],
        query_lens=[1, 1, 1, 1, 1, 1, 1, 1],
    ),
    "medium_prefill": BatchSpec(
        seq_lens=[256, 512, 1024, 2048], query_lens=[16, 16, 16, 16]
    ),
    "mixed_medium": BatchSpec(
        seq_lens=[512, 1024, 2048, 512, 1024, 2048], query_lens=[1, 1, 1, 7, 7, 7]
    ),
    "large_decode": BatchSpec(seq_lens=[2048] * 32, query_lens=[1] * 32),
    "large_prefill": BatchSpec(seq_lens=[4096] * 8, query_lens=[32] * 8),
    "single_decode": BatchSpec(seq_lens=[1024], query_lens=[1]),
    "single_prefill": BatchSpec(seq_lens=[1024], query_lens=[64]),
    "spec_decode_small": BatchSpec(
        seq_lens=[128, 256, 512, 1024], query_lens=[4, 4, 4, 4]
    ),
    "spec_decode_medium": BatchSpec(
        seq_lens=[512, 1024, 2048, 512, 1024, 2048], query_lens=[8, 8, 8, 8, 8, 8]
    ),
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
    randomize_blocks: bool = True,
    kv_cache_dtype: str | None = None,
    scale: float | torch.Tensor = 1.0,
) -> torch.Tensor:
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
        kv_cache_dtype: Optional kv cache dtype string. When set to
                        "fp8_ds_mla" the cache is populated using the
                        fp8 DeepSeek MLA layout via concat_and_cache_mla.
        scale: Scaling factor forwarded to concat_and_cache_mla when the
               fp8 cache layout is requested.

    Returns:
        MLA KV cache tensor
    """
    batch_size = len(kv_c_contexts)
    seq_lens = common_attn_metadata.seq_lens.cpu()
    query_lens = (
        common_attn_metadata.query_start_loc_cpu[1:]
        - common_attn_metadata.query_start_loc_cpu[:-1]
    )
    context_lens = seq_lens - query_lens
    block_table = common_attn_metadata.block_table_tensor
    slot_mapping = common_attn_metadata.slot_mapping

    use_fp8_ds_mla = kv_cache_dtype == "fp8_ds_mla"

    if use_fp8_ds_mla:
        if not kv_c_contexts:
            raise ValueError(
                "kv_c_contexts cannot be empty when using fp8_ds_mla cache dtype"
            )
        kv_lora_rank = kv_c_contexts[0].shape[-1]
        rope_dim = k_pe_contexts[0].shape[-1]
        entry_size = kv_lora_rank + 4 * 4 + 2 * rope_dim
        kv_cache = torch.zeros(
            num_blocks, block_size, entry_size, dtype=torch.uint8, device=device
        )
        scale_tensor = (
            scale
            if isinstance(scale, torch.Tensor)
            else torch.tensor(scale, dtype=torch.float32, device=device)
        )
        scale_tensor = scale_tensor.to(device=device, dtype=torch.float32)
    else:
        # Create MLA KV cache: (num_blocks, block_size, head_size)
        kv_cache = torch.zeros(
            num_blocks, block_size, head_size, dtype=dtype, device=device
        )
        kv_cache_flat = kv_cache.view(-1, head_size)

    # Populate the cache with the context tokens
    # Start from block_id=1 since block_id=0 is considered the null block
    start_block_idx = 1
    for i in range(batch_size):
        kv_c_context, k_pe_context = kv_c_contexts[i], k_pe_contexts[i]
        context_len = kv_c_context.shape[0]
        if context_len == 0:
            start_block_idx += cdiv(int(seq_lens[i]), block_size)
            continue

        start = start_block_idx * block_size

        if use_fp8_ds_mla:
            slots = torch.arange(context_len, device=device, dtype=torch.long) + start
            ops.concat_and_cache_mla(
                kv_c_context,
                k_pe_context.squeeze(1),
                kv_cache,
                slots,
                kv_cache_dtype="fp8_ds_mla",
                scale=scale_tensor,
            )
        else:
            kv_context = torch.cat([kv_c_context, k_pe_context.squeeze(1)], dim=-1)
            end = start + kv_context.shape[0]
            kv_cache_flat[start:end, ...] = kv_context

        # Stay block aligned and allocate enough blocks for the new tokens
        start_block_idx += cdiv(int(seq_lens[i]), block_size)

    blocks_end = start_block_idx

    # Permute the context blocks (excluding block 0 which is null)
    if randomize_blocks:
        perm = (
            torch.randperm(blocks_end - 1) + 1
        )  # Random permutation starting from block 1
    else:
        perm = torch.arange(1, blocks_end)  # Sequential order starting from block 1

    inv_perm = torch.zeros(blocks_end, dtype=torch.long, device=device)
    inv_perm[1:] = torch.argsort(perm) + 1  # Add 1 to account for starting from block 1
    kv_cache[1:blocks_end, ...] = kv_cache[perm, ...]

    # Construct the right block table
    # Start from block_id=1 since block_id=0 is considered the null block
    start_block_idx = 1
    for i in range(batch_size):
        num_blocks_for_seq = cdiv(int(seq_lens[i]), block_size)
        start = start_block_idx
        end = start + num_blocks_for_seq
        block_table[i, :num_blocks_for_seq] = inv_perm[start:end]
        block_table[i, num_blocks_for_seq:] = 0
        start_block_idx += num_blocks_for_seq

        # Create a realistic slot mapping that corresponds to the block table
    for i in range(batch_size):
        token_offsets = torch.arange(int(query_lens[i])) + int(context_lens[i])
        block_indices = token_offsets // block_size
        token_inter_block_offsets = token_offsets % block_size
        start = common_attn_metadata.query_start_loc_cpu[i]
        end = common_attn_metadata.query_start_loc_cpu[i + 1]
        slot_mapping[start:end] = block_table[
            i, block_indices
        ] * block_size + token_inter_block_offsets.to(device)

    return kv_cache


class MockAttentionLayer:
    """A mock attention layer for testing."""

    def __init__(self, device: torch.device):
        self._q_scale = torch.tensor(1.0, device=device)
        self._k_scale = torch.tensor(1.0, device=device)
        self._v_scale = torch.tensor(1.0, device=device)
        self._prob_scale = torch.tensor(1.0, device=device)
        self._q_scale_float = 1.0
        self._k_scale_float = 1.0
        self._v_scale_float = 1.0

    def forward(self, *_args, **_kwargs):
        raise NotImplementedError


class MockMLAAttentionLayer(AttentionLayerBase):
    """A mock MLA attention layer for populating static_forward_context."""

    def __init__(self, impl):
        self.impl = impl

    def get_attn_backend(self):
        raise NotImplementedError

    def get_kv_cache_spec(self, vllm_config):
        raise NotImplementedError


def run_attention_backend(
    backend: AttentionBackendEnum,
    kv_cache_spec: FullAttentionSpec,
    layer_names: list[str],
    vllm_config,
    device: torch.device,
    common_attn_metadata: CommonAttentionMetadata,
    query: torch.Tensor,
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    mock_kv_b_proj,
) -> torch.Tensor:
    """Run attention computation using the specified backend's AttentionImpl."""

    builder_cls, impl_cls = try_get_attention_backend(backend)

    # Set the current vllm config so that get_current_vllm_config() works
    # in the backend implementations
    with set_current_vllm_config(vllm_config):
        # Instantiate MLA implementation
        num_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        num_kv_heads = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        )
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

        # Populate static_forward_context with mock attention layers
        for layer_name in layer_names:
            vllm_config.compilation_config.static_forward_context[layer_name] = (
                MockMLAAttentionLayer(impl)
            )

        # Build metadata
        builder = builder_cls(kv_cache_spec, layer_names, vllm_config, device)
        attn_metadata = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
        )

        # Create mock layer and output buffer
        mock_layer = MockAttentionLayer(device)
        num_tokens = query.shape[0]
        output = torch.empty(
            num_tokens, num_heads * v_head_dim, dtype=query.dtype, device=query.device
        )

        # Run forward pass
        # NOTE: The query, key, and value are already shaped correctly
        # in the calling test function.
        output = impl.forward(
            mock_layer, query, kv_c, k_pe, kv_cache, attn_metadata, output=output
        )

        return output


@pytest.mark.parametrize(
    "batch_spec_name",
    [
        "small_decode",
        "small_prefill",
        "mixed_small",
        "medium_decode",
        "medium_prefill",
        "mixed_medium",
        "large_decode",
        "large_prefill",
        "single_decode",
        "single_prefill",
        "spec_decode_small",
        "spec_decode_medium",
    ],
)
@pytest.mark.parametrize("model", ["deepseek-ai/DeepSeek-R1"])
@pytest.mark.parametrize("tensor_parallel_size", [1, 4, 8, 16])
def test_backend_correctness(
    default_vllm_config,
    dist_init,
    batch_spec_name: str,
    model: str,
    tensor_parallel_size: int,
):
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

    Note: When tensor_parallel_size > 1, we simulate the head partitioning
    by overriding the model config to use fewer heads, without requiring
    multiple GPUs. This tests that backends work correctly with different
    head counts.
    """

    batch_spec = BATCH_SPECS[batch_spec_name]
    is_spec_decode_test = batch_spec_name.startswith("spec_decode")
    unique_block_sizes = sorted(set(BACKEND_BLOCK_SIZES.values()))
    default_block_size = unique_block_sizes[0]
    required_blocks = sum(
        (seq_len + default_block_size - 1) // default_block_size
        for seq_len in batch_spec.seq_lens
    )
    # Add 1 for null block at index 0, and some buffer
    num_gpu_blocks = required_blocks + 1 + 100

    hf_config_override = None
    if tensor_parallel_size > 1:
        from vllm.config import ModelConfig

        temp_config = ModelConfig(model=model, max_model_len=1)
        original_num_heads = temp_config.hf_text_config.num_attention_heads
        original_num_kv_heads = getattr(
            temp_config.hf_text_config, "num_key_value_heads", None
        )
        hf_config_override = {
            "num_attention_heads": original_num_heads // tensor_parallel_size,
        }
        if original_num_kv_heads is not None:
            hf_config_override["num_key_value_heads"] = max(
                1, original_num_kv_heads // tensor_parallel_size
            )

    vllm_config = create_vllm_config(
        model_name=model,
        tensor_parallel_size=1,  # Always use TP=1 to avoid multi-GPU requirements
        max_model_len=max(batch_spec.seq_lens),
        num_gpu_blocks=num_gpu_blocks,
        block_size=default_block_size,
        hf_config_override=hf_config_override,
    )

    # For spec decode tests, add a speculative_config to set the reorder_batch_threshold
    if is_spec_decode_test:
        from vllm.config import SpeculativeConfig

        # Get the query length from the batch spec (they should all be uniform)
        query_len = batch_spec.query_lens[0]
        # Set num_speculative_tokens to query_len - 1
        # (since threshold is 1 + num_spec_tokens)
        # Use ngram method which doesn't require a draft model
        vllm_config.speculative_config = SpeculativeConfig(
            method="ngram", num_speculative_tokens=query_len - 1
        )

    device = torch.device("cuda:0")

    # 1. Setup
    batch_size = batch_spec.batch_size
    seq_lens = batch_spec.seq_lens
    query_lens = batch_spec.query_lens
    num_q_heads = vllm_config.model_config.get_num_attention_heads(
        vllm_config.parallel_config
    )
    head_size = vllm_config.model_config.get_head_size()
    dtype = _convert_dtype_to_torch(vllm_config.model_config.dtype)
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    qk_nope_head_dim = 128
    v_head_dim = 128
    total_head_size = kv_lora_rank + qk_rope_head_dim
    assert kv_lora_rank + qk_rope_head_dim == head_size, (
        f"MLA dimensions don't match: {total_head_size} != {head_size}"
    )
    scale = 1.0 / (total_head_size**0.5)

    # 2. Generate data and compute SDPA reference output for MLA
    all_q_vllm, all_kv_c_vllm, all_k_pe_vllm = [], [], []
    all_sdpa_outputs: list[list[torch.Tensor]] = []
    kv_c_contexts, k_pe_contexts = [], []

    # Create shared MLA weight matrices for consistency across all sequences
    W_UK = torch.randn(
        kv_lora_rank, num_q_heads, qk_nope_head_dim, dtype=dtype, device=device
    )
    W_UV = torch.randn(
        kv_lora_rank, num_q_heads, v_head_dim, dtype=dtype, device=device
    )
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
        q_c = torch.randn(
            q_len,
            num_q_heads,
            qk_nope_head_dim + qk_rope_head_dim,
            dtype=dtype,
            device=device,
        )

        # KV_C (latent K/V): [s_len, kv_lora_rank]
        kv_c_full = torch.randn(s_len, kv_lora_rank, dtype=dtype, device=device)

        # K_PE (rope component): [s_len, 1, qk_rope_head_dim]
        k_pe_full = torch.randn(s_len, 1, qk_rope_head_dim, dtype=dtype, device=device)

        # Determine if this sequence uses the decode pipeline or prefill
        # pipeline for each backend
        # NOTE: For spec decode tests with uniform query_len > 1, backends that
        # support spec decode (FLASH_ATTN_MLA with varlen support, FLASHMLA with
        # uniform support) will use the decode pipeline (MQA-style), while
        # backends that only support single-token queries will use the prefill
        # pipeline (MHA-style). This ensures the reference implementation
        # matches each backend's actual decode/prefill pipeline path.
        is_decode = []
        for backend_idx, backend in enumerate(BACKENDS_TO_TEST):
            builder_cls, _ = try_get_attention_backend(backend)
            if is_spec_decode_test:
                query_len_support = getattr(
                    builder_cls, "query_len_support", QueryLenSupport.SINGLE_ONLY
                )
                supports_spec = query_len_support != QueryLenSupport.SINGLE_ONLY
                is_decode.append(supports_spec)
            else:
                threshold = getattr(builder_cls, "reorder_batch_threshold", None)
                query_len_support = getattr(
                    builder_cls, "query_len_support", QueryLenSupport.SINGLE_ONLY
                )
                within_threshold = q_len <= threshold if threshold else False
                if (
                    within_threshold
                    and query_len_support == QueryLenSupport.UNIFORM
                    and i > 0
                ):
                    first_q_len = query_lens[0]
                    within_threshold = q_len == first_q_len
                is_decode.append(within_threshold)

        # Split q into nope and rope components
        q_nope, q_pe = q_c.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)

        #######################################################
        # Decode path: MQA-style attention in latent space
        # Transform q_nope to latent space: q_nope @ W_UK
        # q_nope: [1, num_heads, qk_nope_head_dim]
        # W_UK: [kv_lora_rank, num_heads, qk_nope_head_dim]
        ql_nope = torch.einsum(
            "qnh,lnh->qnl", q_nope, W_UK
        )  # [1, num_heads, kv_lora_rank]

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
            q_sdpa_in, k_sdpa_in, v_sdpa_in, attn_mask=attn_mask, scale=scale
        )
        sdpa_out_i_decode = sdpa_out_i_decode.transpose(1, 2).squeeze(
            0
        )  # [1, num_heads, kv_lora_rank]

        # Project back to output space: sdpa_out @ W_UV
        sdpa_out_i_decode = torch.einsum("qnl,lnv->qnv", sdpa_out_i_decode, W_UV)
        sdpa_out_i_decode = sdpa_out_i_decode.flatten(start_dim=-2)

        #######################################################
        # Prefill path: MHA-style attention with full sequence
        # Apply kv_b_proj to the full kv_c tensor
        kv_nope_full = torch.einsum("sl,lnh->snh", kv_c_full, kv_b_proj_weight)
        k_nope_full, v_full = kv_nope_full.split([qk_nope_head_dim, v_head_dim], dim=-1)

        # Build attention inputs for full sequence
        q_mha = torch.cat([q_nope, q_pe], dim=-1)  # [q_len, num_heads, total_dim]
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
            q_sdpa_in, k_sdpa_in, v_sdpa_in, attn_mask=attn_mask, scale=scale
        )
        sdpa_out_i_prefill = sdpa_out_i_prefill.transpose(1, 2).squeeze(0)
        sdpa_out_i_prefill = sdpa_out_i_prefill.flatten(start_dim=-2)

        for backend_idx, backend in enumerate(BACKENDS_TO_TEST):
            if is_decode[backend_idx]:
                all_sdpa_outputs[backend_idx].append(sdpa_out_i_decode)
            else:
                all_sdpa_outputs[backend_idx].append(sdpa_out_i_prefill)

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
    sdpa_outputs = {}
    for backend_idx, backend in enumerate(BACKENDS_TO_TEST):
        sdpa_outputs[backend] = torch.cat(all_sdpa_outputs[backend_idx], dim=0)

    # Create mock kv_b_proj using the same weights as reference implementation
    from vllm.model_executor.layers.linear import ColumnParallelLinear

    mock_kv_b_proj = ColumnParallelLinear(
        input_size=kv_lora_rank,
        output_size=num_q_heads * (qk_nope_head_dim + v_head_dim),
        bias=False,
    ).to(device=device, dtype=dtype)

    # Set the mock weights to match our reference implementation
    # Reshape W_UK and W_UV to match the expected kv_b_proj format
    # [kv_lora_rank, num_heads, qk_nope_head_dim + v_head_dim]
    kv_b_proj_weight = kv_b_proj_weight.view(
        kv_lora_rank, num_q_heads * (qk_nope_head_dim + v_head_dim)
    )
    mock_kv_b_proj.weight = torch.nn.Parameter(kv_b_proj_weight.T, requires_grad=False)

    # 3. Create metadata and KV caches for each block size
    # Group backends by block size and test each group
    metadata_per_block_size = {}
    kv_cache_per_block_size = {}

    for block_size in unique_block_sizes:
        # Create metadata for this block size
        common_attn_metadata = create_common_attn_metadata(
            batch_spec, block_size, device
        )

        # Pad block table to meet requirement:
        # block_num % (128 / block_size) == 0
        required_divisor = int(128 / block_size)
        current_block_num = common_attn_metadata.block_table_tensor.shape[1]
        if current_block_num % required_divisor != 0:
            # Pad to next multiple of required_divisor
            padded_block_num = (
                (current_block_num + required_divisor - 1) // required_divisor
            ) * required_divisor
            padding_cols = padded_block_num - current_block_num
            padding = torch.zeros(
                (common_attn_metadata.block_table_tensor.shape[0], padding_cols),
                dtype=torch.int32,
                device=device,
            )
            common_attn_metadata.block_table_tensor = torch.cat(
                [common_attn_metadata.block_table_tensor, padding], dim=1
            )

        metadata_per_block_size[block_size] = common_attn_metadata

        # Create KV cache for this block size
        required_blocks_for_size = sum(
            (seq_len + block_size - 1) // block_size for seq_len in batch_spec.seq_lens
        )
        num_blocks_for_size = required_blocks_for_size + 1 + 100

        kv_cache = create_and_prepopulate_kv_cache(
            kv_c_contexts=kv_c_contexts,
            k_pe_contexts=k_pe_contexts,
            block_size=block_size,
            head_size=head_size,
            dtype=dtype,
            device=device,
            num_blocks=num_blocks_for_size,
            common_attn_metadata=common_attn_metadata,
            randomize_blocks=True,
        )
        kv_cache_per_block_size[block_size] = kv_cache

    # 4. Run vLLM backends and compare
    failures = []
    for backend_idx, backend_name in enumerate(BACKENDS_TO_TEST):
        # Skip backends that don't support spec decode for spec decode tests
        if is_spec_decode_test and backend_name not in SPEC_DECODE_BACKENDS:
            continue

        # Get the appropriate block_size, metadata, and cache for this backend
        block_size = BACKEND_BLOCK_SIZES[backend_name]
        common_attn_metadata = metadata_per_block_size[block_size]
        kv_cache = kv_cache_per_block_size[block_size]

        # Create kv_cache_spec with the correct block_size for this backend
        backend_kv_cache_spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=vllm_config.model_config.get_num_kv_heads(
                vllm_config.parallel_config
            ),
            head_size=vllm_config.model_config.get_head_size(),
            dtype=vllm_config.model_config.dtype,
            sliding_window=vllm_config.model_config.get_sliding_window(),
        )

        backend_output = run_attention_backend(
            backend_name,
            backend_kv_cache_spec,
            ["placeholder"],
            vllm_config,
            device,
            common_attn_metadata,
            query_vllm,
            kv_c_vllm,
            k_pe_vllm,
            kv_cache,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            mock_kv_b_proj,
        )

        # Use backend_idx to get the correct SDPA output for this backend
        expected_output = sdpa_outputs[backend_name]

        # Check shape and dtype consistency
        try:
            assert backend_output.shape == expected_output.shape, (
                f"[{backend_name}] shape {backend_output.shape} != "
                f"SDPA shape {expected_output.shape}"
            )
            assert backend_output.dtype == expected_output.dtype, (
                f"[{backend_name}] dtype {backend_output.dtype} != "
                f"SDPA dtype {expected_output.dtype}"
            )

            assert torch.isfinite(backend_output).all(), (
                f"[{backend_name}] produced non-finite values"
            )

            # Check numerical similarity
            rtol = 1e-2
            atol = 5e-1

            max_diff = torch.max(torch.abs(backend_output - expected_output)).item()
            max_rel_diff = torch.max(
                torch.abs(backend_output - expected_output) / torch.abs(expected_output)
            ).item()
            all_close = torch.allclose(
                backend_output, expected_output, rtol=rtol, atol=atol
            )

            assert all_close, (
                f"[{backend_name}] output differs from SDPA baseline. "
                f"Max diff: {max_diff:.6f}, max rel diff: {max_rel_diff:.6f})"
            )
        except AssertionError as e:
            failures.append(str(e))

    # Report all failures at once
    if failures:
        # Create a summary for the single-line failure message
        backend_names = []
        for f in failures:
            if "[AttentionBackendEnum." in f:
                backend_name = f.split("[")[1].split("]")[0]
                backend_names.append(backend_name)

        summary = f"{len(failures)} backend(s) failed: {', '.join(backend_names)}"
        detailed_msg = "\n".join(failures)
        pytest.fail(f"{summary}\n{detailed_msg}")
