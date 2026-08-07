# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for v1 attention backends without GPUModelRunner dependency."""

from functools import partial
from types import SimpleNamespace

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    create_vllm_config,
    try_backend_includes_kv_cache_update,
    try_get_attention_backend,
)
from vllm.config import ModelConfig, set_current_vllm_config
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    is_quantized_kv_cache,
    is_torch_equal_or_newer,
    set_random_seed,
)
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.b12x_attn import B12XPagedAttentionBackend
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.utils import (
    set_kv_cache_layout,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec

BACKENDS_TO_TEST = [
    AttentionBackendEnum.FLASH_ATTN,
    AttentionBackendEnum.FLASHINFER,
    AttentionBackendEnum.FLEX_ATTENTION,
    AttentionBackendEnum.TRITON_ATTN,
    "FLEX_ATTENTION_SLOW",
]

DEVICE_TYPE = current_platform.device_type

# Use the platform's preferred FP8 type so the stored cache matches what the
# backends reinterpret at runtime. On ROCm gfx94x this is e4m3fnuz, not e4m3fn;
# storing e4m3fn bytes there would be re-read as fnuz and produce NaNs.
FP8_KV_CACHE_DTYPES = {
    "fp8": current_platform.fp8_dtype(),
    "fp8_e4m3": current_platform.fp8_dtype(),
}

# Remove flashinfer from the list if it's not available
try:
    import flashinfer  # noqa: F401
except ImportError:
    BACKENDS_TO_TEST.remove(AttentionBackendEnum.FLASHINFER)


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
    "mixed_large": BatchSpec(
        seq_lens=[1024, 2048, 4096, 1024, 2048, 4096], query_lens=[1, 1, 1, 32, 32, 32]
    ),
    "single_decode": BatchSpec(seq_lens=[1024], query_lens=[1]),
    "single_prefill": BatchSpec(seq_lens=[1024], query_lens=[64]),
    # encoder-only
    "small_encoder_prefill": BatchSpec(
        seq_lens=[32, 64, 128, 256], query_lens=[32, 64, 128, 256]
    ),
    "medium_encoder_prefill": BatchSpec(
        seq_lens=[256, 512, 1024, 2048], query_lens=[256, 512, 1024, 2048]
    ),
}


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
    randomize_blocks: bool = True,
    kv_cache_dtype: str = "auto",
) -> torch.Tensor:
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
    seq_lens = common_attn_metadata.seq_lens.cpu()
    query_lens = (
        common_attn_metadata.query_start_loc_cpu[1:]
        - common_attn_metadata.query_start_loc_cpu[:-1]
    )
    context_lens = seq_lens - query_lens
    block_table = common_attn_metadata.block_table_tensor
    slot_mapping = common_attn_metadata.slot_mapping

    # For an fp8 kv cache, store the cache in the fp8 dtype so that assigning
    # the higher-precision context tensors quantizes them, mirroring runtime.
    fp8_kv_cache = is_quantized_kv_cache(kv_cache_dtype)
    if fp8_kv_cache:
        storage_dtype = FP8_KV_CACHE_DTYPES[kv_cache_dtype]
    elif kv_cache_dtype == "auto":
        storage_dtype = dtype
    else:
        storage_dtype = STR_DTYPE_TO_TORCH_DTYPE[kv_cache_dtype]

    kv_cache = torch.zeros(
        num_blocks,
        block_size,
        num_kv_heads,
        2 * head_size,
        dtype=storage_dtype,
        device=device,
    )
    kv_cache_flat = kv_cache.view(-1, num_kv_heads, 2 * head_size)

    # Populate the cache with the context tokens
    # Start from block_id=1 since block_id=0 is considered the null block
    start_block_idx = 1
    for i in range(batch_size):
        k_context, v_context = k_contexts[i], v_contexts[i]
        start = start_block_idx * block_size
        end = start + k_context.shape[0]
        kv_cache_flat[start:end, :, :head_size] = k_context
        kv_cache_flat[start:end, :, head_size:] = v_context

        # Stay block aligned and allocate enough blocks for the new tokens
        start_block_idx += cdiv(int(seq_lens[i]), block_size)

    blocks_end = start_block_idx

    # Permute the context blocks (excluding block 0 which is null)
    if randomize_blocks:
        # Random permutation starting from block 1
        perm = torch.randperm(blocks_end - 1) + 1
    else:
        # Sequential order starting from block 1
        perm = torch.arange(1, blocks_end)

    inv_perm = torch.zeros(blocks_end, dtype=torch.long, device=device)
    # Add 1 to account for starting from block 1
    inv_perm[1:] = torch.argsort(perm) + 1
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
            i, block_indices
        ] * block_size + token_inter_block_offsets.to(device)

    # Transpose to logical (num_blocks, num_kv_heads, block_size, 2*hs)
    kv_cache = kv_cache.transpose(1, 2).contiguous()

    if fp8_kv_cache:
        kv_cache = kv_cache.view(torch.uint8)

    return kv_cache


class MockAttentionLayer:
    """A mock attention layer for testing."""

    def __init__(self, device: torch.device):
        self._q_scale = torch.tensor(1.0, device=device)
        self._k_scale = torch.tensor(1.0, device=device)
        self._v_scale = torch.tensor(1.0, device=device)
        # Add float versions for flashinfer
        self._q_scale_float = 1.0
        self._k_scale_float = 1.0
        self._v_scale_float = 1.0


def run_attention_backend(
    backend: AttentionBackendEnum,
    kv_cache_spec: FullAttentionSpec,
    layer_names: list[str],
    vllm_config,
    device: torch.device,
    common_attn_metadata: CommonAttentionMetadata,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_type: AttentionType = AttentionType.DECODER,
    sliding_window: int | None = None,
    kv_cache_dtype: str = "auto",
    sinks: torch.Tensor | None = None,
    use_cuda_graph: bool = False,
) -> torch.Tensor:
    """Run attention computation using the specified backend's AttentionImpl."""

    # Handle special case for FLEX_ATTENTION_SLOW
    actual_backend = backend

    use_direct_block_mask = is_torch_equal_or_newer("2.9.0.dev0")
    if backend == "FLEX_ATTENTION_SLOW":
        actual_backend = AttentionBackendEnum.FLEX_ATTENTION
        use_direct_block_mask = False

    builder_cls, impl_cls = try_get_attention_backend(actual_backend)

    # Mock flashinfer's get_per_layer_parameters if needed
    if actual_backend == AttentionBackendEnum.FLASHINFER:
        import unittest.mock

        from vllm.v1.attention.backends.utils import PerLayerParameters

        def mock_get_per_layer_parameters(vllm_config, layer_names, impl_cls):
            # Return mock parameters for a single layer
            head_size = vllm_config.model_config.get_head_size()
            return {
                layer_name: PerLayerParameters(
                    window_left=-1,  # No sliding window
                    logits_soft_cap=0.0,  # No soft cap
                    sm_scale=1.0 / (head_size**0.5),  # Standard scale
                )
                for layer_name in layer_names
            }

        with unittest.mock.patch(
            "vllm.v1.attention.backends.flashinfer.get_per_layer_parameters",
            mock_get_per_layer_parameters,
        ):
            builder = builder_cls(kv_cache_spec, layer_names, vllm_config, device)
            attn_metadata = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
    else:
        # Build metadata
        with set_current_vllm_config(vllm_config):
            builder = builder_cls(kv_cache_spec, layer_names, vllm_config, device)
            if actual_backend == AttentionBackendEnum.FLEX_ATTENTION:
                builder.direct_build = use_direct_block_mask
            attn_metadata = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )

    # Instantiate implementation
    num_heads = vllm_config.model_config.get_num_attention_heads(
        vllm_config.parallel_config
    )
    num_kv_heads = vllm_config.model_config.get_num_kv_heads(
        vllm_config.parallel_config
    )
    head_size = vllm_config.model_config.get_head_size()
    scale = 1.0 / (head_size**0.5)
    extra_impl_kwargs = {"sinks": sinks} if sinks is not None else {}
    with set_current_vllm_config(vllm_config):
        impl = impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=None,
            sliding_window=sliding_window,
            attn_type=attn_type,
            kv_cache_dtype=kv_cache_dtype,
            **extra_impl_kwargs,
        )

    # Create mock layer and output buffer
    mock_layer = MockAttentionLayer(device)
    output = torch.empty_like(query)

    if is_quantized_kv_cache(kv_cache_dtype) and impl.supports_quant_query_input:
        query = query.to(current_platform.fp8_dtype())

    # Run forward pass
    # NOTE: The query, key, and value are already shaped correctly
    # in the calling test function.
    if not try_backend_includes_kv_cache_update(actual_backend):
        impl.do_kv_cache_update(
            mock_layer, key, value, kv_cache, attn_metadata.slot_mapping
        )
    if use_cuda_graph:
        impl.forward(
            mock_layer, query, key, value, kv_cache, attn_metadata, output=output
        )
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            impl.forward(
                mock_layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output=output,
            )
        graph.replay()
        torch.accelerator.synchronize()
    else:
        output = impl.forward(
            mock_layer, query, key, value, kv_cache, attn_metadata, output=output
        )

    return output


def _test_backend_correctness(
    batch_spec: BatchSpec,
    model: str,
    backend_to_test: list[AttentionBackendEnum | str],
    mask_mod,
    *,
    causal: bool = True,
    attn_type: AttentionType = AttentionType.DECODER,
    block_size: int = 16,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    tensor_parallel_size: int = 1,
    kv_cache_dtype: str = "auto",
    sliding_window_override: int | None = None,
    use_attention_sinks: bool = False,
    use_cuda_graph: bool = False,
    num_speculative_tokens: int = 0,
    model_dtype: torch.dtype | None = None,
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
    set_random_seed(42)

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
        dtype=model_dtype or "auto",
        block_size=block_size,
        num_gpu_blocks=8192,
        hf_config_override=hf_config_override,
    )
    if AttentionBackendEnum.B12X_ATTN in backend_to_test:
        vllm_config.scheduler_config.max_num_seqs = batch_spec.batch_size
        vllm_config.scheduler_config.max_num_batched_tokens = max(
            sum(batch_spec.query_lens), 64
        )
        if num_speculative_tokens > 0:
            vllm_config.speculative_config = SimpleNamespace(
                num_speculative_tokens=num_speculative_tokens
            )
    vllm_config.cache_config.cache_dtype = kv_cache_dtype
    device = torch.device(f"{DEVICE_TYPE}:0")

    kv_cache_spec = create_standard_kv_cache_spec(vllm_config, attn_type)

    # 1. Setup
    batch_size = batch_spec.batch_size
    seq_lens = batch_spec.seq_lens
    query_lens = batch_spec.query_lens
    num_q_heads = vllm_config.model_config.get_num_attention_heads(
        vllm_config.parallel_config
    )
    num_kv_heads = vllm_config.model_config.get_num_kv_heads(
        vllm_config.parallel_config
    )
    head_size = vllm_config.model_config.get_head_size()
    sliding_window = (
        sliding_window_override
        if sliding_window_override is not None
        else vllm_config.model_config.get_sliding_window()
    )
    dtype = _convert_dtype_to_torch(vllm_config.model_config.dtype)
    block_size = vllm_config.cache_config.block_size
    scale = 1.0 / (head_size**0.5)
    sinks = (
        torch.linspace(-0.3, 0.3, num_q_heads, dtype=dtype, device=device)
        if use_attention_sinks
        else None
    )

    fp8_kv_cache = is_quantized_kv_cache(kv_cache_dtype)
    if fp8_kv_cache:
        query_fp8_dtype = current_platform.fp8_dtype()
        kv_fp8_dtype = FP8_KV_CACHE_DTYPES[kv_cache_dtype]
        atol = max(atol, 6e-2)
        rtol = max(rtol, 1e-1)

    # 2. Generate data and compute SDPA reference output
    all_q_vllm, all_k_vllm, all_v_vllm = [], [], []
    all_sdpa_outputs = []
    k_contexts, v_contexts = [], []

    for i in range(batch_size):
        s_len = seq_lens[i]
        q_len = query_lens[i]
        context_len = s_len - q_len

        # Generate Q, K, V for the whole sequence to be used in SDPA
        q = torch.randn(q_len, num_q_heads, head_size, dtype=dtype, device=device)
        k_full = torch.randn(s_len, num_kv_heads, head_size, dtype=dtype, device=device)
        v_full = torch.randn(s_len, num_kv_heads, head_size, dtype=dtype, device=device)

        if fp8_kv_cache:
            q_ref = q.to(query_fp8_dtype).to(dtype)
            k_ref = k_full.to(kv_fp8_dtype).to(dtype)
            v_ref = v_full.to(kv_fp8_dtype).to(dtype)
        else:
            q_ref, k_ref, v_ref = q, k_full, v_full

        # SDPA expects (N, H, L, D), so unsqueeze batch and permute
        q_sdpa_in = q_ref.unsqueeze(0).transpose(1, 2)
        k_sdpa_in = k_ref.unsqueeze(0).transpose(1, 2)
        v_sdpa_in = v_ref.unsqueeze(0).transpose(1, 2)

        if num_q_heads != num_kv_heads:
            assert num_q_heads % num_kv_heads == 0, (
                f"num_q_heads ({num_q_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads})"
            )
            repeats = num_q_heads // num_kv_heads
            k_sdpa_in = k_sdpa_in.repeat_interleave(repeats, dim=1)
            v_sdpa_in = v_sdpa_in.repeat_interleave(repeats, dim=1)

        # Create causal mask: query token i attends to positions 0 to
        #  (context_len + i)
        kv_len = s_len

        final_mask_mod = partial(mask_mod, context_len=context_len)
        block_mask = create_block_mask(
            final_mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=device
        )
        if sinks is None:
            sdpa_out_i = flex_attention(
                q_sdpa_in,
                k_sdpa_in,
                v_sdpa_in,
                block_mask=block_mask,
                scale=scale,
                enable_gqa=True,
            )
        else:
            q_idx = torch.arange(q_len, device=device).unsqueeze(1)
            kv_idx = torch.arange(kv_len, device=device).unsqueeze(0)
            mask = final_mask_mod(
                torch.zeros((), device=device),
                torch.zeros((), device=device),
                q_idx,
                kv_idx,
            )
            scores = (
                torch.einsum(
                    "hqd,hkd->hqk",
                    q_sdpa_in.squeeze(0).float(),
                    k_sdpa_in.squeeze(0).float(),
                )
                * scale
            )
            scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
            sink_logits = sinks.float().view(num_q_heads, 1, 1)
            probabilities = torch.softmax(
                torch.cat([scores, sink_logits.expand(num_q_heads, q_len, 1)], dim=-1),
                dim=-1,
            )[..., :kv_len]
            sdpa_out_i = (
                torch.einsum(
                    "hqk,hkd->hqd", probabilities, v_sdpa_in.squeeze(0).float()
                )
                .unsqueeze(0)
                .to(dtype)
            )

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
        batch_spec, vllm_config.cache_config.block_size, device
    )
    common_attn_metadata.causal = causal

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
        randomize_blocks=True,
        kv_cache_dtype=kv_cache_dtype,
    )

    # 4. Run vLLM backends and compare
    # Note: flex_attention has known Triton kernel compatibility issues
    # with test infrastructures
    for backend_name in backend_to_test:
        reset_kv_cache_layout = False

        # Resolve backend class for both enum and string names.
        actual_backend = backend_name
        if backend_name == "FLEX_ATTENTION_SLOW":
            actual_backend = AttentionBackendEnum.FLEX_ATTENTION
        if hasattr(actual_backend, "get_class"):
            backend_cls = actual_backend.get_class()
        else:
            backend_cls = None

        if is_quantized_kv_cache(kv_cache_dtype) and (
            backend_cls is None
            or not backend_cls.supports_kv_cache_dtype(kv_cache_dtype)
        ):
            continue

        if backend_name == AttentionBackendEnum.FLASHINFER:
            set_kv_cache_layout("HND")
            reset_kv_cache_layout = True
        elif backend_name == AttentionBackendEnum.B12X_ATTN:
            set_kv_cache_layout("NHD")
            reset_kv_cache_layout = True

        kv_cache_for_backend = kv_cache
        if backend_name == AttentionBackendEnum.B12X_ATTN:
            cache_dtype = (
                FP8_KV_CACHE_DTYPES[kv_cache_dtype]
                if is_quantized_kv_cache(kv_cache_dtype)
                else kv_cache.dtype
            )
            typed_cache = kv_cache.view(cache_dtype)
            key_cache = typed_cache[..., :head_size].permute(0, 2, 1, 3)
            value_cache = typed_cache[..., head_size:].permute(0, 2, 1, 3)
            kv_cache_for_backend = torch.stack((key_cache, value_cache), dim=1)
            if is_quantized_kv_cache(kv_cache_dtype):
                kv_cache_for_backend = kv_cache_for_backend.view(torch.uint8)
        elif backend_cls is not None:
            try:
                stride_order = backend_cls.get_kv_cache_stride_order()
            except (AttributeError, NotImplementedError):
                stride_order = tuple(range(kv_cache.ndim))
            if stride_order != tuple(range(kv_cache.ndim)):
                # Apply stride order like runtime does in
                # _reshape_kv_cache (attn_utils.py:182-210): permute to physical
                # layout, make contiguous, then permute to logical layout.
                inv_order = [stride_order.index(i) for i in range(len(stride_order))]
                kv_cache_for_backend = (
                    kv_cache.permute(*stride_order).contiguous().permute(*inv_order)
                )

        try:
            backend_output = run_attention_backend(
                backend_name,
                kv_cache_spec,
                ["placeholder"],
                vllm_config,
                device,
                common_attn_metadata,
                query_vllm,
                key_vllm,
                value_vllm,
                kv_cache_for_backend,
                sliding_window=sliding_window,
                attn_type=attn_type,
                kv_cache_dtype=kv_cache_dtype,
                sinks=sinks,
                use_cuda_graph=use_cuda_graph,
            )
        finally:
            if reset_kv_cache_layout:
                set_kv_cache_layout(None)

        # Check shape and dtype consistency
        assert backend_output.shape == sdpa_output.shape, (
            f"[{backend_name}] shape {backend_output.shape} != "
            f"SDPA shape {sdpa_output.shape}"
        )
        assert backend_output.dtype == sdpa_output.dtype, (
            f"[{backend_name}] dtype {backend_output.dtype} != "
            f"SDPA dtype {sdpa_output.dtype}"
        )

        assert torch.isfinite(backend_output).all(), (
            f"[{backend_name}] produced non-finite values"
        )

        # Check numerical similarity
        def error_msg(msg: str, backend_name: str):
            return f"[{backend_name}] output differs from SDPA baseline. {msg}"

        torch.testing.assert_close(
            backend_output,
            sdpa_output,
            rtol=rtol,
            atol=atol,
            msg=partial(error_msg, backend_name=backend_name),
        )


def _require_b12x_paged_attention() -> None:
    capability = current_platform.get_device_capability()
    if (
        not current_platform.is_cuda()
        or capability is None
        or not B12XPagedAttentionBackend.supports_compute_capability(capability)
    ):
        pytest.skip("B12X paged attention requires SM120 or SM121.")

    from b12x.attention import paged

    if not paged.is_supported():
        pytest.skip("B12X paged attention is not available.")


def _b12x_causal_mask(
    b: torch.Tensor,
    h: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    *,
    context_len: int,
):
    return q_idx + context_len >= kv_idx


def _b12x_causal_sliding_window_mask(
    b: torch.Tensor,
    h: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    *,
    context_len: int,
    sliding_window: int,
):
    causal_mask = q_idx + context_len >= kv_idx
    window_mask = q_idx + context_len - kv_idx < sliding_window
    return causal_mask & window_mask


@pytest.mark.parametrize(
    "batch_spec_name",
    ["small_decode", "small_prefill", "mixed_small", "medium_decode"],
)
@pytest.mark.parametrize(
    ("kv_cache_dtype", "model_dtype"),
    [
        ("auto", None),
        ("bfloat16", torch.bfloat16),
        ("fp8_e4m3", torch.bfloat16),
    ],
)
@pytest.mark.parametrize("block_size", [64, 128])
def test_b12x_causal_backend_correctness(
    default_vllm_config,
    workspace_init,
    batch_spec_name: str,
    kv_cache_dtype: str,
    model_dtype: torch.dtype | None,
    block_size: int,
):
    """B12X causal paged attention matches the shared SDPA reference."""
    _require_b12x_paged_attention()

    _test_backend_correctness(
        BATCH_SPECS[batch_spec_name],
        "Qwen/Qwen3-0.6B",
        [AttentionBackendEnum.B12X_ATTN],
        _b12x_causal_mask,
        block_size=block_size,
        kv_cache_dtype=kv_cache_dtype,
        model_dtype=model_dtype,
    )


@pytest.mark.parametrize("batch_spec_name", ["small_decode", "small_prefill"])
def test_b12x_causal_sliding_window_and_sinks(
    default_vllm_config,
    workspace_init,
    batch_spec_name: str,
):
    """B12X preserves causal SWA and attention-sink semantics."""
    _require_b12x_paged_attention()

    sliding_window = 16
    mask = partial(_b12x_causal_sliding_window_mask, sliding_window=sliding_window)

    _test_backend_correctness(
        BATCH_SPECS[batch_spec_name],
        "Qwen/Qwen3-0.6B",
        [AttentionBackendEnum.B12X_ATTN],
        mask,
        block_size=64,
        atol=3e-2,
        rtol=3e-2,
        sliding_window_override=sliding_window,
        use_attention_sinks=True,
    )


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8_e4m3"])
def test_b12x_decode_cuda_graph_replay_with_sliding_window_and_sinks(
    default_vllm_config,
    workspace_init,
    kv_cache_dtype: str,
):
    """B12X causal metadata and sinks remain capture-safe during replay."""
    _require_b12x_paged_attention()

    sliding_window = 16
    mask = partial(_b12x_causal_sliding_window_mask, sliding_window=sliding_window)

    _test_backend_correctness(
        BATCH_SPECS["small_decode"],
        "Qwen/Qwen3-0.6B",
        [AttentionBackendEnum.B12X_ATTN],
        mask,
        block_size=64,
        atol=3e-2,
        rtol=3e-2,
        sliding_window_override=sliding_window,
        use_attention_sinks=True,
        use_cuda_graph=True,
        kv_cache_dtype=kv_cache_dtype,
    )


def test_b12x_speculative_verifier_cuda_graph_replay(
    default_vllm_config,
    workspace_init,
):
    """B12X replays uniform speculative verification through its graph plan."""
    _require_b12x_paged_attention()

    _test_backend_correctness(
        BatchSpec(seq_lens=[32, 40], query_lens=[4, 4]),
        "Qwen/Qwen3-0.6B",
        [AttentionBackendEnum.B12X_ATTN],
        _b12x_causal_mask,
        block_size=128,
        num_speculative_tokens=3,
        use_cuda_graph=True,
    )


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
    ],
)
@pytest.mark.parametrize("model", ["meta-llama/Meta-Llama-3-8B"])
@pytest.mark.parametrize("tensor_parallel_size", [1, 2, 4])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8", "fp8_e4m3"])
def test_causal_backend_correctness(
    default_vllm_config,
    batch_spec_name: str,
    model: str,
    tensor_parallel_size: int,
    kv_cache_dtype: str,
):
    """Test backend's correctness with causal attention."""

    def causal_mask_mod(
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
        *,
        context_len: int,
    ):
        return (q_idx + context_len) >= kv_idx

    batch_spec = BATCH_SPECS[batch_spec_name]
    LARGE_BLOCK_BACKENDS = (
        [AttentionBackendEnum.FLEX_ATTENTION]
        if is_torch_equal_or_newer("2.9.0.dev0")
        else []
    )

    if current_platform.is_rocm():
        SMALL_BLOCK_BACKENDS = [
            x
            for x in BACKENDS_TO_TEST
            if (
                x not in LARGE_BLOCK_BACKENDS
                and x is not AttentionBackendEnum.FLASH_ATTN
            )
        ]
    else:
        SMALL_BLOCK_BACKENDS = [
            x for x in BACKENDS_TO_TEST if x not in LARGE_BLOCK_BACKENDS
        ]

    _test_backend_correctness(
        batch_spec,
        model,
        SMALL_BLOCK_BACKENDS,
        causal_mask_mod,
        tensor_parallel_size=tensor_parallel_size,
        kv_cache_dtype=kv_cache_dtype,
    )

    # Fast FlexAttention needs to run with block_size=128
    if LARGE_BLOCK_BACKENDS:
        _test_backend_correctness(
            batch_spec,
            model,
            LARGE_BLOCK_BACKENDS,
            causal_mask_mod,
            block_size=128,
            tensor_parallel_size=tensor_parallel_size,
            kv_cache_dtype=kv_cache_dtype,
        )


@pytest.mark.skipif(
    AttentionBackendEnum.FLASHINFER not in BACKENDS_TO_TEST,
    reason="FlashInfer is not available.",
)
def test_flashinfer_xqa_bmm1_scale_matches_decode_q_dtype():
    """XQA decode should only apply q_scale when decode Q is FP8."""
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    class MockLayer:
        _q_scale_float = 2.0
        _k_scale_float = 3.0

    impl = object.__new__(flashinfer_backend.FlashInferImpl)
    impl.scale = 0.5
    impl.kv_cache_dtype = "fp8"

    assert impl.get_xqa_bmm1_scale(MockLayer, torch.bfloat16) == 1.5
    assert impl.get_xqa_bmm1_scale(MockLayer, torch.float8_e4m3fn) == 3.0


@pytest.mark.skipif(
    AttentionBackendEnum.FLASHINFER not in BACKENDS_TO_TEST,
    reason="FlashInfer is not available.",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_flashinfer_attention_sinks_refreshed_after_reload(dtype):
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    source_sinks = torch.tensor([1.0, 2.0], dtype=dtype)
    impl = object.__new__(flashinfer_backend.FlashInferImpl)
    impl._sinks_source = source_sinks
    impl.sinks = source_sinks

    impl.process_weights_after_loading(dtype)

    assert impl.sinks is not None
    sinks_ptr = impl.sinks.data_ptr()
    assert impl.sinks.dtype == torch.float32
    torch.testing.assert_close(impl.sinks, source_sinks.float())

    source_sinks.copy_(torch.tensor([3.0, 4.0], dtype=dtype))
    impl.process_weights_after_loading(dtype)

    assert impl.sinks.data_ptr() == sinks_ptr
    torch.testing.assert_close(impl.sinks, source_sinks.float())


@pytest.mark.skipif(
    AttentionBackendEnum.FLASHINFER not in BACKENDS_TO_TEST,
    reason="FlashInfer is not available.",
)
def test_flashinfer_sm90_xqa_decode_correctness(default_vllm_config):
    """FlashInfer should route Hopper decode through XQA and match SDPA."""
    if not current_platform.is_cuda() or not current_platform.is_device_capability(90):
        pytest.skip("FlashInfer XQA decode requires SM90.")

    import unittest.mock

    from vllm.utils.flashinfer import can_use_trtllm_attention
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend
    from vllm.v1.attention.backends.utils import PerLayerParameters

    def mock_get_per_layer_parameters(vllm_config, layer_names, impl_cls):
        return {
            "placeholder": PerLayerParameters(
                window_left=-1,
                logits_soft_cap=0.0,
                sm_scale=1.0,
            )
        }

    def causal_mask_mod(
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
        *,
        context_len: int,
    ):
        return (q_idx + context_len) >= kv_idx

    batch_spec = BATCH_SPECS["small_decode"]
    vllm_config = create_vllm_config(
        model_name="meta-llama/Meta-Llama-3-8B",
        max_model_len=max(batch_spec.seq_lens),
        block_size=16,
    )
    device = torch.device(f"{DEVICE_TYPE}:0")
    kv_cache_spec = FullAttentionSpec(
        block_size=vllm_config.cache_config.block_size,
        num_kv_heads=vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        ),
        head_size=vllm_config.model_config.get_head_size(),
        dtype=vllm_config.model_config.dtype,
    )

    with set_current_vllm_config(vllm_config):
        if not can_use_trtllm_attention(
            vllm_config.model_config.get_num_attention_heads(
                vllm_config.parallel_config
            ),
            kv_cache_spec.num_kv_heads,
            is_prefill=False,
        ):
            pytest.skip("FlashInfer XQA decode is not available in this setup.")

        with unittest.mock.patch(
            "vllm.v1.attention.backends.flashinfer.get_per_layer_parameters",
            mock_get_per_layer_parameters,
        ):
            builder = flashinfer_backend.FlashInferMetadataBuilder(
                kv_cache_spec, ["placeholder"], vllm_config, device
            )
            common_attn_metadata = create_common_attn_metadata(
                batch_spec, vllm_config.cache_config.block_size, device
            )
            attn_metadata = builder.build(0, common_attn_metadata)

    assert (
        flashinfer_backend.FlashInferMetadataBuilder.get_cudagraph_support(
            vllm_config, kv_cache_spec
        )
        == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )
    assert isinstance(
        attn_metadata.decode,
        flashinfer_backend.FlashInferTrtllmAPIDecode,
    )
    assert attn_metadata.decode.kernel == flashinfer_backend.FlashInferDecodeKernel.XQA

    _test_backend_correctness(
        batch_spec,
        "meta-llama/Meta-Llama-3-8B",
        [AttentionBackendEnum.FLASHINFER],
        causal_mask_mod,
    )


if current_platform.is_rocm():
    # FLASH_ATTN is not supported on ROCm
    SLIDING_WINDOW_BACKENDS_TO_TEST = [
        AttentionBackendEnum.FLEX_ATTENTION,
        AttentionBackendEnum.TRITON_ATTN,
        "FLEX_ATTENTION_SLOW",
    ]
else:
    SLIDING_WINDOW_BACKENDS_TO_TEST = [
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.FLEX_ATTENTION,
        AttentionBackendEnum.TRITON_ATTN,
        "FLEX_ATTENTION_SLOW",
    ]


@pytest.mark.parametrize(
    "batch_spec_name",
    [
        "small_decode",
        "small_prefill",
        "mixed_medium",
        "large_decode",
        "large_prefill",
        "mixed_large",
    ],
)
@pytest.mark.parametrize("model", ["microsoft/Phi-tiny-MoE-instruct"])
@pytest.mark.parametrize("tensor_parallel_size", [1, 2, 4])
def test_sliding_window_backend_correctness(
    default_vllm_config, batch_spec_name: str, model: str, tensor_parallel_size: int
):
    """Test backend's correctness with sliding window attention."""

    def sliding_window_mask_mod(
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
        *,
        context_len: int,
        sliding_window: int,
    ):
        causal_mask = q_idx + context_len >= kv_idx
        window_mask = q_idx + context_len - kv_idx < sliding_window
        return causal_mask & window_mask

    batch_spec = BATCH_SPECS[batch_spec_name]
    model_config = ModelConfig(model=model, max_model_len=max(batch_spec.seq_lens))
    sliding_window = model_config.get_sliding_window()
    sliding_window_mask_mod_fn = partial(
        sliding_window_mask_mod, sliding_window=sliding_window
    )

    LARGE_BLOCK_BACKENDS = (
        [AttentionBackendEnum.FLEX_ATTENTION]
        if is_torch_equal_or_newer("2.9.0.dev0")
        else []
    )
    SMALL_BLOCK_BACKENDS = [
        x for x in SLIDING_WINDOW_BACKENDS_TO_TEST if x not in LARGE_BLOCK_BACKENDS
    ]
    _test_backend_correctness(
        batch_spec,
        model,
        SMALL_BLOCK_BACKENDS,
        sliding_window_mask_mod_fn,
        tensor_parallel_size=tensor_parallel_size,
    )

    # Fast FlexAttention needs to run with block_size=128
    if LARGE_BLOCK_BACKENDS:
        _test_backend_correctness(
            batch_spec,
            model,
            LARGE_BLOCK_BACKENDS,
            sliding_window_mask_mod_fn,
            block_size=128,
            tensor_parallel_size=tensor_parallel_size,
        )


@pytest.mark.parametrize(
    "batch_spec_name",
    [
        "small_encoder_prefill",
        "medium_encoder_prefill",
    ],
)
@pytest.mark.parametrize("model", ["google/embeddinggemma-300m"])
@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
def test_sliding_window_encoder_backend_correctness(
    default_vllm_config, batch_spec_name: str, model: str, tensor_parallel_size: int
):
    """Test backend's correctness with sliding window attention."""

    def bidi_sliding_window_mask_mod(
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
        *,
        context_len: int,
        sliding_window: int,
    ):
        return torch.abs(q_idx + context_len - kv_idx) < sliding_window

    batch_spec = BATCH_SPECS[batch_spec_name]
    model_config = ModelConfig(model=model, max_model_len=max(batch_spec.seq_lens))
    sliding_window = model_config.get_sliding_window()
    sliding_window_mask_mod_fn = partial(
        bidi_sliding_window_mask_mod, sliding_window=sliding_window
    )

    _test_backend_correctness(
        batch_spec,
        model,
        SLIDING_WINDOW_BACKENDS_TO_TEST,
        sliding_window_mask_mod_fn,
        causal=False,
        attn_type=AttentionType.ENCODER_ONLY,
        tensor_parallel_size=tensor_parallel_size,
    )


NON_CAUSAL_BACKENDS_TO_TEST = [
    AttentionBackendEnum.FLASH_ATTN,
    AttentionBackendEnum.FLEX_ATTENTION,
    "FLEX_ATTENTION_SLOW",
]

if current_platform.is_rocm():
    NON_CAUSAL_BACKENDS_TO_TEST = [
        x
        for x in NON_CAUSAL_BACKENDS_TO_TEST
        if x is not AttentionBackendEnum.FLASH_ATTN
    ]


@pytest.mark.parametrize(
    "batch_spec_name",
    [
        "small_decode",
        "small_prefill",
        "mixed_small",
    ],
)
@pytest.mark.parametrize("model", ["meta-llama/Meta-Llama-3-8B"])
def test_non_causal_backend_correctness(
    default_vllm_config, batch_spec_name: str, model: str
):
    """Test backend's correctness with non-causal (bidirectional) decoder
    attention, as used by DFlash speculative decoding."""

    def bidirectional_mask_mod(
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
        *,
        context_len: int,
    ):
        return q_idx >= 0  # Always True

    batch_spec = BATCH_SPECS[batch_spec_name]
    LARGE_BLOCK_BACKENDS = (
        [AttentionBackendEnum.FLEX_ATTENTION]
        if is_torch_equal_or_newer("2.9.0.dev0")
        else []
    )

    SMALL_BLOCK_BACKENDS = [
        x for x in NON_CAUSAL_BACKENDS_TO_TEST if x not in LARGE_BLOCK_BACKENDS
    ]

    _test_backend_correctness(
        batch_spec,
        model,
        SMALL_BLOCK_BACKENDS,
        bidirectional_mask_mod,
        causal=False,
    )

    if LARGE_BLOCK_BACKENDS:
        _test_backend_correctness(
            batch_spec,
            model,
            LARGE_BLOCK_BACKENDS,
            bidirectional_mask_mod,
            causal=False,
            block_size=128,
        )
