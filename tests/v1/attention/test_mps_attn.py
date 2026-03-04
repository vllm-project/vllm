# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MPS (Apple Metal) attention backend."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_mps():
    pytest.skip("MPS-only tests", allow_module_level=True)

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.mps_attn import (
    MPSAttentionBackend,
    MPSAttentionBackendImpl,
    MPSAttentionMetadataBuilder,
    _reshape_and_cache,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec

DEVICE = torch.device("mps")

# Batch configurations for testing
BATCH_SPECS = {
    "small_decode": BatchSpec(seq_lens=[32, 40], query_lens=[1, 1]),
    "small_prefill": BatchSpec(seq_lens=[32, 40], query_lens=[8, 8]),
    "mixed_small": BatchSpec(seq_lens=[32, 40, 48, 56], query_lens=[1, 1, 5, 5]),
    "single_decode": BatchSpec(seq_lens=[64], query_lens=[1]),
    "single_prefill": BatchSpec(seq_lens=[64], query_lens=[16]),
    "medium_decode": BatchSpec(seq_lens=[128, 256, 512], query_lens=[1, 1, 1]),
}


def create_kv_cache_hnd(
    num_blocks: int,
    num_kv_heads: int,
    block_size: int,
    head_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Create KV cache in HND layout: (2, num_blocks, num_kv_heads, block_size, head_size)."""
    return torch.zeros(
        2,
        num_blocks,
        num_kv_heads,
        block_size,
        head_size,
        dtype=dtype,
        device=device,
    )


def prepopulate_kv_cache(
    kv_cache: torch.Tensor,
    k_contexts: list[torch.Tensor],
    v_contexts: list[torch.Tensor],
    block_table: torch.Tensor,
    seq_lens: list[int],
    query_lens: list[int],
    block_size: int,
) -> None:
    """Populate KV cache with context data using _reshape_and_cache."""
    key_cache, value_cache = kv_cache.unbind(0)
    for i, (k_ctx, v_ctx) in enumerate(zip(k_contexts, v_contexts)):
        context_len = seq_lens[i] - query_lens[i]
        if context_len == 0:
            continue
        # Create slot mapping for context tokens
        num_blocks_needed = (context_len + block_size - 1) // block_size
        blocks = block_table[i, :num_blocks_needed]
        slots = []
        for b_idx in range(num_blocks_needed):
            block_id = int(blocks[b_idx])
            tokens_in_block = min(block_size, context_len - b_idx * block_size)
            for off in range(tokens_in_block):
                slots.append(block_id * block_size + off)
        slot_mapping = torch.tensor(slots, dtype=torch.int64, device=k_ctx.device)
        _reshape_and_cache(k_ctx, v_ctx, key_cache, value_cache, slot_mapping)


def sdpa_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    seq_lens: list[int],
    query_lens: list[int],
    scale: float,
    num_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    """Compute reference attention output using torch SDPA on contiguous data."""
    output = torch.empty_like(query)
    q_start = 0
    k_start = 0
    for i in range(len(seq_lens)):
        q_len = query_lens[i]
        s_len = seq_lens[i]
        context_len = s_len - q_len

        q = query[q_start : q_start + q_len]  # [q_len, num_heads, head_size]
        # Full key/value includes context + query tokens
        k = key[k_start : k_start + s_len]
        v = value[k_start : k_start + s_len]

        # [1, num_heads, q_len, head_size]
        q_t = q.transpose(0, 1).unsqueeze(0)
        k_t = k.transpose(0, 1).unsqueeze(0)
        v_t = v.transpose(0, 1).unsqueeze(0)

        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=(q_len > 1),
            scale=scale,
            enable_gqa=(num_heads != num_kv_heads),
        )
        output[q_start : q_start + q_len] = attn_out.squeeze(0).transpose(0, 1)

        q_start += q_len
        k_start += s_len

    return output


class TestMPSAttentionBackend:
    """Test MPSAttentionBackend class methods."""

    def test_get_name(self):
        assert MPSAttentionBackend.get_name() == "MPS_ATTN"

    def test_get_supported_dtypes(self):
        dtypes = MPSAttentionBackend.get_supported_dtypes()
        assert torch.float16 in dtypes
        assert torch.float32 in dtypes

    def test_get_supported_head_sizes(self):
        sizes = MPSAttentionBackend.get_supported_head_sizes()
        assert 64 in sizes
        assert 128 in sizes

    def test_supports_decoder(self):
        assert MPSAttentionBackend.supports_attn_type(AttentionType.DECODER)

    def test_supports_encoder(self):
        assert MPSAttentionBackend.supports_attn_type(AttentionType.ENCODER)
        assert MPSAttentionBackend.supports_attn_type(AttentionType.ENCODER_ONLY)

    def test_no_cascade(self):
        assert MPSAttentionBackend.use_cascade_attention() is False

    def test_kv_cache_shape(self):
        shape = MPSAttentionBackend.get_kv_cache_shape(
            num_blocks=100,
            block_size=16,
            num_kv_heads=4,
            head_size=64,
        )
        assert shape == (2, 100, 4, 16, 64)


class TestReshapeAndCache:
    """Test _reshape_and_cache function."""

    @pytest.mark.parametrize("block_size", [16, 32])
    @pytest.mark.parametrize("num_kv_heads", [1, 4])
    def test_basic_cache_write(self, block_size, num_kv_heads):
        head_size = 64
        num_tokens = 8
        num_blocks = 10

        key = torch.randn(num_tokens, num_kv_heads, head_size, device=DEVICE)
        value = torch.randn(num_tokens, num_kv_heads, head_size, device=DEVICE)
        key_cache = torch.zeros(
            num_blocks, num_kv_heads, block_size, head_size, device=DEVICE
        )
        value_cache = torch.zeros(
            num_blocks, num_kv_heads, block_size, head_size, device=DEVICE
        )

        # Place tokens in block 2, offsets 0..num_tokens-1
        slot_mapping = (
            torch.arange(num_tokens, dtype=torch.int64, device=DEVICE) + 2 * block_size
        )

        _reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

        # Verify
        for t in range(num_tokens):
            torch.testing.assert_close(key_cache[2, :, t, :], key[t])
            torch.testing.assert_close(value_cache[2, :, t, :], value[t])

    def test_empty_tokens(self):
        """_reshape_and_cache should handle 0 tokens gracefully."""
        key = torch.empty(0, 4, 64, device=DEVICE)
        value = torch.empty(0, 4, 64, device=DEVICE)
        key_cache = torch.zeros(10, 4, 16, 64, device=DEVICE)
        value_cache = torch.zeros(10, 4, 16, 64, device=DEVICE)
        slot_mapping = torch.empty(0, dtype=torch.int64, device=DEVICE)

        _reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
        # Should not crash


class TestMPSAttentionMetadataBuilder:
    """Test metadata builder."""

    def test_build_metadata(self):
        vllm_config = create_vllm_config(
            model_name="Qwen/Qwen3-0.6B",
            block_size=16,
            num_gpu_blocks=100,
        )
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=vllm_config.model_config.get_num_kv_heads(
                vllm_config.parallel_config
            ),
            head_size=vllm_config.model_config.get_head_size(),
            dtype=vllm_config.model_config.dtype,
            sliding_window=None,
        )
        batch_spec = BatchSpec(seq_lens=[32, 40], query_lens=[1, 1])
        common_meta = create_common_attn_metadata(batch_spec, 16, DEVICE)

        builder = MPSAttentionMetadataBuilder(
            kv_cache_spec,
            ["layer0"],
            vllm_config,
            DEVICE,
        )
        meta = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_meta,
        )

        assert meta.num_actual_tokens == 2
        assert meta.max_query_len == 1
        assert meta.max_seq_len == 40
        assert meta.causal is True


class TestMPSAttentionCorrectness:
    """Test MPS attention produces correct results vs reference SDPA."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    @pytest.mark.parametrize(
        "batch_name",
        [
            "small_decode",
            "small_prefill",
            "mixed_small",
            "single_decode",
            "single_prefill",
        ],
    )
    @pytest.mark.parametrize("num_kv_heads,num_heads", [(4, 4), (2, 8)])
    def test_attention_correctness(
        self,
        dtype,
        batch_name,
        num_kv_heads,
        num_heads,
    ):
        head_size = 64
        block_size = 16
        scale = 1.0 / (head_size**0.5)
        batch_spec = BATCH_SPECS[batch_name]

        num_tokens = sum(batch_spec.query_lens)
        total_context_tokens = sum(
            s - q for s, q in zip(batch_spec.seq_lens, batch_spec.query_lens)
        )

        # Generate full Q, K, V for reference computation
        # Full K, V = context + query tokens for each sequence
        total_kv_tokens = sum(batch_spec.seq_lens)
        full_key = torch.randn(
            total_kv_tokens, num_kv_heads, head_size, dtype=dtype, device=DEVICE
        )
        full_value = torch.randn(
            total_kv_tokens, num_kv_heads, head_size, dtype=dtype, device=DEVICE
        )

        # Query tokens (what the model is computing attention for)
        query = torch.randn(
            num_tokens, num_heads, head_size, dtype=dtype, device=DEVICE
        )

        # Extract the query portion of K, V (new tokens being added to cache)
        new_key_parts = []
        new_value_parts = []
        context_key_parts = []
        context_value_parts = []
        kv_offset = 0
        for i in range(batch_spec.batch_size):
            s_len = batch_spec.seq_lens[i]
            q_len = batch_spec.query_lens[i]
            ctx_len = s_len - q_len
            context_key_parts.append(full_key[kv_offset : kv_offset + ctx_len])
            context_value_parts.append(full_value[kv_offset : kv_offset + ctx_len])
            new_key_parts.append(full_key[kv_offset + ctx_len : kv_offset + s_len])
            new_value_parts.append(full_value[kv_offset + ctx_len : kv_offset + s_len])
            kv_offset += s_len

        new_key = torch.cat(new_key_parts, dim=0)
        new_value = torch.cat(new_value_parts, dim=0)

        # Reference output (contiguous SDPA)
        ref_output = sdpa_reference(
            query,
            full_key,
            full_value,
            batch_spec.seq_lens,
            batch_spec.query_lens,
            scale,
            num_heads,
            num_kv_heads,
        )

        # Now test through MPS attention backend
        max_blocks_per_seq = max(
            (s + block_size - 1) // block_size for s in batch_spec.seq_lens
        )
        total_blocks = (
            batch_spec.batch_size * max_blocks_per_seq + 1
        )  # +1 for null block
        kv_cache = create_kv_cache_hnd(
            total_blocks,
            num_kv_heads,
            block_size,
            head_size,
            dtype,
            DEVICE,
        )

        # Build block table — assign blocks sequentially starting from 1
        block_table = torch.zeros(
            batch_spec.batch_size,
            max_blocks_per_seq,
            dtype=torch.int32,
            device=DEVICE,
        )
        next_block = 1
        for i in range(batch_spec.batch_size):
            n_blocks = (batch_spec.seq_lens[i] + block_size - 1) // block_size
            for b in range(n_blocks):
                block_table[i, b] = next_block
                next_block += 1

        # Prepopulate cache with context
        prepopulate_kv_cache(
            kv_cache,
            context_key_parts,
            context_value_parts,
            block_table,
            batch_spec.seq_lens,
            batch_spec.query_lens,
            block_size,
        )

        # Build slot mapping for new tokens
        slot_list = []
        for i in range(batch_spec.batch_size):
            ctx_len = batch_spec.seq_lens[i] - batch_spec.query_lens[i]
            for t in range(batch_spec.query_lens[i]):
                token_pos = ctx_len + t
                block_idx = token_pos // block_size
                block_off = token_pos % block_size
                block_id = int(block_table[i, block_idx])
                slot_list.append(block_id * block_size + block_off)
        slot_mapping = torch.tensor(slot_list, dtype=torch.int64, device=DEVICE)

        # Build query_start_loc and seq_lens tensors
        query_start_loc = torch.zeros(
            batch_spec.batch_size + 1,
            dtype=torch.int32,
            device=DEVICE,
        )
        for i in range(batch_spec.batch_size):
            query_start_loc[i + 1] = query_start_loc[i] + batch_spec.query_lens[i]
        seq_lens_tensor = torch.tensor(
            batch_spec.seq_lens,
            dtype=torch.int32,
            device=DEVICE,
        )

        from vllm.v1.attention.backends.mps_attn import MPSAttentionMetadata

        attn_metadata = MPSAttentionMetadata(
            num_actual_tokens=num_tokens,
            max_query_len=max(batch_spec.query_lens),
            query_start_loc=query_start_loc,
            max_seq_len=max(batch_spec.seq_lens),
            seq_lens=seq_lens_tensor,
            block_table=block_table,
            slot_mapping=slot_mapping,
            num_reqs=batch_spec.batch_size,
            causal=True,
        )

        impl = MPSAttentionBackendImpl(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
        )

        output = torch.empty_like(query)

        # Mock layer
        class MockLayer:
            pass

        output = impl.forward(
            MockLayer(),
            query,
            new_key,
            new_value,
            kv_cache,
            attn_metadata,
            output=output,
        )

        # Compare with tolerance appropriate for dtype
        if dtype == torch.float16:
            atol, rtol = 1e-2, 1e-2
        else:
            atol, rtol = 1e-4, 1e-4

        torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol)


class TestMPSPlatformDetection:
    """Test that MPS platform is correctly detected."""

    def test_platform_is_mps(self):
        assert current_platform.is_mps()

    def test_device_type(self):
        assert current_platform.device_type == "mps"

    def test_dispatch_key(self):
        assert current_platform.dispatch_key == "MPS"


class TestMPSBackendSelection:
    """Test that MPS backend is selected correctly."""

    def test_mps_attn_in_registry(self):
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        assert hasattr(AttentionBackendEnum, "MPS_ATTN")

    def test_get_attn_backend_returns_mps(self):
        from unittest.mock import patch

        from vllm.config import AttentionConfig, VllmConfig, set_current_vllm_config
        from vllm.platforms.mps import MpsPlatform
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        from vllm.v1.attention.selector import (
            _cached_get_attn_backend,
            get_attn_backend,
        )

        _cached_get_attn_backend.cache_clear()
        attention_config = AttentionConfig(backend=AttentionBackendEnum.MPS_ATTN)
        vllm_config = VllmConfig(attention_config=attention_config)

        with set_current_vllm_config(vllm_config):
            with patch("vllm.platforms.current_platform", MpsPlatform()):
                backend = get_attn_backend(64, torch.float16, None)
        assert backend.get_name() == "MPS_ATTN"
