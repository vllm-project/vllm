# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for INT8 per-tensor KV cache quantization."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode, get_kv_quant_mode

DEVICE_TYPE = current_platform.device_type

NUM_HEADS = [(4, 4), (8, 2)]
HEAD_SIZES = [64, 128]
BLOCK_SIZES = [16]
SEQ_THRESHOLD_3D_VALUES = [0, 8]


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Reference paged attention in pure PyTorch."""
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len]
        q = q * scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)

        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        mask = torch.triu(
            torch.ones(query_len, kv_len), diagonal=kv_len - query_len + 1
        ).bool()
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)
        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


class TestKVQuantModeMapping:
    def test_int8_per_tensor_mapping(self):
        mode = get_kv_quant_mode("int8_per_tensor")
        assert mode == KVQuantMode.INT8_PER_TENSOR
        assert mode.value == 5

    def test_int8_per_tensor_not_per_token_head(self):
        mode = get_kv_quant_mode("int8_per_tensor")
        assert not mode.is_per_token_head

    def test_int8_per_tensor_value(self):
        assert KVQuantMode.INT8_PER_TENSOR.value == 5


class TestInt8ReshapeAndCache:
    @pytest.mark.parametrize("head_size", HEAD_SIZES)
    @pytest.mark.parametrize("num_heads", [4, 8])
    @torch.inference_mode()
    def test_quantize_dequantize_roundtrip(self, head_size, num_heads):
        """Verify quantize -> dequantize error is within 1 LSB."""
        from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
            triton_reshape_and_cache_flash,
        )

        torch.set_default_device(DEVICE_TYPE)
        set_random_seed(0)

        num_tokens = 16
        block_size = 16
        num_blocks = 4

        key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16) * 2.0
        value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16) * 2.0

        key_cache = torch.zeros(
            num_blocks, block_size, num_heads, head_size, dtype=torch.int8
        )
        value_cache = torch.zeros(
            num_blocks, block_size, num_heads, head_size, dtype=torch.int8
        )

        k_scale_val = key.abs().max().float() / 127.0
        v_scale_val = value.abs().max().float() / 127.0
        k_scale = torch.tensor([k_scale_val], dtype=torch.float32)
        v_scale = torch.tensor([v_scale_val], dtype=torch.float32)

        slot_mapping = torch.arange(num_tokens, dtype=torch.long)

        triton_reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            "int8_per_tensor",
            k_scale,
            v_scale,
        )

        # Dequantize and check error
        key_dequant = key_cache[:1, :num_tokens].float() * k_scale_val
        value_dequant = value_cache[:1, :num_tokens].float() * v_scale_val

        key_err = (key_dequant - key.float()).abs().max()
        value_err = (value_dequant - value.float()).abs().max()

        # Max error <= 0.5 * scale (half a quantization step)
        assert key_err <= k_scale_val * 0.6, (
            f"Key error {key_err:.6f} > threshold {k_scale_val * 0.6:.6f}"
        )
        assert value_err <= v_scale_val * 0.6, (
            f"Value error {value_err:.6f} > threshold {v_scale_val * 0.6:.6f}"
        )

    @torch.inference_mode()
    def test_scale_one_lossless(self):
        """With scale=1.0 and integer values in [-127,127], quant is lossless."""
        from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
            triton_reshape_and_cache_flash,
        )

        torch.set_default_device(DEVICE_TYPE)

        num_tokens, num_heads, head_size = 8, 4, 64
        block_size, num_blocks = 16, 2

        key = torch.randint(
            -50, 50, (num_tokens, num_heads, head_size), dtype=torch.float16
        )
        value = torch.randint(
            -50, 50, (num_tokens, num_heads, head_size), dtype=torch.float16
        )

        key_cache = torch.zeros(
            num_blocks, block_size, num_heads, head_size, dtype=torch.int8
        )
        value_cache = torch.zeros(
            num_blocks, block_size, num_heads, head_size, dtype=torch.int8
        )

        k_scale = torch.tensor([1.0], dtype=torch.float32)
        v_scale = torch.tensor([1.0], dtype=torch.float32)
        slot_mapping = torch.arange(num_tokens, dtype=torch.long)

        triton_reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            "int8_per_tensor",
            k_scale,
            v_scale,
        )

        key_back = key_cache[0, :num_tokens].to(torch.float16)
        assert torch.equal(key_back, key), (
            "INT8 with scale=1.0 should be lossless for integer values"
        )


@pytest.mark.parametrize("seq_lens", [[(1, 1328), (5, 18)], [(1, 523), (1, 2011)]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("seq_threshold_3D", SEQ_THRESHOLD_3D_VALUES)
@torch.inference_mode()
def test_int8_per_tensor_attention(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    block_size: int,
    seq_threshold_3D: int,
) -> None:
    """Test INT8 per-tensor attention against reference implementation.

    Follows the same pattern as test_triton_unified_attn for FP8.
    """
    torch.set_default_device(DEVICE_TYPE)
    set_random_seed(0)

    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5
    num_blocks = 32768

    # Generate FP16 reference data
    query = torch.randn(
        sum(query_lens), num_query_heads, head_size, dtype=torch.bfloat16
    )
    key_cache_fp16 = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=torch.bfloat16
    )
    value_cache_fp16 = torch.randn_like(key_cache_fp16)

    # Quantize KV cache to INT8 with proper scaling
    k_scale_val = key_cache_fp16.abs().max().float().item() / 127.0
    v_scale_val = value_cache_fp16.abs().max().float().item() / 127.0

    key_cache_int8 = torch.clamp(
        torch.round(key_cache_fp16.float() / k_scale_val), -128, 127
    ).to(torch.int8)
    value_cache_int8 = torch.clamp(
        torch.round(value_cache_fp16.float() / v_scale_val), -128, 127
    ).to(torch.int8)

    # Dequantized version for reference (what INT8 effectively represents)
    key_cache_dequant = key_cache_int8.to(torch.bfloat16) * k_scale_val
    value_cache_dequant = value_cache_int8.to(torch.bfloat16) * v_scale_val

    k_descale = torch.full((num_seqs, num_kv_heads), k_scale_val, dtype=torch.float32)
    v_descale = torch.full((num_seqs, num_kv_heads), v_scale_val, dtype=torch.float32)

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    # Buffers for 3D kernel
    num_par_softmax_segments = 16
    head_size_padded = next_power_of_2(head_size)
    softmax_segm_output = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments, head_size_padded),
        dtype=torch.float32,
    )
    softmax_segm_max = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )
    softmax_segm_expsum = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )

    # Run INT8 attention via Triton kernel
    output_int8 = torch.empty_like(query)
    unified_attention(
        q=query,
        k=key_cache_int8,
        v=value_cache_int8,
        out=output_int8,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_tensor,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0,
        q_descale=None,
        k_descale=k_descale,
        v_descale=v_descale,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
        kv_quant_mode=KVQuantMode.INT8_PER_TENSOR,
    )

    # Reference: run attention on the dequantized cache (pure PyTorch)
    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache_dequant,
        value_cache=value_cache_dequant,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
    )

    # Tolerance: INT8 quantization introduces rounding error that propagates
    # through softmax. Use same tolerance as FP8 test.
    atol, rtol = 1.5e-1, 1.5e-1
    torch.testing.assert_close(output_int8, ref_output, atol=atol, rtol=rtol)


class TestTritonAttentionBackend:
    def test_int8_per_tensor_in_supported_dtypes(self):
        from vllm.v1.attention.backends.triton_attn import (
            TritonAttentionBackend,
        )

        assert "int8_per_tensor" in TritonAttentionBackend.supported_kv_cache_dtypes
