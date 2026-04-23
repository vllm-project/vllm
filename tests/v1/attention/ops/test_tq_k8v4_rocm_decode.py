# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the TQ k8v4 HIP paged attention kernel.

Validates the HIP kernel output against a PyTorch reference implementation
of paged attention with TQ k8v4 quantization (FP8 keys + 4-bit values).

The kernel only runs on gfx942 (MI300X) and gfx950 (MI355X).
"""

import pytest
import torch

from vllm.platforms import current_platform

# Skip entire module if not on ROCm.
pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="TQ k8v4 HIP kernel requires ROCm (gfx942/gfx950)",
)

# Constants matching the HIP kernel.
HEAD_DIM = 128
BLOCK_SIZE = 16  # KV cache block size (tokens per block)

# TQ k8v4 slot layout for D=128:
#   [0..127]   = FP8 key (128 bytes)
#   [128..191] = 4-bit packed value (64 bytes for 128 dims)
#   [192..195] = value scale (fp16) + zero (fp16)
KEY_PACKED_SIZE = 128
VAL_DATA_OFFSET = 128
VAL_SCALE_OFFSET = 192
SLOT_SIZE = 196  # total bytes per token per KV head


def _check_arch_supported():
    """Check if current GPU is gfx942 or gfx950."""
    try:
        props = torch.cuda.get_device_properties(0)
        arch = getattr(props, "gcnArchName", "").split(":")[0]
        return arch in ("gfx942", "gfx950")
    except Exception:
        return False


def _quantize_key_fp8(key_fp32: torch.Tensor) -> torch.Tensor:
    """Quantize a float32 key vector [D] to FP8 E4M3 bytes [D].

    Uses the hardware's native FP8 E4M3 format. Returns uint8 tensor.
    """
    # Clamp to FP8 E4M3 range: [-448, 448]
    clamped = key_fp32.clamp(-448.0, 448.0)
    # Convert via float8_e4m3fnuz on ROCm (native format)
    try:
        fp8 = clamped.to(torch.float8_e4m3fnuz)
    except Exception:
        # Fallback: use float8_e4m3fn
        fp8 = clamped.to(torch.float8_e4m3fn)
    return fp8.view(torch.uint8)


def _dequantize_key_fp8(key_bytes: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 key bytes back to float32 for reference computation."""
    try:
        return key_bytes.view(torch.float8_e4m3fnuz).float()
    except Exception:
        return key_bytes.view(torch.float8_e4m3fn).float()


def _quantize_value_int4(
    value_fp32: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a float32 value vector [D] to 4-bit with per-token scale+zero.

    Returns:
        packed: uint8 tensor [D//2] with two nibbles per byte
        scale: fp16 scalar
        zero: fp16 scalar
    """
    vmin = value_fp32.min()
    vmax = value_fp32.max()
    # Compute scale and zero for mapping [vmin, vmax] -> [0, 15]
    if vmax - vmin < 1e-8:
        scale = torch.tensor(1.0, dtype=torch.float16)
        zero = vmin.to(torch.float16)
        quantized = torch.zeros(HEAD_DIM, dtype=torch.int32, device=value_fp32.device)
    else:
        scale_f32 = (vmax - vmin) / 15.0
        zero_f32 = vmin
        scale = scale_f32.to(torch.float16)
        zero = zero_f32.to(torch.float16)
        # Quantize: nibble = round((val - zero) / scale)
        quantized = ((value_fp32 - zero.float()) / scale.float()).round().clamp(0, 15).int()

    # Pack two nibbles per byte: low nibble = even dim, high nibble = odd dim
    even = quantized[0::2]  # dims 0,2,4,...
    odd = quantized[1::2]   # dims 1,3,5,...
    packed = (odd.to(torch.uint8) << 4) | even.to(torch.uint8)
    return packed, scale, zero


def _dequantize_value_int4(
    packed: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor
) -> torch.Tensor:
    """Dequantize 4-bit packed values back to float32 [D].

    This mirrors the HIP kernel's dequantization logic.
    """
    result = torch.zeros(HEAD_DIM, dtype=torch.float32, device=packed.device)
    for i in range(packed.shape[0]):
        low_nibble = (packed[i].item()) & 0xF
        high_nibble = (packed[i].item() >> 4) & 0xF
        result[2 * i] = low_nibble * scale.float().item() + zero.float().item()
        result[2 * i + 1] = high_nibble * scale.float().item() + zero.float().item()
    return result


def _build_tq_kv_cache(
    keys: torch.Tensor,     # [num_tokens, Hk, D] float32
    values: torch.Tensor,   # [num_tokens, Hk, D] float32
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack keys/values into TQ k8v4 KV cache format.

    Returns:
        kv_cache: [num_blocks, block_size, Hk, SLOT_SIZE] uint8
        block_table: [1, num_blocks_used] int32 (identity mapping)
    """
    device = keys.device
    num_tokens = keys.shape[0]
    kv_cache = torch.zeros(
        num_blocks, block_size, num_kv_heads, SLOT_SIZE,
        dtype=torch.uint8, device=device,
    )

    for t in range(num_tokens):
        block_idx = t // block_size
        block_offset = t % block_size
        for h in range(num_kv_heads):
            slot = kv_cache[block_idx, block_offset, h]
            # Pack key as FP8
            key_fp8 = _quantize_key_fp8(keys[t, h])
            slot[:KEY_PACKED_SIZE] = key_fp8
            # Pack value as 4-bit
            val_packed, val_scale, val_zero = _quantize_value_int4(values[t, h])
            slot[VAL_DATA_OFFSET:VAL_DATA_OFFSET + HEAD_DIM // 2] = val_packed
            # Store scale and zero as fp16 packed into 4 bytes
            scale_bytes = val_scale.view(torch.uint8)
            zero_bytes = val_zero.view(torch.uint8)
            slot[VAL_SCALE_OFFSET] = scale_bytes[0]
            slot[VAL_SCALE_OFFSET + 1] = scale_bytes[1]
            slot[VAL_SCALE_OFFSET + 2] = zero_bytes[0]
            slot[VAL_SCALE_OFFSET + 3] = zero_bytes[1]

    num_blocks_used = (num_tokens + block_size - 1) // block_size
    block_table = torch.arange(
        num_blocks_used, dtype=torch.int32, device=device
    ).unsqueeze(0)

    return kv_cache, block_table


def _ref_paged_attention_tq(
    query: torch.Tensor,      # [B, Hq, D] float32
    kv_cache: torch.Tensor,   # [num_blocks, block_size, Hk, SLOT_SIZE] uint8
    block_table: torch.Tensor, # [B, max_blocks] int32
    seq_lens: torch.Tensor,    # [B] int32
    scale: float,
    num_kv_heads: int,
) -> torch.Tensor:
    """PyTorch reference implementation of paged attention with TQ k8v4 cache.

    Performs standard scaled dot-product attention by dequantizing keys and
    values from the TQ k8v4 packed format.
    """
    B, Hq, D = query.shape
    gqa_ratio = Hq // num_kv_heads
    output = torch.zeros_like(query)

    for b in range(B):
        ctx_len = seq_lens[b].item()
        if ctx_len <= 0:
            continue

        for kv_h in range(num_kv_heads):
            # Gather keys and values for this KV head
            all_keys = []
            all_values = []
            for t in range(ctx_len):
                block_idx = block_table[b, t // BLOCK_SIZE].item()
                block_offset = t % BLOCK_SIZE
                slot = kv_cache[block_idx, block_offset, kv_h]

                # Dequantize key
                key_bytes = slot[:KEY_PACKED_SIZE]
                key_f32 = _dequantize_key_fp8(key_bytes)
                all_keys.append(key_f32)

                # Dequantize value
                val_packed = slot[VAL_DATA_OFFSET:VAL_DATA_OFFSET + D // 2]
                scale_bytes = slot[VAL_SCALE_OFFSET:VAL_SCALE_OFFSET + 2]
                zero_bytes = slot[VAL_SCALE_OFFSET + 2:VAL_SCALE_OFFSET + 4]
                val_scale = scale_bytes.view(torch.float16)
                val_zero = zero_bytes.view(torch.float16)
                val_f32 = _dequantize_value_int4(val_packed, val_scale, val_zero)
                all_values.append(val_f32)

            K = torch.stack(all_keys)   # [ctx_len, D]
            V = torch.stack(all_values) # [ctx_len, D]

            # Compute attention for each Q head mapped to this KV head
            for q_offset in range(gqa_ratio):
                q_h = kv_h * gqa_ratio + q_offset
                q = query[b, q_h].float()  # [D]

                # scores = Q @ K^T * scale
                scores = (q @ K.T) * scale  # [ctx_len]
                # softmax
                scores = scores - scores.max()
                weights = torch.exp(scores)
                weights = weights / weights.sum()
                # weighted sum
                output[b, q_h] = (weights.unsqueeze(-1) * V).sum(0)

    return output


@pytest.fixture(autouse=True)
def skip_if_unsupported():
    """Skip tests if the GPU doesn't support the TQ k8v4 kernel."""
    if not _check_arch_supported():
        pytest.skip(
            "TQ k8v4 HIP kernel requires gfx942 or gfx950 GPU"
        )


def _load_kernel():
    """Load the TQ k8v4 kernel, skip test if unavailable."""
    from vllm.v1.attention.ops.tq_k8v4_rocm_decode import (
        _load_tq_k8v4_kernel,
    )
    if not _load_tq_k8v4_kernel():
        pytest.skip("TQ k8v4 HIP kernel failed to load")


# ─── Test Cases ────────────────────────────────────────────────────────


@pytest.mark.parametrize("B", [1, 4, 16])
@pytest.mark.parametrize("seq_len", [32, 128, 512])
@pytest.mark.parametrize("num_q_heads,num_kv_heads", [(32, 8), (8, 8), (16, 1)])
def test_tq_k8v4_vs_reference(B, seq_len, num_q_heads, num_kv_heads):
    """Compare TQ k8v4 HIP kernel output against PyTorch reference."""
    _load_kernel()

    from vllm.v1.attention.ops.tq_k8v4_rocm_decode import (
        tq_k8v4_rocm_decode_attention,
    )

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    scale = 1.0 / (HEAD_DIM ** 0.5)

    torch.manual_seed(42)

    # Generate random query, keys, values
    query = torch.randn(B, num_q_heads, HEAD_DIM, dtype=dtype, device=device)

    num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE + 4  # extra padding
    keys_f32 = torch.randn(seq_len, num_kv_heads, HEAD_DIM, device=device)
    values_f32 = torch.randn(seq_len, num_kv_heads, HEAD_DIM, device=device)

    # Build TQ k8v4 KV cache
    kv_cache, block_table_single = _build_tq_kv_cache(
        keys_f32, values_f32, num_blocks, BLOCK_SIZE, num_kv_heads
    )
    # Expand block table for batch
    max_blocks = block_table_single.shape[1]
    block_table = block_table_single.expand(B, -1).contiguous()

    seq_lens = torch.full((B,), seq_len, dtype=torch.int32, device=device)

    # Run HIP kernel
    output_hip = tq_k8v4_rocm_decode_attention(
        query, kv_cache, block_table, seq_lens,
        scale=scale,
        key_packed_size=KEY_PACKED_SIZE,
        max_num_kv_splits=32,
    )

    # Run reference
    output_ref = _ref_paged_attention_tq(
        query.float().cpu(), kv_cache.cpu(), block_table.cpu(),
        seq_lens.cpu(), scale, num_kv_heads,
    )

    # Compare with relaxed tolerance due to FP8/INT4 quantization
    # and BF16 accumulation differences.
    output_hip_f32 = output_hip.float().cpu()
    torch.testing.assert_close(
        output_hip_f32, output_ref,
        atol=0.15, rtol=0.1,
        msg=lambda msg: (
            f"TQ k8v4 HIP vs reference mismatch "
            f"(B={B}, seq_len={seq_len}, "
            f"Hq={num_q_heads}, Hk={num_kv_heads}): {msg}"
        ),
    )


@pytest.mark.parametrize("B", [1, 2])
def test_tq_k8v4_variable_seq_lens(B):
    """Test with different sequence lengths per batch element."""
    _load_kernel()

    from vllm.v1.attention.ops.tq_k8v4_rocm_decode import (
        tq_k8v4_rocm_decode_attention,
    )

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    num_q_heads = 32
    num_kv_heads = 8
    scale = 1.0 / (HEAD_DIM ** 0.5)
    max_seq_len = 256

    torch.manual_seed(123)

    query = torch.randn(B, num_q_heads, HEAD_DIM, dtype=dtype, device=device)

    num_blocks = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE + 4
    keys_f32 = torch.randn(max_seq_len, num_kv_heads, HEAD_DIM, device=device)
    values_f32 = torch.randn(max_seq_len, num_kv_heads, HEAD_DIM, device=device)

    kv_cache, block_table_single = _build_tq_kv_cache(
        keys_f32, values_f32, num_blocks, BLOCK_SIZE, num_kv_heads
    )
    block_table = block_table_single.expand(B, -1).contiguous()

    # Variable sequence lengths
    if B == 1:
        seq_lens = torch.tensor([64], dtype=torch.int32, device=device)
    else:
        seq_lens = torch.tensor([64, 128], dtype=torch.int32, device=device)

    output_hip = tq_k8v4_rocm_decode_attention(
        query, kv_cache, block_table, seq_lens,
        scale=scale,
        key_packed_size=KEY_PACKED_SIZE,
        max_num_kv_splits=32,
    )

    output_ref = _ref_paged_attention_tq(
        query.float().cpu(), kv_cache.cpu(), block_table.cpu(),
        seq_lens.cpu(), scale, num_kv_heads,
    )

    torch.testing.assert_close(
        output_hip.float().cpu(), output_ref,
        atol=0.15, rtol=0.1,
    )


def test_tq_k8v4_fp16_input():
    """Test with fp16 query input (vs default bf16)."""
    _load_kernel()

    from vllm.v1.attention.ops.tq_k8v4_rocm_decode import (
        tq_k8v4_rocm_decode_attention,
    )

    device = torch.device("cuda:0")
    B = 2
    num_q_heads = 8
    num_kv_heads = 8
    seq_len = 64
    scale = 1.0 / (HEAD_DIM ** 0.5)

    torch.manual_seed(7)

    query = torch.randn(B, num_q_heads, HEAD_DIM, dtype=torch.float16, device=device)
    num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE + 2
    keys_f32 = torch.randn(seq_len, num_kv_heads, HEAD_DIM, device=device)
    values_f32 = torch.randn(seq_len, num_kv_heads, HEAD_DIM, device=device)

    kv_cache, block_table_single = _build_tq_kv_cache(
        keys_f32, values_f32, num_blocks, BLOCK_SIZE, num_kv_heads
    )
    block_table = block_table_single.expand(B, -1).contiguous()
    seq_lens = torch.full((B,), seq_len, dtype=torch.int32, device=device)

    output_hip = tq_k8v4_rocm_decode_attention(
        query, kv_cache, block_table, seq_lens,
        scale=scale,
        key_packed_size=KEY_PACKED_SIZE,
        max_num_kv_splits=32,
    )

    assert output_hip.dtype == torch.float16
    assert output_hip.shape == (B, num_q_heads, HEAD_DIM)

    output_ref = _ref_paged_attention_tq(
        query.float().cpu(), kv_cache.cpu(), block_table.cpu(),
        seq_lens.cpu(), scale, num_kv_heads,
    )

    torch.testing.assert_close(
        output_hip.float().cpu(), output_ref,
        atol=0.15, rtol=0.1,
    )


def test_tq_k8v4_output_buffer_reuse():
    """Test with pre-allocated output buffer."""
    _load_kernel()

    from vllm.v1.attention.ops.tq_k8v4_rocm_decode import (
        tq_k8v4_rocm_decode_attention,
    )

    device = torch.device("cuda:0")
    B = 4
    num_q_heads = 16
    num_kv_heads = 4
    seq_len = 128
    scale = 1.0 / (HEAD_DIM ** 0.5)

    torch.manual_seed(99)

    query = torch.randn(B, num_q_heads, HEAD_DIM, dtype=torch.bfloat16, device=device)
    num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE + 2
    keys_f32 = torch.randn(seq_len, num_kv_heads, HEAD_DIM, device=device)
    values_f32 = torch.randn(seq_len, num_kv_heads, HEAD_DIM, device=device)

    kv_cache, block_table_single = _build_tq_kv_cache(
        keys_f32, values_f32, num_blocks, BLOCK_SIZE, num_kv_heads
    )
    block_table = block_table_single.expand(B, -1).contiguous()
    seq_lens = torch.full((B,), seq_len, dtype=torch.int32, device=device)

    # Pre-allocate output buffer (larger than needed)
    output_buf = torch.empty(
        B + 4, num_q_heads, HEAD_DIM,
        dtype=torch.bfloat16, device=device,
    )

    output = tq_k8v4_rocm_decode_attention(
        query, kv_cache, block_table, seq_lens,
        scale=scale,
        key_packed_size=KEY_PACKED_SIZE,
        output_buf=output_buf,
    )

    assert output.data_ptr() == output_buf.data_ptr()
    assert output.shape == (B, num_q_heads, HEAD_DIM)


# ─── Validation Tests (should raise) ──────────────────────────────────


def test_tq_k8v4_head_dim_validation():
    """Verify ValueError when head_dim != 128."""
    _load_kernel()

    from vllm.v1.attention.ops.tq_k8v4_rocm_decode import (
        tq_k8v4_rocm_decode_attention,
    )

    device = torch.device("cuda:0")
    # head_dim = 64 (unsupported)
    query = torch.randn(1, 8, 64, dtype=torch.bfloat16, device=device)
    kv_cache = torch.zeros(4, 16, 8, SLOT_SIZE, dtype=torch.uint8, device=device)
    block_table = torch.zeros(1, 4, dtype=torch.int32, device=device)
    seq_lens = torch.tensor([16], dtype=torch.int32, device=device)

    with pytest.raises(ValueError, match="head_dim=128"):
        tq_k8v4_rocm_decode_attention(
            query, kv_cache, block_table, seq_lens,
            scale=0.1, key_packed_size=64,
        )


def test_tq_k8v4_gqa_ratio_validation():
    """Verify ValueError when GQA ratio > 16."""
    _load_kernel()

    from vllm.v1.attention.ops.tq_k8v4_rocm_decode import (
        tq_k8v4_rocm_decode_attention,
    )

    device = torch.device("cuda:0")
    # Hq=64, Hk=2 → GQA ratio=32 (exceeds 16)
    query = torch.randn(1, 64, HEAD_DIM, dtype=torch.bfloat16, device=device)
    kv_cache = torch.zeros(4, 16, 2, SLOT_SIZE, dtype=torch.uint8, device=device)
    block_table = torch.zeros(1, 4, dtype=torch.int32, device=device)
    seq_lens = torch.tensor([16], dtype=torch.int32, device=device)

    with pytest.raises(ValueError, match="GQA ratio"):
        tq_k8v4_rocm_decode_attention(
            query, kv_cache, block_table, seq_lens,
            scale=0.1, key_packed_size=KEY_PACKED_SIZE,
        )


def test_tq_k8v4_arch_guard():
    """Verify is_tq_k8v4_supported() returns True on supported hardware."""
    from vllm.v1.attention.ops.tq_k8v4_rocm_decode import (
        is_tq_k8v4_supported,
    )
    # On the test machine (gfx942/gfx950), this should be True.
    # The skip_if_unsupported fixture already ensures we're on the right arch.
    assert is_tq_k8v4_supported()


@pytest.mark.parametrize("B", [1, 8, 32])
@pytest.mark.parametrize("seq_len", [64, 512, 2048])
def test_tq_k8v4_large_batch_mfma_path(B, seq_len):
    """Exercise the MFMA GQA kernel path (B > 8)."""
    _load_kernel()

    from vllm.v1.attention.ops.tq_k8v4_rocm_decode import (
        tq_k8v4_rocm_decode_attention,
    )

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    num_q_heads = 32
    num_kv_heads = 8
    scale = 1.0 / (HEAD_DIM ** 0.5)

    torch.manual_seed(42)

    query = torch.randn(B, num_q_heads, HEAD_DIM, dtype=dtype, device=device)
    num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE + 4
    keys_f32 = torch.randn(seq_len, num_kv_heads, HEAD_DIM, device=device)
    values_f32 = torch.randn(seq_len, num_kv_heads, HEAD_DIM, device=device)

    kv_cache, block_table_single = _build_tq_kv_cache(
        keys_f32, values_f32, num_blocks, BLOCK_SIZE, num_kv_heads
    )
    block_table = block_table_single.expand(B, -1).contiguous()
    seq_lens_t = torch.full((B,), seq_len, dtype=torch.int32, device=device)

    output = tq_k8v4_rocm_decode_attention(
        query, kv_cache, block_table, seq_lens_t,
        scale=scale,
        key_packed_size=KEY_PACKED_SIZE,
        max_num_kv_splits=32,
    )

    assert output.shape == (B, num_q_heads, HEAD_DIM)
    assert output.dtype == dtype
    # Sanity check: output should not be all zeros or NaN
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    assert output.abs().sum() > 0, "Output is all zeros"

    # For small enough configs, also check against reference
    if B <= 4 and seq_len <= 512:
        output_ref = _ref_paged_attention_tq(
            query.float().cpu(), kv_cache.cpu(), block_table.cpu(),
            seq_lens_t.cpu(), scale, num_kv_heads,
        )
        torch.testing.assert_close(
            output.float().cpu(), output_ref,
            atol=0.15, rtol=0.1,
        )
