# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.v1.attention.backends.mla.sparse_utils import (
    build_rotated_dcp_peer_block_table,
)

# DeepSeek V3 MLA dimensions
NOPE_DIM = 512  # NoPE latent dimension (FP8 quantized in cache)
ROPE_DIM = 64  # RoPE dimension (stored as BF16 in cache)
NUM_TILES = 4  # NOPE_DIM / GROUP_SIZE = 512 / 128
GROUP_SIZE = 128  # FP8 quantization group size (one scale per group)
ENTRY_BYTES = 656  # 512 (FP8) + 16 (4×float32 scales) + 128 (64×BF16 RoPE)


def _build_test_case(seq_lens, block_size, seed=42):
    """Build a synthetic FP8 cache and compute the expected BF16 output.

    This simulates what concat_and_cache_ds_mla_kernel writes into the
    KV cache, then computes what cp_gather_and_upconvert should produce.

    Args:
        seq_lens: List of sequence lengths, one per request.
        block_size: Number of tokens per physical cache block.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (cache, block_table, workspace_starts_t, num_reqs,
                  total_tokens, expected_output).
    """
    torch.manual_seed(seed)

    num_reqs = len(seq_lens)
    total_tokens = sum(seq_lens)

    # workspace_starts[r] = sum of seq_lens[0..r-1]
    # This tells the kernel where in the output buffer each request's
    # gathered tokens should be written.
    workspace_starts = []
    s = 0
    for sl in seq_lens:
        workspace_starts.append(s)
        s += sl

    # How many physical cache blocks each request needs
    blocks_per_req = [math.ceil(s / block_size) for s in seq_lens]
    total_blocks = sum(blocks_per_req)
    max_blocks = max(blocks_per_req)

    # Block table maps (request, logical_block_idx) -> physical_block_id.
    # Here we assign blocks contiguously: request 0 gets blocks [0, 1, ...],
    # request 1 gets the next set, etc.
    block_table = torch.zeros(num_reqs, max_blocks, dtype=torch.int32, device="cuda")
    block_idx = 0
    for r in range(num_reqs):
        for b in range(blocks_per_req[r]):
            block_table[r, b] = block_idx
            block_idx += 1

    # The raw paged cache: [num_blocks, block_size, 656] as uint8
    cache = torch.zeros(
        total_blocks, block_size, ENTRY_BYTES, dtype=torch.uint8, device="cuda"
    )
    # Expected kernel output: [total_tokens, 576] as BF16
    expected = torch.zeros(
        total_tokens, NOPE_DIM + ROPE_DIM, dtype=torch.bfloat16, device="cuda"
    )

    # Fill each token's cache entry and compute expected output
    for r in range(num_reqs):
        for t in range(seq_lens[r]):
            out_idx = workspace_starts[r] + t
            # Map token position -> (physical_block, offset_within_block)
            phys = block_table[r, t // block_size].item()
            off = t % block_size

            # --- NoPE section: 4 tiles of 128 FP8 values, each with a scale ---
            for tile in range(NUM_TILES):
                start = tile * GROUP_SIZE

                # Generate random data and quantize to FP8 e4m3
                fp8_vals = torch.randn(GROUP_SIZE, device="cuda").to(
                    torch.float8_e4m3fn
                )
                # Pack FP8 bytes into cache at bytes [start : start+128]
                cache[phys, off, start : start + GROUP_SIZE] = fp8_vals.view(
                    torch.uint8
                )

                # Random positive scale in [0.1, 2.1]
                scale = (torch.rand(1, device="cuda") * 2.0 + 0.1).item()
                scale_t = torch.tensor([scale], dtype=torch.float32, device="cuda")
                # Pack scale as 4 raw bytes at bytes [512 + tile*4 : ...]
                cache[phys, off, NOPE_DIM + tile * 4 : NOPE_DIM + (tile + 1) * 4] = (
                    scale_t.view(torch.uint8)
                )

                # Reference dequant: fp8 -> float32, multiply scale, -> bf16.
                # This matches the CUDA path: fp8 -> half -> float * scale -> bf16.
                # (fp8 -> half is exact, half -> float is exact, so fp8 -> float
                # gives the same result regardless of intermediate type.)
                expected[out_idx, start : start + GROUP_SIZE] = (
                    fp8_vals.float() * scale
                ).bfloat16()

            # --- RoPE section: 64 BF16 values, direct copy (no dequant) ---
            rope = torch.randn(ROPE_DIM, dtype=torch.bfloat16, device="cuda")
            # Pack RoPE bytes into cache at bytes [528 : 656]
            cache[phys, off, NOPE_DIM + 16 :] = rope.view(torch.uint8)
            # Expected output: exact copy
            expected[out_idx, NOPE_DIM:] = rope

    workspace_starts_t = torch.tensor(
        workspace_starts, dtype=torch.int32, device="cuda"
    )

    return (
        cache,
        block_table,
        workspace_starts_t,
        num_reqs,
        total_tokens,
        expected,
    )


def _build_test_case_fast(seq_lens, block_size, seed=42):
    """Vectorized test-case builder for large sequence lengths.

    Same logic as _build_test_case but uses tensor operations instead of
    per-token Python loops, making it practical for seq_lens up to 128K+.
    """
    torch.manual_seed(seed)

    num_reqs = len(seq_lens)
    total_tokens = sum(seq_lens)

    workspace_starts = []
    s = 0
    for sl in seq_lens:
        workspace_starts.append(s)
        s += sl

    blocks_per_req = [math.ceil(sl / block_size) for sl in seq_lens]
    total_blocks = sum(blocks_per_req)
    max_blocks = max(blocks_per_req)

    # Contiguous block allocation
    block_table = torch.zeros(num_reqs, max_blocks, dtype=torch.int32, device="cuda")
    block_idx = 0
    for r in range(num_reqs):
        for b in range(blocks_per_req[r]):
            block_table[r, b] = block_idx
            block_idx += 1

    cache = torch.zeros(
        total_blocks, block_size, ENTRY_BYTES, dtype=torch.uint8, device="cuda"
    )

    # Generate all data vectorized
    nope_fp8 = torch.randn(total_tokens, NOPE_DIM, device="cuda").to(
        torch.float8_e4m3fn
    )
    scales = (torch.rand(total_tokens, NUM_TILES, device="cuda") * 2.0 + 0.1).float()
    rope = torch.randn(total_tokens, ROPE_DIM, dtype=torch.bfloat16, device="cuda")

    # Compute expected output vectorized (same dequant logic as kernel)
    expected = torch.zeros(
        total_tokens, NOPE_DIM + ROPE_DIM, dtype=torch.bfloat16, device="cuda"
    )
    for tile in range(NUM_TILES):
        start = tile * GROUP_SIZE
        expected[:, start : start + GROUP_SIZE] = (
            nope_fp8[:, start : start + GROUP_SIZE].float() * scales[:, tile : tile + 1]
        ).bfloat16()
    expected[:, NOPE_DIM:] = rope

    # Build per-token cache entries as [total_tokens, 656] uint8
    token_data = torch.zeros(
        total_tokens, ENTRY_BYTES, dtype=torch.uint8, device="cuda"
    )
    token_data[:, :NOPE_DIM] = nope_fp8.view(torch.uint8)
    token_data[:, NOPE_DIM : NOPE_DIM + 16] = scales.view(torch.uint8)
    token_data[:, NOPE_DIM + 16 :] = rope.view(torch.uint8)

    # Scatter into paged cache (loop over requests, not tokens)
    block_start = 0
    for r in range(num_reqs):
        sl = seq_lens[r]
        nb = blocks_per_req[r]
        ws = workspace_starts[r]
        flat_cache = cache[block_start : block_start + nb].reshape(-1, ENTRY_BYTES)
        flat_cache[:sl] = token_data[ws : ws + sl]
        block_start += nb

    workspace_starts_t = torch.tensor(
        workspace_starts, dtype=torch.int32, device="cuda"
    )

    return (
        cache,
        block_table,
        workspace_starts_t,
        num_reqs,
        total_tokens,
        expected,
    )


@pytest.mark.parametrize(
    "seq_lens,block_size",
    [
        # Production block_size=64 (only supported value for FlashMLA sparse).
        # Realistic prefill scenarios with varying request counts.
        ([1], 64),  # single token edge case
        ([64], 64),  # 1 req, exactly one block
        ([128], 64),  # 1 req, crosses block boundary
        ([512], 64),  # 1 req, longer prefill
        ([256, 128, 384], 64),  # 3 reqs, varying lengths
        ([128] * 4, 64),  # 4 reqs, equal lengths
        ([64] * 16, 64),  # 16 reqs, shorter prefills
    ],
)
def test_cp_gather_and_upconvert_fp8_kv_cache(seq_lens, block_size):
    """Core correctness test: build cache, run kernel, compare output."""
    (
        cache,
        block_table,
        workspace_starts_t,
        num_reqs,
        total_tokens,
        expected,
    ) = _build_test_case(seq_lens, block_size)

    dst = torch.zeros(
        total_tokens, NOPE_DIM + ROPE_DIM, dtype=torch.bfloat16, device="cuda"
    )

    ops.cp_gather_and_upconvert_fp8_kv_cache(
        cache, dst, block_table, workspace_starts_t, num_reqs
    )

    # NoPE: fp8 dequant has rounding error, so we allow small tolerance.
    # The fp8 -> float -> bf16 path can differ by up to ~1 ULP of bf16.
    torch.testing.assert_close(
        dst[:, :NOPE_DIM], expected[:, :NOPE_DIM], atol=1e-3, rtol=1e-2
    )

    # RoPE: pure bf16 copy, must be bit-exact
    assert torch.equal(dst[:, NOPE_DIM:], expected[:, NOPE_DIM:])


def test_cp_gather_fp8_shuffled_blocks():
    """Test that the kernel correctly follows the block table when
    physical blocks are non-contiguous and out of order.

    Here we allocate 4 physical blocks but map the request's 2 logical
    blocks to physical blocks [3, 1] (reversed, with gaps).
    """
    torch.manual_seed(123)
    block_size = 4
    seq_lens = [8]  # needs 2 blocks (tokens 0-3 in block 0, 4-7 in block 1)
    total_tokens = 8

    # 4 physical blocks, but only blocks 3 and 1 are used (in that order).
    # Tokens 0-3 -> physical block 3, tokens 4-7 -> physical block 1.
    num_phys_blocks = 4
    cache = torch.zeros(
        num_phys_blocks, block_size, ENTRY_BYTES, dtype=torch.uint8, device="cuda"
    )
    block_table = torch.tensor([[3, 1]], dtype=torch.int32, device="cuda")
    workspace_starts = torch.tensor([0], dtype=torch.int32, device="cuda")
    expected = torch.zeros(
        total_tokens, NOPE_DIM + ROPE_DIM, dtype=torch.bfloat16, device="cuda"
    )

    # Fill cache at the shuffled physical locations
    for t in range(total_tokens):
        # Follow the same block_table lookup the kernel will use
        phys = block_table[0, t // block_size].item()
        off = t % block_size

        for tile in range(NUM_TILES):
            start = tile * GROUP_SIZE
            fp8_vals = torch.randn(GROUP_SIZE, device="cuda").to(torch.float8_e4m3fn)
            cache[phys, off, start : start + GROUP_SIZE] = fp8_vals.view(torch.uint8)

            # Use a fixed scale to keep this test simple
            scale = 1.5
            scale_t = torch.tensor([scale], dtype=torch.float32, device="cuda")
            cache[phys, off, NOPE_DIM + tile * 4 : NOPE_DIM + (tile + 1) * 4] = (
                scale_t.view(torch.uint8)
            )

            expected[t, start : start + GROUP_SIZE] = (
                fp8_vals.float() * scale
            ).bfloat16()

        rope = torch.randn(ROPE_DIM, dtype=torch.bfloat16, device="cuda")
        cache[phys, off, NOPE_DIM + 16 :] = rope.view(torch.uint8)
        expected[t, NOPE_DIM:] = rope

    dst = torch.zeros(
        total_tokens, NOPE_DIM + ROPE_DIM, dtype=torch.bfloat16, device="cuda"
    )

    ops.cp_gather_and_upconvert_fp8_kv_cache(
        cache, dst, block_table, workspace_starts, len(seq_lens)
    )

    torch.testing.assert_close(
        dst[:, :NOPE_DIM], expected[:, :NOPE_DIM], atol=1e-3, rtol=1e-2
    )
    assert torch.equal(dst[:, NOPE_DIM:], expected[:, NOPE_DIM:])


def test_cp_gather_fp8_rank_major_owner_translated_full_prefix():
    """Compose the owner block-table translation with FP8 prefix gathering.

    A single CUDA allocation models the flattened rank-major VMM view.  Each of
    four owner segments has padding between its two shuffled physical blocks.
    The translated table then gathers a full eight-page logical prefix through
    all four segments.
    """
    block_size = 64
    dcp_size = 4
    peer_block_stride = 6
    owner_local_blocks = torch.tensor([[2, 0]], dtype=torch.int32, device="cuda")
    owner_block_tables = owner_local_blocks.unsqueeze(0).expand(dcp_size, -1, -1)
    translated_block_table = build_rotated_dcp_peer_block_table(
        owner_block_tables,
        local_rank=0,
        peer_block_stride=peer_block_stride,
        cp_kv_cache_interleave_size=block_size,
        block_size=block_size,
        BLOCK_N=8,
    )
    expected_blocks = torch.tensor(
        [[2, 8, 14, 20, 0, 6, 12, 18]],
        dtype=torch.int32,
        device="cuda",
    )
    torch.testing.assert_close(translated_block_table, expected_blocks)

    num_logical_pages = expected_blocks.shape[1]
    full_prefix_len = num_logical_pages * block_size
    (
        canonical_cache,
        _canonical_block_table,
        _canonical_workspace_starts,
        _num_reqs,
        total_tokens,
        expected,
    ) = _build_test_case([full_prefix_len], block_size, seed=314)
    assert total_tokens == full_prefix_len

    # Model a rank-major peer view with allocation padding.  Copy each logical
    # page into the translated owner segment while leaving every other block as
    # a decoy.  Reading an unrotated or unpadded block ID therefore cannot
    # accidentally match the reference.
    peer_cache = torch.full(
        (dcp_size * peer_block_stride, block_size, ENTRY_BYTES),
        0xA5,
        dtype=torch.uint8,
        device="cuda",
    )
    for logical_page, peer_block in enumerate(expected_blocks[0].tolist()):
        peer_cache[peer_block].copy_(canonical_cache[logical_page])

    workspace = torch.empty(
        total_tokens,
        NOPE_DIM + ROPE_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    workspace_starts = torch.zeros(1, dtype=torch.int32, device="cuda")
    ops.cp_gather_and_upconvert_fp8_kv_cache(
        peer_cache,
        workspace,
        translated_block_table,
        workspace_starts,
        1,
    )

    torch.testing.assert_close(
        workspace[:, :NOPE_DIM],
        expected[:, :NOPE_DIM],
        atol=1e-3,
        rtol=1e-2,
    )
    assert torch.equal(workspace[:, NOPE_DIM:], expected[:, NOPE_DIM:])

    # Make the intended boundary coverage explicit: first/last token in a page
    # and the first token after each owner/page transition.
    boundary_rows = torch.tensor(
        [0, 63, 64, 127, 255, 256, 319, 320, 511],
        device="cuda",
    )
    torch.testing.assert_close(
        workspace[boundary_rows],
        expected[boundary_rows],
        atol=1e-3,
        rtol=1e-2,
    )


@pytest.mark.parametrize(
    "gather_seq_lens,seq_starts",
    [
        ([6, 5, 3], [3, 4, 5]),
        ([0, 5, 3], [12, 4, 5]),
    ],
)
def test_cp_gather_fp8_with_sequence_starts(gather_seq_lens, seq_starts):
    """Gather request slices beginning at arbitrary cache positions."""
    full_seq_lens = [12, 11, 9]
    (
        cache,
        block_table,
        _workspace_starts_t,
        num_reqs,
        _total_tokens,
        full_expected,
    ) = _build_test_case(full_seq_lens, block_size=4)

    workspace_starts = torch.tensor(
        [0, gather_seq_lens[0], sum(gather_seq_lens[:2])],
        dtype=torch.int32,
        device="cuda",
    )
    seq_starts_t = torch.tensor(seq_starts, dtype=torch.int32, device="cuda")
    dst = torch.empty(
        sum(gather_seq_lens),
        NOPE_DIM + ROPE_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )

    ops.cp_gather_and_upconvert_fp8_kv_cache(
        cache,
        dst,
        block_table,
        workspace_starts,
        num_reqs,
        seq_starts_t,
    )

    full_workspace_starts = [0, full_seq_lens[0], sum(full_seq_lens[:2])]
    expected = torch.cat(
        [
            full_expected[
                full_workspace_starts[i] + seq_starts[i] : full_workspace_starts[i]
                + seq_starts[i]
                + gather_seq_lens[i]
            ]
            for i in range(num_reqs)
        ]
    )
    torch.testing.assert_close(
        dst[:, :NOPE_DIM], expected[:, :NOPE_DIM], atol=1e-3, rtol=1e-2
    )
    assert torch.equal(dst[:, NOPE_DIM:], expected[:, NOPE_DIM:])


@pytest.mark.parametrize(
    "seq_lens,block_size",
    [
        # Large sequence lengths matching end-to-end benchmark scenarios.
        # Uses vectorized builder since per-token Python loops would be too slow.
        ([8000], 64),
        ([16000], 64),
        ([32000], 64),
        ([64000], 64),
        ([96000], 64),
        ([128000], 64),
    ],
)
def test_cp_gather_fp8_large_seqlens(seq_lens, block_size):
    """Correctness test with large sequence lengths matching benchmark
    scenarios (8K-128K prefill)."""
    (
        cache,
        block_table,
        workspace_starts_t,
        num_reqs,
        total_tokens,
        expected,
    ) = _build_test_case_fast(seq_lens, block_size)

    dst = torch.zeros(
        total_tokens, NOPE_DIM + ROPE_DIM, dtype=torch.bfloat16, device="cuda"
    )

    ops.cp_gather_and_upconvert_fp8_kv_cache(
        cache, dst, block_table, workspace_starts_t, num_reqs
    )

    torch.testing.assert_close(
        dst[:, :NOPE_DIM], expected[:, :NOPE_DIM], atol=1e-3, rtol=1e-2
    )
    assert torch.equal(dst[:, NOPE_DIM:], expected[:, NOPE_DIM:])
