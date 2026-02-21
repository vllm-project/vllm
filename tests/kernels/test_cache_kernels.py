# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for CUDA kernels in cache_kernels.cu."""

import pytest
import torch

try:
    from vllm import _custom_ops as ops
except ImportError:
    pytest.skip(
        "Could not import vllm._custom_ops. (pip install -e .)", allow_module_level=True
    )


def _is_debug_build() -> bool:
    """Detect whether the C++ extension was built without NDEBUG (debug mode).

    Some metadata checks use GPU reductions with .item() and are gated behind
    #ifndef NDEBUG to avoid host-device sync in production builds.  Tests that
    rely on those checks should be skipped in release builds.
    """
    if not torch.cuda.is_available():
        return False
    try:
        # Trigger a debug-only check: negative seq_starts should raise in
        # debug builds but be silently ignored in release builds.
        e = 576
        ops.gather_cache(
            torch.randn((4, 16, e), dtype=torch.float16, device="cuda"),
            torch.empty((1, e), dtype=torch.float16, device="cuda"),
            torch.zeros((1, 4), dtype=torch.int32, device="cuda"),
            torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
            torch.zeros((1,), dtype=torch.int32, device="cuda"),
            1,
            -1,
            "auto",
            torch.tensor([1.0], dtype=torch.float32, device="cuda"),
            torch.tensor([-1], dtype=torch.int32, device="cuda"),
        )
        return False  # No error → release build (check was compiled out)
    except RuntimeError:
        return True  # Error fired → debug build


requires_debug_build = pytest.mark.skipif(
    not _is_debug_build(),
    reason="Metadata value checks are gated behind #ifndef NDEBUG",
)


@requires_debug_build
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Need CUDA device")
def test_gather_cache_oob():
    """
    Tests for OOB read in gather_cache token-major mode (Issue #27909).
    This test constructs a boundary case identified in the issue where
    seq_starts causes the block_table offset to read out of bounds.
    """

    block_size = 64
    entry_size = 128

    block_table = torch.tensor([[1, 2]], dtype=torch.int32, device="cuda")

    # This will result in offset = 128 / block_size = 128 / 64 = 2
    # This will cause the kernel to try to read from
    # block_table[0, 2], but its size is only 2.
    seq_starts = torch.tensor([128], dtype=torch.int32, device="cuda")

    seq_len = 65
    cu_seq_lens = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")

    # src_cache: [num_blocks, block_size, entry_size]
    num_blocks = 5
    src_cache = torch.randn(
        (num_blocks, block_size, entry_size), dtype=torch.float16, device="cuda"
    )

    dst = torch.empty((seq_len, entry_size), dtype=torch.float16, device="cuda")

    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    token_to_seq = torch.zeros((seq_len,), dtype=torch.int32, device="cuda")

    # The metadata drives an OOB read: (seq_starts[0] + seq_len) / block_size
    # = (128 + 65) / 64 = 4 blocks needed, but block_table width is only 2.
    # The host-side bounds check must catch this before kernel launch.
    with pytest.raises(RuntimeError, match="exceeds block_table width"):
        ops.gather_cache(
            src_cache,
            dst,
            block_table,
            cu_seq_lens,
            token_to_seq,
            seq_len,
            -1,
            "auto",  # kv_cache_dtype
            scale,
            seq_starts,
        )


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Need CUDA device")
def test_gather_cache_token_major_zero_tokens_noop():
    block_size = 16
    entry_size = 576
    src_cache = torch.randn(
        (4, block_size, entry_size), dtype=torch.float16, device="cuda"
    )
    dst = torch.empty((0, entry_size), dtype=torch.float16, device="cuda")
    block_table = torch.zeros((1, 4), dtype=torch.int32, device="cuda")
    cu_seq_lens = torch.zeros((2,), dtype=torch.int32, device="cuda")
    token_to_seq = torch.empty((0,), dtype=torch.int32, device="cuda")
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    ops.gather_cache(
        src_cache,
        dst,
        block_table,
        cu_seq_lens,
        token_to_seq,
        0,
        -1,
        "auto",
        scale,
        None,
    )
    torch.cuda.synchronize()


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Need CUDA device")
def test_gather_cache_requires_mode_metadata():
    block_size = 16
    entry_size = 576
    src_cache = torch.randn(
        (4, block_size, entry_size), dtype=torch.float16, device="cuda"
    )
    dst = torch.empty((1, entry_size), dtype=torch.float16, device="cuda")
    block_table = torch.zeros((1, 4), dtype=torch.int32, device="cuda")
    cu_seq_lens = torch.tensor([0, 1], dtype=torch.int32, device="cuda")

    with pytest.raises(RuntimeError, match="requires either token-major metadata"):
        ops.gather_cache(
            src_cache,
            dst,
            block_table,
            cu_seq_lens,
            None,
            -1,
            -1,
            "auto",
            None,
            None,
        )


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Need CUDA device")
def test_gather_cache_batch_major_zero_batches_noop():
    src = torch.randn((4, 16, 576), dtype=torch.float16, device="cuda")
    dst = torch.empty((0, 576), dtype=torch.float16, device="cuda")
    bt = torch.zeros((0, 4), dtype=torch.int32, device="cuda")
    cu = torch.zeros((1,), dtype=torch.int32, device="cuda")

    ops.gather_cache(src, dst, bt, cu, None, 0, 0, "auto", None, None)
    torch.cuda.synchronize()


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Need CUDA device")
def test_gather_cache_batch_major_rejects_dtype_mismatch():
    """Batch-major must reject src/dst with different dtypes.

    This must hold even when element_size matches.
    """
    src = torch.randn((4, 16, 576), dtype=torch.float16, device="cuda")
    dst = torch.empty((2, 576), dtype=torch.bfloat16, device="cuda")
    bt = torch.zeros((1, 4), dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, 2], dtype=torch.int32, device="cuda")

    with pytest.raises(RuntimeError, match="same dtype"):
        ops.gather_cache(src, dst, bt, cu, None, 0, 1, "auto", None, None)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Need CUDA device")
def test_gather_cache_token_major_rejects_dst_too_small():
    """Token-major must reject when num_tokens > dst.size(0)."""
    entry_size = 576
    src = torch.randn((4, 16, entry_size), dtype=torch.float16, device="cuda")
    dst = torch.empty((2, entry_size), dtype=torch.float16, device="cuda")
    bt = torch.zeros((1, 4), dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, 5], dtype=torch.int32, device="cuda")
    t2s = torch.zeros((5,), dtype=torch.int32, device="cuda")
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    with pytest.raises(RuntimeError, match="exceeds dst.size"):
        ops.gather_cache(src, dst, bt, cu, t2s, 5, -1, "auto", scale, None)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Need CUDA device")
def test_gather_cache_rejects_bad_seq_starts_length():
    """seq_starts must have length >= batch dimension."""
    entry_size = 576
    src = torch.randn((4, 16, entry_size), dtype=torch.float16, device="cuda")
    dst = torch.empty((3, entry_size), dtype=torch.float16, device="cuda")
    # block_table has 2 batch rows
    bt = torch.zeros((2, 4), dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, 1, 3], dtype=torch.int32, device="cuda")
    t2s = torch.zeros((3,), dtype=torch.int32, device="cuda")
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    # seq_starts has only 1 element but batch dimension is 2
    bad_seq_starts = torch.tensor([0], dtype=torch.int32, device="cuda")

    with pytest.raises(RuntimeError, match="seq_starts length"):
        ops.gather_cache(src, dst, bt, cu, t2s, 3, -1, "auto", scale, bad_seq_starts)


@requires_debug_build
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Need CUDA device")
def test_gather_cache_token_major_rejects_negative_token_to_seq():
    """token_to_seq with negative batch ids must be rejected."""
    entry_size = 576
    src = torch.randn((4, 16, entry_size), dtype=torch.float16, device="cuda")
    dst = torch.empty((2, entry_size), dtype=torch.float16, device="cuda")
    bt = torch.zeros((1, 4), dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    t2s = torch.tensor([-1, 0], dtype=torch.int32, device="cuda")
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    with pytest.raises(RuntimeError, match="out-of-range batch ids"):
        ops.gather_cache(src, dst, bt, cu, t2s, 2, -1, "auto", scale, None)


@requires_debug_build
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Need CUDA device")
def test_gather_cache_token_major_rejects_oob_token_to_seq():
    """token_to_seq with batch ids >= batch_dim must be rejected."""
    entry_size = 576
    src = torch.randn((4, 16, entry_size), dtype=torch.float16, device="cuda")
    dst = torch.empty((2, entry_size), dtype=torch.float16, device="cuda")
    bt = torch.zeros((1, 4), dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    # batch_dim is 1 (block_table has 1 row), but token_to_seq has value 1
    t2s = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    with pytest.raises(RuntimeError, match="out-of-range batch ids"):
        ops.gather_cache(src, dst, bt, cu, t2s, 2, -1, "auto", scale, None)


@requires_debug_build
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Need CUDA device")
def test_gather_cache_batch_major_rejects_block_table_oob():
    """Batch-major must reject when seq_starts + seq_len exceeds block_table width."""
    block_size = 16
    entry_size = 576
    src = torch.randn((4, block_size, entry_size), dtype=torch.float16, device="cuda")
    # block_table width is 1 (only 1 block index per batch row)
    bt = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    # seq_len = 32 tokens → needs ceil(32/16) = 2 blocks, but width is 1
    cu = torch.tensor([0, 32], dtype=torch.int32, device="cuda")
    dst = torch.empty((32, entry_size), dtype=torch.float16, device="cuda")

    with pytest.raises(RuntimeError, match="exceeds block_table width"):
        ops.gather_cache(src, dst, bt, cu, None, 0, 1, "auto", None, None)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Need CUDA device")
def test_gather_cache_token_major_rejects_scale_wrong_device():
    """Token-major must reject when scale is on a different device (CPU)."""
    entry_size = 576
    src = torch.randn((4, 16, entry_size), dtype=torch.float16, device="cuda")
    dst = torch.empty((1, entry_size), dtype=torch.float16, device="cuda")
    bt = torch.zeros((1, 4), dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    t2s = torch.zeros((1,), dtype=torch.int32, device="cuda")
    # scale on CPU
    scale = torch.tensor([1.0], dtype=torch.float32, device="cpu")

    with pytest.raises(RuntimeError, match="same device"):
        ops.gather_cache(src, dst, bt, cu, t2s, 1, -1, "auto", scale, None)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Need CUDA device")
def test_gather_cache_token_major_rejects_token_to_seq_wrong_device():
    """Token-major must reject when token_to_seq is on a different device (CPU)."""
    entry_size = 576
    src = torch.randn((4, 16, entry_size), dtype=torch.float16, device="cuda")
    dst = torch.empty((1, entry_size), dtype=torch.float16, device="cuda")
    bt = torch.zeros((1, 4), dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    # token_to_seq on CPU
    t2s = torch.zeros((1,), dtype=torch.int32, device="cpu")
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    with pytest.raises(RuntimeError, match="same device"):
        ops.gather_cache(src, dst, bt, cu, t2s, 1, -1, "auto", scale, None)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Need CUDA device")
def test_gather_cache_token_major_seq_starts_longer_than_batch_dim():
    """seq_starts with extra entries beyond batch_dim should still work
    (sliced internally) without shape-mismatch errors."""
    entry_size = 576
    block_size = 16
    num_blocks = 8
    src = torch.randn(
        (num_blocks, block_size, entry_size), dtype=torch.float16, device="cuda"
    )
    # 1 batch, 4 tokens
    bt = torch.zeros((1, 4), dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
    t2s = torch.zeros((4,), dtype=torch.int32, device="cuda")
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    dst = torch.empty((4, entry_size), dtype=torch.float16, device="cuda")
    # seq_starts has 3 entries but batch_dim is 1 — extra entries should be ignored
    seq_starts = torch.tensor([0, 99, 99], dtype=torch.int32, device="cuda")

    ops.gather_cache(src, dst, bt, cu, t2s, 4, -1, "auto", scale, seq_starts)
    torch.cuda.synchronize()


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Need CUDA device")
def test_gather_cache_token_major_inconsistent_token_to_seq_mapping():
    """token_to_seq maps a token to a batch whose cu_seq_lens range does not
    contain that token index.  The kernel lower-bound guard must skip it
    (no crash, no OOB write)."""
    entry_size = 576
    block_size = 16
    num_blocks = 8
    src = torch.randn(
        (num_blocks, block_size, entry_size), dtype=torch.float16, device="cuda"
    )
    # 2 batches: batch 0 has tokens [0,2), batch 1 has tokens [2,4)
    bt = torch.zeros((2, 4), dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    dst = torch.zeros((4, entry_size), dtype=torch.float16, device="cuda")
    # token 0 is mapped to batch 1 (cu_seq_lens[1]=2, so token 0 < batch_start=2)
    # kernel should skip this token via the lower-bound guard
    t2s = torch.tensor([1, 0, 1, 1], dtype=torch.int32, device="cuda")

    ops.gather_cache(src, dst, bt, cu, t2s, 4, -1, "auto", scale, None)
    torch.cuda.synchronize()
    # Token 0 was mapped to batch 1 but token_id=0 < batch_start=2,
    # so the kernel skips it; dst[0] should remain zero.
    assert torch.all(dst[0] == 0), "Token with inconsistent mapping should be skipped"


@requires_debug_build
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Need CUDA device")
def test_gather_cache_token_major_rejects_negative_seq_starts():
    """Token-major must reject negative seq_starts values."""
    entry_size = 576
    src = torch.randn((4, 16, entry_size), dtype=torch.float16, device="cuda")
    dst = torch.empty((2, entry_size), dtype=torch.float16, device="cuda")
    bt = torch.zeros((1, 4), dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    t2s = torch.zeros((2,), dtype=torch.int32, device="cuda")
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    bad_starts = torch.tensor([-1], dtype=torch.int32, device="cuda")

    with pytest.raises(RuntimeError, match="non-negative"):
        ops.gather_cache(src, dst, bt, cu, t2s, 2, -1, "auto", scale, bad_starts)


@requires_debug_build
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Need CUDA device")
def test_gather_cache_batch_major_rejects_negative_seq_starts():
    """Batch-major must reject negative seq_starts values."""
    entry_size = 576
    src = torch.randn((4, 16, entry_size), dtype=torch.float16, device="cuda")
    dst = torch.empty((2, entry_size), dtype=torch.float16, device="cuda")
    bt = torch.zeros((1, 4), dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    bad_starts = torch.tensor([-1], dtype=torch.int32, device="cuda")

    with pytest.raises(RuntimeError, match="non-negative"):
        ops.gather_cache(src, dst, bt, cu, None, 0, 1, "auto", None, bad_starts)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Need CUDA device")
def test_gather_cache_token_major_rejects_empty_scale():
    """Token-major must reject an empty scale tensor."""
    entry_size = 576
    src = torch.randn((4, 16, entry_size), dtype=torch.float16, device="cuda")
    dst = torch.empty((1, entry_size), dtype=torch.float16, device="cuda")
    bt = torch.zeros((1, 4), dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    t2s = torch.zeros((1,), dtype=torch.int32, device="cuda")
    empty_scale = torch.empty(0, dtype=torch.float32, device="cuda")

    with pytest.raises(RuntimeError, match="at least 1 element"):
        ops.gather_cache(src, dst, bt, cu, t2s, 1, -1, "auto", empty_scale, None)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Need CUDA device")
def test_gather_cache_token_major_kernel_guard_oob_token_to_seq():
    """Kernel-level guard: OOB token_to_seq values are skipped (no crash).

    This test runs in all builds (debug and release).  The kernel guard
    (batch_id < 0 || batch_id >= batch_dim) prevents OOB access.
    """
    entry_size = 576
    src = torch.randn((4, 16, entry_size), dtype=torch.float16, device="cuda")
    dst = torch.zeros((2, entry_size), dtype=torch.float16, device="cuda")
    bt = torch.zeros((1, 4), dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    # batch_dim is 1, but token_to_seq has value 5 (OOB) for token 0
    t2s = torch.tensor([5, 0], dtype=torch.int32, device="cuda")
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    try:
        ops.gather_cache(src, dst, bt, cu, t2s, 2, -1, "auto", scale, None)
        torch.cuda.synchronize()
        # In release: kernel skips OOB token, no crash. Token 0 stays zero.
        assert torch.all(dst[0] == 0), "OOB token should be skipped by kernel guard"
    except RuntimeError:
        # In debug: host-side check catches it first — also acceptable.
        pass


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Need CUDA device")
def test_gather_cache_token_major_kernel_guard_negative_seq_starts():
    """Kernel-level guard: negative seq_starts are skipped (no crash).

    This test runs in all builds.  The kernel guard
    (seq_start_val < 0) prevents negative offset arithmetic.
    """
    entry_size = 576
    src = torch.randn((4, 16, entry_size), dtype=torch.float16, device="cuda")
    dst = torch.zeros((2, entry_size), dtype=torch.float16, device="cuda")
    bt = torch.zeros((1, 4), dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    t2s = torch.zeros((2,), dtype=torch.int32, device="cuda")
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    bad_starts = torch.tensor([-1], dtype=torch.int32, device="cuda")

    try:
        ops.gather_cache(src, dst, bt, cu, t2s, 2, -1, "auto", scale, bad_starts)
        torch.cuda.synchronize()
        # In release: kernel skips tokens with negative seq_starts.
    except RuntimeError:
        # In debug: host-side check catches it first.
        pass


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Need CUDA device")
def test_gather_cache_batch_major_kernel_guard_block_table_oob():
    """Kernel-level guard: block_table column OOB is caught (no crash).

    This test runs in all builds.  The kernel guard
    (offset_div >= block_table_width) prevents OOB block_table read.
    """
    block_size = 16
    entry_size = 576
    src = torch.randn((4, block_size, entry_size), dtype=torch.float16, device="cuda")
    # block_table width is 1, but seq_len=32 needs 2 blocks
    bt = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, 32], dtype=torch.int32, device="cuda")
    dst = torch.zeros((32, entry_size), dtype=torch.float16, device="cuda")

    try:
        ops.gather_cache(src, dst, bt, cu, None, 0, 1, "auto", None, None)
        torch.cuda.synchronize()
        # In release: kernel breaks out of loop when offset_div >= width.
    except RuntimeError:
        # In debug: host-side check catches it first.
        pass


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Need CUDA device")
def test_gather_cache_token_major_kernel_guard_invalid_block_id():
    """Kernel-level guard: invalid block ids are skipped (no crash)."""
    entry_size = 576
    src = torch.randn((4, 1, entry_size), dtype=torch.float16, device="cuda")
    dst = torch.zeros((2, entry_size), dtype=torch.float16, device="cuda")
    # block_table[0, 0] is invalid; block_table[0, 1] is valid.
    bt = torch.tensor([[99, 0, 0, 0]], dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    t2s = torch.zeros((2,), dtype=torch.int32, device="cuda")
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    ops.gather_cache(src, dst, bt, cu, t2s, 2, -1, "auto", scale, None)
    torch.cuda.synchronize()

    assert torch.all(dst[0] == 0), "Token with invalid block_id should be skipped"
    torch.testing.assert_close(dst[1], src[0, 0], atol=0, rtol=0)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Need CUDA device")
def test_gather_cache_batch_major_kernel_guard_invalid_block_id():
    """Kernel-level guard: batch-major invalid block ids are skipped."""
    entry_size = 576
    src = torch.randn((4, 1, entry_size), dtype=torch.float16, device="cuda")
    dst = torch.zeros((2, entry_size), dtype=torch.float16, device="cuda")
    # First token maps to invalid block id; second token maps to valid block id.
    bt = torch.tensor([[99, 0]], dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, 2], dtype=torch.int32, device="cuda")

    ops.gather_cache(src, dst, bt, cu, None, 0, 1, "auto", None, None)
    torch.cuda.synchronize()

    assert torch.all(dst[0] == 0), "Token with invalid block_id should be skipped"
    torch.testing.assert_close(dst[1], src[0, 0], atol=0, rtol=0)


if __name__ == "__main__":
    pytest.main([__file__])
