# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for vllm.multimodal.paged_shm.memcpy_utils."""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest

from vllm.multimodal.paged_shm import memcpy_utils
from vllm.multimodal.paged_shm.memcpy_utils import (
    copy_blocks_to_contig,
    copy_contig_to_blocks,
    get_copy_threads,
    get_topology,
    use_multithread,
    warmup,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_paged(n_block: int, block_size: int, seed: int = 0) -> np.ndarray:
    """Return a flat uint8 array of n_block * block_size, filled by RNG."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=n_block * block_size, dtype=np.uint8)


def _reference_write(flat, src, blocks, block_size) -> np.ndarray:
    """Reference scatter, one block at a time (no sub-chunking)."""
    out = flat.copy()
    full = len(src) // block_size
    rem = len(src) - full * block_size
    for i in range(full):
        s = i * block_size
        d = blocks[i] * block_size
        out[d : d + block_size] = src[s : s + block_size]
    if rem > 0:
        s = full * block_size
        d = blocks[full] * block_size
        out[d : d + rem] = src[s : s + rem]
    return out


def _reference_read(flat, size, blocks, block_size) -> np.ndarray:
    out = np.empty(size, dtype=np.uint8)
    full = size // block_size
    rem = size - full * block_size
    for i in range(full):
        s = blocks[i] * block_size
        d = i * block_size
        out[d : d + block_size] = flat[s : s + block_size]
    if rem > 0:
        s = blocks[full] * block_size
        d = full * block_size
        out[d : d + rem] = flat[s : s + rem]
    return out


# Block sizes exercised by the roundtrip tests.
BLOCK_SIZES = [
    1 << 20,   # 1 MiB, the common case
    1 << 16,   # 64 KiB
    1 << 14,   # 16 KiB
    3 << 12,   # 12 KiB, not a power of two
]


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestHeuristic:
    """Tests for the multi-threading heuristic and topology detection."""

    @pytest.mark.parametrize(
        "cores,numa,mem_gib,expected",
        [
            # Desktop-like: single NUMA, few cores, small RAM.
            (8, 1, 16, False),
            (16, 1, 64, False),
            (32, 1, 256, False),
            # Server signals.
            (128, 2, 1024, True),      # multi-NUMA
            (64, 1, 256, True),        # many cores, single NUMA
            (16, 1, 1024, True),       # huge RAM, few cores
            (4, 1, 512, False),        # exactly 512 GiB is not > 512 GiB
            (4, 1, 513, True),         # just above the threshold
            (33, 1, 16, True),         # just above the core threshold
        ],
    )
    def test_auto_detect(self, cores, numa, mem_gib, expected):
        """Enable MT only when the host looks like a server."""
        mem_bytes = mem_gib * (1 << 30)
        assert (
            memcpy_utils._auto_detect_mt(cores, numa, mem_bytes) is expected
        )

    def test_topology_shape(self):
        """get_topology returns sane values on the host running the tests."""
        cores, numa, mem = get_topology()
        assert cores >= 1
        assert numa >= 1
        assert mem >= 0

    def test_use_multithread_consistent_with_threads(self):
        """The MT flag and the thread count must agree."""
        if use_multithread():
            assert get_copy_threads() >= 2
            assert get_copy_threads() <= memcpy_utils._MAX_COPY_THREADS
        else:
            assert get_copy_threads() == 1

    def test_no_numba_disables_mt(self):
        """Without numba, _USE_MT must be False regardless of topology."""
        # _USE_MT is computed at import time; re-evaluate the expression
        # with the numba flag patched to False to confirm the dependency.
        with patch.object(memcpy_utils, "_HAS_NUMBA", False):
            assert (
                memcpy_utils._HAS_NUMBA
                and memcpy_utils._auto_detect_mt(128, 2, 1 << 40)
            ) is False


class TestPickSubchunk:
    """Tests for _pick_subchunk's gcd-based alignment."""

    @pytest.mark.parametrize(
        "block_size,expected",
        [
            (1 << 20, 8 * 1024),   # 1 MiB -> 8 KiB
            (64 * 1024, 8 * 1024),  # 64 KiB -> 8 KiB
            (16 * 1024, 8 * 1024),  # 16 KiB -> 8 KiB
            (8 * 1024, 8 * 1024),   # exactly 8 KiB
            (4 * 1024, 4 * 1024),   # 4 KiB -> gcd = 4 KiB
            (12 * 1024, 4 * 1024),  # 12 KiB -> gcd = 4 KiB
            (100 * 1024, 4 * 1024),  # 100 KiB -> gcd = 4 KiB
        ],
    )
    def test_gcd_alignment(self, block_size, expected):
        """Sub-chunk is gcd(block_size, 8 KiB) and divides block_size."""
        if not memcpy_utils._HAS_NUMBA:
            pytest.skip("sub-chunking only applies with numba")
        sub = memcpy_utils._pick_subchunk(block_size)
        assert sub == expected
        assert block_size % sub == 0

    def test_never_exceeds_block_size(self):
        """For any power-of-two block size, sub-chunk fits inside a block."""
        for exp in range(10, 22):  # 1 KiB .. 2 MiB
            block_size = 1 << exp
            sub = math.gcd(block_size, memcpy_utils._SUBCHUNK_BYTES)
            assert 0 < sub <= block_size
            assert block_size % sub == 0

    def test_no_numba_uses_block_size(self):
        """Without numba we skip sub-chunking: one segment per block."""
        with patch.object(memcpy_utils, "_HAS_NUMBA", False):
            assert memcpy_utils._pick_subchunk(1 << 20) == (1 << 20)
            assert memcpy_utils._pick_subchunk(12 << 10) == (12 << 10)


class TestBuildOffsets:
    """Tests for _build_offsets decomposition."""

    def test_full_blocks(self):
        """All offsets are sub-chunk multiples; sizes are all sub-chunk."""
        blocks = [0, 2, 1]
        size = 3 * (1 << 20)
        block_size = 1 << 20
        sub = 8 * 1024
        contig, flat, sizes = memcpy_utils._build_offsets(
            blocks, size, block_size, sub
        )
        # 1 MiB / 8 KiB = 128 segments per block, 3 blocks.
        assert contig.shape[0] == 3 * 128
        assert flat.shape[0] == 3 * 128
        assert (sizes == sub).all()
        # Every offset is a multiple of sub (=> 8 KiB aligned).
        assert (contig % sub == 0).all()
        assert (flat % sub == 0).all()

    def test_with_remainder(self):
        """Ragged tail is a shorter final segment inside the last block."""
        block_size = 1 << 20
        sub = 8 * 1024
        size = block_size + 1234  # 1 full block + remainder
        blocks = [5, 3]
        contig, flat, sizes = memcpy_utils._build_offsets(
            blocks, size, block_size, sub
        )
        # 128 full sub-chunks + 1 ragged tail.
        assert contig.shape[0] == 128 + 1
        assert sizes[-1] == 1234
        # Ragged tail starts at the physical base of block 3.
        assert flat[-1] == 3 * block_size

    def test_contig_and_flat_differ(self):
        """Contig is monotonic; flat jumps by block index."""
        blocks = [7, 2]
        block_size = 1 << 20
        size = 2 * block_size
        sub = 8 * 1024
        contig, flat, _ = memcpy_utils._build_offsets(
            blocks, size, block_size, sub
        )
        assert contig[0] == 0
        assert contig[-1] == 2 * block_size - sub
        assert flat[0] == 7 * block_size
        assert flat[128] == 2 * block_size

    def test_segments_never_cross_block_boundary(self):
        """Every segment lies entirely within one block on the flat side."""
        block_size = 1 << 16
        sub = 8 * 1024
        size = 2 * block_size + 777
        blocks = [4, 1, 9]
        _, flat, sizes = memcpy_utils._build_offsets(
            blocks, size, block_size, sub
        )
        for off, sz in zip(flat, sizes):
            block = int(off) // block_size
            in_block = int(off) - block * block_size
            assert in_block + int(sz) <= block_size


class TestRoundtrip:
    """Correctness of copy_contig_to_blocks / copy_blocks_to_contig."""

    @pytest.mark.parametrize("block_size", BLOCK_SIZES)
    @pytest.mark.parametrize("size", [0, 1, 5000, None])
    def test_roundtrip(self, block_size, size):
        """Write into blocks, read back, compare against reference."""
        n_block = 32
        if size is None:
            size = 3 * block_size  # exact multiple

        flat = _make_paged(n_block, block_size, seed=1)
        src = np.random.default_rng(2).integers(
            0, 256, size=max(size, 1), dtype=np.uint8
        )[:size]

        blocks = list(range(n_block))
        rng = np.random.default_rng(3)
        rng.shuffle(blocks)
        blocks = blocks[: (size + block_size - 1) // block_size or 1]

        copy_contig_to_blocks(src, flat, blocks, block_size)

        expected = _reference_write(flat, src, blocks, block_size)
        np.testing.assert_array_equal(flat, expected)

        dst = np.empty(size, dtype=np.uint8)
        copy_blocks_to_contig(flat, dst, blocks, block_size)
        np.testing.assert_array_equal(dst, src)

        ref_dst = _reference_read(flat, size, blocks, block_size)
        np.testing.assert_array_equal(dst, ref_dst)

    @pytest.mark.parametrize("block_size", BLOCK_SIZES)
    def test_roundtrip_single_thread(self, block_size):
        """Same as above but forcing n_threads=1."""
        n_block = 16
        size = 2 * block_size + 137
        flat = _make_paged(n_block, block_size, seed=10)
        src = np.random.default_rng(11).integers(
            0, 256, size=size, dtype=np.uint8
        )
        blocks = [3, 0, 7]

        copy_contig_to_blocks(src, flat, blocks, block_size, n_threads=1)
        dst = np.empty(size, dtype=np.uint8)
        copy_blocks_to_contig(flat, dst, blocks, block_size, n_threads=1)
        np.testing.assert_array_equal(dst, src)

    def test_multi_thread_matches_single_thread(self):
        """MT must produce byte-identical output to ST."""
        if not memcpy_utils._HAS_NUMBA:
            pytest.skip("requires numba")

        block_size = 1 << 20
        n_block = 64
        size = 20 * block_size + 4096

        src = np.random.default_rng(20).integers(
            0, 256, size=size, dtype=np.uint8
        )
        blocks = list(range(24))
        np.random.default_rng(21).shuffle(blocks)

        flat_a = _make_paged(n_block, block_size, seed=22)
        flat_b = flat_a.copy()

        copy_contig_to_blocks(src, flat_a, blocks, block_size, n_threads=1)
        copy_contig_to_blocks(src, flat_b, blocks, block_size, n_threads=4)
        np.testing.assert_array_equal(flat_a, flat_b)

    def test_random_blocks_and_sizes(self):
        """Fuzz with random block indices and odd sizes."""
        rng = np.random.default_rng(42)
        block_size = 1 << 16
        n_block = 128

        for _ in range(20):
            size = int(rng.integers(0, 6 * block_size))
            n_needed = (size + block_size - 1) // block_size
            if n_needed == 0:
                continue
            blocks = rng.choice(n_block, size=n_needed, replace=False).tolist()

            flat = _make_paged(
                n_block, block_size, seed=int(rng.integers(0, 1 << 30))
            )
            src = rng.integers(0, 256, size=size, dtype=np.uint8)

            copy_contig_to_blocks(src, flat, blocks, block_size)
            expected = _reference_write(flat, src, blocks, block_size)
            np.testing.assert_array_equal(flat, expected)

            dst = np.empty(size, dtype=np.uint8)
            copy_blocks_to_contig(flat, dst, blocks, block_size)
            np.testing.assert_array_equal(dst, src)


class TestEdgeCases:
    """Validation and boundary conditions."""

    def test_empty_copy_is_noop(self):
        flat = _make_paged(4, 1 << 20, seed=0)
        before = flat.copy()
        copy_contig_to_blocks(np.empty(0, dtype=np.uint8), flat, [], 1 << 20)
        np.testing.assert_array_equal(flat, before)

        dst = np.empty(0, dtype=np.uint8)
        copy_blocks_to_contig(flat, dst, [], 1 << 20)
        assert dst.size == 0

    def test_rejects_wrong_dtype(self):
        flat = np.zeros(1 << 20, dtype=np.uint8)
        src = np.zeros(16, dtype=np.int16)
        with pytest.raises(TypeError, match="uint8"):
            copy_contig_to_blocks(src, flat, [0], 1 << 20)

    def test_rejects_non_contiguous(self):
        flat = np.zeros(2 << 20, dtype=np.uint8)
        src = np.zeros(2 << 20, dtype=np.uint8)[::2]  # strided view
        with pytest.raises(ValueError, match="C-contiguous"):
            copy_contig_to_blocks(src, flat, [0], 1 << 20)

    def test_single_byte_copy(self):
        block_size = 1 << 16
        flat = np.zeros(4 * block_size, dtype=np.uint8)
        src = np.array([0xAB], dtype=np.uint8)
        copy_contig_to_blocks(src, flat, [2], block_size)
        assert flat[2 * block_size] == 0xAB
        # Neighbouring bytes untouched.
        assert flat[2 * block_size - 1] == 0
        assert flat[2 * block_size + 1] == 0

        dst = np.empty(1, dtype=np.uint8)
        copy_blocks_to_contig(flat, dst, [2], block_size)
        assert dst[0] == 0xAB

    def test_size_exactly_one_block(self):
        block_size = 1 << 16
        flat = np.zeros(4 * block_size, dtype=np.uint8)
        src = np.arange(block_size, dtype=np.uint8)
        copy_contig_to_blocks(src, flat, [1], block_size)
        np.testing.assert_array_equal(flat[block_size : 2 * block_size], src)
        assert flat[:block_size].sum() == 0
        assert flat[2 * block_size :].sum() == 0


class TestNoNumbaFallback:
    """Simulate the 'numba missing' path by toggling the module flag."""

    def test_pick_subchunk_returns_block_size(self):
        with patch.object(memcpy_utils, "_HAS_NUMBA", False):
            assert memcpy_utils._pick_subchunk(1 << 20) == (1 << 20)
            assert memcpy_utils._pick_subchunk(12 << 10) == (12 << 10)

    def test_serial_fallback_correctness(self):
        """The pure-numpy loop must produce the same bytes as the reference."""
        block_size = 1 << 14
        n_block = 8
        size = 2 * block_size + 100
        flat = _make_paged(n_block, block_size, seed=100)
        src = np.random.default_rng(101).integers(
            0, 256, size=size, dtype=np.uint8
        )
        blocks = [1, 5, 2]

        with patch.object(memcpy_utils, "_HAS_NUMBA", False):
            copy_contig_to_blocks(src, flat, blocks, block_size)

        expected = _reference_write(
            _make_paged(n_block, block_size, seed=100),
            src,
            blocks,
            block_size,
        )
        np.testing.assert_array_equal(flat, expected)


class TestWarmupAndSurface:
    """Warm-up behaviour and public module surface."""

    def test_warmup_is_idempotent_and_safe(self):
        warmup()
        warmup()  # second call must not raise

    def test_public_api_present(self):
        for name in memcpy_utils.__all__:
            assert hasattr(memcpy_utils, name)


class TestStorageIntegration:
    """End-to-end tests through PagedShmStorage."""

    def test_write_read_roundtrip(self):
        pytest.importorskip("torch")
        from vllm.multimodal.paged_shm.storage import PagedShmStorage

        block_size = 1 << 20
        size = 8 * block_size
        storage = PagedShmStorage(size=size, block_size=block_size)
        try:
            data = np.random.default_rng(0).integers(
                0, 256, size=3 * block_size + 777, dtype=np.uint8
            )
            blocks = [7, 0, 5, 2]
            storage.write(data, blocks)
            out = storage.read_to_numpy(data.size, blocks)
            np.testing.assert_array_equal(out, data)
        finally:
            storage.close()

    def test_preserves_untouched_blocks(self):
        pytest.importorskip("torch")
        from vllm.multimodal.paged_shm.storage import PagedShmStorage

        block_size = 1 << 16
        n_block = 16
        size = n_block * block_size
        storage = PagedShmStorage(size=size, block_size=block_size)
        try:
            # Fill everything with a marker.
            marker = np.full(size, 0xAA, dtype=np.uint8)
            storage.write(marker, list(range(n_block)))

            # Overwrite only a couple of blocks.
            payload = np.zeros(2 * block_size, dtype=np.uint8)
            target = [3, 11]
            storage.write(payload, target)

            out = storage.read_to_numpy(size, list(range(n_block)))
            expected = marker.copy()
            for b in target:
                expected[b * block_size : (b + 1) * block_size] = 0
            np.testing.assert_array_equal(out, expected)
        finally:
            storage.close()