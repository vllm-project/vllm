# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the multi-threaded memcpy."""

from unittest import mock

import numpy as np
import pytest

from vllm.utils import memcpy_utils as mod

class TestMemcpyMt:

    def test_zero_size_is_noop(self):
        src = np.arange(16, dtype=np.uint8)
        dst = np.zeros_like(src)
        mod.memcpy_mt(src, dst, 0)
        np.testing.assert_array_equal(dst, np.zeros_like(src))

    def test_negative_size_is_noop(self):
        src = np.arange(16, dtype=np.uint8)
        dst = np.zeros_like(src)
        mod.memcpy_mt(src, dst, -5)
        np.testing.assert_array_equal(dst, np.zeros_like(src))

    def test_basic_copy_full(self):
        src = np.arange(4096, dtype=np.uint8)
        dst = np.empty_like(src)
        mod.memcpy_mt(src, dst, 4096)
        np.testing.assert_array_equal(src, dst)

    def test_basic_copy_partial(self):
        src = np.arange(1024, dtype=np.uint8)
        dst = np.full_like(src, 0xFF)
        mod.memcpy_mt(src, dst, 100)
        np.testing.assert_array_equal(dst[:100], src[:100])
        # Tail should be untouched.
        np.testing.assert_array_equal(dst[100:], np.full(924, 0xFF, dtype=np.uint8))

    def test_large_copy_multithread(self):
        # Large enough to exercise the kernel regardless of MT heuristic.
        n = 1 << 20  # 1 MiB
        rng = np.random.default_rng(0)
        src = rng.integers(0, 256, size=n, dtype=np.uint8)
        dst = np.empty_like(src)
        mod.memcpy_mt(src, dst, n)
        np.testing.assert_array_equal(src, dst)

    def test_sub_chunk_boundary(self):
        # size == sub_chunk_bytes uses the numpy fast path.
        n = mod._SUBCHUNK_BYTES
        src = np.arange(n, dtype=np.uint8)
        dst = np.empty_like(src)
        mod.memcpy_mt(src, dst, n)
        np.testing.assert_array_equal(src, dst)

    def test_sub_chunk_plus_one(self):
        n = mod._SUBCHUNK_BYTES + 1
        src = np.arange(n, dtype=np.uint8)
        dst = np.empty_like(src)
        mod.memcpy_mt(src, dst, n)
        np.testing.assert_array_equal(src, dst)

    def test_custom_sub_chunk_bytes(self):
        n = 64 * 1024
        src = np.arange(n, dtype=np.uint8)
        dst = np.empty_like(src)
        mod.memcpy_mt(src, dst, n, sub_chunk_bytes=1024)
        np.testing.assert_array_equal(src, dst)

    def test_max_copy_threads_one_forces_serial(self):
        n = 128 * 1024
        src = np.arange(n, dtype=np.uint8)
        dst = np.empty_like(src)
        mod.memcpy_mt(src, dst, n, max_copy_threads=1)
        np.testing.assert_array_equal(src, dst)

    def test_max_copy_threads_high(self):
        n = 128 * 1024
        src = np.arange(n, dtype=np.uint8)
        dst = np.empty_like(src)
        mod.memcpy_mt(src, dst, n, max_copy_threads=999)
        np.testing.assert_array_equal(src, dst)

    def test_invalid_sub_chunk_bytes(self):
        src = np.arange(16, dtype=np.uint8)
        dst = np.empty_like(src)
        with pytest.raises(ValueError, match="sub_chunk_bytes must be positive"):
            mod.memcpy_mt(src, dst, 16, sub_chunk_bytes=0)
        with pytest.raises(ValueError, match="sub_chunk_bytes must be positive"):
            mod.memcpy_mt(src, dst, 16, sub_chunk_bytes=-1)

    def test_invalid_max_copy_threads(self):
        src = np.arange(16, dtype=np.uint8)
        dst = np.empty_like(src)
        with pytest.raises(ValueError, match="max_copy_threads must be >= 1"):
            mod.memcpy_mt(src, dst, 16, max_copy_threads=0)

    def test_size_exceeds_src(self):
        src = np.zeros(4, dtype=np.uint8)
        dst = np.zeros(8, dtype=np.uint8)
        with pytest.raises(ValueError, match="size exceeds buffer capacity"):
            mod.memcpy_mt(src, dst, 5)

    def test_size_exceeds_dst(self):
        src = np.zeros(8, dtype=np.uint8)
        dst = np.zeros(4, dtype=np.uint8)
        with pytest.raises(ValueError, match="size exceeds buffer capacity"):
            mod.memcpy_mt(src, dst, 5)

    def test_non_contiguous_dst_rejected(self):
        src = np.arange(12, dtype=np.uint8)
        dst = np.zeros((4, 3), dtype=np.uint8).T  # non-contiguous view
        assert not dst.flags.c_contiguous
        with pytest.raises(ValueError, match="C-contiguous"):
            mod.memcpy_mt(src, dst, 12)

    def test_non_contiguous_src_allowed(self):
        src = np.arange(12, dtype=np.uint8).reshape(3, 4).T  # non-contiguous
        assert not src.flags.c_contiguous
        dst = np.empty(12, dtype=np.uint8)
        mod.memcpy_mt(src, dst, 12)
        np.testing.assert_array_equal(dst, src.reshape(-1))

    def test_multidim_buffers(self):
        src = np.arange(64, dtype=np.uint8).reshape(8, 8)
        dst = np.zeros_like(src)
        mod.memcpy_mt(src, dst, 64)
        np.testing.assert_array_equal(src, dst)

    def test_int32_dtype_round_trip(self):
        src = np.arange(256, dtype=np.int32)
        dst = np.zeros_like(src)
        mod.memcpy_mt(src, dst, src.nbytes)
        np.testing.assert_array_equal(src, dst)

    def test_size_larger_than_view(self):
        # Sanity: size is interpreted as bytes when buffers are non-uint8.
        src = np.arange(256, dtype=np.int32)
        dst = np.zeros_like(src)
        # only copy first 10 elements' worth of bytes
        mod.memcpy_mt(src, dst, 10 * 4)
        np.testing.assert_array_equal(dst[:10], src[:10])
        np.testing.assert_array_equal(dst[10:], np.zeros(246, dtype=np.int32))

    def test_example_from_docstring(self):
        src = np.arange(4096, dtype=np.uint8)
        dst = np.empty_like(src)
        mod.memcpy_mt(src, dst, 4096)
        assert np.array_equal(src, dst)

    @pytest.mark.skipif(not mod._HAS_NUMBA, reason="numba not installed")
    def test_thread_pool_restored_on_exit(self):
        n = 256 * 1024
        src = np.arange(n, dtype=np.uint8)
        dst = np.empty_like(src)
        prev = mod.get_num_threads()
        mod.memcpy_mt(src, dst, n, max_copy_threads=2)
        # Pool should be restored to its previous size.
        assert mod.get_num_threads() == prev
        np.testing.assert_array_equal(src, dst)

    @pytest.mark.skipif(not mod._HAS_NUMBA, reason="numba not installed")
    def test_thread_pool_restored_on_kernel_error(self):
        n = 256 * 1024
        src = np.arange(n, dtype=np.uint8)
        dst = np.empty_like(src)
        prev = mod.get_num_threads()
        with mock.patch.object(mod, "_copy_kernel",
                               side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                mod.memcpy_mt(src, dst, n, max_copy_threads=2)
        assert mod.get_num_threads() == prev