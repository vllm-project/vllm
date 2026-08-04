# SPDX-License-Identifier: Apache-2.0
"""Tests for TokenHasher."""

# Standard
import builtins

# Third Party
import numpy as np
import pytest

# First Party
from lmcache.v1.multiprocess.token_hasher import (
    TokenHasher,
    chunk_hash_windows_numba,
    rolling_hash_windows_numba,
    unique_hits_direct_id_numba,
    update_table_id_numba,
)


@pytest.fixture
def hasher() -> TokenHasher:
    """TokenHasher with small chunk_size for testing."""
    return TokenHasher(chunk_size=4, hash_algorithm="blake3")


class TestTokenHasher:
    def test_init_blake3(self) -> None:
        hasher = TokenHasher(chunk_size=256, hash_algorithm="blake3")
        assert hasher.chunk_size == 256
        assert hasher.none_hash is not None

    def test_init_blake3_does_not_import_vllm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "vllm" or name.startswith("vllm."):
                raise AssertionError("blake3 TokenHasher should not import vLLM")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        hasher = TokenHasher(chunk_size=256, hash_algorithm="blake3")

        assert hasher.none_hash is not None

    def test_init_none_hash_vllm_runtime_error_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_import = builtins.__import__

        def failing_vllm_import(name, *args, **kwargs):
            if name == "vllm.v1.core":
                raise RuntimeError("already a kernel registered")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", failing_vllm_import)
        hasher = TokenHasher.__new__(TokenHasher)
        hasher.hash_algorithm_name = "sha256_cbor"
        hasher.hash_func = lambda _: b"fallback"

        assert hasher._init_none_hash() == b"fallback"

    def test_hash_tokens_returns_bytes(self, hasher: TokenHasher) -> None:
        h = hasher.hash_tokens([1, 2, 3, 4])
        assert isinstance(h, bytes)

    def test_hash_tokens_deterministic(self, hasher: TokenHasher) -> None:
        h1 = hasher.hash_tokens([1, 2, 3, 4])
        h2 = hasher.hash_tokens([1, 2, 3, 4])
        assert h1 == h2

    def test_hash_tokens_different_input(self, hasher: TokenHasher) -> None:
        h1 = hasher.hash_tokens([1, 2, 3, 4])
        h2 = hasher.hash_tokens([5, 6, 7, 8])
        assert h1 != h2

    def test_hash_tokens_with_prefix(self, hasher: TokenHasher) -> None:
        """Rolling hash: same tokens with different prefix produces different hash."""
        h_no_prefix = hasher.hash_tokens([1, 2, 3, 4])
        h_with_prefix = hasher.hash_tokens([1, 2, 3, 4], prefix_hash=h_no_prefix)
        assert h_no_prefix != h_with_prefix

    def test_compute_chunk_hashes_exact_chunks(self, hasher: TokenHasher) -> None:
        """8 tokens with chunk_size=4 produces 2 hashes."""
        tokens = list(range(8))
        hashes = hasher.compute_chunk_hashes(tokens)
        assert len(hashes) == 2

    def test_compute_chunk_hashes_partial_chunk_discarded(
        self, hasher: TokenHasher
    ) -> None:
        """10 tokens with chunk_size=4 produces 2 hashes (last 2 tokens discarded)."""
        tokens = list(range(10))
        hashes = hasher.compute_chunk_hashes(tokens)
        assert len(hashes) == 2

    def test_compute_chunk_hashes_too_few_tokens(self, hasher: TokenHasher) -> None:
        """3 tokens with chunk_size=4 produces 0 hashes."""
        hashes = hasher.compute_chunk_hashes([1, 2, 3])
        assert len(hashes) == 0

    def test_compute_chunk_hashes_empty(self, hasher: TokenHasher) -> None:
        hashes = hasher.compute_chunk_hashes([])
        assert len(hashes) == 0

    def test_compute_chunk_hashes_rolling(self, hasher: TokenHasher) -> None:
        """Second chunk hash depends on the first (rolling property)."""
        tokens = list(range(8))
        hashes = hasher.compute_chunk_hashes(tokens)
        # Hash of chunk [4,5,6,7] alone (no prefix) should differ
        standalone = hasher.hash_tokens([4, 5, 6, 7])
        assert hashes[1] != standalone

    def test_compute_chunk_hashes_matches_manual(self, hasher: TokenHasher) -> None:
        """compute_chunk_hashes should match manual rolling hash_tokens calls."""
        tokens = list(range(12))  # 3 chunks
        auto_hashes = hasher.compute_chunk_hashes(tokens)

        h0 = hasher.hash_tokens(tokens[0:4])
        h1 = hasher.hash_tokens(tokens[4:8], prefix_hash=h0)
        h2 = hasher.hash_tokens(tokens[8:12], prefix_hash=h1)
        assert auto_hashes == [h0, h1, h2]

    def test_hash_to_bytes_from_bytes(self) -> None:
        val = b"\x01\x02\x03"
        assert TokenHasher.hash_to_bytes(val) is val

    def test_hash_to_bytes_from_int(self) -> None:
        val = 42
        result = TokenHasher.hash_to_bytes(val)
        assert isinstance(result, bytes)
        assert len(result) == 8


_BASE = np.uint64(31)


class TestRollingHashWindowsNumba:
    def test_output_length(self) -> None:
        arr = np.arange(10, dtype=np.uint64)
        out = rolling_hash_windows_numba(arr, 4, _BASE)
        assert len(out) == 7  # 10 - 4 + 1

    def test_single_window(self) -> None:
        arr = np.array([1, 2, 3], dtype=np.uint64)
        out = rolling_hash_windows_numba(arr, 3, _BASE)
        assert len(out) == 1

    def test_k_equals_1(self) -> None:
        """With window size 1 each output equals the corresponding input element."""
        arr = np.array([7, 13, 42], dtype=np.uint64)
        out = rolling_hash_windows_numba(arr, 1, _BASE)
        np.testing.assert_array_equal(out, arr)

    def test_deterministic(self) -> None:
        arr = np.array([1, 2, 3, 4, 5], dtype=np.uint64)
        out1 = rolling_hash_windows_numba(arr, 3, _BASE)
        out2 = rolling_hash_windows_numba(arr, 3, _BASE)
        np.testing.assert_array_equal(out1, out2)

    def test_different_inputs_different_outputs(self) -> None:
        arr1 = np.array([1, 2, 3, 4, 5], dtype=np.uint64)
        arr2 = np.array([5, 4, 3, 2, 1], dtype=np.uint64)
        out1 = rolling_hash_windows_numba(arr1, 3, _BASE)
        out2 = rolling_hash_windows_numba(arr2, 3, _BASE)
        assert not np.array_equal(out1, out2)

    def test_different_base_different_output(self) -> None:
        arr = np.array([1, 2, 3, 4, 5], dtype=np.uint64)
        out1 = rolling_hash_windows_numba(arr, 3, np.uint64(31))
        out2 = rolling_hash_windows_numba(arr, 3, np.uint64(37))
        assert not np.array_equal(out1, out2)

    def test_manual_values(self) -> None:
        """Verify polynomial hash values against manual computation.

        For arr=[1,2,3,4,5], k=3, base=31:
          power = 31^2 = 961
          window [1,2,3]: h = ((0*31+1)*31+2)*31+3 = 1026
          window [2,3,4]: h = 1026 - 1*961 = 65; 65*31+4 = 2019
          window [3,4,5]: h = 2019 - 2*961 = 97; 97*31+5 = 3012
        """
        arr = np.array([1, 2, 3, 4, 5], dtype=np.uint64)
        out = rolling_hash_windows_numba(arr, 3, _BASE)
        expected = np.array([1026, 2019, 3012], dtype=np.uint64)
        np.testing.assert_array_equal(out, expected)

    def test_output_dtype(self) -> None:
        arr = np.array([1, 2, 3], dtype=np.uint64)
        out = rolling_hash_windows_numba(arr, 2, _BASE)
        assert out.dtype == np.uint64


class TestChunkHashWindowsNumba:
    def test_output_length(self) -> None:
        arr = np.arange(12, dtype=np.uint64)
        out = chunk_hash_windows_numba(arr, 4, _BASE)
        assert len(out) == 3  # 12 // 4

    def test_partial_chunk_ignored(self) -> None:
        """Trailing tokens that don't fill a full window are dropped."""
        arr = np.arange(10, dtype=np.uint64)
        out = chunk_hash_windows_numba(arr, 4, _BASE)
        assert len(out) == 2  # 10 // 4

    def test_no_full_windows(self) -> None:
        arr = np.arange(3, dtype=np.uint64)
        out = chunk_hash_windows_numba(arr, 4, _BASE)
        assert len(out) == 0

    def test_deterministic(self) -> None:
        arr = np.arange(8, dtype=np.uint64)
        out1 = chunk_hash_windows_numba(arr, 4, _BASE)
        out2 = chunk_hash_windows_numba(arr, 4, _BASE)
        np.testing.assert_array_equal(out1, out2)

    def test_different_inputs_different_outputs(self) -> None:
        arr1 = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint64)
        arr2 = np.array([6, 5, 4, 3, 2, 1], dtype=np.uint64)
        out1 = chunk_hash_windows_numba(arr1, 3, _BASE)
        out2 = chunk_hash_windows_numba(arr2, 3, _BASE)
        assert not np.array_equal(out1, out2)

    def test_first_window_matches_rolling(self) -> None:
        """First chunk hash must equal the first rolling-hash window."""
        arr = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint64)
        chunk_out = chunk_hash_windows_numba(arr, 3, _BASE)
        rolling_out = rolling_hash_windows_numba(arr, 3, _BASE)
        assert chunk_out[0] == rolling_out[0]

    def test_manual_values(self) -> None:
        """Verify chunk hashes against manual computation.

        For arr=[1,2,3,4,5,6], k=3, base=31:
          window 0 [1,2,3]: ((0*31+1)*31+2)*31+3 = 1026
          window 1 [4,5,6]: ((0*31+4)*31+5)*31+6 = 4005
        """
        arr = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint64)
        out = chunk_hash_windows_numba(arr, 3, _BASE)
        expected = np.array([1026, 4005], dtype=np.uint64)
        np.testing.assert_array_equal(out, expected)

    def test_output_dtype(self) -> None:
        arr = np.array([1, 2, 3, 4], dtype=np.uint64)
        out = chunk_hash_windows_numba(arr, 2, _BASE)
        assert out.dtype == np.uint64


class TestUpdateTableIdNumba:
    def _empty_table(self, size: int = 8) -> np.ndarray:
        return np.full(size, -1, dtype=np.int64)

    def test_basic_update(self) -> None:
        table = self._empty_table()
        hashes = np.array([0], dtype=np.uint64)  # 0 & 7 = 0
        vals = np.array([99], dtype=np.int64)
        update_table_id_numba(hashes, table, vals)
        assert table[0] == 99

    def test_index_from_hash_lower_bits(self) -> None:
        """idx = hash & (table_size - 1)."""
        table = self._empty_table(8)  # mask = 7
        hashes = np.array([9], dtype=np.uint64)  # 9 & 7 = 1
        vals = np.array([42], dtype=np.int64)
        update_table_id_numba(hashes, table, vals)
        assert table[1] == 42
        assert all(table[i] == -1 for i in range(8) if i != 1)

    def test_multiple_updates(self) -> None:
        table = self._empty_table()
        hashes = np.array([0, 1, 2], dtype=np.uint64)
        vals = np.array([10, 20, 30], dtype=np.int64)
        update_table_id_numba(hashes, table, vals)
        assert table[0] == 10
        assert table[1] == 20
        assert table[2] == 30

    def test_collision_last_write_wins(self) -> None:
        """When two hashes map to the same slot, the last value is stored."""
        table = self._empty_table()
        # hash 0 → idx 0, hash 8 → idx 0 (8 & 7 = 0)
        hashes = np.array([0, 8], dtype=np.uint64)
        vals = np.array([10, 20], dtype=np.int64)
        update_table_id_numba(hashes, table, vals)
        assert table[0] == 20

    def test_modifies_in_place(self) -> None:
        table = self._empty_table()
        ptr = table.ctypes.data
        hashes = np.array([3], dtype=np.uint64)
        vals = np.array([7], dtype=np.int64)
        update_table_id_numba(hashes, table, vals)
        assert table.ctypes.data == ptr  # same buffer
        assert table[3] == 7

    def test_empty_hashes_no_change(self) -> None:
        table = self._empty_table()
        hashes = np.empty(0, dtype=np.uint64)
        vals = np.empty(0, dtype=np.int64)
        update_table_id_numba(hashes, table, vals)
        assert all(v == -1 for v in table)


class TestUniqueHitsDirectIdNumba:
    def _table(self, size: int = 8) -> np.ndarray:
        return np.full(size, -1, dtype=np.int64)

    def test_no_hits_all_empty(self) -> None:
        table = self._table()
        hashes = np.array([0, 1, 2], dtype=np.uint64)
        out = unique_hits_direct_id_numba(hashes, table, np.uint64(7), 4)
        assert len(out) == 0

    def test_single_hit(self) -> None:
        table = self._table()
        table[1] = 5
        hashes = np.array([1], dtype=np.uint64)  # 1 & 7 = 1 → ID 5
        out = unique_hits_direct_id_numba(hashes, table, np.uint64(7), 10)
        assert len(out) == 1
        assert out[0] == 5

    def test_deduplication(self) -> None:
        """Same ID reached via two different hashes is returned only once."""
        table = self._table()
        table[1] = 3  # index 1 → ID 3
        # hash 1 and hash 9 both map to index 1 (& 7 == 1)
        hashes = np.array([1, 9], dtype=np.uint64)
        out = unique_hits_direct_id_numba(hashes, table, np.uint64(7), 5)
        assert len(out) == 1
        assert out[0] == 3

    def test_multiple_unique_ids(self) -> None:
        table = self._table()
        table[0] = 0
        table[1] = 1
        table[2] = 2
        hashes = np.array([0, 1, 2], dtype=np.uint64)
        out = unique_hits_direct_id_numba(hashes, table, np.uint64(7), 5)
        assert len(out) == 3
        assert set(out) == {0, 1, 2}

    def test_mixed_hits_and_misses(self) -> None:
        table = self._table()
        table[2] = 10
        # indices 0 (miss), 2 (hit → 10), 5 (miss)
        hashes = np.array([0, 2, 5], dtype=np.uint64)
        out = unique_hits_direct_id_numba(hashes, table, np.uint64(7), 15)
        assert len(out) == 1
        assert out[0] == 10

    def test_order_of_first_encounter_preserved(self) -> None:
        """IDs appear in the order they are first seen in the hash stream."""
        table = self._table()
        table[0] = 2
        table[1] = 0
        table[2] = 1
        hashes = np.array([0, 1, 2], dtype=np.uint64)
        out = unique_hits_direct_id_numba(hashes, table, np.uint64(7), 3)
        np.testing.assert_array_equal(out, [2, 0, 1])

    def test_empty_hashes(self) -> None:
        table = self._table()
        hashes = np.empty(0, dtype=np.uint64)
        out = unique_hits_direct_id_numba(hashes, table, np.uint64(7), 4)
        assert len(out) == 0

    def test_output_dtype(self) -> None:
        table = self._table()
        table[0] = 0
        hashes = np.array([0], dtype=np.uint64)
        out = unique_hits_direct_id_numba(hashes, table, np.uint64(7), 1)
        assert out.dtype == np.int64
