# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the skip_first_n_tokens logic in CacheServer.retrieve.

These tests verify the chunk-skip arithmetic without requiring GPU,
ensuring that:
  - Tokens within the skip range are not written (chunks are skipped)
  - Tokens outside the skip range are written with correct skip_in_chunk
  - Partial-chunk skip works (skip_first_n_tokens = chunk_size / 2)
  - Full-chunk skip works (skip_first_n_tokens >= chunk_size)
"""

# Third Party
import pytest


def compute_retrieve_plan(
    num_chunks: int,
    chunk_size: int,
    skip_first_n_tokens: int,
) -> list[tuple[int, int]]:
    """
    Reproduce the skip logic from CacheServer.retrieve._retrieve_loop.

    Returns a list of (chunk_index, skip_in_chunk) for each chunk that
    would actually be written to the paged buffer (i.e., not fully skipped).
    """
    written_chunks = []
    for idx in range(num_chunks):
        chunk_start = idx * chunk_size
        chunk_end = chunk_start + chunk_size

        effective_start = max(chunk_start, skip_first_n_tokens)
        effective_start = min(effective_start, chunk_end)
        if effective_start >= chunk_end:
            continue

        skip_in_chunk = max(0, min(effective_start - chunk_start, chunk_size - 1))
        written_chunks.append((idx, skip_in_chunk))
    return written_chunks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

CHUNK_SIZE = 256


class TestSkipFirstNTokensZero:
    """skip_first_n_tokens == 0  →  every chunk is written, no skip."""

    def test_all_chunks_written(self):
        plan = compute_retrieve_plan(
            num_chunks=4, chunk_size=CHUNK_SIZE, skip_first_n_tokens=0
        )
        assert len(plan) == 4
        for idx, skip in plan:
            assert skip == 0, f"chunk {idx} should have skip_in_chunk=0"


class TestFullChunkSkip:
    """skip_first_n_tokens >= chunk_size  →  first chunk(s) fully skipped."""

    def test_skip_exactly_one_chunk(self):
        """skip == chunk_size  →  chunk 0 skipped, chunk 1+ written."""
        plan = compute_retrieve_plan(
            num_chunks=4, chunk_size=CHUNK_SIZE, skip_first_n_tokens=CHUNK_SIZE
        )
        indices = [idx for idx, _ in plan]
        assert 0 not in indices, "chunk 0 should be fully skipped"
        assert indices == [1, 2, 3]
        for _, skip in plan:
            assert skip == 0

    def test_skip_multiple_chunks(self):
        """skip == 2 * chunk_size  →  chunks 0 and 1 skipped."""
        plan = compute_retrieve_plan(
            num_chunks=4,
            chunk_size=CHUNK_SIZE,
            skip_first_n_tokens=2 * CHUNK_SIZE,
        )
        indices = [idx for idx, _ in plan]
        assert indices == [2, 3]
        for _, skip in plan:
            assert skip == 0

    def test_skip_all_chunks(self):
        """skip >= total tokens  →  everything skipped, nothing written."""
        plan = compute_retrieve_plan(
            num_chunks=4,
            chunk_size=CHUNK_SIZE,
            skip_first_n_tokens=4 * CHUNK_SIZE,
        )
        assert plan == []

    def test_skip_more_than_all_tokens(self):
        """skip far exceeds total tokens  →  still nothing written."""
        plan = compute_retrieve_plan(
            num_chunks=2,
            chunk_size=CHUNK_SIZE,
            skip_first_n_tokens=100 * CHUNK_SIZE,
        )
        assert plan == []


class TestPartialChunkSkip:
    """skip_first_n_tokens lands in the middle of a chunk."""

    def test_skip_half_chunk(self):
        """skip == chunk_size / 2  →  chunk 0 written with half skip."""
        half = CHUNK_SIZE // 2
        plan = compute_retrieve_plan(
            num_chunks=4,
            chunk_size=CHUNK_SIZE,
            skip_first_n_tokens=half,
        )
        assert len(plan) == 4
        # chunk 0 partially skipped
        assert plan[0] == (0, half)
        # remaining chunks fully written
        for idx, skip in plan[1:]:
            assert skip == 0

    def test_skip_one_token(self):
        """skip == 1  →  chunk 0 written with skip_in_chunk=1."""
        plan = compute_retrieve_plan(
            num_chunks=3, chunk_size=CHUNK_SIZE, skip_first_n_tokens=1
        )
        assert plan[0] == (0, 1)
        assert plan[1] == (1, 0)
        assert plan[2] == (2, 0)

    def test_skip_last_token_of_chunk(self):
        """skip == chunk_size - 1  →  chunk 0 almost entirely skipped."""
        plan = compute_retrieve_plan(
            num_chunks=2,
            chunk_size=CHUNK_SIZE,
            skip_first_n_tokens=CHUNK_SIZE - 1,
        )
        assert len(plan) == 2
        assert plan[0] == (0, CHUNK_SIZE - 1)
        assert plan[1] == (1, 0)

    def test_skip_into_second_chunk(self):
        """skip lands in the middle of chunk 1."""
        skip = CHUNK_SIZE + CHUNK_SIZE // 4  # 1.25 chunks
        plan = compute_retrieve_plan(
            num_chunks=4,
            chunk_size=CHUNK_SIZE,
            skip_first_n_tokens=skip,
        )
        indices = [idx for idx, _ in plan]
        assert 0 not in indices, "chunk 0 should be fully skipped"
        # chunk 1 partially written
        assert plan[0] == (1, CHUNK_SIZE // 4)
        # chunks 2, 3 fully written
        assert plan[1] == (2, 0)
        assert plan[2] == (3, 0)


class TestEdgeCases:
    """Boundary and edge cases."""

    def test_single_chunk_full_skip(self):
        plan = compute_retrieve_plan(
            num_chunks=1,
            chunk_size=CHUNK_SIZE,
            skip_first_n_tokens=CHUNK_SIZE,
        )
        assert plan == []

    def test_single_chunk_no_skip(self):
        plan = compute_retrieve_plan(
            num_chunks=1,
            chunk_size=CHUNK_SIZE,
            skip_first_n_tokens=0,
        )
        assert plan == [(0, 0)]

    def test_single_chunk_partial_skip(self):
        plan = compute_retrieve_plan(
            num_chunks=1,
            chunk_size=CHUNK_SIZE,
            skip_first_n_tokens=100,
        )
        assert plan == [(0, 100)]

    def test_skip_in_chunk_never_negative(self):
        """skip_in_chunk must always be >= 0 regardless of inputs."""
        for skip in range(0, 5 * CHUNK_SIZE, 17):
            plan = compute_retrieve_plan(
                num_chunks=4,
                chunk_size=CHUNK_SIZE,
                skip_first_n_tokens=skip,
            )
            for _, skip_in_chunk in plan:
                assert skip_in_chunk >= 0

    def test_skip_in_chunk_never_exceeds_chunk_size_minus_one(self):
        """skip_in_chunk must always be <= chunk_size - 1."""
        for skip in range(0, 5 * CHUNK_SIZE, 17):
            plan = compute_retrieve_plan(
                num_chunks=4,
                chunk_size=CHUNK_SIZE,
                skip_first_n_tokens=skip,
            )
            for _, skip_in_chunk in plan:
                assert skip_in_chunk <= CHUNK_SIZE - 1

    @pytest.mark.parametrize("chunk_size", [1, 16, 64, 256, 1024])
    def test_various_chunk_sizes(self, chunk_size):
        """The logic should work correctly for any chunk_size."""
        half = chunk_size // 2
        plan = compute_retrieve_plan(
            num_chunks=3,
            chunk_size=chunk_size,
            skip_first_n_tokens=half,
        )
        if half == 0:
            # chunk_size=1, half=0 → no skip
            assert len(plan) == 3
            assert all(skip == 0 for _, skip in plan)
        else:
            assert len(plan) == 3
            assert plan[0] == (0, half)
            assert plan[1][1] == 0
            assert plan[2][1] == 0
