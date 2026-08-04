# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the fold / unfold prefix-cache hit logic."""

# Third Party
import pytest

# First Party
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import TrimPolicy
from lmcache.v1.distributed.bitmap_ops import (
    FULL_ATTENTION_WINDOW,
    fold,
    fold_unfold,
    fold_unfold_ranked,
    highest_set_bit,
    merge_bitmaps,
    select_retained,
    unfold,
    unfold_range,
)
from lmcache.v1.distributed.bitmap_ops.fold import _fold_python, _unfold_python


def _make_presence(num_chunks: int, present_per_group: list[list[int]]) -> Bitmap:
    """Build a group-major presence bitmap.

    Args:
        num_chunks: chunks per group.
        present_per_group: present_per_group[g] is the list of chunk indices
            available for object group g.

    Returns:
        A group-major Bitmap of length ``len(present_per_group) * num_chunks``.
    """
    bm = Bitmap(len(present_per_group) * num_chunks)
    for group_idx, chunks in enumerate(present_per_group):
        base = group_idx * num_chunks
        for j in chunks:
            bm.set(base + j)
    return bm


# --------------------------------------------------------------------------- #
# unfold_range                                                                 #
# --------------------------------------------------------------------------- #


def test_unfold_full_attention_needs_whole_prefix():
    assert unfold_range(4, FULL_ATTENTION_WINDOW) == (0, 4)
    assert unfold_range(4, 0) == (0, 4)


def test_unfold_window_needs_only_last_w():
    assert unfold_range(4, 2) == (2, 4)
    assert unfold_range(1, 2) == (0, 1)  # window larger than prefix
    assert unfold_range(5, 1) == (4, 5)  # mamba: last chunk only


def test_unfold_empty_prefix():
    assert unfold_range(0, FULL_ATTENTION_WINDOW) == (0, 0)
    assert unfold_range(0, 2) == (0, 0)


# --------------------------------------------------------------------------- #
# fold_unfold — single group reduces to leading-ones                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "present,expected_hit",
    [
        ([0, 1, 2], 3),  # full contiguous prefix
        ([0, 1, 3], 2),  # gap at 2 caps the prefix
        ([], 0),  # nothing present
        ([1, 2], 0),  # missing chunk 0 -> empty prefix
    ],
)
def test_single_full_group_equals_leading_ones(present, expected_hit):
    num_chunks = 4
    found = _make_presence(num_chunks, [present])
    hit, mask = fold_unfold(found, num_chunks, [FULL_ATTENTION_WINDOW])
    assert hit == expected_hit
    # equals the plain PREFIX leading-ones count on the same bitmap
    assert hit == found.count_leading_ones()
    # retained mask is exactly the first `hit` chunks
    assert mask.get_indices_list() == list(range(expected_hit))


# --------------------------------------------------------------------------- #
# fold_unfold — worked full + sliding-window example                           #
# --------------------------------------------------------------------------- #


def test_full_plus_sliding_window_worked_example():
    # N=5; group A full present {0,1,2,3}; group B sliding-window w=2 {2,3,4}.
    # A blocks length 5 (chunk 4 missing); B's last-2 window at L=4 is {2,3} (present).
    num_chunks = 5
    found = _make_presence(num_chunks, [[0, 1, 2, 3], [2, 3, 4]])
    hit, mask = fold_unfold(found, num_chunks, [FULL_ATTENTION_WINDOW, 2])
    assert hit == 4
    # A (full) needs chunks 0..3 -> flat 0,1,2,3 ; B (w=2) needs 2..3 -> flat 7,8
    assert mask.get_indices_list() == [0, 1, 2, 3, 7, 8]


def test_sliding_window_does_not_block_long_prefix_when_tail_present():
    # SW group missing early chunks but holding the tail still serves a long hit.
    num_chunks = 6
    found = _make_presence(num_chunks, [[0, 1, 2, 3, 4, 5], [4, 5]])
    hit, mask = fold_unfold(found, num_chunks, [FULL_ATTENTION_WINDOW, 2])
    assert hit == 6
    # full needs 0..5 ; window-2 needs 4..5 -> flat 6*1 + {4,5} = {10,11}
    assert mask.get_indices_list() == [0, 1, 2, 3, 4, 5, 10, 11]


def test_mamba_window_one():
    # mamba == window 1: only the last chunk of the prefix is needed.
    num_chunks = 4
    found = _make_presence(num_chunks, [[0, 1, 2, 3], [3]])
    hit, mask = fold_unfold(found, num_chunks, [FULL_ATTENTION_WINDOW, 1])
    assert hit == 4
    assert mask.get_indices_list() == [0, 1, 2, 3, 7]  # full 0..3 + mamba {3}


# --------------------------------------------------------------------------- #
# fold_unfold — all-full reduces to require-all intersection                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "group_a,group_b,expected_hit",
    [
        ([0, 1, 2, 3], [0, 1, 2, 3], 4),  # both full -> full
        ([0, 1, 2, 3], [0, 1], 2),  # B caps at 2
        ([0, 1], [0, 1, 2, 3], 2),  # A caps at 2
        ([0, 2, 3], [0, 1, 2, 3], 1),  # A gap at 1 caps at 1
    ],
)
def test_all_full_is_require_all_intersection(group_a, group_b, expected_hit):
    num_chunks = 4
    found = _make_presence(num_chunks, [group_a, group_b])
    windows = [FULL_ATTENTION_WINDOW, FULL_ATTENTION_WINDOW]
    hit, mask = fold_unfold(found, num_chunks, windows)
    assert hit == expected_hit
    # both groups retain the same first `hit` chunks
    expected = list(range(expected_hit)) + [num_chunks + j for j in range(expected_hit)]
    assert mask.get_indices_list() == expected


# --------------------------------------------------------------------------- #
# fold_unfold — edges                                                          #
# --------------------------------------------------------------------------- #


def test_zero_chunks():
    found = Bitmap(0)
    hit, mask = fold_unfold(found, 0, [FULL_ATTENTION_WINDOW, 2])
    assert hit == 0
    assert mask.get_indices_list() == []


# --------------------------------------------------------------------------- #
# fold_unfold_ranked — group x chunk x kv_rank layout                          #
# --------------------------------------------------------------------------- #


def _make_ranked(
    num_chunks: int,
    num_ranks: int,
    present_per_group: list[list[tuple[int, int]]],
) -> Bitmap:
    """Build a group-major / chunk-major / rank-minor presence bitmap.

    present_per_group[g] is the list of ``(chunk, rank)`` present for group g.
    """
    num_groups = len(present_per_group)
    stride = num_chunks * num_ranks
    bm = Bitmap(num_groups * stride)
    for group_idx, cells in enumerate(present_per_group):
        gbase = group_idx * stride
        for chunk, rank in cells:
            bm.set(gbase + chunk * num_ranks + rank)
    return bm


def test_ranked_chunk_present_only_if_all_ranks_present():
    # 1 full group, 2 ranks, 3 chunks. chunk1 is missing rank 1 -> not present.
    present = [[(0, 0), (0, 1), (1, 0), (2, 0), (2, 1)]]
    found = _make_ranked(3, 2, present)
    hit, mask = fold_unfold_ranked(found, 3, 2, [FULL_ATTENTION_WINDOW])
    assert hit == 1  # only chunk 0 has both ranks; chunk1 gap caps the prefix
    assert mask.get_indices_list() == [0, 1]  # both ranks of chunk 0


def test_ranked_reduces_to_unranked_when_one_rank():
    # num_ranks == 1 must match fold_unfold exactly.
    found_unranked = _make_presence(5, [[0, 1, 2, 3], [2, 3, 4]])
    found_ranked = _make_ranked(
        5, 1, [[(c, 0) for c in [0, 1, 2, 3]], [(c, 0) for c in [2, 3, 4]]]
    )
    hit_u, mask_u = fold_unfold(found_unranked, 5, [FULL_ATTENTION_WINDOW, 2])
    hit_r, mask_r = fold_unfold_ranked(found_ranked, 5, 1, [FULL_ATTENTION_WINDOW, 2])
    assert hit_u == hit_r == 4
    assert mask_u.get_indices_list() == mask_r.get_indices_list()


def test_ranked_full_plus_sw_expands_all_ranks():
    # 2 groups, 2 ranks, 4 chunks. group0 full all present; group1 SW w=1 all present.
    g0 = [(c, r) for c in range(4) for r in range(2)]
    g1 = [(c, r) for c in range(4) for r in range(2)]
    found = _make_ranked(4, 2, [g0, g1])
    hit, mask = fold_unfold_ranked(found, 4, 2, [FULL_ATTENTION_WINDOW, 1])
    assert hit == 4
    # group0 full -> chunks 0..3 (ranks 0,1): flat 0..7
    # group1 w=1 -> chunk 3 only (ranks 0,1): group base = 4*2 = 8, chunk3 -> 8+6,8+7
    assert mask.get_indices_list() == [0, 1, 2, 3, 4, 5, 6, 7, 14, 15]


def test_ranked_invalid_num_ranks_raises():
    with pytest.raises(ValueError):
        fold_unfold_ranked(Bitmap(0), 0, 0, [FULL_ATTENTION_WINDOW])


def test_empty_group_windows_raises():
    with pytest.raises(ValueError):
        fold_unfold(Bitmap(0), 0, [])


def test_negative_num_chunks_raises():
    with pytest.raises(ValueError):
        fold_unfold(Bitmap(0), -1, [FULL_ATTENTION_WINDOW])


def _bm(num_keys: int, set_indices: list[int]) -> Bitmap:
    bm = Bitmap(num_keys)
    for i in set_indices:
        bm.set(i)
    return bm


class TestSelectRetained:
    """select_retained picks the retained subset per policy: PREFIX trims at the
    first gap; any other policy keeps every set bit (gaps and all)."""

    def test_prefix_trims_at_first_gap(self):
        found = _bm(5, [0, 1, 3, 4])  # gap at index 2
        assert select_retained(found, 5, TrimPolicy.PREFIX).get_indices_list() == [0, 1]

    def test_sparse_keeps_all_found(self):
        found = _bm(5, [0, 2, 4])
        result = select_retained(found, 5, TrimPolicy.SPARSE).get_indices_list()
        assert result == [0, 2, 4]

    def test_segmented_prefix_keeps_all_found(self):
        found = _bm(5, [0, 1, 3, 4])  # gap at index 2
        result = select_retained(
            found, 5, TrimPolicy.SEGMENTED_PREFIX
        ).get_indices_list()
        assert result == [0, 1, 3, 4]


class TestMergeBitmaps:
    """merge_bitmaps always returns a num_keys-sized bitmap."""

    def test_empty_input_returns_sized_bitmap(self):
        """Empty input -> num_keys-sized all-zeros bitmap (not Bitmap(0)), so a
        downstream ``&`` with a same-sized mask never hits a size mismatch."""
        merged = merge_bitmaps([], 5)
        assert merged.popcount() == 0
        mask = Bitmap(5)
        mask.set(2)
        assert (merged & mask).popcount() == 0  # would raise on size mismatch

    def test_empty_generator_returns_sized_bitmap(self):
        """A generator is truthy even when empty; the result is still size-5."""
        merged = merge_bitmaps((b for b in []), 5)
        assert merged.popcount() == 0
        assert (merged & Bitmap(5)).popcount() == 0

    def test_union_of_bitmaps(self):
        """Non-empty inputs are OR-merged into one num_keys-sized bitmap."""
        a, b = Bitmap(5), Bitmap(5)
        a.set(0)
        b.set(3)
        assert merge_bitmaps([a, b], 5).get_indices_list() == [0, 3]


# --------------------------------------------------------------------------- #
# Separated operators: fold / highest_set_bit / unfold                      #
# --------------------------------------------------------------------------- #


class TestFoldOperator:
    """``fold`` produces the servable bitmap (bit ``j`` = every group can serve
    a length-``j + 1`` prefix)."""

    def test_full_attention_servable_is_downward_closed(self):
        # full group present {0,1,2} of 4 -> servable lengths {1,2,3} -> bits
        # {0,1,2} (bit j == length j+1).
        found = _make_ranked(4, 1, [[(0, 0), (1, 0), (2, 0)]])
        servable = fold(found, 4, 1, [FULL_ATTENTION_WINDOW])
        assert servable.get_indices_list() == [0, 1, 2]

    def test_sliding_window_servable_is_gappy(self):
        # window-2 group present chunks {0,1,3,4} of 5. A length L is servable
        # iff chunks [L-2, L) present: L=1 ok(0), 2 ok(0,1), 3 no(1,2),
        # 4 no(2,3), 5 ok(3,4) -> lengths {1,2,5} -> bits {0,1,4}.
        found = _make_ranked(5, 1, [[(0, 0), (1, 0), (3, 0), (4, 0)]])
        servable = fold(found, 5, 1, [2])
        assert servable.get_indices_list() == [0, 1, 4]

    def test_nothing_present_is_empty(self):
        # No chunk present -> no length servable -> empty bitmap (highest_set_bit
        # returns -1, so the pipeline reports hit length 0).
        found = _make_ranked(3, 2, [[]])  # nothing present
        servable = fold(found, 3, 2, [FULL_ATTENTION_WINDOW])
        assert servable.get_indices_list() == []


class TestHighestSetBit:
    """``highest_set_bit`` returns the highest set bit, -1 if none."""

    def test_basic(self):
        bm = Bitmap(10)
        for i in (1, 4, 7):
            bm.set(i)
        assert highest_set_bit(bm) == 7

    def test_empty_returns_minus_one(self):
        assert highest_set_bit(Bitmap(10)) == -1
        assert highest_set_bit(Bitmap(0)) == -1

    def test_single_and_last_bit(self):
        bm = Bitmap(9)
        bm.set(8)
        assert highest_set_bit(bm) == 8


class TestUnfoldOperator:
    """``unfold`` expands a hit length into the ranked retain mask."""

    def test_full_plus_sliding_window(self):
        # hit=4, full group keeps [0,4), window-2 group keeps [2,4); 2 ranks.
        mask = unfold(4, 5, 2, [FULL_ATTENTION_WINDOW, 2])
        stride = 5 * 2
        expected = [0, 1, 2, 3, 4, 5, 6, 7]  # full: chunks 0..3 x 2 ranks
        expected += [stride + 4, stride + 5, stride + 6, stride + 7]  # win: c2,c3
        assert mask.get_indices_list() == expected

    def test_zero_hit_is_empty(self):
        assert unfold(0, 5, 2, [FULL_ATTENTION_WINDOW, 2]).get_indices_list() == []

    def test_hit_clamped_to_num_chunks(self):
        # hit beyond num_chunks is clamped; full group keeps every chunk.
        mask = unfold(99, 3, 1, [FULL_ATTENTION_WINDOW])
        assert mask.get_indices_list() == [0, 1, 2]


# --------------------------------------------------------------------------- #
# Native ops must match the pure-Python reference (_fold_python/_unfold_python) #
# bit-for-bit, on deterministic constructed inputs.                            #
# --------------------------------------------------------------------------- #


class TestNativeMatchesReference:
    # (num_chunks, num_ranks, group_windows) shapes, small to large.
    CASES = [
        (64, 1, [FULL_ATTENTION_WINDOW]),
        (64, 4, [FULL_ATTENTION_WINDOW, FULL_ATTENTION_WINDOW]),
        (100, 3, [FULL_ATTENTION_WINDOW, 2, 5, 1]),
        (300, 4, [FULL_ATTENTION_WINDOW, FULL_ATTENTION_WINDOW, 8, 32, 1]),
    ]

    def test_fold_matches_reference(self):
        for num_chunks, num_ranks, gw in self.CASES:
            nk = len(gw) * num_chunks * num_ranks
            bm = Bitmap(nk)
            # Deterministic irregular gap pattern (no RNG): drop ~1/7 of bits on
            # an irregular stride so windows and rank-reduction are exercised.
            bm.batched_set([i for i in range(nk) if (i * 5 + i // num_ranks) % 7 != 0])
            assert (
                fold(bm, num_chunks, num_ranks, gw).get_indices_list()
                == _fold_python(bm, num_chunks, num_ranks, gw).get_indices_list()
            ), f"fold mismatch C={num_chunks} R={num_ranks} gw={gw}"

    def test_unfold_matches_reference_at_boundaries(self):
        for num_chunks, num_ranks, gw in self.CASES:
            # Cover empty, both ends, and interior hit lengths.
            for hit in (0, 1, num_chunks // 3, num_chunks - 1, num_chunks):
                assert (
                    unfold(hit, num_chunks, num_ranks, gw).get_indices_list()
                    == _unfold_python(hit, num_chunks, num_ranks, gw).get_indices_list()
                ), f"unfold mismatch hit={hit} C={num_chunks} R={num_ranks} gw={gw}"


# --------------------------------------------------------------------------- #
# End-to-end: full fold -> highest_set_bit -> unfold pipeline against an    #
# independent reference modeling vLLM's hybrid prefix-cache hit logic.         #
# --------------------------------------------------------------------------- #


def _reference_longest_hit(num_chunks, group_present, group_windows):
    """Longest model-wide prefix hit, mirroring vLLM's per-group
    ``find_longest_cache_hit`` combined across a hybrid model (independent
    brute force; no vLLM import).

    A length-``L`` prefix is a model-wide hit iff every object group can serve
    it under its rule:

    * full attention (``window <= 0``): chunks ``[0, L)`` all present
      (vLLM ``FullAttentionManager``);
    * sliding window ``w``: chunks ``[max(0, L - w), L)`` all present
      (vLLM ``SlidingWindowManager``).

    Args:
        num_chunks: number of chunks.
        group_present: ``group_present[g]`` = set of chunk indices present for
            object group ``g`` (after requiring every kv_rank present).
        group_windows: per-group window size; ``<= 0`` means full attention.

    Returns:
        The largest ``L`` in ``[0, num_chunks]`` servable by all groups.
    """
    best = 0
    for length in range(num_chunks + 1):
        servable_by_all = True
        for present, window in zip(group_present, group_windows, strict=True):
            lo = 0 if window <= 0 else max(0, length - window)
            if not all(j in present for j in range(lo, length)):
                servable_by_all = False
                break
        if servable_by_all:
            best = length
    return best


def _expected_retained_indices(hit, num_chunks, num_ranks, group_windows):
    """The ranked retain-mask indices the pipeline should produce for ``hit``."""
    indices = []
    stride = num_chunks * num_ranks
    for g, window in enumerate(group_windows):
        lo, hi = unfold_range(hit, window)
        for j in range(lo, hi):
            base = g * stride + j * num_ranks
            indices.extend(range(base, base + num_ranks))
    return sorted(indices)


class TestEndToEndAgainstVllmStyleReference:
    """Drive the full fold/highest_set_bit/unfold pipeline and compare the
    hit length and retain mask against an independent vLLM-style oracle."""

    def _run(
        self, num_chunks, num_ranks, group_windows, present_cells, expected_hit=None
    ):
        # present_cells[g] = set of (chunk, rank) present for group g.
        stride = num_chunks * num_ranks
        bm = Bitmap(len(group_windows) * stride)
        for g, cells in enumerate(present_cells):
            for chunk, rank in cells:
                bm.set(g * stride + chunk * num_ranks + rank)
        hit, mask = fold_unfold_ranked(bm, num_chunks, num_ranks, group_windows)

        # Reference: a chunk is present for a group only if all ranks present.
        group_present = [
            {
                chunk
                for chunk in range(num_chunks)
                if all((chunk, r) in cells for r in range(num_ranks))
            }
            for cells in present_cells
        ]
        ref_hit = _reference_longest_hit(num_chunks, group_present, group_windows)
        assert hit == ref_hit, (
            f"hit {hit} != reference {ref_hit} "
            f"(windows={group_windows}, present={group_present})"
        )
        if expected_hit is not None:
            assert hit == expected_hit, f"hit {hit} != hand-derived {expected_hit}"
        assert mask.get_indices_list() == _expected_retained_indices(
            hit, num_chunks, num_ranks, group_windows
        )

    def test_full_attention_only_is_contiguous_prefix(self):
        # Two full-attention groups; hit is the shortest contiguous prefix.
        self._run(
            num_chunks=6,
            num_ranks=2,
            group_windows=[FULL_ATTENTION_WINDOW, FULL_ATTENTION_WINDOW],
            present_cells=[
                {(j, r) for j in range(5) for r in range(2)},  # chunks 0..4
                {(j, r) for j in range(3) for r in range(2)},  # chunks 0..2
            ],
        )  # -> hit 3

    def test_sliding_window_tail_extends_hit(self):
        # Full group has 0..5; window-2 group only has the tail {4,5} -> hit 6.
        self._run(
            num_chunks=6,
            num_ranks=1,
            group_windows=[FULL_ATTENTION_WINDOW, 2],
            present_cells=[
                {(j, 0) for j in range(6)},
                {(4, 0), (5, 0)},
            ],
        )

    def test_mamba_window_one(self):
        self._run(
            num_chunks=4,
            num_ranks=1,
            group_windows=[FULL_ATTENTION_WINDOW, 1],
            present_cells=[{(j, 0) for j in range(4)}, {(3, 0)}],
        )  # -> hit 4

    def test_missing_rank_breaks_chunk(self):
        # chunk 2 missing one rank -> not present -> caps the full-attn prefix.
        self._run(
            num_chunks=5,
            num_ranks=2,
            group_windows=[FULL_ATTENTION_WINDOW],
            present_cells=[
                {(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (3, 0), (3, 1)},
            ],
        )  # -> hit 2

    def test_large_adversarial_hybrid(self):
        # A large, deterministic scenario engineered so the hit is decided by a
        # mid-window sliding-window gap, with decoy later gaps a wrong algorithm
        # might trip on. 300 chunks x 4 ranks x 5 groups.
        #
        # windows: [full, full, SW8, SW32, mamba]
        #   - g0 full: gap at chunk 150          -> full prefix capped at 150
        #   - g1 full: one rank of chunk 220 gone -> chunk 220 absent (rank test)
        #   - g2 SW8:  gaps at 10,11,12 (old)     -> must NOT affect a hit > 20
        #   - g3 SW32: gap at chunk 130           -> lengths 131..162 unservable
        #   - g4 mamba: fully present
        # The only length servable by all groups and <= 150 is 130 (g3's gap at
        # 130 blocks 131..162; g3 is servable again only at >= 163, beyond g0's
        # 150 cap). So the model-wide hit is exactly 130.
        num_chunks, num_ranks = 300, 4
        group_windows = [FULL_ATTENTION_WINDOW, FULL_ATTENTION_WINDOW, 8, 32, 1]
        cells = [
            {(j, r) for j in range(num_chunks) for r in range(num_ranks)}
            for _ in group_windows
        ]
        cells[0] -= {(150, r) for r in range(num_ranks)}
        cells[1].discard((220, 2))
        cells[2] -= {(j, r) for j in (10, 11, 12) for r in range(num_ranks)}
        cells[3] -= {(130, r) for r in range(num_ranks)}
        self._run(num_chunks, num_ranks, group_windows, cells, expected_hit=130)

    def test_dense_deterministic_pattern(self):
        # Wide grid with a deterministic irregular gap pattern (no RNG): drops
        # ~1/9 of cells on an irregular stride so many window/intersection
        # boundaries are exercised. Validated against the reference oracle.
        num_chunks, num_ranks = 128, 3
        group_windows = [FULL_ATTENTION_WINDOW, 2, 5, 1]
        cells = []
        for g in range(len(group_windows)):
            present = {
                (j, r)
                for j in range(num_chunks)
                for r in range(num_ranks)
                if (j * 7 + r * 3 + g * 5) % 9 != 0
            }
            cells.append(present)
        self._run(num_chunks, num_ranks, group_windows, cells)
