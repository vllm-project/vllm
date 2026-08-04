# SPDX-License-Identifier: Apache-2.0
"""Optimized vs current V3 lookup — pure-Python module test.

Compares two lookup designs over the same V3 matcher state:

* ``current_v3_lookup`` — single ``match_sub_sequence`` call returning one
  flat list; each result is classified as prefix/non-prefix at retrieve
  time via ``old_st == cur_st``.

* ``optimal_lookup`` — chain-walk first (contiguous aligned prefix),
  matcher probe filtered to ``cur_chunk_idx >= K``, partitioned into
  ``prefix_hits`` / ``cb_aligned`` / ``cb_shifted``.

Both paths share the same matcher (same registrations, same hashes), so the
union of optimal's buckets must equal the V3 result set (after dedupe).
Re-rope decisions must match: ``cb_shifted`` ↔ ``old_st != cur_st``,
``prefix_hits ∪ cb_aligned`` ↔ ``old_st == cur_st``.
"""

# Standard
from collections.abc import Callable
from dataclasses import dataclass
import time

# Third Party
import numpy as np
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.custom_types import CBMatchResult
from lmcache.v1.multiprocess.modules.blend_v3 import BlendTokenRangeMatcherV3
from lmcache.v1.multiprocess.token_hasher import rolling_hash_windows_numba

CHUNK_SIZE = 256
# vLLM v1's typical paged-block size; retrieve drops cur_st % BLOCK_SIZE != 0.
BLOCK_SIZE = 16


# ---------------------------------------------------------------------------
# Lookup implementations under test
# ---------------------------------------------------------------------------


def _greedy_leftmost_dedupe(matches: list[CBMatchResult]) -> list[CBMatchResult]:
    """Drop overlapping matches keeping the leftmost — same as V2/V3 lookup."""
    matches = sorted(matches, key=lambda r: r.cur_st)
    out: list[CBMatchResult] = []
    covered_end = -1
    for r in matches:
        if r.cur_st >= covered_end:
            out.append(r)
            covered_end = r.cur_ed
    return out


def current_v3_lookup(
    matcher: BlendTokenRangeMatcherV3,
    query: list[int],
) -> list[CBMatchResult]:
    """Mirror of the V3 lookup body (without the storage-prefetch round)."""
    matches = matcher.match_sub_sequence(query)
    return _greedy_leftmost_dedupe(matches)


@dataclass
class OptimalLookupResult:
    """Three-bucket plan emitted by the optimal lookup."""

    prefix_hits: list[CBMatchResult]  # chain walk; old_st == cur_st, contiguous from 0
    cb_aligned: list[CBMatchResult]  # matcher hit; old_st == cur_st, past chain break
    cb_shifted: list[CBMatchResult]  # matcher hit; old_st != cur_st


def _simulate_chain_walk(
    matches_by_chunk: dict[int, CBMatchResult],
    num_chunks: int,
) -> list[CBMatchResult]:
    """Longest contiguous prefix where each chunk i has an aligned match.

    Stand-in for the real chain walker, which probes storage by chained
    chunk hash. Here we use ``matches_by_chunk`` (built from the matcher's
    output) as the ground truth for "chunk i is in storage and aligns".
    """
    out: list[CBMatchResult] = []
    for i in range(num_chunks):
        m = matches_by_chunk.get(i)
        if m is None or m.old_st != m.cur_st:
            break
        out.append(m)
    return out


def optimal_lookup(
    matcher: BlendTokenRangeMatcherV3,
    query: list[int],
    chunk_size: int,
    chain_walk_fn=None,
) -> OptimalLookupResult:
    """Chain walk + filtered matcher probe + three-bucket partition.

    Args:
        matcher: V3 matcher with registered chunks.
        query: Query token sequence.
        chunk_size: Chunk size.
        chain_walk_fn: Optional callable ``(query, chunk_size) -> list[CBMatchResult]``
            that simulates the chain-walk storage probe **without** touching the
            matcher. When provided and it covers the full query, the matcher
            probe is skipped (early-out). When ``None``, the chain walk is
            derived from the matcher's own output (no early-out — used by
            equivalence tests to keep the implementation closed-form).
    """
    num_chunks = len(query) // chunk_size

    if chain_walk_fn is not None:
        prefix_hits = chain_walk_fn(query, chunk_size)
        if len(prefix_hits) == num_chunks:
            return OptimalLookupResult(
                prefix_hits=prefix_hits, cb_aligned=[], cb_shifted=[]
            )
        K = len(prefix_hits)
        matches = _greedy_leftmost_dedupe(matcher.match_sub_sequence(query))
    else:
        matches = _greedy_leftmost_dedupe(matcher.match_sub_sequence(query))
        aligned_by_chunk: dict[int, CBMatchResult] = {}
        for m in matches:
            if m.cur_st % chunk_size == 0 and m.old_st == m.cur_st:
                aligned_by_chunk.setdefault(m.cur_st // chunk_size, m)
        prefix_hits = _simulate_chain_walk(aligned_by_chunk, num_chunks)
        K = len(prefix_hits)

    cb_aligned: list[CBMatchResult] = []
    cb_shifted: list[CBMatchResult] = []
    for m in matches:
        if m.cur_st // chunk_size < K:
            continue  # already covered by prefix_hits
        if m.old_st == m.cur_st:
            cb_aligned.append(m)
        else:
            cb_shifted.append(m)

    return OptimalLookupResult(
        prefix_hits=prefix_hits,
        cb_aligned=cb_aligned,
        cb_shifted=cb_shifted,
    )


def make_chain_walk_storage(
    stored_chunks: list[tuple[int, list[int]]],
) -> Callable:
    """Build a chain-walk callable backed by a dict.

    Models the real chain-walk path: walks contiguous chunks from position 0,
    probes a fast in-memory map keyed by ``(chunk_idx, tuple(tokens))``, stops
    at the first miss. **Does not** touch the matcher.

    Args:
        stored_chunks: List of ``(chunk_idx, tokens)`` pairs known to be in
            storage at that chunk index.

    Returns:
        Function ``(query, chunk_size) -> list[CBMatchResult]``.
    """
    storage: dict[tuple[int, tuple[int, ...]], bytes] = {}
    for idx, toks in stored_chunks:
        storage[(idx, tuple(toks))] = ObjectKey.IntHash2Bytes(900000 + idx)

    def walk(query: list[int], chunk_size: int) -> list[CBMatchResult]:
        out: list[CBMatchResult] = []
        for i in range(len(query) // chunk_size):
            toks = tuple(query[i * chunk_size : (i + 1) * chunk_size])
            h = storage.get((i, toks))
            if h is None:
                break
            out.append(
                CBMatchResult(
                    old_st=i * chunk_size,
                    old_ed=(i + 1) * chunk_size,
                    cur_st=i * chunk_size,
                    cur_ed=(i + 1) * chunk_size,
                    hash=h,
                )
            )
        return out

    return walk


def match_strided(
    matcher: BlendTokenRangeMatcherV3,
    query: list[int],
    chunk_size: int,
    stride: int,
    start_pos: int = 0,
) -> list[CBMatchResult]:
    """Aligned-only probe — same hash function as V3 but only probes
    positions ``{start_pos, start_pos + stride, start_pos + 2*stride, ...}``.

    Equivalent to V3 ``match_sub_sequence`` post-filtered to
    ``cur_st % stride == 0 and cur_st >= start_pos``, except this version
    does not visit non-strided positions in its probe loop, so it scales
    as ``O(N / stride)`` instead of ``O(N)``.

    Accesses matcher internals directly — module test only.

    Args:
        matcher: V3 matcher with registered chunks.
        query: Query token sequence.
        chunk_size: Chunk size (must match matcher.chunk_size).
        stride: Step between probe positions (e.g., BLOCK_SIZE).
        start_pos: First position to probe (used for trim — skip the
            chain-walk-covered prefix).
    """
    if len(query) < chunk_size:
        return []
    arr = np.array(query, dtype=np.uint64)
    rolling = rolling_hash_windows_numba(arr, chunk_size, matcher._BASE)
    n_positions = rolling.shape[0]

    with matcher._lock:
        if not matcher._chunk_token_hash:
            return []

        results: list[CBMatchResult] = []
        seen_cids: set[int] = set()
        mask = int(matcher._mask)
        for q_pos in range(start_pos, n_positions, stride):
            r = int(rolling[q_pos])
            cid = int(matcher._table_id[r & mask])
            if cid < 0 or cid in seen_cids:
                continue
            if r != matcher._chunk_poly_hash[cid]:
                continue  # bucket collision
            th = matcher._chunk_token_hash[cid]
            if th is None:
                continue  # evicted
            old_st = matcher._token_hash_to_start.get(th)
            if old_st is None:
                continue
            seen_cids.add(cid)
            results.append(
                CBMatchResult(
                    old_st=old_st,
                    old_ed=old_st + chunk_size,
                    cur_st=q_pos,
                    cur_ed=q_pos + chunk_size,
                    hash=th,
                )
            )
        return results


def optimal_lookup_aligned(
    matcher: BlendTokenRangeMatcherV3,
    query: list[int],
    chunk_size: int,
    block_size: int,
    chain_walk_fn,
) -> OptimalLookupResult:
    """Chain walk + block-aligned probe + chain-walk trim + fused partition.

    Implements Opts 1–3 from the design discussion:
    * Probe only at ``block_size`` stride (Opt 1) — zero coverage loss vs
      V3 because retrieve already drops non-block-aligned ``cur_st``.
    * Trim probe range to skip ``[0, K * chunk_size)`` (Opt 2) — the
      chain walk already covers it.
    * Partition into aligned/shifted in the same pass as the probe
      (Opt 3) — no extra Python loop.
    """
    num_chunks = len(query) // chunk_size

    prefix_hits = chain_walk_fn(query, chunk_size)
    if len(prefix_hits) == num_chunks:
        return OptimalLookupResult(
            prefix_hits=prefix_hits, cb_aligned=[], cb_shifted=[]
        )
    K = len(prefix_hits)
    start_pos = K * chunk_size

    raw = match_strided(matcher, query, chunk_size, block_size, start_pos=start_pos)
    raw = _greedy_leftmost_dedupe(raw)

    cb_aligned: list[CBMatchResult] = []
    cb_shifted: list[CBMatchResult] = []
    for m in raw:
        if m.old_st == m.cur_st:
            cb_aligned.append(m)
        else:
            cb_shifted.append(m)

    return OptimalLookupResult(
        prefix_hits=prefix_hits, cb_aligned=cb_aligned, cb_shifted=cb_shifted
    )


# ---------------------------------------------------------------------------
# Equivalence assertions
# ---------------------------------------------------------------------------


def _match_key(m: CBMatchResult) -> tuple[int, int, bytes]:
    return (m.cur_st, m.old_st, m.hash)


def assert_equivalent(
    v3_result: list[CBMatchResult],
    opt: OptimalLookupResult,
) -> None:
    """Optimal's three buckets cover the V3 list with no overlap and same classes."""
    union = opt.prefix_hits + opt.cb_aligned + opt.cb_shifted

    # 1. Union == V3 (same set of match keys).
    v3_keys = {_match_key(m) for m in v3_result}
    union_keys = {_match_key(m) for m in union}
    assert union_keys == v3_keys, (
        f"Union mismatch:\n  V3 only:   {v3_keys - union_keys}\n  "
        f"Opt only:  {union_keys - v3_keys}"
    )

    # 2. No bucket overlap.
    p_keys = {_match_key(m) for m in opt.prefix_hits}
    a_keys = {_match_key(m) for m in opt.cb_aligned}
    s_keys = {_match_key(m) for m in opt.cb_shifted}
    assert not (p_keys & a_keys), "prefix_hits ∩ cb_aligned non-empty"
    assert not (p_keys & s_keys), "prefix_hits ∩ cb_shifted non-empty"
    assert not (a_keys & s_keys), "cb_aligned ∩ cb_shifted non-empty"

    # 3. cb_shifted ↔ old_st != cur_st; prefix_hits ∪ cb_aligned ↔ aligned.
    for m in opt.prefix_hits:
        assert m.old_st == m.cur_st, "prefix_hits contains shifted match"
    for m in opt.cb_aligned:
        assert m.old_st == m.cur_st, "cb_aligned contains shifted match"
    for m in opt.cb_shifted:
        assert m.old_st != m.cur_st, "cb_shifted contains aligned match"

    # 4. prefix_hits is contiguous from chunk 0 and strictly ordered.
    for i, m in enumerate(opt.prefix_hits):
        assert m.cur_st == i * CHUNK_SIZE, (
            f"prefix_hits[{i}].cur_st = {m.cur_st}, expected {i * CHUNK_SIZE}"
        )


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _register(
    matcher: BlendTokenRangeMatcherV3,
    tokens: list[int],
    chunk_hash_seeds: list[int],
    position_offset: int = 0,
    start_chunk_idx: int = 0,
) -> None:
    hashes = [ObjectKey.IntHash2Bytes(s) for s in chunk_hash_seeds]
    matcher.on_new_token_hashes(
        tokens,
        hashes,
        start_chunk_idx=start_chunk_idx,
        position_offset=position_offset,
    )


def _chunk(seed: int) -> list[int]:
    """Deterministic chunk_size tokens derived from seed; unique per seed."""
    return [seed * CHUNK_SIZE + i + 1 for i in range(CHUNK_SIZE)]


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestOptimizedLookupEquivalence:
    """Same matcher, two lookup paths — outputs must be equivalent."""

    def test_full_prefix_hit(self):
        """Query == stored sequence verbatim → all chunks become prefix_hits."""
        matcher = BlendTokenRangeMatcherV3(chunk_size=CHUNK_SIZE)
        stored = _chunk(1) + _chunk(2) + _chunk(3) + _chunk(4)
        _register(matcher, stored, [101, 102, 103, 104])

        v3 = current_v3_lookup(matcher, stored)
        opt = optimal_lookup(matcher, stored, CHUNK_SIZE)

        assert_equivalent(v3, opt)
        assert len(opt.prefix_hits) == 4
        assert opt.cb_aligned == []
        assert opt.cb_shifted == []

    def test_no_hits(self):
        """Disjoint query → empty result, empty buckets."""
        matcher = BlendTokenRangeMatcherV3(chunk_size=CHUNK_SIZE)
        _register(matcher, _chunk(1) + _chunk(2), [201, 202])

        query = _chunk(9) + _chunk(8)
        v3 = current_v3_lookup(matcher, query)
        opt = optimal_lookup(matcher, query, CHUNK_SIZE)

        assert v3 == []
        assert opt.prefix_hits == []
        assert opt.cb_aligned == []
        assert opt.cb_shifted == []

    def test_segmented_prefix_aligned_gap(self):
        """First chunk differs (chain breaks at 0), later chunks align positionally.

        Stored: [A B C D] at positions 0..4*c
        Query:  [X B C D] — chunk 0 differs, chunks 1..3 are aligned hits.
        Expected: prefix_hits=[], cb_aligned=[chunk1..chunk3], cb_shifted=[].
        """
        matcher = BlendTokenRangeMatcherV3(chunk_size=CHUNK_SIZE)
        a, b, c, d, x = _chunk(1), _chunk(2), _chunk(3), _chunk(4), _chunk(99)
        stored = a + b + c + d
        _register(matcher, stored, [301, 302, 303, 304])

        query = x + b + c + d
        v3 = current_v3_lookup(matcher, query)
        opt = optimal_lookup(matcher, query, CHUNK_SIZE)

        assert_equivalent(v3, opt)
        assert opt.prefix_hits == []
        assert len(opt.cb_aligned) == 3
        assert opt.cb_shifted == []
        for i, m in enumerate(opt.cb_aligned, start=1):
            assert m.cur_st == i * CHUNK_SIZE
            assert m.old_st == m.cur_st

    def test_pure_non_prefix_shifted(self):
        """Stored chunk reappears at a different position → cb_shifted only."""
        matcher = BlendTokenRangeMatcherV3(chunk_size=CHUNK_SIZE)
        a, b = _chunk(1), _chunk(2)
        stored = a + b
        _register(matcher, stored, [401, 402])

        # Insert two filler chunks before chunk 'b' — pushes it to position 3*c.
        f1, f2, f3 = _chunk(50), _chunk(51), _chunk(52)
        query = f1 + f2 + f3 + b
        v3 = current_v3_lookup(matcher, query)
        opt = optimal_lookup(matcher, query, CHUNK_SIZE)

        assert_equivalent(v3, opt)
        assert opt.prefix_hits == []
        assert opt.cb_aligned == []
        assert len(opt.cb_shifted) == 1
        m = opt.cb_shifted[0]
        assert m.old_st == CHUNK_SIZE  # was second chunk in stored seq
        assert m.cur_st == 3 * CHUNK_SIZE  # now at position 3 in query

    def test_mixed_prefix_then_shifted(self):
        """Contiguous prefix hit, chain break, shifted hit past the break."""
        matcher = BlendTokenRangeMatcherV3(chunk_size=CHUNK_SIZE)
        a, b, c, d = _chunk(1), _chunk(2), _chunk(3), _chunk(4)
        stored_main = a + b + c + d  # registered as a single 4-chunk run
        _register(matcher, stored_main, [501, 502, 503, 504])
        # Separately register chunk 'd' as if it had been stored at position 0
        # — gives us a registration with old_st=0 we can shift in a query.
        _register(matcher, d, [505])

        # Query: a, b, unrelated, d. Prefix hits = [a, b]; shifted = [d@0 → 3c].
        g1 = _chunk(60)
        query = a + b + g1 + d
        v3 = current_v3_lookup(matcher, query)
        opt = optimal_lookup(matcher, query, CHUNK_SIZE)

        assert_equivalent(v3, opt)
        # 'd' has multiple stored registrations; matcher picks one. Just check shapes.
        assert len(opt.prefix_hits) == 2
        # 'd' at query position 3c could match either old_st=3c (aligned) or
        # old_st=0 (shifted).
        assert len(opt.cb_aligned) + len(opt.cb_shifted) == 1

    def test_mixed_prefix_gap_shifted(self):
        """Prefix hit at chunk 0, gap, aligned hit, shifted hit — all three buckets."""
        matcher = BlendTokenRangeMatcherV3(chunk_size=CHUNK_SIZE)
        a, b, c = _chunk(1), _chunk(2), _chunk(3)
        # Register one sequence [a, b, c] with positions 0, c, 2c
        _register(matcher, a + b + c, [601, 602, 603])
        # Register chunk 'c' alone, recorded at position 0 — shifted candidate.
        _register(matcher, c, [604])

        # Query: a (prefix), unrelated (gap), c at 2c (aligned), c again at 3c.
        g = _chunk(70)
        query = a + g + c + c
        v3 = current_v3_lookup(matcher, query)
        opt = optimal_lookup(matcher, query, CHUNK_SIZE)

        assert_equivalent(v3, opt)
        # Prefix walk reaches chunk 0 only (chunk 1 is unrelated).
        assert len(opt.prefix_hits) == 1
        assert opt.prefix_hits[0].cur_st == 0
        assert opt.prefix_hits[0].old_st == 0

    def test_partial_sequence_registration(self):
        """Registration with position_offset > 0; queries find at that offset."""
        matcher = BlendTokenRangeMatcherV3(chunk_size=CHUNK_SIZE)
        b, c = _chunk(2), _chunk(3)
        # Pretend b, c were stored at absolute positions [c, 2c] — register with offset.
        _register(matcher, b + c, [701, 702], position_offset=CHUNK_SIZE)

        # Query containing b, c at positions c, 2c — both aligned (cb_aligned).
        a = _chunk(80)
        query = a + b + c
        v3 = current_v3_lookup(matcher, query)
        opt = optimal_lookup(matcher, query, CHUNK_SIZE)

        assert_equivalent(v3, opt)
        # Chunk 0 (a) has no registration → prefix walk stops at 0.
        assert opt.prefix_hits == []
        # b, c found at their original positions → both aligned.
        assert len(opt.cb_aligned) == 2
        assert opt.cb_shifted == []


class TestOptimizedLookupTiming:
    """Timing comparison — informational only, no asserts.

    The optimal path's win comes from its **early-out**: when the chain walk
    covers the whole query, the matcher is skipped entirely. The tests below
    drive ``optimal_lookup`` with a chain-walk callable backed by a dict, so
    the timing reflects the real cost model (chain walk = O(K) dict probes,
    matcher = O(N) numba probe).
    """

    @pytest.mark.parametrize(
        "n_chunks,scenario",
        [
            (128, "full_prefix"),  # K = N → early-out, matcher skipped
            (
                128,
                "no_prefix_shifted",
            ),  # K = 0 → chain walk fails fast, matcher does all
            (128, "half_prefix"),  # K = N/2 → both pay something
        ],
    )
    def test_timing(self, n_chunks, scenario, capsys):
        matcher = BlendTokenRangeMatcherV3(chunk_size=CHUNK_SIZE)
        stored: list[int] = []
        seeds = []
        chunks_for_storage: list[tuple[int, list[int]]] = []
        for i in range(n_chunks):
            c = _chunk(i)
            stored.extend(c)
            seeds.append(1000 + i)
            chunks_for_storage.append((i, c))
        _register(matcher, stored, seeds)
        chain_walk = make_chain_walk_storage(chunks_for_storage)

        if scenario == "full_prefix":
            query = list(stored)
        elif scenario == "no_prefix_shifted":
            rotated: list[int] = []
            for i in range(n_chunks):
                rotated.extend(_chunk((i + 1) % n_chunks))
            query = rotated
        elif scenario == "half_prefix":
            half = n_chunks // 2
            query = stored[: half * CHUNK_SIZE]
            for i in range(half, n_chunks):
                query.extend(_chunk((i + 1) % n_chunks))
        else:
            raise ValueError(scenario)

        # Warm up numba JIT.
        matcher.match_sub_sequence(query)
        chain_walk(query, CHUNK_SIZE)

        n_iter = 50
        t0 = time.perf_counter()
        for _ in range(n_iter):
            current_v3_lookup(matcher, query)
        t_v3 = (time.perf_counter() - t0) / n_iter * 1e6  # µs

        t0 = time.perf_counter()
        for _ in range(n_iter):
            optimal_lookup(matcher, query, CHUNK_SIZE, chain_walk_fn=chain_walk)
        t_opt = (time.perf_counter() - t0) / n_iter * 1e6  # µs

        with capsys.disabled():
            print(
                f"\n  [{scenario:>20s}] N={n_chunks:>4d} chunks  "
                f"V3={t_v3:8.1f}µs  OPT={t_opt:8.1f}µs  "
                f"OPT/V3={t_opt / t_v3:.2f}x"
            )


# =============================================================================
# Block-aligned probe (Opt 1) + trim (Opt 2) + fused partition (Opt 3)
# =============================================================================


def _v3_filtered_to_block_aligned(
    v3_result: list[CBMatchResult], block_size: int
) -> set[tuple[int, int, bytes]]:
    """Set of V3 matches that survive retrieve's ``cur_st % block_size == 0`` drop.

    Strided probe's correctness target: it must find at least every
    block-aligned chunk-content occurrence in the query. V3's first-cur_st
    behavior may keep a non-block-aligned position for a chunk that also
    has a block-aligned occurrence; the strided probe will find the
    block-aligned occurrence instead. So the strided probe is a superset
    by **content** (hash) of what V3 surfaces after the retrieve filter.
    """
    return {_match_key(m) for m in v3_result if m.cur_st % block_size == 0}


class TestStridedProbe:
    """Block-aligned probe must give zero-loss coverage vs V3 + retrieve filter."""

    def test_returns_only_block_aligned_cur_st(self):
        """Every match has cur_st divisible by BLOCK_SIZE."""
        matcher = BlendTokenRangeMatcherV3(chunk_size=CHUNK_SIZE)
        a, b = _chunk(1), _chunk(2)
        _register(matcher, a + b, [801, 802])

        # Insert a 32-token shim (block-aligned, not chunk-aligned).
        shim = list(range(70000, 70032))  # length 32 = 2 * BLOCK_SIZE
        query = shim + a + b
        results = match_strided(matcher, query, CHUNK_SIZE, BLOCK_SIZE)

        assert len(results) >= 1
        for r in results:
            assert r.cur_st % BLOCK_SIZE == 0

    def test_finds_block_aligned_shifted_match(self):
        """Stored chunk at block-aligned, non-chunk-aligned query pos → found.

        V3 (stride=1) finds these too; strided (stride=block) must as well.
        """
        matcher = BlendTokenRangeMatcherV3(chunk_size=CHUNK_SIZE)
        a = _chunk(1)
        _register(matcher, a, [901])

        # Query: 32-token prefix (block-aligned but not chunk-aligned start).
        shim = list(range(80000, 80032))
        query = shim + a
        results = match_strided(matcher, query, CHUNK_SIZE, BLOCK_SIZE)

        assert len(results) == 1
        assert results[0].old_st == 0
        assert results[0].cur_st == 32
        assert results[0].cur_st % BLOCK_SIZE == 0

    def test_zero_loss_vs_v3_filtered(self):
        """Strided probe content-hash set ⊇ V3 set filtered to block-aligned cur_st."""
        matcher = BlendTokenRangeMatcherV3(chunk_size=CHUNK_SIZE)
        chunks = [_chunk(i) for i in range(5)]
        stored = sum(chunks, [])
        _register(matcher, stored, [1100 + i for i in range(5)])

        # Mixed query: shifted by 48 tokens (3 * BLOCK_SIZE, not chunk-aligned).
        shim = list(range(90000, 90048))
        query = shim + stored

        v3 = current_v3_lookup(matcher, query)
        v3_block_aligned = _v3_filtered_to_block_aligned(v3, BLOCK_SIZE)
        strided = match_strided(matcher, query, CHUNK_SIZE, BLOCK_SIZE)
        strided_hashes = {m.hash for m in strided}
        v3_block_aligned_hashes = {key[2] for key in v3_block_aligned}

        # Every block-aligned-cur_st hit that V3 surfaces must be discoverable
        # by the strided probe (same hash). Strided may discover additional
        # block-aligned occurrences that V3 dropped in favor of an earlier
        # non-block-aligned cur_st.
        assert v3_block_aligned_hashes <= strided_hashes, (
            f"Strided missed hashes V3 found:\n"
            f"  v3 block-aligned: {v3_block_aligned_hashes}\n"
            f"  strided:          {strided_hashes}"
        )

    def test_trim_skips_prefix_range(self):
        """``start_pos > 0`` skips the chain-walk-covered prefix."""
        matcher = BlendTokenRangeMatcherV3(chunk_size=CHUNK_SIZE)
        a, b, c = _chunk(1), _chunk(2), _chunk(3)
        _register(matcher, a + b + c, [1201, 1202, 1203])

        query = a + b + c
        # start_pos at chunk 1's beginning: must skip chunk 0's match.
        strided = match_strided(
            matcher, query, CHUNK_SIZE, BLOCK_SIZE, start_pos=CHUNK_SIZE
        )

        assert len(strided) == 2
        for r in strided:
            assert r.cur_st >= CHUNK_SIZE


class TestOptimalLookupAligned:
    """End-to-end equivalence for optimal_lookup_aligned."""

    def _make(self):
        matcher = BlendTokenRangeMatcherV3(chunk_size=CHUNK_SIZE)
        chunks = [_chunk(i) for i in range(4)]
        stored = sum(chunks, [])
        _register(matcher, stored, [1300 + i for i in range(4)])
        chain_walk = make_chain_walk_storage(list(enumerate(chunks)))
        return matcher, stored, chunks, chain_walk

    def test_full_prefix_early_out(self):
        matcher, stored, _, chain_walk = self._make()
        opt = optimal_lookup_aligned(
            matcher, stored, CHUNK_SIZE, BLOCK_SIZE, chain_walk
        )
        assert len(opt.prefix_hits) == 4
        assert opt.cb_aligned == []
        assert opt.cb_shifted == []

    def test_segmented_prefix(self):
        """Chunk 0 mismatch → cb_aligned picks up the rest with no re-rope."""
        matcher, _, chunks, chain_walk = self._make()
        x = _chunk(99)
        query = x + chunks[1] + chunks[2] + chunks[3]
        opt = optimal_lookup_aligned(matcher, query, CHUNK_SIZE, BLOCK_SIZE, chain_walk)
        assert opt.prefix_hits == []
        assert len(opt.cb_aligned) == 3
        assert opt.cb_shifted == []
        for m in opt.cb_aligned:
            assert m.old_st == m.cur_st
            assert m.cur_st % BLOCK_SIZE == 0

    def test_shifted_at_block_boundary(self):
        """Stored chunk appears at block-aligned but not chunk-aligned cur_st."""
        matcher, _, chunks, chain_walk = self._make()
        shim = list(range(80000, 80032))  # 32 tokens = 2 * BLOCK_SIZE
        query = shim + chunks[1] + chunks[2]
        opt = optimal_lookup_aligned(matcher, query, CHUNK_SIZE, BLOCK_SIZE, chain_walk)

        # Chunk 0 of query is the shim → chain walk stops at 0.
        assert opt.prefix_hits == []
        # chunks[1], chunks[2] appear at cur_st=32, 288 — both shifted from stored.
        assert opt.cb_aligned == []
        assert len(opt.cb_shifted) == 2
        for m in opt.cb_shifted:
            assert m.cur_st % BLOCK_SIZE == 0
            assert m.old_st != m.cur_st


class TestOptimalLookupAlignedTiming:
    """Cost of optimal_lookup_aligned vs current V3, with chain-walk early-out."""

    @pytest.mark.parametrize(
        "n_chunks,scenario",
        [
            (128, "full_prefix"),
            (128, "no_prefix_shifted"),
            (128, "half_prefix"),
        ],
    )
    def test_timing(self, n_chunks, scenario, capsys):
        matcher = BlendTokenRangeMatcherV3(chunk_size=CHUNK_SIZE)
        stored: list[int] = []
        seeds = []
        chunks_for_storage: list[tuple[int, list[int]]] = []
        for i in range(n_chunks):
            c = _chunk(i)
            stored.extend(c)
            seeds.append(2000 + i)
            chunks_for_storage.append((i, c))
        _register(matcher, stored, seeds)
        chain_walk = make_chain_walk_storage(chunks_for_storage)

        if scenario == "full_prefix":
            query = list(stored)
        elif scenario == "no_prefix_shifted":
            rotated: list[int] = []
            for i in range(n_chunks):
                rotated.extend(_chunk((i + 1) % n_chunks))
            query = rotated
        elif scenario == "half_prefix":
            half = n_chunks // 2
            query = stored[: half * CHUNK_SIZE]
            for i in range(half, n_chunks):
                query.extend(_chunk((i + 1) % n_chunks))
        else:
            raise ValueError(scenario)

        # Warm up numba JIT for both kernels.
        matcher.match_sub_sequence(query)
        match_strided(matcher, query, CHUNK_SIZE, BLOCK_SIZE)
        chain_walk(query, CHUNK_SIZE)

        n_iter = 50
        t0 = time.perf_counter()
        for _ in range(n_iter):
            current_v3_lookup(matcher, query)
        t_v3 = (time.perf_counter() - t0) / n_iter * 1e6

        t0 = time.perf_counter()
        for _ in range(n_iter):
            optimal_lookup_aligned(matcher, query, CHUNK_SIZE, BLOCK_SIZE, chain_walk)
        t_opt = (time.perf_counter() - t0) / n_iter * 1e6

        with capsys.disabled():
            print(
                f"\n  [{scenario:>20s}] N={n_chunks:>4d} chunks  "
                f"V3={t_v3:8.1f}µs  ALIGNED={t_opt:8.1f}µs  "
                f"speedup={t_v3 / t_opt:.1f}x"
            )
