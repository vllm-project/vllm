# SPDX-License-Identifier: Apache-2.0
"""Tests for the coordinator global CacheBlend fingerprint directory."""

# Standard
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.mp_coordinator.blend_directory import (
    GlobalBlendMatcher,
    StoreRange,
)
from lmcache.v1.mp_coordinator.schemas import decode_tokens, encode_tokens

CHUNK = 3
SCOPE = "model-a"


def store_range(
    prefix: str, tokens: list[int], *, scope: str = SCOPE, old_st_base: int = 0
) -> StoreRange:
    """A StoreRange with one object_key per complete chunk (mirrors publisher)."""
    n_chunks = len(tokens) // CHUNK
    return StoreRange(
        model_scope=scope,
        tokens=tokens,
        object_keys=[f"{prefix}{i}" for i in range(n_chunks)],
        old_st_base=old_st_base,
    )


class TestRegisterMatch:
    def test_full_reuse(self) -> None:
        m = GlobalBlendMatcher(chunk_size=CHUNK)
        doc = [1, 2, 3, 4, 5, 6]  # 2 chunks
        assert m.register([store_range("K", doc)]) == 2
        matches = m.match(SCOPE, doc)
        assert [(x.object_key, x.old_st, x.cur_st) for x in matches] == [
            ("K0", 0, 0),
            ("K1", 3, 3),
        ]

    def test_stride_controls_offset(self) -> None:
        """A non-chunk-aligned offset is found at stride 1, missed at stride=chunk."""
        doc = [1, 2, 3, 4, 5, 6]
        req = [9, 1, 2, 3, 4, 5, 6]  # doc shifted by 1 (preamble 9)

        fine = GlobalBlendMatcher(chunk_size=CHUNK, probe_stride=1)
        fine.register([store_range("K", doc)])
        out = fine.match(SCOPE, req)
        assert [(x.object_key, x.cur_st) for x in out] == [("K0", 1), ("K1", 4)]

        coarse = GlobalBlendMatcher(chunk_size=CHUNK, probe_stride=CHUNK)
        coarse.register([store_range("K", doc)])  # probes pos 0,3,6 -> misses
        assert coarse.match(SCOPE, req) == []

    def test_dedup_by_object_key(self) -> None:
        m = GlobalBlendMatcher(chunk_size=CHUNK)
        m.register([store_range("K", [1, 2, 3])])  # single chunk K0
        matches = m.match(SCOPE, [1, 2, 3, 1, 2, 3])  # content repeats
        assert len(matches) == 1 and matches[0].object_key == "K0"

    def test_scope_isolation(self) -> None:
        m = GlobalBlendMatcher(chunk_size=CHUNK)
        m.register([store_range("K", [1, 2, 3], scope="model-a")])
        assert m.match("model-b", [1, 2, 3]) == []

    def test_request_shorter_than_chunk(self) -> None:
        m = GlobalBlendMatcher(chunk_size=CHUNK)
        m.register([store_range("K", [1, 2, 3])])
        assert m.match(SCOPE, [1, 2]) == []

    def test_no_match(self) -> None:
        m = GlobalBlendMatcher(chunk_size=CHUNK)
        m.register([store_range("K", [1, 2, 3])])
        assert m.match(SCOPE, [7, 8, 9]) == []


class TestIdempotencyEviction:
    def test_register_idempotent(self) -> None:
        m = GlobalBlendMatcher(chunk_size=CHUNK)
        rng = store_range("K", [1, 2, 3, 4, 5, 6])
        assert m.register([rng]) == 2
        assert m.register([rng]) == 0
        assert len(m.match(SCOPE, [1, 2, 3, 4, 5, 6])) == 2

    def test_remove_evicts(self) -> None:
        m = GlobalBlendMatcher(chunk_size=CHUNK)
        m.register([store_range("K", [1, 2, 3, 4, 5, 6])])  # K0, K1
        assert m.remove(["K0"]) == 1
        matches = m.match(SCOPE, [1, 2, 3, 4, 5, 6])
        assert [x.object_key for x in matches] == ["K1"]  # only K0 gone

    def test_remove_unknown_is_noop(self) -> None:
        assert GlobalBlendMatcher(chunk_size=CHUNK).remove(["nope"]) == 0

    def test_growth_and_compaction(self) -> None:
        """Matching survives a table grow rebuild and a mass-evict compaction."""
        m = GlobalBlendMatcher(chunk_size=CHUNK)
        n = 400  # past the initial table's load factor, forcing a grow rebuild
        docs = [[1000 + 3 * i, 1001 + 3 * i, 1002 + 3 * i] for i in range(n)]
        for d, doc in enumerate(docs):
            m.register([store_range(f"D{d}_", doc)])
        # Mass eviction triggers tombstone compaction.
        assert m.remove([f"D{d}_0" for d in range(n - 1)]) == n - 1
        out = m.match(SCOPE, docs[n - 1])
        assert [x.object_key for x in out] == [f"D{n - 1}_0"]
        assert m.match(SCOPE, docs[0]) == []

    def test_count_mismatch_range_skipped(self) -> None:
        """A range whose object_keys count != chunk count is skipped, not partial."""
        m = GlobalBlendMatcher(chunk_size=CHUNK)
        doc = [1, 2, 3, 4, 5, 6]  # 2 chunks
        bad = StoreRange(
            model_scope=SCOPE,
            tokens=doc,
            object_keys=["K0"],  # only 1 key for 2 chunks -> mismatch
            old_st_base=0,
        )
        assert m.register([bad]) == 0
        assert m.match(SCOPE, doc) == []


class TestValidation:
    def test_bad_chunk_size(self) -> None:
        with pytest.raises(ValueError):
            GlobalBlendMatcher(chunk_size=0)

    def test_bad_probe_stride(self) -> None:
        with pytest.raises(ValueError):
            GlobalBlendMatcher(chunk_size=CHUNK, probe_stride=0)


class TestVectorizedMatch:
    """Guards the vectorized probe against a brute-force reference."""

    def test_matches_reference_on_large_random_input(self) -> None:
        # Standard
        import random

        rng = random.Random(1234)
        chunk = 8
        m = GlobalBlendMatcher(chunk_size=chunk, probe_stride=1)

        # Build a corpus of distinct docs; register each as one object_key/chunk.
        windows: dict[tuple[int, ...], tuple[str, int]] = {}
        for d in range(40):
            doc = [rng.randint(0, 50) for _ in range(chunk * 4)]
            n_chunks = len(doc) // chunk
            m.register(
                [
                    StoreRange(
                        model_scope=SCOPE,
                        tokens=doc,
                        object_keys=[f"D{d}_{c}" for c in range(n_chunks)],
                        old_st_base=0,
                    )
                ]
            )
            for c in range(len(doc) // chunk):
                w = tuple(doc[c * chunk : (c + 1) * chunk])
                windows.setdefault(w, (f"D{d}_{c}", c * chunk))

        # A request stitched from random tokens and some embedded known windows.
        req = [rng.randint(0, 50) for _ in range(200)]
        for _ in range(5):
            embed = list(rng.choice(list(windows.keys())))
            pos = rng.randint(0, len(req) - chunk)
            req[pos : pos + chunk] = embed

        # Reference: probe every position, first-writer-wins object_key dedup.
        expected: list[tuple[str, int]] = []
        seen: set[str] = set()
        for p in range(len(req) - chunk + 1):
            w = tuple(req[p : p + chunk])
            loc = windows.get(w)
            if loc is None or loc[0] in seen:
                continue
            seen.add(loc[0])
            expected.append((loc[0], p))

        got = [(x.object_key, x.cur_st) for x in m.match(SCOPE, req)]
        assert got == expected
        # Sanity: the embedded windows are actually found.
        assert len(got) >= 1

    def test_cur_st_ascending_with_stride(self) -> None:
        doc = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 3 chunks of size 3
        m = GlobalBlendMatcher(chunk_size=CHUNK, probe_stride=CHUNK)
        m.register([store_range("K", doc)])
        out = m.match(SCOPE, doc)
        cur_sts = [x.cur_st for x in out]
        assert cur_sts == sorted(cur_sts)
        assert cur_sts == [0, 3, 6]


class TestConcurrency:
    def test_concurrent_register_and_match(self) -> None:
        """Interleaved publishes and queries stay correct and never crash."""
        m = GlobalBlendMatcher(chunk_size=CHUNK)
        docs = [[i, i + 1, i + 2] for i in range(0, 300, 3)]
        errors: list[Exception] = []

        def publisher() -> None:
            try:
                for d, doc in enumerate(docs):
                    m.register([store_range(f"D{d}_", doc)])
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        def matcher() -> None:
            try:
                for doc in docs:
                    m.match(SCOPE, doc)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=publisher) for _ in range(2)]
        threads += [threading.Thread(target=matcher) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # After all publishes, every doc's chunk is matchable.
        for d, doc in enumerate(docs):
            out = m.match(SCOPE, doc)
            assert any(x.object_key == f"D{d}_0" for x in out)


class TestTokenCodec:
    def test_round_trip(self) -> None:
        tokens = [0, 1, 2, 65535, 2**31, 2**32 - 1]
        decoded = decode_tokens(encode_tokens(tokens))
        assert decoded.tolist() == tokens

    def test_empty(self) -> None:
        assert decode_tokens(encode_tokens([])).tolist() == []

    def test_decoded_feeds_match(self) -> None:
        m = GlobalBlendMatcher(chunk_size=CHUNK)
        doc = [1, 2, 3, 4, 5, 6]
        m.register([store_range("K", doc)])
        out = m.match(SCOPE, decode_tokens(encode_tokens(doc)))
        assert [x.object_key for x in out] == ["K0", "K1"]

    def test_malformed_base64_raises(self) -> None:
        with pytest.raises(ValueError):
            decode_tokens("not valid base64 !!!")

    def test_bad_byte_length_raises(self) -> None:
        # Standard
        import base64

        with pytest.raises(ValueError):
            decode_tokens(base64.b64encode(b"abc").decode())  # 3 bytes, not /4
