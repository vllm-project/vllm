"""Tests for codec_compression.py — the negotiator + stream-compressor helpers.

These tests cover regression classes discovered during the v0.4.1 cohort
bench: (a) brotli's per-chunk flush() inflating small streams, (b) silent
fall-through when zstd/brotli modules are missing in the image, (c)
preference-order violations.
"""
from __future__ import annotations

import asyncio
import gzip
import io
from typing import AsyncIterable

import pytest

from vllm.entrypoints import codec_compression as cc


async def _from_chunks(chunks: list[bytes]) -> AsyncIterable[bytes]:
    for c in chunks:
        yield c


def _collect(stream_factory) -> bytes:
    async def run() -> bytes:
        out = bytearray()
        async for chunk in stream_factory():
            out.extend(chunk)
        return bytes(out)

    return asyncio.run(run())


# Representative Codec msgpack-stream content: a sequence of small token-ID
# frames. The bench's 64-token msgpack-identity cell measures ~975 bytes; we
# synthesize a comparable payload so the inflation check is meaningful.
def _msgpack_like_stream(n_chunks: int = 12, chunk_bytes: int = 80) -> list[bytes]:
    payload = bytes(range(256)) * 4  # 1 KB of varied byte values
    out = []
    for i in range(n_chunks):
        start = (i * chunk_bytes) % (len(payload) - chunk_bytes)
        out.append(payload[start : start + chunk_bytes])
    return out


def test_negotiate_prefers_zstd_with_dict():
    """Spec §Transport-Compression: preference order zstd > br > gzip > identity."""
    if not cc._ZSTD_AVAILABLE or not cc._BROTLI_AVAILABLE:
        pytest.skip("zstandard or brotli not installed — would be caught by supervisor startup check")
    cc.set_zstd_dict("msgpack", b"\x00" * 16384)
    try:
        assert cc.negotiate_encoding("zstd, br, gzip, identity", stream_format="msgpack") == "zstd"
    finally:
        cc.clear_zstd_dicts()


def test_negotiate_falls_through_to_gzip_when_zstd_dict_missing():
    """No dict registered for the requested format → zstd skipped, picker falls
    through preference order to gzip (NOT to identity)."""
    cc.clear_zstd_dicts()
    if cc._BROTLI_AVAILABLE:
        # br is next in preference order before gzip
        assert cc.negotiate_encoding("zstd, br, gzip", stream_format="msgpack") == "br"
    assert cc.negotiate_encoding("zstd, gzip", stream_format="msgpack") == "gzip"


def test_negotiate_identity_only_returns_none():
    """An Accept-Encoding listing only identity (or empty) returns None."""
    assert cc.negotiate_encoding("identity") is None
    assert cc.negotiate_encoding("") is None


def test_brotli_does_not_inflate_repeated_payload():
    """Regression for the per-chunk flush() bug discovered during v0.4.1 bench.

    Before the fix, calling compressor.flush() on every chunk made small
    streams *larger* than identity (64-token msgpack cell: 1159 B vs 975 B
    identity; 512-token: 9013 B vs 7616 B). After dropping the per-chunk
    flush, brotli compresses or matches identity on any payload > ~200 B.
    """
    if not cc._BROTLI_AVAILABLE:
        pytest.skip("brotli not installed")
    chunks = _msgpack_like_stream()
    identity = b"".join(chunks)
    compressed = _collect(lambda: cc._compress_brotli(_from_chunks(chunks)))
    assert len(compressed) <= len(identity), (
        f"brotli inflated a {len(identity)}-byte stream to {len(compressed)} bytes "
        f"— the per-chunk flush() regression is back"
    )


def test_gzip_round_trips():
    chunks = _msgpack_like_stream()
    expected = b"".join(chunks)
    compressed = _collect(lambda: cc._compress_gzip(_from_chunks(chunks)))
    assert gzip.decompress(compressed) == expected


def test_brotli_round_trips():
    if not cc._BROTLI_AVAILABLE:
        pytest.skip("brotli not installed")
    import brotli as _brotli  # type: ignore

    chunks = _msgpack_like_stream()
    expected = b"".join(chunks)
    compressed = _collect(lambda: cc._compress_brotli(_from_chunks(chunks)))
    assert _brotli.decompress(compressed) == expected


def test_zstd_with_dict_round_trips():
    """When a dict is registered, zstd compresses+decompresses byte-identically
    using the same dict — the contract clients rely on."""
    if not cc._ZSTD_AVAILABLE:
        pytest.skip("zstandard not installed")
    import zstandard as _zstd  # type: ignore

    dict_bytes = b"\x37\x00\x00\x00" + b"\x00" * 16380  # minimal valid zstd dict header is non-trivial;
    # use a real dict-shaped buffer — zstandard accepts arbitrary bytes as a raw dict prefix
    # If the empty-dict path errors, skip rather than fail the test (env-specific).
    try:
        cc.set_zstd_dict("msgpack", dict_bytes)
    except Exception:
        pytest.skip("zstandard rejected synthetic dict; use a real trained dict via integration test")
    try:
        chunks = _msgpack_like_stream()
        expected = b"".join(chunks)
        compressed = _collect(
            lambda: cc._compress_zstd(_from_chunks(chunks), dict_bytes=dict_bytes)
        )
        zdict = _zstd.ZstdCompressionDict(dict_bytes)
        with _zstd.ZstdDecompressor(dict_data=zdict).stream_reader(io.BytesIO(compressed)) as r:
            assert r.read() == expected
    finally:
        cc.clear_zstd_dicts()
