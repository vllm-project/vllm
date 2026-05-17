# SPDX-License-Identifier: Apache-2.0
"""
Negotiated transport compression for Codec binary streaming responses.

Compression is opt-in (PKCE-style): clients advertise supported encodings
via the standard ``Accept-Encoding`` header, servers respond with whatever
overlap they choose, and the response is tagged with ``Content-Encoding``.
Clients that don't want compression simply omit the header and receive
identity-encoded frames as before.

This is layered *outside* the Codec frame format — frames themselves are
unchanged. The compression covers the entire HTTP response stream so a
single compression context spans many frames (much better ratio than
per-frame compression for small frames).

Supported encodings, in server preference order:
  1. ``zstd``     — Zstandard. Best ratio, fastest streaming. Requires the
                    optional ``zstandard`` package; gracefully skipped if
                    absent. Browsers: Chrome 123+, Firefox 126+.
  2. ``br``       — Brotli. Slightly better ratio than gzip, similar speed
                    at quality 4-6. Universal browser support (Chrome 50+,
                    Firefox 44+, Safari 11+). Requires the optional
                    ``brotli`` package; gracefully skipped if absent.
  3. ``gzip``     — Universal fallback. Pure stdlib, always available.
                    Supported in 100% of browsers and Node 18+ via fetch.
  4. ``identity`` — No compression. Always available, used when none of
                    the above appears in ``Accept-Encoding``.

Usage:

    from vllm.entrypoints.codec_compression import wrap_streaming_response

    async def codec_handler(request: Request):
        gen = build_codec_stream(...)  # AsyncIterable[bytes]
        return wrap_streaming_response(
            request.headers.get("accept-encoding", ""),
            gen,
            background=...,
        )
"""

from __future__ import annotations

import hashlib
import zlib
from typing import AsyncIterable, Optional

from fastapi import BackgroundTasks
from fastapi.responses import StreamingResponse

# Soft deps — both are graceful no-ops if the package isn't installed.
try:
    import zstandard as zstd

    _ZSTD_AVAILABLE = True
except ImportError:
    _ZSTD_AVAILABLE = False

try:
    import brotli

    _BROTLI_AVAILABLE = True
except ImportError:
    _BROTLI_AVAILABLE = False


# ── Pre-trained ZSTD dictionary registry ─────────────────────────────────────
#
# Per the Codec protocol (spec/PROTOCOL.md "Pre-trained ZSTD dictionaries"),
# **the dict is the precondition for using zstd at all**, not an optimization
# layered on top. Without a matching pre-trained dict, no-dict zstd's wire-byte
# advantage over gzip is essentially zero on Codec streams (RESULTS.md §1f:
# 3.4 B/token vs 3.4 B/token within noise) but its TTFB cost on the shipped
# buffered middleware is catastrophic (§1d: 334× at 2K tokens). So no-dict
# zstd is the *worst of both worlds* — same bytes as gzip, much worse TTFB.
#
# The dict registry is keyed by ``stream_format`` because zstd dictionaries
# are not interchangeable across formats — a dict trained on msgpack Codec
# frames captures a different byte distribution than one trained on protobuf.
# Operators load the appropriate dict at server start (e.g. fetched from the
# tokenizer map's ``zstd_dictionaries[]`` entry whose ``format`` matches), and
# the negotiator then unlocks zstd for that format only.
#
# Default state: empty registry → zstd never selected → server falls through
# to gzip on every request that advertises zstd. This is the correct default:
# gzip works on every middleware stack, ships in stdlib, and matches the wire
# performance of no-dict zstd.

_ZSTD_DICTS: dict[str, bytes] = {}
# Parallel registry of hashes — sha256(dict_bytes) computed once at
# registration so emit-time is a constant-time map lookup. Goes onto every
# zstd response as the Codec-Zstd-Dict header so clients can validate (or
# fetch) the right dict before decompressing. See spec/PROTOCOL.md
# "Codec-Zstd-Dict response header".
_ZSTD_DICT_HASHES: dict[str, str] = {}


def _hash_dict(dict_bytes: bytes) -> str:
    """sha256 hex digest of the dict, prefixed `sha256:` so the value is
    self-describing and matches the `hash` shape in tokenizer-map
    `zstd_dictionaries[]` entries."""
    return "sha256:" + hashlib.sha256(dict_bytes).hexdigest()


def set_zstd_dict(stream_format: str, dict_bytes: bytes) -> None:
    """Register a pre-trained zstd dictionary for ``stream_format``.

    ``stream_format`` is one of ``"msgpack"`` or ``"protobuf"`` (matches the
    Codec request's ``stream_format`` field). ``dict_bytes`` is the raw
    bytes of a zstd dictionary as produced by ``zstd --train`` or
    ``packages/bench/scripts/train-zstd-dict.py``.

    Replaces any previously-registered dict for that format. Call once at
    server startup, e.g.::

        with open("qwen2.5-msgpack-v1.dict", "rb") as f:
            set_zstd_dict("msgpack", f.read())
        with open("qwen2.5-protobuf-v1.dict", "rb") as f:
            set_zstd_dict("protobuf", f.read())

    No-op if the ``zstandard`` package isn't installed — the negotiator
    won't pick zstd in that case anyway.
    """
    if not _ZSTD_AVAILABLE:
        return
    _ZSTD_DICTS[stream_format] = dict_bytes
    _ZSTD_DICT_HASHES[stream_format] = _hash_dict(dict_bytes)


def clear_zstd_dicts() -> None:
    """Drop all registered dictionaries. Mostly for tests."""
    _ZSTD_DICTS.clear()
    _ZSTD_DICT_HASHES.clear()


def has_zstd_dict(stream_format: Optional[str]) -> bool:
    """Is there a registered dict for ``stream_format``?

    Returns False when ``stream_format`` is None — callers that don't know
    the response format (e.g. legacy code paths) can't safely use a
    format-keyed dict, so we drop them off the zstd path.
    """
    return bool(stream_format) and stream_format in _ZSTD_DICTS


def get_zstd_dict_hash(stream_format: Optional[str]) -> Optional[str]:
    """sha256 hex digest of the registered dict for ``stream_format``,
    formatted as ``sha256:<hex>`` for the ``Codec-Zstd-Dict`` response
    header. Returns None when no dict is registered."""
    if not stream_format:
        return None
    return _ZSTD_DICT_HASHES.get(stream_format)


def _parse_accept_encoding(header: str) -> list[str]:
    """Return the encodings the client lists, in the order they appear.

    We don't bother with q-values for now — clients that explicitly want
    a non-default ordering are rare, and the server preference order
    (zstd, gzip) covers the realistic case. Identity is always implicitly
    acceptable per RFC 9110 §12.5.3.
    """
    if not header:
        return []
    parts = []
    for part in header.split(","):
        name = part.strip().split(";", 1)[0].strip().lower()
        if name:
            parts.append(name)
    return parts


def negotiate_encoding(
    accept_encoding: str,
    *,
    stream_format: Optional[str] = None,
) -> Optional[str]:
    """Pick the best encoding both sides can speak.

    Returns one of ``"zstd"``, ``"br"``, ``"gzip"``, or ``None`` (identity).
    Order of preference: zstd > br > gzip > identity. ``"*"`` in
    Accept-Encoding is treated as accepting any encoding the server has.

    **zstd is gated on a pre-trained dict being registered for the request's
    ``stream_format``.** Without a dict, this falls through to gzip even
    when the client advertises zstd — see the dict registry comment above
    and spec/PROTOCOL.md "Pre-trained ZSTD dictionaries" for the rationale.
    ``stream_format`` defaults to None, which always disables zstd — keeps
    legacy callers safe.
    """
    encs = _parse_accept_encoding(accept_encoding)
    if not encs:
        return None
    has_wildcard = "*" in encs

    if (
        _ZSTD_AVAILABLE
        and has_zstd_dict(stream_format)
        and ("zstd" in encs or has_wildcard)
    ):
        return "zstd"
    if _BROTLI_AVAILABLE and ("br" in encs or has_wildcard):
        return "br"
    if "gzip" in encs or has_wildcard:
        return "gzip"
    return None


async def _compress_zstd(
    stream: AsyncIterable[bytes],
    *,
    dict_bytes: bytes,
) -> AsyncIterable[bytes]:
    """Stream-compress with Zstandard, using a pre-trained dict.

    Per the Codec protocol the encoder MUST load the dict; ``negotiate_encoding``
    only selects zstd when a dict is registered, so we pass the bytes through
    here rather than re-looking it up from the registry (avoids a TOCTOU
    where the dict gets cleared mid-request).
    """
    zdict = zstd.ZstdCompressionDict(dict_bytes)
    cctx = zstd.ZstdCompressor(level=3, dict_data=zdict)
    chunker = cctx.chunker(chunk_size=16384)
    async for chunk in stream:
        for out in chunker.compress(chunk):
            yield out
    for out in chunker.finish():
        yield out


async def _compress_gzip(stream: AsyncIterable[bytes]) -> AsyncIterable[bytes]:
    """Stream-compress with gzip. wbits=31 = gzip wrapper (vs raw deflate)."""
    compressor = zlib.compressobj(level=6, wbits=31)
    async for chunk in stream:
        out = compressor.compress(chunk)
        if out:
            yield out
    final = compressor.flush(zlib.Z_FINISH)
    if final:
        yield final


async def _compress_brotli(stream: AsyncIterable[bytes]) -> AsyncIterable[bytes]:
    """Stream-compress with Brotli. quality=4 balances speed/ratio for
    server-side dynamic compression — ratio close to the default quality 11
    but at gzip-level CPU cost (default 11 is 10-50x slower for streams).

    Per-chunk flush() was removed after the v0.4.1 bench showed it inflated
    small streams (each flush emits a complete brotli block + header,
    forfeiting between-chunk dictionary sharing). The remaining finish()
    closes the stream once at end-of-input."""
    compressor = brotli.Compressor(
        quality=4, mode=brotli.MODE_GENERIC, lgwin=22
    )
    async for chunk in stream:
        out = compressor.process(chunk)
        if out:
            yield out
    final = compressor.finish()
    if final:
        yield final


def wrap_streaming_response(
    accept_encoding: str,
    body_stream: AsyncIterable[bytes],
    *,
    media_type: str,
    background: Optional[BackgroundTasks] = None,
    extra_headers: Optional[dict[str, str]] = None,
    stream_format: Optional[str] = None,
    client_version: Optional[str] = None,
) -> StreamingResponse:
    """Build a StreamingResponse with the right compression based on the
    client's Accept-Encoding header.

    The Codec frame format is unchanged — compression is purely transport.
    Clients that don't include zstd/gzip in Accept-Encoding receive an
    uncompressed (identity-encoded) stream, which is the previous behavior.

    ``stream_format`` is the request's ``stream_format`` field
    (``"msgpack"`` / ``"protobuf"`` / ``"json"``) and gates the zstd path
    via the dict registry — see ``negotiate_encoding``.

    ``client_version`` is the request's ``Codec-Client-Version``. When set,
    Codec-* response headers are filtered to that version's floor per
    `spec/versions/v0.4.md § Graceful downgrade`.
    """
    from vllm.entrypoints.codec_version import filter_codec_headers

    encoding = negotiate_encoding(accept_encoding, stream_format=stream_format)
    headers: dict[str, str] = {"Vary": "Accept-Encoding"}
    if extra_headers:
        headers.update(extra_headers)

    if encoding == "zstd":
        # has_zstd_dict() was checked inside negotiate_encoding, so the
        # lookup here always hits — but assert defensively in case of a
        # registry mutation between negotiation and use.
        dict_bytes = _ZSTD_DICTS.get(stream_format or "")
        dict_hash = _ZSTD_DICT_HASHES.get(stream_format or "")
        if dict_bytes is None or dict_hash is None:
            # Registry was cleared mid-request — fall through to gzip.
            body = _compress_gzip(body_stream)
            headers["Content-Encoding"] = "gzip"
        else:
            body = _compress_zstd(body_stream, dict_bytes=dict_bytes)
            headers["Content-Encoding"] = "zstd"
            # Tell the client which dict we used so it can pick the
            # matching one before decompressing. See spec/PROTOCOL.md
            # "Codec-Zstd-Dict response header".
            headers["Codec-Zstd-Dict"] = dict_hash
    elif encoding == "br":
        body = _compress_brotli(body_stream)
        headers["Content-Encoding"] = "br"
    elif encoding == "gzip":
        body = _compress_gzip(body_stream)
        headers["Content-Encoding"] = "gzip"
    else:
        body = body_stream

    # Graceful downgrade — strip v0.4+ headers for older clients.
    if client_version is not None:
        headers = filter_codec_headers(headers, client_version)

    return StreamingResponse(
        body,
        media_type=media_type,
        headers=headers,
        background=background,
    )
