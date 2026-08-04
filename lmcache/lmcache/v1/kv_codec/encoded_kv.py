# SPDX-License-Identifier: Apache-2.0
"""On-disk encoded KV format and metadata."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Tuple, Union
import struct
import zlib

# Third Party
import torch

# First Party
from lmcache.v1.kv_codec.errors import (
    CorruptEncodedKVError,
    UnsupportedConfigError,
)

# Magic prefix used to identify a v1 EncodedKV blob.  Chosen so
# `head -c 8 file.bin` is human-readable.  Do not change this value;
# every reader on every node identifies blobs by this byte sequence.
CODEC_MAGIC: bytes = b"LMCKV\x01\x00\x01"  # "LMCKV", version major.minor.patch
CODEC_MAGIC_LEN: int = len(CODEC_MAGIC)
ASSERT_MAGIC_BYTES = 8
assert CODEC_MAGIC_LEN == ASSERT_MAGIC_BYTES, "magic must be 8 bytes"


class CodecVersion(IntEnum):
    """Bumped when the on-disk header format changes incompatibly.

    The minor/patch bytes inside CODEC_MAGIC are reserved for future
    backwards-compatible additions (a v2 reader can read v1 blobs by
    interpreting unknown trailing fields as zero/None).
    """

    V1 = 1


class ScaleScope(IntEnum):
    """How V scales are organized in the encoded blob.

    `per_tensor`     one FP32 scalar for the entire V tensor.
    `per_layer_head` one FP32 scalar per head, applied to all pages
                     of that layer.
    `per_page_head`  one FP32 scalar per (page, head).  Recommended
                     default for asymmetric storage because paged KV
                     chunks do not naturally respect "whole layer
                     tensor" assumptions; the overhead is tiny vs
                     V-byte volume.
    `external`       scales not stored in this blob; expected to be
                     attached out-of-band by the caller.
    """

    PER_TENSOR = 0
    PER_LAYER_HEAD = 1
    PER_PAGE_HEAD = 2
    EXTERNAL = 3


# Header packing: little-endian throughout regardless of host.
#
#   8 B  magic ("LMCKV" + 3-byte version triplet)
#   2 B  header version (CodecVersion)
#   2 B  scale_scope (ScaleScope)
#   2 B  k_dtype_id   (DTYPE_TO_INT)
#   2 B  v_dtype_id   (DTYPE_TO_INT)
#   2 B  scale_dtype_id (DTYPE_TO_INT, typically float32 or float16)
#   2 B  reserved (zero)
#   8 B  layer_id  (int64; signed sentinel value -1 means unset)
#   8 B  chunk_id  (int64; -1 unset)
#   8 B  chunk_size (int64; tokens per chunk for paged caches)
#   8 B  page_size (int64; tokens per page)
#   8 B  kv_head_count (int64)
#   8 B  head_dim (int64)
#   8 B  scale_shape_n (int64; number of dimensions in scale tensor)
#   N×8B scale_shape values
#   8 B  k_payload_len (int64, bytes)
#   8 B  v_payload_len (int64, bytes)
#   8 B  scale_payload_len (int64, bytes)
#   2 B  hash_strs_n (number of (key, value) string pairs that follow)
#   For each str pair: 2-byte key_len, key, 2-byte val_len, val.
#     Currently used keys: model_id, model_revision_hash,
#     tokenizer_hash, rope_config_hash, attention_backend, kv_layout.
#   4 B  payload_crc32c        (CRC of the K + V + scales payload)
#
# Followed by: K_payload (k_payload_len bytes), V_payload, scales.
#
# Field order is fixed; new fields added in V2+ append after CRC.

_FIXED_HEADER_FMT = "<8sHHHHHH" + "qqqqqqq"  # magic + 6 shorts + 7 int64
_FIXED_HEADER_LEN = struct.calcsize(_FIXED_HEADER_FMT)
# 8 + 2*6 + 8*7 = 76 bytes
assert _FIXED_HEADER_LEN == 76, _FIXED_HEADER_LEN


def _pack_str(s: str) -> bytes:
    if s is None:
        s = ""
    encoded = s.encode("utf-8")
    if len(encoded) > 0xFFFF:
        raise UnsupportedConfigError(
            f"hash string longer than 65535 bytes: {len(encoded)}"
        )
    return struct.pack("<H", len(encoded)) + encoded


def _unpack_str(buf: memoryview, off: int) -> Tuple[str, int]:
    if off + 2 > len(buf):
        raise CorruptEncodedKVError(f"truncated string-length field at offset {off}")
    (length,) = struct.unpack_from("<H", buf, off)
    off += 2
    if off + length > len(buf):
        raise CorruptEncodedKVError(
            f"string at offset {off} declares length {length} but only "
            f"{len(buf) - off} bytes remain"
        )
    try:
        s = bytes(buf[off : off + length]).decode("utf-8")
    except UnicodeDecodeError as e:
        raise CorruptEncodedKVError(
            f"non-utf-8 bytes in string field at offset {off}: {e}"
        ) from None
    return s, off + length


@dataclass
class CodecHashes:
    """Identifying hashes that gate cross-config cache poisoning.

    Empty strings mean "not provided"; serialization treats empty
    string and None identically.  Any non-empty mismatch on read
    raises `CodecMismatchError` (see codec.AsymK16V8Codec.decode).
    """

    model_id: str = ""
    model_revision_hash: str = ""
    tokenizer_hash: str = ""
    rope_config_hash: str = ""
    attention_backend: str = ""
    kv_layout: str = ""

    # Fields are checked in this order during decode mismatch detection.
    _CHECK_ORDER = (
        "model_id",
        "model_revision_hash",
        "tokenizer_hash",
        "rope_config_hash",
        "attention_backend",
        "kv_layout",
    )


@dataclass
class EncodedKV:
    """A self-describing encoded KV blob.

    `payload` carries `K_bytes ‖ V_bytes ‖ scale_bytes` in that
    order.  Lengths are in `k_payload_len`, `v_payload_len`,
    `scale_payload_len`.  All numeric fields are little-endian on
    disk regardless of host.
    """

    # Logical shape/dtype information (same fields as MemoryObjMetadata.dtypes/shapes)
    k_dtype: torch.dtype
    v_dtype: torch.dtype
    scale_dtype: torch.dtype = torch.float32
    scale_scope: ScaleScope = ScaleScope.PER_PAGE_HEAD

    # Identity / cache-poisoning gates
    hashes: CodecHashes = field(default_factory=CodecHashes)

    # Layout / shape parameters (-1 = unset)
    layer_id: int = -1
    chunk_id: int = -1
    chunk_size: int = -1
    page_size: int = -1
    kv_head_count: int = -1
    head_dim: int = -1

    # Scale tensor shape; e.g., () for per_tensor, (kv_head_count,)
    # for per_layer_head, (n_pages, kv_head_count) for per_page_head.
    scale_shape: Tuple[int, ...] = ()

    # Payload byte counts
    k_payload_len: int = 0
    v_payload_len: int = 0
    scale_payload_len: int = 0

    # The actual encoded byte blob (header + payload).  When the
    # EncodedKV is freshly assembled from tensors, callers populate
    # `payload` (the K+V+scale bytes) and `header_bytes` is computed
    # at serialize time.  When parsed from a buffer via
    # ``deserialize_header``, ``payload`` is a memoryview slice over
    # the source buffer to avoid a full payload copy on the read
    # path.
    payload: Union[bytes, memoryview] = b""
    header_bytes: Optional[bytes] = None

    @property
    def total_bytes(self) -> int:
        """Total encoded size: header + payload."""
        if self.header_bytes is None:
            return -1
        return len(self.header_bytes) + len(self.payload)

    def expected_payload_len(self) -> int:
        return self.k_payload_len + self.v_payload_len + self.scale_payload_len


def _dtype_to_int(dtype: Optional[torch.dtype]) -> int:
    """Local re-export of the protocol DTYPE_TO_INT to avoid a
    circular import at module-load time."""
    # First Party
    from lmcache.v1.protocol import DTYPE_TO_INT

    if dtype not in DTYPE_TO_INT:
        raise UnsupportedConfigError(
            f"dtype {dtype} not in protocol DTYPE_TO_INT mapping"
        )
    return DTYPE_TO_INT[dtype]


def _int_to_dtype(idx: int) -> Optional[torch.dtype]:
    # First Party
    from lmcache.v1.protocol import INT_TO_DTYPE

    if idx not in INT_TO_DTYPE:
        raise CorruptEncodedKVError(f"unknown dtype index {idx} in encoded header")
    return INT_TO_DTYPE[idx]


def serialize_header(enc: EncodedKV) -> bytes:
    """Pack the EncodedKV header into bytes (excludes payload).

    Computes payload CRC32 from `enc.payload` and writes it as the
    last field.  Caller is responsible for ensuring `enc.payload`
    matches `expected_payload_len()`.
    """
    if len(enc.payload) != enc.expected_payload_len():
        raise UnsupportedConfigError(
            f"payload length {len(enc.payload)} does not match "
            f"declared K({enc.k_payload_len}) + V({enc.v_payload_len}) "
            f"+ scales({enc.scale_payload_len})"
        )

    fixed = struct.pack(
        _FIXED_HEADER_FMT,
        CODEC_MAGIC,
        int(CodecVersion.V1),
        int(enc.scale_scope),
        _dtype_to_int(enc.k_dtype),
        _dtype_to_int(enc.v_dtype),
        _dtype_to_int(enc.scale_dtype),
        0,  # reserved
        enc.layer_id,
        enc.chunk_id,
        enc.chunk_size,
        enc.page_size,
        enc.kv_head_count,
        enc.head_dim,
        len(enc.scale_shape),
    )
    scale_shape_bytes = struct.pack(f"<{len(enc.scale_shape)}q", *enc.scale_shape)
    payload_lens = struct.pack(
        "<qqq", enc.k_payload_len, enc.v_payload_len, enc.scale_payload_len
    )
    # Hash strings
    hash_keys = list(CodecHashes._CHECK_ORDER)
    hash_blob = struct.pack("<H", len(hash_keys))
    for k in hash_keys:
        hash_blob += _pack_str(k) + _pack_str(getattr(enc.hashes, k))

    # CRC over the actual payload bytes (the K+V+scales blob), not
    # over the header itself.  zlib.crc32 is CRC32/IEEE; if a future
    # reader needs CRC32C specifically we bump CodecVersion.
    crc = zlib.crc32(enc.payload) & 0xFFFFFFFF
    crc_bytes = struct.pack("<I", crc)

    return fixed + scale_shape_bytes + payload_lens + hash_blob + crc_bytes


def deserialize_header(buf: bytes) -> EncodedKV:
    """Parse an EncodedKV header from a contiguous byte buffer.

    The buffer must contain header + payload.  Returns an EncodedKV
    with `header_bytes` and `payload` both populated.

    Raises:
        CorruptEncodedKVError on magic/version/CRC mismatch or any
        truncated field.
    """
    mv = memoryview(buf)
    if len(mv) < _FIXED_HEADER_LEN:
        raise CorruptEncodedKVError(
            f"buffer too short for fixed header: {len(mv)} < {_FIXED_HEADER_LEN}"
        )

    fixed = struct.unpack_from(_FIXED_HEADER_FMT, mv, 0)
    (
        magic,
        version,
        scale_scope,
        k_dtype_id,
        v_dtype_id,
        scale_dtype_id,
        _reserved,
        layer_id,
        chunk_id,
        chunk_size,
        page_size,
        kv_head_count,
        head_dim,
        scale_shape_n,
    ) = fixed
    if magic != CODEC_MAGIC:
        raise CorruptEncodedKVError(
            f"bad magic: got {magic!r}, expected {CODEC_MAGIC!r}"
        )
    if version != int(CodecVersion.V1):
        raise CorruptEncodedKVError(
            f"unsupported codec version {version}; this build supports "
            f"{int(CodecVersion.V1)}"
        )
    if scale_shape_n < 0 or scale_shape_n > 8:
        raise CorruptEncodedKVError(f"implausible scale_shape_n: {scale_shape_n}")

    off = _FIXED_HEADER_LEN
    if off + 8 * scale_shape_n > len(mv):
        raise CorruptEncodedKVError(
            f"buffer too short for scale_shape: need {8 * scale_shape_n} bytes"
        )
    scale_shape = tuple(struct.unpack_from(f"<{scale_shape_n}q", mv, off))
    off += 8 * scale_shape_n

    if off + 24 > len(mv):
        raise CorruptEncodedKVError("truncated payload-lengths field")
    k_len, v_len, s_len = struct.unpack_from("<qqq", mv, off)
    off += 24

    if off + 2 > len(mv):
        raise CorruptEncodedKVError("truncated hashes-count field")
    (n_hashes,) = struct.unpack_from("<H", mv, off)
    off += 2

    hashes = CodecHashes()
    expected_keys = set(CodecHashes._CHECK_ORDER)
    for _ in range(n_hashes):
        key, off = _unpack_str(mv, off)
        val, off = _unpack_str(mv, off)
        if key in expected_keys:
            setattr(hashes, key, val)

    if off + 4 > len(mv):
        raise CorruptEncodedKVError("truncated CRC field")
    (crc_declared,) = struct.unpack_from("<I", mv, off)
    off += 4

    expected_payload = k_len + v_len + s_len
    if off + expected_payload > len(mv):
        raise CorruptEncodedKVError(
            f"buffer truncated mid-payload: header ends at {off}, "
            f"declared payload = {expected_payload}, buffer total "
            f"= {len(mv)}"
        )

    payload = mv[off : off + expected_payload]
    crc_computed = zlib.crc32(payload) & 0xFFFFFFFF
    if crc_computed != crc_declared:
        raise CorruptEncodedKVError(
            f"payload CRC mismatch: declared {crc_declared:#x}, "
            f"computed {crc_computed:#x}"
        )

    try:
        scope_enum = ScaleScope(scale_scope)
    except ValueError:
        raise CorruptEncodedKVError(
            f"unknown scale_scope index {scale_scope} in encoded header"
        ) from None
    enc = EncodedKV(
        k_dtype=_int_to_dtype(k_dtype_id),
        v_dtype=_int_to_dtype(v_dtype_id),
        scale_dtype=_int_to_dtype(scale_dtype_id),
        scale_scope=scope_enum,
        hashes=hashes,
        layer_id=layer_id,
        chunk_id=chunk_id,
        chunk_size=chunk_size,
        page_size=page_size,
        kv_head_count=kv_head_count,
        head_dim=head_dim,
        scale_shape=scale_shape,
        k_payload_len=k_len,
        v_payload_len=v_len,
        scale_payload_len=s_len,
        payload=payload,
        header_bytes=bytes(mv[:off]),
    )
    return enc
