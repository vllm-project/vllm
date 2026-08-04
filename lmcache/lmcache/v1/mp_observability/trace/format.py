# SPDX-License-Identifier: Apache-2.0

"""On-disk format for trace files.

A trace file is a length-prefixed msgpack stream:

    [4-byte big-endian frame length][msgpack frame]
    [4-byte big-endian frame length][msgpack frame]
    ...

The first frame is always a :class:`Header`.  All subsequent frames
are :class:`Record` objects.  Length-prefixing keeps the reader
simple and supports concurrent appenders (each frame is atomic on
local filesystems for sizes below ``PIPE_BUF``).

Format version is policed by the reader.  Unknown versions are
rejected to make corrupt or future-format files fail loudly rather
than silently producing garbage.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any

# Third Party
import msgspec

#: Magic bytes at the start of every file for sanity checking.  The
#: reader rejects files that do not begin with these bytes.
MAGIC: bytes = b"LMCT"

#: Bumped whenever the on-wire framing layout changes in a backwards-
#: incompatible way (length prefix, header/record struct shape, etc.).
FORMAT_VERSION: int = 1

#: Bumped whenever the captured API surface changes in a way that makes
#: older traces undecodable or incorrect to replay — e.g. a traced
#: StorageManager method gains/loses an argument, an argument type's
#: codec wire form changes, or a new codec tag is introduced.  Owned by
#: the trace subsystem, independent of the LMCache package version,
#: because ``lmcache.__version__`` bumps cover many changes irrelevant
#: to the trace contract.
TRACE_SCHEMA_VERSION: int = 1


class Header(msgspec.Struct, tag="header", omit_defaults=True):
    """One-per-file metadata block."""

    magic: bytes
    """Always :data:`MAGIC`."""

    format_version: int
    """File format version; readers reject unknown values."""

    level: str
    """Trace level — currently ``"storage"``.  Future levels (``"mq"``,
    ``"gpu"``) will share this format."""

    trace_schema_version: int
    """:data:`TRACE_SCHEMA_VERSION` at record time.  Replay drivers may
    refuse mismatched schemas rather than silently misinterpreting old
    traces."""

    t_mono_start: float
    """``time.monotonic()`` at recorder construction.  Record
    timestamps are relative to this."""

    t_wall_start: float
    """``time.time()`` at recorder construction.  Used to correlate
    with external logs / metrics in absolute wall-clock time."""

    sm_config_json: str
    """JSON dump of ``StorageManagerConfig`` at record time, or an
    empty string when not available."""

    sm_config_digest: str
    """SHA-256 hex digest of :attr:`sm_config_json`.  Replay drivers
    use this to detect mismatched configurations."""


class Record(msgspec.Struct, tag="record", omit_defaults=True):
    """One captured function call.

    All records share the same shape; the ``qualname`` field
    differentiates operations.  Future trace levels can introduce new
    ``qualname`` values without bumping the format version.
    """

    t_mono: float
    """Monotonic seconds since :attr:`Header.t_mono_start`."""

    t_wall: float
    """Wall-clock ``time.time()`` at the moment the event was
    published."""

    qualname: str
    """Fully-qualified call-site name (e.g.
    ``lmcache.v1.distributed.storage_manager.StorageManager.reserve_write``)."""

    args: dict[str, Any]
    """Codec-encoded argument dict.  See
    :mod:`lmcache.v1.mp_observability.trace.codecs` for the codec
    contract."""


# msgspec encoders/decoders.  Reused per-process; both are thread-safe
# after construction.
_ENCODER = msgspec.msgpack.Encoder()
_DECODER_HEADER = msgspec.msgpack.Decoder(Header)
_DECODER_RECORD = msgspec.msgpack.Decoder(Record)


def encode_header(h: Header) -> bytes:
    """Serialize a header to msgpack bytes."""
    return _ENCODER.encode(h)


def encode_record(r: Record) -> bytes:
    """Serialize a record to msgpack bytes."""
    return _ENCODER.encode(r)


def decode_header(buf: bytes) -> Header:
    """Parse a header from msgpack bytes."""
    return _DECODER_HEADER.decode(buf)


def decode_record(buf: bytes) -> Record:
    """Parse a record from msgpack bytes."""
    return _DECODER_RECORD.decode(buf)
