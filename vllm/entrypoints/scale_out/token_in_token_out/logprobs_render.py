# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Object-free rendering of ``/inference/v1/generate`` sample logprobs.

:func:`render_tokens_logprobs` reads the engine rows of a
:class:`~vllm.logprobs.FlatLogprobs` and produces exactly the JSON that
``ServingTokens._create_tokens_logprobs`` + ``model_dump`` + Starlette's
``JSONResponse.render`` produce for the same rows, without a Python object
per entry. :func:`render_json_with_fragments` splices such pre-rendered values
into the JSON of the remaining (small) response.
"""

import json
import secrets
import threading
from collections.abc import Mapping, Sequence
from typing import Any

import msgspec
import numpy as np

from vllm.logprobs import FlatLogprobs

# Rows rendered per batch; bounds the per-entry lists and index arrays.
_RENDER_BLOCK_ROWS = 1024


class _LeadTable:
    """token id -> ``prefix + str(id) + suffix`` as a numpy object array, so a
    block of ids maps to its JSON pieces with one fancy index.

    Covers ids below ``max_ids``, kept for the process lifetime (a few MB per
    table at a 150k vocabulary); blocks with other ids are formatted per call.
    """

    max_ids = 1 << 18

    def __init__(self, prefix: bytes, suffix: bytes):
        self.prefix = prefix
        self.suffix = suffix
        # (values, filled), replaced as one tuple so a lock-free reader always
        # sees a consistent pair.
        self._table: tuple[np.ndarray, np.ndarray] = (
            np.empty(0, dtype=object),
            np.zeros(0, dtype=bool),
        )
        # Renders run on the event loop and in the build threads.
        self._lock = threading.Lock()

    def _formatted(self, ids: np.ndarray) -> np.ndarray:
        result = np.empty(ids.size, dtype=object)
        result[:] = [b"%s%d%s" % (self.prefix, i, self.suffix) for i in ids.tolist()]
        return result

    def lookup(self, ids: np.ndarray) -> np.ndarray:
        """The pieces of a non-empty array of ids, in its shape."""
        lo, hi = int(ids.min()), int(ids.max())
        if lo < 0 or hi >= self.max_ids:
            return self._formatted(ids.ravel()).reshape(ids.shape)
        values, filled = self._table
        if hi < len(values):
            missing = np.unique(ids[~filled[ids]])
            if not missing.size:
                return values[ids]
        else:
            missing = np.unique(ids)
        # Format outside the lock, publish under it.
        formatted = self._formatted(missing)
        with self._lock:
            values, filled = self._table
            if hi >= len(values):
                size = min(self.max_ids, max(hi + 1, 2 * len(values)))
                grown = np.empty(size, dtype=object)
                grown[: len(values)] = values
                grown_filled = np.zeros(size, dtype=bool)
                grown_filled[: len(filled)] = filled
                values, filled = grown, grown_filled
            # Values before flags: a reader that sees a flag sees its value.
            values[missing] = formatted
            filled[missing] = True
            self._table = (values, filled)
            return values[ids]


# Pieces before each logprob value: the content entry, the first top-k entry
# (the sampled token again) and every further top-k entry.
_CONTENT_LEADS = _LeadTable(b'{"token_id":', b',"logprob":')
_FIRST_TOP_LEADS = _LeadTable(b',"top_logprobs":[{"token_id":', b',"logprob":')
_NEXT_TOP_LEADS = _LeadTable(b'},{"token_id":', b',"logprob":')


def _rank_pieces(ranks: list[int]) -> list[bytes]:
    """``,"rank":<rank>`` with a rank of 0 sent as null, like ``rank or None``."""
    return [b',"rank":%d' % r if r else b',"rank":null' for r in ranks]


def format_float_reprs(values: np.ndarray) -> list[bytes]:
    """``[repr(float(v)).encode() for v in values]`` for a non-empty 1-D
    float64 array of finite values that float32 represents exactly.

    ``repr`` is what ``json.dumps`` emits. msgspec's shortest round-trip
    encoder is used instead where it gives the same digits: for
    ``1e-4 <= |v| < 1e16`` and zeros; other magnitudes, where the exponent
    notation differs (``1e-05`` vs ``0.00001``), use ``repr``. The fast path
    is disabled if an import-time probe finds msgspec formatting that differs
    from ``repr``.
    """
    floats = values.tolist()
    if not _MSGSPEC_FLOATS_MATCH_REPR:
        return [repr(v).encode("ascii") for v in floats]
    return _format_fast(values, floats)


def _format_fast(values: np.ndarray, floats: list[float]) -> list[bytes]:
    out = msgspec.json.encode(floats)[1:-1].split(b",")
    magnitude = np.abs(values)
    for i in np.flatnonzero(
        ~(((magnitude >= 1e-4) & (magnitude < 1e16)) | (magnitude == 0))
    ).tolist():
        out[i] = repr(floats[i]).encode("ascii")
    return out


def _msgspec_floats_match_repr() -> bool:
    """Probe the msgspec fast path against ``repr`` once at import: float32
    boundaries of the fast range, powers of ten, extremes and a fixed
    pseudo-random sample."""
    probe: list[float] = [0.0, -0.0, 0.1, 0.5, 1.0, 9999.0, 123.456, -1.2e-7]
    for edge in (1e-4, 1e16, 1.0, 1e-3, 1e6, 1e15, 3.4028235e38, 1e-45):
        e = np.float32(edge)
        probe += [
            float(e),
            float(np.nextafter(e, np.float32(0))),
            float(np.nextafter(e, np.float32(3.4028235e38))),
        ]
    probe += [float(np.float32(10.0**p)) for p in range(-45, 39)]
    bits = np.random.default_rng(1234).integers(0, 2**32, 4096, dtype=np.uint64)
    sample = bits.astype(np.uint32).view(np.float32)
    probe += sample[np.isfinite(sample)].astype(np.float64).tolist()
    probe += [-v for v in probe]
    values = np.array(probe, dtype=np.float64)
    values = values[np.isfinite(values)]
    probe = values.tolist()
    try:
        return _format_fast(values, probe) == [repr(v).encode("ascii") for v in probe]
    except Exception:
        return False


_MSGSPEC_FLOATS_MATCH_REPR = _msgspec_floats_match_repr()


def render_tokens_logprobs(
    sampled_token_ids: Sequence[int],
    container: FlatLogprobs,
    num_output_top_logprobs: int,
) -> list[bytes] | None:
    """Render the ``GenerateLogProbs`` JSON of one choice, as parts whose
    concatenation is the JSON.

    Per position, the per-entry path builds a dict from the row (keys keep
    their first occurrence's order, values come from the last occurrence;
    slot 0 has the engine rank, slot ``j`` rank ``j``), looks the sampled
    entry up by the sampled id, and lists the first ``max(k, 1)`` dict items
    as ``top_logprobs``; logprobs are clamped (NaN and values below -9999 are
    -9999.0) and a rank of 0 is null.

    Returns None when the rows do not map to that shape directly (sampled id
    not in slot 0, repeated top-k ids, fewer entries than requested, a value
    JSON cannot represent): the caller then uses the per-entry path.
    """
    engine_rows = container.rows()
    if engine_rows is None:
        return None
    token_ids, logprobs, engine_ranks = engine_rows
    n = len(sampled_token_ids)
    if n != len(token_ids):
        return None
    if n == 0:
        return [b'{"content":[]}']
    if not np.array_equal(token_ids[:, 0], np.asarray(sampled_token_ids)):
        return None

    num_slots = token_ids.shape[1]
    limit = max(num_output_top_logprobs, 1)
    cols = np.arange(limit)
    rank_pieces = np.array(_rank_pieces(list(range(num_slots + 1))), dtype=object)
    parts: list[bytes] = [b'{"content":[']
    for start in range(0, n, _RENDER_BLOCK_ROWS):
        ids = token_ids[start : start + _RENDER_BLOCK_ROWS]
        rows = len(ids)
        if num_slots > 2:
            # Repeated top-k ids: logprob_token_ids may repeat an id, and a
            # co-batched logprob_token_ids request pads other rows with id 0.
            top = np.sort(ids[:, 1:], axis=1)
            if (top[:, 1:] == top[:, :-1]).any():
                return None
        # The slot (>= 1) repeating the sampled id, at most one per row here.
        if num_slots > 1:
            eq = ids[:, 1:] == ids[:, :1]
            has_dup = eq.any(axis=1)
            dup_col = np.where(has_dup, eq.argmax(axis=1) + 1, num_slots)
        else:
            has_dup = np.zeros(rows, dtype=bool)
            dup_col = np.full(rows, num_slots)
        if num_slots - int(has_dup.any()) < limit:
            return None
        # Slots of the top-k entries: the duplicate's slot is skipped.
        src = cols[None, :] + (cols[None, :] >= dup_col[:, None])
        # The sampled entry's value and rank come from the duplicate's slot.
        sampled_src = np.where(has_dup, dup_col, 0)
        value_src = np.where(src == 0, sampled_src[:, None], src)
        values = np.take_along_axis(
            logprobs[start : start + rows], value_src, axis=1
        ).astype(np.float64)
        values = np.where(np.isnan(values), -9999.0, np.maximum(values, -9999.0))
        if not np.isfinite(values).all():
            return None
        sampled_ranks = np.where(has_dup, dup_col, engine_ranks[start : start + rows])

        # Per row: content lead, value, rank; first top lead, value, rank;
        # lead, value, rank per further top entry; the row end. Built as one
        # object array: per-row lists would be GC-tracked objects.
        floats = np.empty(rows * limit, dtype=object)
        floats[:] = format_float_reprs(values.ravel())
        floats = floats.reshape(rows, limit)
        sampled_rank_pieces = np.empty(rows, dtype=object)
        sampled_rank_pieces[:] = _rank_pieces(sampled_ranks.tolist())
        out = np.empty((rows, 3 * (limit + 1) + 1), dtype=object)
        out[:, 0] = _CONTENT_LEADS.lookup(ids[:, 0])
        out[:, 1] = floats[:, 0]
        out[:, 2] = sampled_rank_pieces
        out[:, 3] = _FIRST_TOP_LEADS.lookup(ids[:, 0])
        out[:, 4] = floats[:, 0]
        out[:, 5] = sampled_rank_pieces
        if limit > 1:
            top_src = src[:, 1:]
            out[:, 6:-1:3] = _NEXT_TOP_LEADS.lookup(
                np.take_along_axis(ids, top_src, axis=1)
            )
            out[:, 7:-1:3] = floats[:, 1:]
            out[:, 8:-1:3] = rank_pieces[top_src]
        out[:, -1] = b"}]},"
        parts.append(b"".join(out.ravel().tolist()))
    parts[-1] = parts[-1][:-1]  # the last row has no separator
    parts.append(b"]}")
    return parts


def _dumps(content: Any) -> bytes:
    """Same serialization as ``starlette.responses.JSONResponse.render``."""
    return json.dumps(
        content,
        ensure_ascii=False,
        allow_nan=False,
        indent=None,
        separators=(",", ":"),
    ).encode("utf-8")


def render_json_with_fragments(
    content: dict[str, Any], key: str, fragments: Mapping[int, list[bytes]]
) -> list[bytes]:
    """Render ``content`` like ``JSONResponse``, with the value of
    ``content["choices"][i][key]`` replaced by the pre-rendered JSON
    ``fragments[i]`` (parts to concatenate), as parts whose concatenation is
    the body. ``content`` is modified in place."""
    token = secrets.token_hex(16)
    for index in fragments:
        content["choices"][index][key] = token
    pieces = _dumps(content).split(f'"{token}"'.encode())
    if len(pieces) != len(fragments) + 1:
        raise AssertionError("Fragment placeholder collision")
    # Choices, hence placeholders, appear in index order.
    out = [pieces[0]]
    for index, piece in zip(sorted(fragments), pieces[1:]):
        out += fragments[index]
        out.append(piece)
    return out
