# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Renders `/inference/v1/generate` sample logprobs from `FlatLogprobs`
engine rows without a Python object per entry: the same JSON values as
`ServingTokens._create_tokens_logprobs` + `JSONResponse` (orjson may write a
float in other notation, e.g. `-0.00001` for `-1e-05`)."""

import asyncio
import contextlib
import json
import secrets
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import orjson

from vllm.logprobs import FlatLogprobs

# Logprob entries rendered between yields to the event loop (about 2 ms).
RENDER_BLOCK_ENTRIES = 1 << 13


def _object_array(items: list) -> np.ndarray:
    array = np.empty(len(items), dtype=object)
    array[:] = items
    return array


class _LeadTable:
    """Cached `prefix + str(id) + suffix` per token id below `max_ids`, so a
    block of ids maps to its JSON pieces with one fancy index."""

    max_ids = 1 << 18

    def __init__(self, prefix: bytes, suffix: bytes):
        self.prefix, self.suffix = prefix, suffix
        self._values: np.ndarray | None = None
        self._filled = np.zeros(self.max_ids, dtype=bool)

    def _formatted(self, ids: np.ndarray) -> np.ndarray:
        return _object_array(
            [b"%s%d%s" % (self.prefix, i, self.suffix) for i in ids.tolist()]
        )

    def lookup(self, ids: np.ndarray) -> np.ndarray:
        """The pieces of a non-empty array of ids, in its shape."""
        if ids.min() < 0 or ids.max() >= self.max_ids:
            return self._formatted(ids.ravel()).reshape(ids.shape)
        if self._values is None:
            self._values = np.empty(self.max_ids, dtype=object)
        missing = np.unique(ids[~self._filled[ids]])
        if missing.size:
            self._values[missing] = self._formatted(missing)
            self._filled[missing] = True
        return self._values[ids]


# Pieces before each logprob value: the content entry, the first top-k entry
# (the sampled token again) and every further top-k entry.
_CONTENT_LEADS = _LeadTable(b'{"token_id":', b',"logprob":')
_FIRST_TOP_LEADS = _LeadTable(b',"top_logprobs":[{"token_id":', b',"logprob":')
_NEXT_TOP_LEADS = _LeadTable(b'},{"token_id":', b',"logprob":')


def _rank_pieces(ranks: list[int]) -> list[bytes]:
    """`,"rank":<rank>` with a rank of 0 sent as null, like `rank or None`."""
    return [b',"rank":%d' % r if r else b',"rank":null' for r in ranks]


async def render_tokens_logprobs(
    sampled_token_ids: Sequence[int],
    container: FlatLogprobs,
    lock: asyncio.Lock | None = None,
) -> list[bytes] | None:
    """See `_render_tokens_logprobs`. Renders of more than one block hold
    `lock`: on one event loop, interleaving them would only hold more partial
    output at once."""
    rows = container.rows()
    if rows is None:
        return None
    async with (
        lock
        if lock is not None and rows[0].size > RENDER_BLOCK_ENTRIES
        else contextlib.nullcontext()
    ):
        return await _render_tokens_logprobs(sampled_token_ids, rows)


async def _render_tokens_logprobs(
    sampled_token_ids: Sequence[int],
    engine_rows: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> list[bytes] | None:
    """The `GenerateLogProbs` JSON of one choice, as parts to concatenate.

    Matches the per-entry path, which builds a dict per row (first
    occurrence's key order, last occurrence's value; slot 0 has the engine
    rank, slot `j` rank `j`), looks the sampled entry up in it and lists all
    its items as `top_logprobs`, clamping logprobs to -9999.0 (NaN too) and
    sending rank 0 as null. Returns None for rows it cannot map directly
    (sampled id not in slot 0, repeated top-k ids, non-finite values)."""
    token_ids, logprobs, engine_ranks = engine_rows
    n = len(sampled_token_ids)
    if n != len(token_ids):
        return None
    if n == 0:
        return [b'{"content":[]}']
    if not np.array_equal(token_ids[:, 0], np.asarray(sampled_token_ids)):
        return None

    num_slots = token_ids.shape[1]
    cols = np.arange(num_slots)
    rank_pieces = _object_array(_rank_pieces(list(range(num_slots + 1))))
    block_rows = max(1, RENDER_BLOCK_ENTRIES // num_slots)
    parts: list[bytes] = [b'{"content":[']
    for start in range(0, n, block_rows):
        if start:
            await asyncio.sleep(0)
        ids = token_ids[start : start + block_rows]
        rows = len(ids)
        # Repeated top-k ids (e.g. from logprob_token_ids).
        top = np.sort(ids[:, 1:], axis=1)
        if (top[:, 1:] == top[:, :-1]).any():
            return None
        # The slot (>= 1) repeating the sampled id, or num_slots if none.
        dup_col = np.where(ids[:, 1:] == ids[:, :1], cols[1:], num_slots).min(
            axis=1, initial=num_slots
        )
        has_dup = dup_col < num_slots
        # Slots of the top-k entries: a row with a duplicate skips its slot
        # and so has no last entry.
        src = cols[None, :] + (cols[None, :] >= dup_col[:, None])
        missing = src == num_slots
        src = np.minimum(src, num_slots - 1)
        # The sampled entry's value and rank come from the duplicate's slot.
        sampled_src = np.where(has_dup, dup_col, 0)
        value_src = np.where(src == 0, sampled_src[:, None], src)
        values = np.take_along_axis(
            logprobs[start : start + rows], value_src, axis=1
        ).astype(np.float64)
        values = np.where(np.isnan(values), -9999.0, np.maximum(values, -9999.0))
        # orjson would write inf as null; the per-entry path rejects it.
        if not np.isfinite(values).all():
            return None
        sampled_ranks = np.where(has_dup, dup_col, engine_ranks[start : start + rows])

        # Per row: lead, value and rank of the content entry, of the first
        # top entry and of each further one, then the row end, in one object
        # array (per-row lists would be GC-tracked).
        encoded = orjson.dumps(values.ravel(), option=orjson.OPT_SERIALIZE_NUMPY)
        floats = _object_array(encoded[1:-1].split(b",")).reshape(rows, num_slots)
        sampled_rank_pieces = _object_array(_rank_pieces(sampled_ranks.tolist()))
        out = np.empty((rows, 3 * (num_slots + 1) + 1), dtype=object)
        out[:, 0] = _CONTENT_LEADS.lookup(ids[:, 0])
        out[:, 1] = floats[:, 0]
        out[:, 2] = sampled_rank_pieces
        out[:, 3] = _FIRST_TOP_LEADS.lookup(ids[:, 0])
        out[:, 4] = floats[:, 0]
        out[:, 5] = sampled_rank_pieces
        if num_slots > 1:
            top_src = src[:, 1:]
            out[:, 6:-1:3] = _NEXT_TOP_LEADS.lookup(
                np.take_along_axis(ids, top_src, axis=1)
            )
            out[:, 7:-1:3] = floats[:, 1:]
            out[:, 8:-1:3] = rank_pieces[top_src]
            for start_col in (6, 7, 8):
                out[:, start_col:-1:3][missing[:, 1:]] = b""
        out[:, -1] = b"}]},"
        parts.append(b"".join(out.ravel().tolist()))
    parts[-1] = parts[-1][:-1]  # the last row has no separator
    parts.append(b"]}")
    return parts


def _dumps(content: Any) -> bytes:
    """Same serialization as `starlette.responses.JSONResponse.render`."""
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
    """`content` rendered like `JSONResponse`, as parts to concatenate, with
    `content["choices"][i][key]` (modified in place) replaced by the
    pre-rendered parts `fragments[i]`."""
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
