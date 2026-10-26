# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The row table: how a row-tier decode kernel learns which rows of a step are live."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

#: Header words before the row to request map.
HEADER_WORDS = 2
#: Word indices of the header.
ROWS_WORD = 0
REQUESTS_WORD = 1
#: Rows of one worker group and of one selector segment.
SEGMENT_ROWS = 4
SEGMENT_SLACK = 3
#: Selector slots of each packed tier, also its request capacity (a request takes at
#: least one slot; on tier 16 a selector rank serves requests ``r`` and ``r + 8``).
PACKED_SLOTS = {8: 8, 16: 16}


@dataclass(frozen=True)
class RowTableLayout:
    """Word offsets of the row table of one tier (``rmax`` rows, ``bmax`` requests)."""

    rmax: int
    bmax: int

    def __post_init__(self) -> None:
        if self.rmax < 1 or not 1 <= self.bmax <= self.rmax:
            raise ValueError(f"row table of {self.rmax} rows, {self.bmax} requests")

    @property
    def words(self) -> int:
        return self.block_row_word(self.bmax - 1) + 1

    @property
    def bytes(self) -> int:
        return 4 * self.words

    @property
    def step_word(self) -> int:
        return self.words

    @property
    def table_words(self) -> int:
        return self.words + 1

    def row_request_word(self, row: int) -> int:
        return HEADER_WORDS + row

    def first_row_word(self, request: int) -> int:
        return HEADER_WORDS + self.rmax + request

    def context_word(self, request: int) -> int:
        return HEADER_WORDS + self.rmax + self.bmax + request

    def block_row_word(self, request: int) -> int:
        return HEADER_WORDS + self.rmax + 2 * self.bmax + request


def header(
    layout: RowTableLayout,
    rows_per_request: Sequence[int],
    block_rows: Sequence[int] | None = None,
) -> list[int]:
    """Return the table a step whose requests own the given row counts starts with:
    everything the host knows, with the context words 0.
    """
    requests = len(rows_per_request)
    rows = sum(rows_per_request)
    if not 1 <= requests <= layout.bmax or not 1 <= rows <= layout.rmax:
        raise ValueError(
            f"{requests} requests / {rows} rows do not fit the tier "
            f"({layout.bmax} / {layout.rmax})"
        )
    if min(rows_per_request) < 1:
        raise ValueError("every live request owns at least one row")
    words = [0] * layout.words
    words[ROWS_WORD], words[REQUESTS_WORD] = rows, requests
    block_rows = range(requests) if block_rows is None else block_rows
    if len(block_rows) != requests:
        raise ValueError("one block-table row per request")
    first = 0
    for request, count in enumerate(rows_per_request):
        words[layout.first_row_word(request)] = first
        words[layout.block_row_word(request)] = block_rows[request]
        for row in range(first, first + count):
            words[layout.row_request_word(row)] = request
        first += count
    return words


def segments_per_request(rows_per_request: int) -> int:
    """Return how many selector segments (``SEGMENT_ROWS`` rows each, the last one
    shorter) a request of ``rows_per_request`` rows has.
    """
    return -(-rows_per_request // SEGMENT_ROWS)


@dataclass(frozen=True)
class Segment:
    """One selector slot of a packed step: ``rows`` consecutive rows of ``request``,
    starting ``offset`` rows into it.
    """

    request: int
    sub: int
    offset: int
    rows: int


def segment(slot: int, rows_per_request: int) -> Segment:
    """Return the segment selector slot ``slot`` serves in a uniform step: request
    ``slot // m`` and its ``slot % m``-th group of four rows (``m`` = segments per
    request).
    """
    count = segments_per_request(rows_per_request)
    request, sub = divmod(slot, count)
    offset = SEGMENT_ROWS * sub
    return Segment(request, sub, offset, min(SEGMENT_ROWS, rows_per_request - offset))
