# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import operator
from collections.abc import Callable, Iterable, Iterator, MutableSequence, Sequence
from dataclasses import dataclass
from typing import Any, overload

import numpy as np


# We use dataclass for now because it is used for
# openai server output, and msgspec is not serializable.
# TODO(sang): Fix it.
@dataclass
class Logprob:
    """Infos for supporting OpenAI compatible logprobs and token ranks.

    Attributes:
        logprob: The logprob of chosen token
        rank: The vocab rank of chosen token (>=1)
        decoded_token: The decoded chosen token index

    """

    logprob: float
    rank: int | None = None
    decoded_token: str | None = None


LogprobsOnePosition = dict[int, Logprob]


_NO_RANK = np.iinfo(np.int64).min


def _fits(values: np.ndarray, dtype: np.dtype) -> bool:
    """Whether ``values`` round-trip through ``dtype``."""
    if dtype.kind == values.dtype.kind and dtype.itemsize >= values.dtype.itemsize:
        return True
    with np.errstate(over="ignore", invalid="ignore"):
        cast = values.astype(dtype)
    # values != values marks NaN, which round-trips as NaN.
    return bool(((cast == values) | (values != values)).all())


class _Column:
    """An append-only 1-D array in a buffer that doubles when full. Values
    that ``dtype`` cannot hold exactly widen the column to ``wide``.

    Appends only write past the current length and growing allocates a new
    buffer, so views returned earlier stay valid.
    """

    def __init__(self, dtype: str, wide: str | None = None) -> None:
        self.dtype = np.dtype(dtype)
        self.wide = np.dtype(wide or dtype)
        self._buf = np.empty(0, dtype=self.dtype)
        self._len = 0

    def __len__(self) -> int:
        return self._len

    def append(self, values: np.ndarray | Sequence[int] | Sequence[float]) -> None:
        values = np.asarray(values)
        if not values.size:
            return
        start = self._len
        end = start + values.size
        if values.dtype != self.dtype and not _fits(values, self.dtype):
            self.dtype = self.wide
            self._buf = self.view().astype(self.wide)
        if end > self._buf.size:
            buf = np.empty(max(end, 2 * self._buf.size), dtype=self.dtype)
            buf[:start] = self._buf[:start]
            self._buf = buf
        self._buf[start:end] = values
        self._len = end

    def view(self) -> np.ndarray:
        return self._buf[: self._len]

    def item(self, index: int) -> int | float:
        return self._buf[index].item()

    def read(self, start: int, stop: int) -> list:
        return self._buf[start:stop].tolist()


class _ColumnView(Sequence[Any]):
    """A read-only view of one ``FlatLogprobs`` column that converts only the
    elements read, so indexing it in a loop costs O(1) per element and no
    copy is kept. It follows later appends."""

    # Elements converted per step while iterating.
    _BLOCK = 1024

    def __init__(
        self,
        length: Callable[[], int],
        read: Callable[[int, int], list],
        item: Callable[[int], Any] | None = None,
    ) -> None:
        self._length = length
        # read(start, stop) -> elements [start, stop) as Python values.
        self._read = read
        self._item = item or (lambda index: read(index, index + 1)[0])

    def __len__(self) -> int:
        return self._length()

    def __getitem__(self, index):
        n = len(self)
        if isinstance(index, slice):
            start, stop, step = index.indices(n)
            if step == 1:
                return self._read(start, max(start, stop))
            return [self[i] for i in range(start, stop, step)]
        i = operator.index(index)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError("FlatLogprobs column index out of range")
        return self._item(i)

    def __iter__(self) -> Iterator[Any]:
        n = len(self)
        for start in range(0, n, self._BLOCK):
            yield from self[start : min(start + self._BLOCK, n)]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sequence) or isinstance(other, (str, bytes)):
            return NotImplemented
        return len(self) == len(other) and list(self) == list(other)

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return repr(list(self))


class FlatLogprobs(MutableSequence[LogprobsOnePosition | None]):
    """Logprobs of a request stored as flat numpy columns.

    Compared to list[dict[int, Logprob]], this creates no Python object per
    position or entry: entries (position, candidate) are kept in a few
    append-only numpy columns, about 8.5 bytes per entry.

    Token ids and logprobs are kept as int32 / float32, the engine's dtypes,
    and a column widens to int64 / float64 when a value would not round-trip.
    The rank of each position's first entry is kept per position, and entry
    ``j > 0`` has rank ``j``, as ``append_logprobs_for_next_position`` gives;
    other ranks switch to a per-entry rank column. Decoded tokens are only
    kept once one is not None.

    NOTE: To reduce the migration overhead and improve backward compatibility,
    we support the key Sequence APIs of list, so it could act as
    list[LogprobsOnePosition]. The column attributes (``start_indices``,
    ``end_indices``, ``token_ids``, ``logprobs``, ``ranks``,
    ``decoded_tokens``) are read-only views that convert only the elements
    read; use ``list(...)`` for a list.
    """

    def __init__(
        self,
        start_indices: Sequence[int] | None = None,
        end_indices: Sequence[int] | None = None,
        token_ids: Sequence[int] | None = None,
        logprobs: Sequence[float] | None = None,
        ranks: Sequence[int | None] | None = None,
        decoded_tokens: Sequence[str | None] | None = None,
    ) -> None:
        self._num_positions = 0
        # Entries per position while all positions have the same number.
        self._width = 0
        # Position i spans entries [ends[i - 1], ends[i]); kept once
        # positions have different widths.
        self._ends: _Column | None = None
        self._token_ids = _Column("<i4", "<i8")
        self._logprobs = _Column("<f4", "<f8")
        # Rank of each position's first entry.
        self._first_ranks = _Column("<i8")
        # Rank of every entry (_NO_RANK for None), once an entry j > 0 of
        # some position had another rank than j.
        self._ranks: _Column | None = None
        self._decoded: list[str | None] | None = None
        if start_indices is not None:
            assert end_indices is not None
            assert token_ids is not None and logprobs is not None
            assert ranks is not None and decoded_tokens is not None
            for start, end in zip(start_indices, end_indices):
                self._append_position(
                    token_ids[start:end],
                    logprobs[start:end],
                    ranks[start:end],
                    decoded_tokens[start:end],
                )

    def append(self, logprobs_one_position: LogprobsOnePosition | None) -> None:
        """Appends the container with logprobs for the next position."""
        entries = logprobs_one_position or {}
        self._append_position(
            list(entries),
            [logprob.logprob for logprob in entries.values()],
            [logprob.rank for logprob in entries.values()],
            [logprob.decoded_token for logprob in entries.values()],
        )

    def append_fast(
        self,
        token_ids: list[int],
        logprobs: list[float],
        ranks: Iterable[int | None],
        decoded_tokens: Iterable[str | None],
    ) -> None:
        """Appends logprobs for the next position without creating
        the intermediate logprob dictionary. Like ``zip``, stops at the
        shortest input.
        """
        ranks = list(ranks)
        n = min(len(token_ids), len(logprobs), len(ranks))
        decoded = list(itertools.islice(decoded_tokens, n))
        n = len(decoded)
        self._append_position(token_ids[:n], logprobs[:n], ranks[:n], decoded)

    def append_rows(
        self,
        token_ids: np.ndarray,
        logprobs: np.ndarray,
        first_ranks: np.ndarray,
        decoded_tokens: Sequence[str | None] | None = None,
    ) -> None:
        """Appends ``n`` positions given as engine rows: ``[n, S]`` token ids
        and logprobs and the ``[n]`` ranks of slot 0 (slot ``j > 0`` has rank
        ``j``), optionally with the ``n * S`` decoded tokens."""
        self._append(
            token_ids.shape[1],
            token_ids.reshape(-1),
            logprobs.reshape(-1),
            first_ranks,
            decoded_tokens=decoded_tokens,
        )

    def _append_position(
        self,
        token_ids: Sequence[int],
        logprobs: Sequence[float],
        ranks: Sequence[int | None],
        decoded_tokens: Sequence[str | None],
    ) -> None:
        width = len(token_ids)
        canonical = width == 0 or (
            ranks[0] is not None and list(ranks[1:]) == list(range(1, width))
        )
        first_rank = ranks[0] if width and canonical else 0
        entry_ranks = (
            None
            if canonical
            else np.array([_NO_RANK if r is None else r for r in ranks], np.int64)
        )
        self._append(
            width,
            token_ids,
            logprobs,
            np.array([first_rank]),
            entry_ranks,
            decoded_tokens,
        )

    def _append(
        self,
        widths: int | np.ndarray,
        token_ids: np.ndarray | Sequence[int],
        logprobs: np.ndarray | Sequence[float],
        first_ranks: np.ndarray,
        ranks: np.ndarray | None = None,
        decoded_tokens: Sequence[str | None] | None = None,
    ) -> None:
        """Appends ``len(first_ranks)`` positions with ``widths`` entries
        (one for all, or per position). ``ranks`` gives every entry's rank
        when they are not the canonical ones."""
        n = len(first_ranks)
        if n == 0:
            return
        base = len(self._token_ids)
        if ranks is not None and self._ranks is None:
            self._ranks = _Column("<i8")
            self._ranks.append(self._canonical_ranks(0, base))
        if (
            self._ends is None
            and isinstance(widths, int)
            and (self._num_positions == 0 or widths == self._width)
        ):
            self._width = widths
        else:
            if self._ends is None:
                uniform_ends = self._ends_range(0, self._num_positions)
                self._ends = _Column("<i8")
                self._ends.append(uniform_ends)
            if not isinstance(widths, int):
                ends = base + np.cumsum(widths)
            elif n == 1:
                ends = np.array([base + widths])
            else:
                ends = base + widths * np.arange(1, n + 1)
            self._ends.append(ends)
        self._num_positions += n
        self._token_ids.append(token_ids)
        self._logprobs.append(logprobs)
        self._first_ranks.append(first_ranks)
        count = len(self._token_ids) - base
        if self._ranks is not None:
            self._ranks.append(
                self._canonical_ranks(base, base + count) if ranks is None else ranks
            )
        if (
            self._decoded is None
            and decoded_tokens is not None
            and any(token is not None for token in decoded_tokens)
        ):
            self._decoded = [None] * base
        if self._decoded is not None:
            self._decoded.extend(
                itertools.repeat(None, count)
                if decoded_tokens is None
                else decoded_tokens
            )

    def _ends_range(self, start: int, stop: int) -> np.ndarray:
        """End offsets of positions ``[start, stop)``."""
        if self._ends is None:
            return self._width * np.arange(start + 1, stop + 1)
        return self._ends.view()[start:stop]

    def _start(self, position: int) -> int:
        if self._ends is None:
            return position * self._width
        return int(self._ends.view()[position - 1]) if position else 0

    def _entry_ranks(self, start: int, stop: int) -> np.ndarray:
        """Ranks of entries ``[start, stop)``, int64 with ``_NO_RANK`` for
        None."""
        if self._ranks is not None:
            return self._ranks.view()[start:stop]
        return self._canonical_ranks(start, stop)

    def _canonical_ranks(self, start: int, stop: int) -> np.ndarray:
        """Ranks of entries ``[start, stop)`` from their position's first
        rank: entry ``j > 0`` of a position has rank ``j``."""
        entries = np.arange(start, stop)
        if self._ends is None:
            positions = entries // max(self._width, 1)
            begins = positions * self._width
        else:
            ends = self._ends.view()
            positions = np.searchsorted(ends, entries, side="right")
            begins = np.where(positions > 0, ends[positions - 1], 0)
        return np.where(
            entries == begins, self._first_ranks.view()[positions], entries - begins
        )

    def _extend_from(self, source: "FlatLogprobs", start: int, stop: int) -> None:
        """Appends positions ``[start, stop)`` of ``source``."""
        if start >= stop:
            return
        begin, end = source._start(start), source._start(stop)
        widths: int | np.ndarray = source._width
        if source._ends is not None:
            widths = np.diff(source._ends.view()[start:stop], prepend=begin)
            if (widths == widths[0]).all():
                widths = int(widths[0])
        self._append(
            widths,
            source._token_ids.view()[begin:end],
            source._logprobs.view()[begin:end],
            source._first_ranks.view()[start:stop],
            None if source._ranks is None else source._ranks.view()[begin:end],
            None if source._decoded is None else source._decoded[begin:end],
        )

    def extend(self, logprobs_multi_positions) -> None:
        """Extends the container with logprobs for the next multiple positions."""
        if isinstance(logprobs_multi_positions, FlatLogprobs):
            self._extend_from(
                logprobs_multi_positions, 0, len(logprobs_multi_positions)
            )
            return
        for logprobs_one_position in logprobs_multi_positions:
            self.append(logprobs_one_position)

    @property
    def num_entries(self) -> int:
        """Number of (position, candidate) entries."""
        return len(self._token_ids)

    def rows(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """The entries as engine rows: ``token_ids[N, S]`` (int32),
        ``logprobs[N, S]`` (float32) and the ranks of slot 0 ``[N]`` (int64).

        None unless every position has the same ``S > 0`` entries, entry
        ``j > 0`` of each has rank ``j`` and no column was widened.
        """
        n = len(self)
        if n and (
            self._ends is not None
            or self._width < 1
            or self._ranks is not None
            or self._token_ids.dtype != np.dtype("<i4")
            or self._logprobs.dtype != np.dtype("<f4")
        ):
            return None
        return (
            self._token_ids.view().reshape(n, self._width),
            self._logprobs.view().reshape(n, self._width),
            self._first_ranks.view(),
        )

    def first_token_ids(self, start: int, stop: int) -> list[int]:
        """Token id of the first entry of each position in ``[start, stop)``
        that has entries."""
        if start >= stop:
            return []
        ends = self._ends_range(start, stop)
        begins = np.concatenate(([self._start(start)], ends[:-1]))
        return self._token_ids.view()[begins[begins < ends]].tolist()

    @property
    def start_indices(self) -> Sequence[int]:
        return _ColumnView(self.__len__, self._start_list, self._start)

    @property
    def end_indices(self) -> Sequence[int]:
        return _ColumnView(self.__len__, self._end_list, self._end)

    @property
    def token_ids(self) -> Sequence[int]:
        column = self._token_ids
        return _ColumnView(column.__len__, column.read, column.item)

    @property
    def logprobs(self) -> Sequence[float]:
        column = self._logprobs
        return _ColumnView(column.__len__, column.read, column.item)

    @property
    def ranks(self) -> Sequence[int | None]:
        return _ColumnView(self._token_ids.__len__, self._rank_list)

    @property
    def decoded_tokens(self) -> Sequence[str | None]:
        return _ColumnView(self._token_ids.__len__, self._decoded_list)

    def _start_list(self, start: int, stop: int) -> list[int]:
        if start >= stop:
            return []
        return [self._start(start), *self._ends_range(start, stop - 1).tolist()]

    def _end_list(self, start: int, stop: int) -> list[int]:
        return self._ends_range(start, stop).tolist()

    def _end(self, position: int) -> int:
        return self._start(position + 1)

    def _rank_list(self, start: int, stop: int) -> list[int | None]:
        ranks = self._entry_ranks(start, stop).tolist()
        return [None if rank == _NO_RANK else rank for rank in ranks]

    def _decoded_list(self, start: int, stop: int) -> list[str | None]:
        if self._decoded is None:
            return [None] * (stop - start)
        return self._decoded[start:stop]

    def __len__(self) -> int:
        """Gets number of positions stored in the container."""
        return self._num_positions

    @overload
    def __getitem__(self, position: int) -> LogprobsOnePosition: ...

    @overload
    def __getitem__(self, s: slice, /) -> "FlatLogprobs": ...

    def __getitem__(self, index: int | slice):
        """Extracts logprobs of a given position or slice."""
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            sliced = FlatLogprobs()
            if step == 1:
                sliced._extend_from(self, start, stop)
            else:
                for position in range(start, stop, step):
                    sliced._extend_from(self, position, position + 1)
            return sliced
        if not isinstance(index, int):
            raise TypeError(f"Invalid index type: {type(index)}")
        position = range(len(self))[index]
        return self._position(position, self._start(position), self._end(position))

    def _position(self, position: int, begin: int, end: int) -> LogprobsOnePosition:
        """Position ``position``, whose entries are ``[begin, end)``."""
        if self._ranks is not None:
            ranks = self._ranks.read(begin, end)
        else:
            ranks = [self._first_ranks.item(position), *range(1, end - begin)]
        decoded: Iterable[str | None] = (
            itertools.repeat(None)
            if self._decoded is None
            else self._decoded[begin:end]
        )
        return {
            token_id: Logprob(
                logprob=logprob,
                rank=None if rank == _NO_RANK else rank,
                decoded_token=decoded_token,
            )
            for token_id, logprob, rank, decoded_token in zip(
                self._token_ids.read(begin, end),
                self._logprobs.read(begin, end),
                ranks,
                decoded,
            )
        }

    def __setitem__(self, item, value) -> None:
        raise TypeError("Cannot set logprobs in FlatLogprobs")

    def __delitem__(self, item) -> None:
        raise TypeError("Cannot delete logprobs from FlatLogprobs")

    def insert(self, index: int, value: dict[int, Logprob] | None) -> None:
        raise TypeError("Cannot insert logprobs to FlatLogprobs")

    def __iter__(self) -> Iterator[LogprobsOnePosition]:
        """Iterates the container and yields LogprobsOnePosition for
        each position.
        """
        begin = 0
        for position, end in enumerate(self._ends_range(0, len(self)).tolist()):
            yield self._position(position, begin, end)
            begin = end

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FlatLogprobs):
            return NotImplemented
        return (
            np.array_equal(
                self._ends_range(0, len(self)), other._ends_range(0, len(other))
            )
            and np.array_equal(self._token_ids.view(), other._token_ids.view())
            and np.array_equal(
                self._logprobs.view(), other._logprobs.view(), equal_nan=True
            )
            and np.array_equal(
                self._entry_ranks(0, len(self)), other._entry_ranks(0, len(other))
            )
            and self.decoded_tokens == other.decoded_tokens
        )

    def __repr__(self) -> str:
        return (
            f"FlatLogprobs(num_positions={len(self)}, num_entries={self.num_entries})"
        )


# {token_id -> logprob} per each sequence group. None if the corresponding
# sequence group doesn't require prompt logprob.
PromptLogprobs = FlatLogprobs | list[LogprobsOnePosition | None]
# {token_id -> logprob} for each sequence group.
SampleLogprobs = FlatLogprobs | list[LogprobsOnePosition]


def create_prompt_logprobs(flat_logprobs: bool) -> PromptLogprobs:
    """Creates a container to store prompt logprobs for a request."""
    logprobs: PromptLogprobs = FlatLogprobs() if flat_logprobs else []
    # NOTE: logprob of first prompt token is None.
    logprobs.append(None)
    return logprobs


def create_sample_logprobs(flat_logprobs: bool) -> SampleLogprobs:
    """Creates a container to store decode logprobs for a request."""
    return FlatLogprobs() if flat_logprobs else []


def append_logprobs_for_next_position(
    request_logprobs: PromptLogprobs | SampleLogprobs,
    token_ids: list[int],
    logprobs: list[float],
    decoded_tokens: Iterable[str | None],
    rank: int,
    num_logprobs: int,
) -> None:
    """Appends logprobs for the next position."""
    if num_logprobs == -1:
        num_logprobs = len(logprobs)
    # We do not need a special case for the sampled token
    # being in the topk, since inserting duplicated data
    # into a dictionary twice is the same as doing it once.
    topk_ranks = range(1, num_logprobs + 1)
    ranks = itertools.chain((rank,), topk_ranks)

    if isinstance(request_logprobs, FlatLogprobs):
        request_logprobs.append_fast(token_ids, logprobs, ranks, decoded_tokens)
    else:
        request_logprobs.append(
            {
                token_id: Logprob(
                    logprob=logprob,
                    rank=rank,
                    decoded_token=token,
                )
                for token_id, logprob, rank, token in zip(
                    token_ids, logprobs, ranks, decoded_tokens
                )
            }
        )
