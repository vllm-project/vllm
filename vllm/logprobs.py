# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from collections.abc import Callable, Iterable, Iterator, MutableSequence, Sequence
from dataclasses import dataclass
from typing import Any, overload

import numpy as np


# We use dataclass for now because it is used for
# openai server output, and msgspec is not serializable.
# TODO(sang): Fix it.
@dataclass(slots=True)
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
    """Whether `values` round-trip through `dtype` (NaN included)."""
    if dtype.kind == values.dtype.kind and dtype.itemsize >= values.dtype.itemsize:
        return True
    with np.errstate(over="ignore", invalid="ignore"):
        cast = values.astype(dtype)
    return bool(((cast == values) | (values != values)).all())


class _Column:
    """An append-only 1-D array in a buffer that doubles when full, widened
    from `dtype` to `wide` when a value would not round-trip. Growing
    allocates a new buffer, so earlier views stay valid."""

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
        start, end = self._len, self._len + values.size
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

    def copy(self, start: int, stop: int) -> "_Column":
        """A new column holding elements `[start, stop)`."""
        column = _Column.__new__(_Column)
        column.dtype, column.wide = self.dtype, self.wide
        column._buf = self._buf[start:stop].copy()
        column._len = column._buf.size
        return column


class _ColumnView(Sequence[Any]):
    """A read-only view of one `FlatLogprobs` column that follows later
    appends and converts only the elements read."""

    def __init__(
        self,
        length: Callable[[], int],
        read: Callable[[int, int], list],
        item: Callable[[int], Any] | None = None,
    ) -> None:
        self._length = length
        self._read = read  # (start, stop) -> those elements as Python values
        self._item = item or (lambda index: read(index, index + 1)[0])

    def __len__(self) -> int:
        return self._length()

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step == 1:
                return self._read(start, max(start, stop))
            return [self._item(i) for i in range(start, stop, step)]
        return self._item(range(len(self))[index])

    def __iter__(self) -> Iterator[Any]:
        n = len(self)
        for start in range(0, n, 1024):
            yield from self._read(start, min(start + 1024, n))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sequence) or isinstance(other, (str, bytes)):
            return NotImplemented
        return len(self) == len(other) and list(self) == list(other)

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return repr(list(self))


class FlatLogprobs(MutableSequence[LogprobsOnePosition | None]):
    """Logprobs of a request in a few append-only numpy columns, with no
    Python object per position or entry (about 8.5 bytes per entry).

    Token ids and logprobs are int32 / float32, the engine's dtypes, widened
    to int64 / float64 if a value would not round-trip. A position's first
    entry keeps its rank and entry `j > 0` has rank `j`, as
    `append_logprobs_for_next_position` gives; any other rank switches to a
    per-entry rank column. Decoded tokens are kept once one is not None.

    It supports the key Sequence APIs of list[LogprobsOnePosition]. The
    column attributes (`start_indices`, `end_indices`, `token_ids`,
    `logprobs`, `ranks`, `decoded_tokens`) are read-only views.
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
        # Entries per position, while all positions have the same number.
        self._width = 0
        # End offset of each position, once positions have different widths.
        self._ends: _Column | None = None
        self._token_ids = _Column("<i4", "<i8")
        self._logprobs = _Column("<f4", "<f8")
        self._first_ranks = _Column("<i8")
        # Rank of every entry (_NO_RANK for None), once a rank is irregular.
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
        the intermediate logprob dictionary. Like `zip`, stops at the
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
        """Appends `n` positions given as engine rows: `[n, S]` token ids
        and logprobs, the `[n]` ranks of slot 0 (or the `[n, >=S]` ranks of
        every slot) and optionally the `n * S` decoded tokens."""
        width = token_ids.shape[1]
        ranks = None
        if first_ranks.ndim == 2:
            entry_ranks = first_ranks[:, :width]
            if not (entry_ranks[:, 1:] == np.arange(1, width)).all():
                ranks = entry_ranks.reshape(-1).astype(np.int64)
            first_ranks = entry_ranks[:, 0] if width else first_ranks[:, 0]
        self._append(
            width,
            token_ids.reshape(-1),
            logprobs.reshape(-1),
            first_ranks,
            ranks,
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
        self._append(
            width,
            token_ids,
            logprobs,
            np.array([ranks[0] if width and canonical else 0]),
            None
            if canonical
            else np.array([_NO_RANK if r is None else r for r in ranks], np.int64),
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
        """Appends `len(first_ranks)` positions of `widths` entries (one for
        all, or each), with every entry's rank in `ranks` if not canonical."""
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
                uniform_ends = self._bounds(0, self._num_positions)[1:]
                self._ends = _Column("<i8")
                self._ends.append(uniform_ends)
            self._ends.append(base + np.cumsum(np.broadcast_to(widths, n)))
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

    def _bounds(self, start: int, stop: int) -> np.ndarray:
        """Entry offsets `b` of positions `[start, stop)`: position
        `start + i` spans entries `[b[i], b[i + 1])`."""
        if self._ends is None:
            return self._width * np.arange(start, stop + 1)
        ends = self._ends.view()
        return np.concatenate(
            (ends[start - 1 : start] if start else [0], ends[start:stop])
        )

    def _start(self, position: int) -> int:
        if self._ends is None:
            return position * self._width
        return int(self._ends.view()[position - 1]) if position else 0

    def _entry_ranks(self, start: int, stop: int) -> np.ndarray:
        """Ranks of entries `[start, stop)`, with `_NO_RANK` for None."""
        if self._ranks is not None:
            return self._ranks.view()[start:stop]
        return self._canonical_ranks(start, stop)

    def _canonical_ranks(self, start: int, stop: int) -> np.ndarray:
        """Ranks of entries `[start, stop)` from their position's first
        rank: entry `j > 0` of a position has rank `j`."""
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
        """Appends positions `[start, stop)` of `source`."""
        if start >= stop:
            return
        bounds = source._bounds(start, stop)
        begin, end = int(bounds[0]), int(bounds[-1])
        widths = np.diff(bounds)
        self._append(
            int(widths[0]) if (widths == widths[0]).all() else widths,
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
        """The entries as engine rows: int32 `token_ids[N, S]`, float32
        `logprobs[N, S]` and the int64 ranks of slot 0 `[N]`; None unless
        all positions have `S > 0` entries, canonical ranks and no column
        was widened."""
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
        """Token id of the first entry of each position in `[start, stop)`
        that has entries."""
        bounds = self._bounds(start, stop)
        begins, ends = bounds[:-1], bounds[1:]
        return self._token_ids.view()[begins[begins < ends]].tolist()

    @property
    def start_indices(self) -> Sequence[int]:
        return _ColumnView(
            self.__len__, lambda a, b: self._bounds(a, b)[:-1].tolist(), self._start
        )

    @property
    def end_indices(self) -> Sequence[int]:
        return _ColumnView(
            self.__len__,
            lambda a, b: self._bounds(a, b)[1:].tolist(),
            lambda i: self._start(i + 1),
        )

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
        return _ColumnView(
            self._token_ids.__len__,
            lambda a, b: [
                None if r == _NO_RANK else r for r in self._entry_ranks(a, b).tolist()
            ],
        )

    @property
    def decoded_tokens(self) -> Sequence[str | None]:
        return _ColumnView(
            self._token_ids.__len__,
            lambda a, b: (
                [None] * (b - a) if self._decoded is None else self._decoded[a:b]
            ),
        )

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
            if step == 1 and self._ends is None:
                return self._uniform_slice(start, max(start, stop))
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
        return self._position(
            position, self._start(position), self._start(position + 1)
        )

    def _uniform_slice(self, start: int, stop: int) -> "FlatLogprobs":
        """Positions `[start, stop)` of same-width positions, copying each
        column directly (e.g. the per-step `[-n:]` of DELTA outputs)."""
        begin, end = start * self._width, stop * self._width
        sliced = FlatLogprobs()
        sliced._num_positions = stop - start
        sliced._width = self._width
        sliced._token_ids = self._token_ids.copy(begin, end)
        sliced._logprobs = self._logprobs.copy(begin, end)
        sliced._first_ranks = self._first_ranks.copy(start, stop)
        if self._ranks is not None:
            sliced._ranks = self._ranks.copy(begin, end)
        if self._decoded is not None:
            sliced._decoded = self._decoded[begin:end]
        return sliced

    def _position(self, position: int, begin: int, end: int) -> LogprobsOnePosition:
        """Position `position`, whose entries are `[begin, end)`."""
        if self._ranks is not None:
            ranks = self._ranks.read(begin, end)
        else:
            ranks = [self._first_ranks.item(position), *range(1, end - begin)]
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
                itertools.repeat(None)
                if self._decoded is None
                else self._decoded[begin:end],
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
        bounds = self._bounds(0, len(self)).tolist()
        for position in range(len(self)):
            yield self._position(position, bounds[position], bounds[position + 1])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FlatLogprobs):
            return NotImplemented
        return (
            np.array_equal(self._bounds(0, len(self)), other._bounds(0, len(other)))
            and np.array_equal(self._token_ids.view(), other._token_ids.view())
            and np.array_equal(
                self._logprobs.view(), other._logprobs.view(), equal_nan=True
            )
            and np.array_equal(
                self._entry_ranks(0, self.num_entries),
                other._entry_ranks(0, other.num_entries),
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
    rank: int | list[int],
    num_logprobs: int,
) -> None:
    """Appends logprobs for the next position."""
    if num_logprobs == -1:
        num_logprobs = len(logprobs) - 1
    ranks: Iterable[int]
    if isinstance(rank, list):
        ranks = rank[: num_logprobs + 1]
    else:
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
