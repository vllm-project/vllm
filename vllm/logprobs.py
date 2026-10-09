# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import bisect
import itertools
from collections.abc import Callable, Iterable, Iterator, MutableSequence, Sequence
from dataclasses import dataclass
from typing import overload

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
    if values.dtype == dtype:
        return True
    if dtype.kind == "f":
        with np.errstate(over="ignore"):
            return bool(np.array_equal(values.astype(dtype), values, equal_nan=True))
    info = np.iinfo(dtype)
    return bool(values.min() >= info.min and values.max() <= info.max)


class _Column:
    """An append-only 1-D array kept in numpy blocks, so growing it never
    copies stored values. Block sizes grow geometrically up to
    ``BLOCK_BYTES``. Values that ``dtype`` cannot hold exactly widen the
    column to ``wide``."""

    BLOCK_BYTES = 8 << 20

    def __init__(self, dtype: str, wide: str | None = None) -> None:
        self.dtype = np.dtype(dtype)
        self.wide = np.dtype(wide or dtype)
        self._blocks: list[np.ndarray] = []
        # Index of each block's first value; all blocks but the last are full.
        self._starts: list[int] = []
        self._len = 0

    def __len__(self) -> int:
        return self._len

    def append(self, values: np.ndarray) -> None:
        n = len(values)
        if n == 0:
            return
        if self.dtype != self.wide and not _fits(values, self.dtype):
            self.dtype = self.wide
            self._blocks = [block.astype(self.wide) for block in self._blocks]
        if not self._blocks:
            self._blocks.append(np.array(values, dtype=self.dtype))
            self._starts.append(0)
            self._len = n
            return
        block = self._blocks[-1]
        fill = self._len - self._starts[-1]
        if n <= len(block) - fill:
            block[fill : fill + n] = values
            self._len += n
            return
        pos = 0
        while pos < n:
            fill = self._len - self._starts[-1]
            if fill == len(self._blocks[-1]):
                self._new_block(n - pos)
                fill = 0
            block = self._blocks[-1]
            take = min(n - pos, len(block) - fill)
            block[fill : fill + take] = values[pos : pos + take]
            self._len += take
            pos += take

    def append_list(self, values: Sequence[int] | Sequence[float]) -> None:
        """Appends Python numbers, widening only if one does not round-trip."""
        try:
            array = np.array(values, dtype=self.dtype)
            exact = array.tolist() == list(values)
        except OverflowError:
            exact = False
        if not exact:
            array = np.array(values, dtype=self.wide)
        self.append(array)

    def _new_block(self, remaining: int) -> None:
        max_size = max(1, self.BLOCK_BYTES // self.dtype.itemsize)
        size = max(remaining, min(max_size, 2 * len(self._blocks[-1])))
        self._starts.append(self._len)
        self._blocks.append(np.empty(size, dtype=self.dtype))

    def view(self) -> np.ndarray:
        """All values as one array. Blocks are concatenated once and kept as
        one block afterwards; the result is never written to again."""
        if not self._blocks:
            return np.empty(0, dtype=self.dtype)
        values = self._blocks[-1][: self._len - self._starts[-1]]
        if len(self._blocks) > 1:
            values = np.concatenate([*self._blocks[:-1], values])
        self._blocks = [values]
        self._starts = [0]
        return values

    def range(self, start: int, stop: int) -> np.ndarray:
        """Values ``[start, stop)``; a view when they are in one block."""
        if start >= stop:
            return np.empty(0, dtype=self.dtype)
        first = bisect.bisect_right(self._starts, start) - 1
        offset = self._starts[first]
        block = self._blocks[first]
        if stop - offset <= len(block):
            return block[start - offset : stop - offset]
        last = bisect.bisect_right(self._starts, stop - 1)
        pieces = [
            block[max(start - offset, 0) : stop - offset]
            for block, offset in zip(self._blocks[first:last], self._starts[first:last])
        ]
        return pieces[0] if len(pieces) == 1 else np.concatenate(pieces)

    def item(self, index: int) -> int | float:
        return self.range(index, index + 1)[0].item()


class FlatLogprobs(MutableSequence[LogprobsOnePosition | None]):
    """Logprobs of a request stored as flat numpy columns.

    Compared to list[dict[int, Logprob]], this creates no Python object per
    position or entry: entries (position, candidate) are kept in a few
    append-only numpy columns, so storage is about 8 bytes per entry and the
    number of objects grows only with ``log(N)`` and ``N * k / BLOCK_BYTES``.

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
    ``decoded_tokens``) are read-only and return new lists.
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
        # Entries per position, -1 once positions differ; None while empty.
        self._width: int | None = None
        # End of each position's entries (position i spans [end(i - 1),
        # end(i))), kept once positions differ; until then end(i) is
        # (i + 1) * width.
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
        n, width = token_ids.shape
        if n == 0:
            return
        base = len(self._token_ids)
        self._add_positions(n, width, lambda: base + width * np.arange(1, n + 1))
        self._token_ids.append(token_ids.reshape(-1))
        self._logprobs.append(logprobs.reshape(-1))
        self._first_ranks.append(first_ranks)
        if self._ranks is not None and width:
            ranks = np.tile(np.arange(width, dtype=np.int64), (n, 1))
            ranks[:, 0] = first_ranks
            self._ranks.append(ranks.reshape(-1))
        if decoded_tokens is not None or self._decoded is not None:
            self._append_decoded(decoded_tokens, base, n * width)

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
        if not canonical and self._ranks is None:
            self._materialize_ranks()
        base = len(self._token_ids)
        self._add_positions(1, width, lambda: np.array([base + width]))
        if width:
            self._token_ids.append_list(token_ids)
            self._logprobs.append_list(logprobs)
        first_rank = ranks[0] if width and canonical else 0
        self._first_ranks.append(np.array([first_rank]))
        if self._ranks is not None and width:
            self._ranks.append(
                np.array([_NO_RANK if r is None else r for r in ranks], dtype=np.int64)
            )
        self._append_decoded(decoded_tokens, base, width)

    def _add_positions(
        self, n: int, width: int, ends: Callable[[], np.ndarray]
    ) -> None:
        """Counts ``n`` new positions, ``width`` entries each (-1: not all
        the same), ending at the entry offsets ``ends()``."""
        if self._ends is None:
            if width >= 0 and self._width in (None, width):
                self._width = width
                self._num_positions += n
                return
            ends_so_far = self._ends_range(0, self._num_positions)
            self._ends = _Column("<i8")
            self._ends.append(ends_so_far)
            self._width = -1
        self._ends.append(ends())
        self._num_positions += n

    def _ends_range(self, start: int, stop: int) -> np.ndarray:
        """End offsets of positions ``[start, stop)``."""
        if self._ends is not None:
            return self._ends.range(start, stop)
        return np.arange(start + 1, stop + 1, dtype=np.int64) * (self._width or 0)

    def _append_decoded(
        self, decoded_tokens: Sequence[str | None] | None, base: int, count: int
    ) -> None:
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

    def _materialize_ranks(self) -> None:
        ranks = _Column("<i8")
        ranks.append(self._entry_ranks(0, len(self)))
        self._ranks = ranks

    def _start(self, position: int) -> int:
        if self._ends is None:
            return position * (self._width or 0)
        return int(self._ends.item(position - 1)) if position else 0

    def _entry_ranks(self, start: int, stop: int) -> np.ndarray:
        """Ranks of the entries of positions ``[start, stop)``, int64 with
        ``_NO_RANK`` for None."""
        if start >= stop:
            return np.empty(0, dtype=np.int64)
        begin = self._start(start)
        ends = self._ends_range(start, stop)
        if self._ranks is not None:
            return self._ranks.range(begin, int(ends[-1]))
        starts = np.concatenate(([begin], ends[:-1]))
        position = np.repeat(np.arange(len(ends)), ends - starts)
        offsets = np.arange(begin, ends[-1]) - starts[position]
        first_ranks = self._first_ranks.range(start, stop)
        return np.where(offsets == 0, first_ranks[position], offsets)

    def _extend_from(self, source: "FlatLogprobs", start: int, stop: int) -> None:
        """Appends positions ``[start, stop)`` of ``source``."""
        if start >= stop:
            return
        begin = source._start(start)
        ends = source._ends_range(start, stop)
        end = int(ends[-1])
        if source._ranks is not None and self._ranks is None:
            self._materialize_ranks()
        base = len(self._token_ids)
        if source._ends is None:
            width = source._width or 0
        else:
            widths = np.unique(np.diff(ends, prepend=begin))
            width = int(widths[0]) if len(widths) == 1 else -1
        self._add_positions(stop - start, width, lambda: ends - begin + base)
        self._token_ids.append(source._token_ids.range(begin, end))
        self._logprobs.append(source._logprobs.range(begin, end))
        self._first_ranks.append(source._first_ranks.range(start, stop))
        if self._ranks is not None:
            self._ranks.append(source._entry_ranks(start, stop))
        self._append_decoded(
            None if source._decoded is None else source._decoded[begin:end],
            base,
            end - begin,
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
        if n == 0:
            return (
                np.empty((0, 0), dtype="<i4"),
                np.empty((0, 0), dtype="<f4"),
                np.empty(0, dtype="<i8"),
            )
        if (
            self._width is None
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
        ends = self._ends_range(start, stop).tolist()
        begins = [self._start(start), *ends[:-1]]
        return [
            int(self._token_ids.item(begin))
            for begin, end in zip(begins, ends)
            if begin < end
        ]

    @property
    def start_indices(self) -> list[int]:
        ends = self._ends_range(0, len(self))
        return [0, *ends[:-1].tolist()] if len(ends) else []

    @property
    def end_indices(self) -> list[int]:
        return self._ends_range(0, len(self)).tolist()

    @property
    def token_ids(self) -> list[int]:
        return self._token_ids.view().tolist()

    @property
    def logprobs(self) -> list[float]:
        return self._logprobs.view().tolist()

    @property
    def ranks(self) -> list[int | None]:
        return [
            None if rank == _NO_RANK else rank
            for rank in self._entry_ranks(0, len(self)).tolist()
        ]

    @property
    def decoded_tokens(self) -> list[str | None]:
        if self._decoded is None:
            return [None] * self.num_entries
        return list(self._decoded)

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
        begin, end = self._start(position), self._start(position + 1)
        if self._ranks is not None:
            ranks = self._ranks.range(begin, end).tolist()
        elif end > begin:
            ranks = [self._first_ranks.item(position), *range(1, end - begin)]
        else:
            ranks = []
        return self._position(
            self._token_ids.range(begin, end).tolist(),
            self._logprobs.range(begin, end).tolist(),
            ranks,
            begin,
        )

    def _position(
        self, token_ids: list[int], logprobs: list[float], ranks: list[int], begin: int
    ) -> LogprobsOnePosition:
        decoded: Iterable[str | None] = (
            itertools.repeat(None)
            if self._decoded is None
            else self._decoded[begin : begin + len(token_ids)]
        )
        return {
            token_id: Logprob(
                logprob=logprob,
                rank=None if rank == _NO_RANK else rank,
                decoded_token=decoded_token,
            )
            for token_id, logprob, rank, decoded_token in zip(
                token_ids, logprobs, ranks, decoded
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
        token_ids = self._token_ids.view()
        logprobs = self._logprobs.view()
        first_ranks = self._first_ranks.view().tolist()
        ranks = None if self._ranks is None else self._ranks.view()
        begin = 0
        for position, end in enumerate(self._ends_range(0, len(self)).tolist()):
            if ranks is not None:
                position_ranks = ranks[begin:end].tolist()
            elif end > begin:
                position_ranks = [first_ranks[position], *range(1, end - begin)]
            else:
                position_ranks = []
            yield self._position(
                token_ids[begin:end].tolist(),
                logprobs[begin:end].tolist(),
                position_ranks,
                begin,
            )
            begin = end

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FlatLogprobs):
            return NotImplemented
        return (
            np.array_equal(
                self._ends_range(0, len(self)), other._ends_range(0, len(other))
            )
            and np.array_equal(self._token_ids.view(), other._token_ids.view())
            and np.array_equal(self._logprobs.view(), other._logprobs.view())
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
