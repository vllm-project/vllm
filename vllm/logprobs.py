# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from collections.abc import Iterable, Iterator, MutableSequence, Sequence
from dataclasses import dataclass, field
from typing import overload

import numpy as np

from vllm.logger import init_logger

logger = init_logger(__name__)


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


@dataclass
class FlatLogprobs(MutableSequence[LogprobsOnePosition | None]):
    """Flat logprobs of a request into multiple primitive type lists.

    Compared to list[dict[int, Logprob]], this data structure reduced GC
    overhead significantly. As it flattened logprob information for
    all positions and ranks in to multiple primitive type lists (i.e.
    logprobs, token_ids, ranks per token_ids, decoded_tokens).
    So regardless of the sequence length and top_logprobs setup,
    FlatLogprobs would only introduce a constant amount of objects.

    As each position might contains different amount of ranks,
    start_indices_per_position would be used to access the logprob ranges
    for different positions.

    NOTE: To reduce the migration overhead and improve backward compatibility,
    we support the key Sequence APIs of list, so it could act as
    list[LogprobsOnePosition]
    """

    # Start / end indices to indicate the range of logprobs for each position.
    start_indices: list[int] = field(default_factory=list)
    end_indices: list[int] = field(default_factory=list)

    # Flatten Logprob information for (each position, rank).
    # For position <i>, the logprobs are ranged
    # from self.start_indices[i] to self.end_indices[i] (exclusive).
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    ranks: list[int | None] = field(default_factory=list)
    decoded_tokens: list[str | None] = field(default_factory=list)

    def append(self, logprobs_one_position: LogprobsOnePosition | None) -> None:
        """Appends the container with logprobs for the next position."""
        self.start_indices.append(len(self.logprobs))
        if logprobs_one_position:
            for token_id, logprob in logprobs_one_position.items():
                self.token_ids.append(token_id)
                self.logprobs.append(logprob.logprob)
                self.ranks.append(logprob.rank)
                self.decoded_tokens.append(logprob.decoded_token)
        self.end_indices.append(len(self.logprobs))

    def append_fast(
        self,
        token_ids: list[int],
        logprobs: list[float],
        ranks: itertools.chain[int],
        decoded_tokens: Iterable[str | None],
    ) -> None:
        """Appends logprobs for the next position without creating
        the intermediate logprob dictionary.
        """
        self.start_indices.append(len(self.logprobs))
        for token_id, logprob, rank, decoded_token in zip(
            token_ids, logprobs, ranks, decoded_tokens
        ):
            self.token_ids.append(token_id)
            self.logprobs.append(logprob)
            self.ranks.append(rank)
            self.decoded_tokens.append(decoded_token)
        self.end_indices.append(len(self.logprobs))

    def extend(self, logprobs_multi_positions) -> None:
        """Extends the container with logprobs for the next multiple positions."""
        for logprobs_one_position in logprobs_multi_positions:
            self.append(logprobs_one_position)

    def __len__(self) -> int:
        """Gets number of positions stored in the container."""
        return len(self.start_indices)

    @overload
    def __getitem__(self, position: int) -> LogprobsOnePosition: ...

    @overload
    def __getitem__(self, s: slice, /) -> "FlatLogprobs": ...

    def __getitem__(self, index: int | slice):
        """Extracts logprobs of a given position or slice."""
        if isinstance(index, int):
            return {
                self.token_ids[i]: Logprob(
                    logprob=self.logprobs[i],
                    rank=self.ranks[i],
                    decoded_token=self.decoded_tokens[i],
                )
                for i in range(self.start_indices[index], self.end_indices[index])
            }
        elif isinstance(index, slice):
            selected_starts = self.start_indices[index]
            selected_ends = self.end_indices[index]
            # Empty slices have no source offset to normalize.
            if not selected_starts:
                return FlatLogprobs()
            min_index = selected_starts[0]
            max_index = selected_ends[-1]
            return FlatLogprobs(
                # Shift updated start_indices and end_indices to
                # be 0-indexed
                start_indices=[i - min_index for i in selected_starts],
                end_indices=[i - min_index for i in selected_ends],
                token_ids=self.token_ids[min_index:max_index],
                logprobs=self.logprobs[min_index:max_index],
                ranks=self.ranks[min_index:max_index],
                decoded_tokens=self.decoded_tokens[min_index:max_index],
            )
        else:
            raise TypeError(f"Invalid index type: {type(index)}")

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
        for i in range(0, len(self.start_indices)):
            yield self.__getitem__(i)


def _fits_int32(values: np.ndarray) -> bool:
    """Whether an integer array holds only values that int32 represents."""
    dtype = values.dtype
    if dtype.kind not in "iu":
        return False
    if dtype.itemsize < 4 or (dtype.kind == "i" and dtype.itemsize == 4):
        return True
    if values.size == 0:
        return True
    info = np.iinfo(np.int32)
    return info.min <= int(values.min()) and int(values.max()) <= info.max


class ArrayLogprobs(Sequence[LogprobsOnePosition]):
    """Sample logprobs of a request kept as the engine's rows.

    Row ``i`` is the engine row of position ``i``: slot 0 holds the sampled
    token, slots ``1..k`` the top-k candidates in engine order. Rows are
    copied into little-endian int32 / float32 / int32 blocks whose capacity
    grows geometrically up to ``BLOCK_BYTES``, so ``N`` positions create
    ``O(log N + N * row_bytes / BLOCK_BYTES)`` objects instead of
    ``O(N * k)``. Candidate tokens are never detokenized and logprobs keep the
    raw engine values.

    Rows of another width than the first (e.g. when a co-batched request's
    ``logprob_token_ids`` replaced the batch's logprob tensors) are kept from
    then on as ``dict[int, Logprob]`` entries, and :attr:`is_regular` becomes
    False. Any storage failure marks the container :attr:`broken` instead of
    raising in the shared output processing loop.

    Positions read as ``dict[int, Logprob]`` (``decoded_token`` None), like
    the list representation, so any consumer works, slowly; the generate
    endpoint renders from :meth:`arrays` instead. Only used for
    ``FINAL_ONLY`` outputs: there is no ``extend`` for DELTA aggregation.
    """

    BLOCK_BYTES = 8 << 20

    def __init__(self) -> None:
        # Only the first ``_tail_fill`` rows of the last block are used.
        self._token_ids: list[np.ndarray] = []
        self._logprobs: list[np.ndarray] = []
        self._ranks: list[np.ndarray] = []
        self._tail_fill = 0
        self._num_rows = 0
        # Positions after the array rows, once an irregular row was seen.
        self._irregular: list[LogprobsOnePosition] | None = None
        self.broken = False

    @property
    def is_regular(self) -> bool:
        """Whether every position is stored as an array row."""
        return self._irregular is None and not self.broken

    @property
    def num_slots(self) -> int | None:
        """Slots per array row, or None if no row was stored."""
        return self._token_ids[0].shape[1] if self._token_ids else None

    def append_rows(
        self, token_ids: np.ndarray, logprobs: np.ndarray, ranks: np.ndarray
    ) -> None:
        """Append ``n`` positions given as ``[n, S]``, ``[n, S]`` and ``[n]``
        engine arrays (copied). Never raises: a failure marks the container
        broken, failing only its request when it is rendered."""
        if self.broken:
            return
        try:
            self._append_rows(token_ids, logprobs, ranks)
        except Exception:
            logger.exception("Storing sample logprobs failed; failing the request")
            self.broken = True
            self._token_ids, self._logprobs, self._ranks = [], [], []
            self._irregular = None

    def _append_rows(
        self, token_ids: np.ndarray, logprobs: np.ndarray, ranks: np.ndarray
    ) -> None:
        n = len(ranks)
        if (
            token_ids.ndim != 2
            or logprobs.shape != token_ids.shape
            or ranks.shape != (n,)
            or token_ids.shape[0] != n
        ):
            raise ValueError(
                f"Inconsistent logprob rows: token_ids {token_ids.shape}, "
                f"logprobs {logprobs.shape}, ranks {ranks.shape}"
            )
        if not (
            logprobs.dtype.kind == "f"
            and logprobs.dtype.itemsize <= 4
            and _fits_int32(token_ids)
            and _fits_int32(ranks)
        ):
            raise TypeError(
                f"Unsupported logprob rows: token_ids {token_ids.dtype}, "
                f"logprobs {logprobs.dtype}, ranks {ranks.dtype}"
            )
        if n == 0:
            return
        width = token_ids.shape[1]
        if self._irregular is not None or width != (self.num_slots or width):
            if self._irregular is None:
                self._irregular = []
            self._irregular.extend(
                _row_dict(token_ids[i], logprobs[i], ranks[i]) for i in range(n)
            )
            return
        pos = 0
        while pos < n:
            if not self._ranks or self._tail_fill == len(self._ranks[-1]):
                self._new_block(n - pos, width)
            fill = self._tail_fill
            take = min(n - pos, len(self._ranks[-1]) - fill)
            self._token_ids[-1][fill : fill + take] = token_ids[pos : pos + take]
            self._logprobs[-1][fill : fill + take] = logprobs[pos : pos + take]
            self._ranks[-1][fill : fill + take] = ranks[pos : pos + take]
            self._tail_fill = fill + take
            pos += take
        self._num_rows += n

    def _new_block(self, remaining: int, width: int) -> None:
        max_rows = max(1, self.BLOCK_BYTES // (width * 8 + 4))
        previous = len(self._ranks[-1]) if self._ranks else 0
        rows = max(remaining, min(max_rows, max(1, 2 * previous)))
        self._token_ids.append(np.empty((rows, width), dtype="<i4"))
        self._logprobs.append(np.empty((rows, width), dtype="<f4"))
        self._ranks.append(np.empty((rows,), dtype="<i4"))
        self._tail_fill = 0

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The rows as contiguous ``(token_ids[N, S], logprobs[N, S],
        ranks[N])``, little-endian int32 / float32 / int32.

        Blocks are concatenated once and kept as one block afterwards.
        Raises ValueError unless :attr:`is_regular`.
        """
        if not self.is_regular:
            raise ValueError("Sample logprobs are not stored as regular rows")
        if not self._token_ids:
            return (
                np.empty((0, 0), dtype="<i4"),
                np.empty((0, 0), dtype="<f4"),
                np.empty((0,), dtype="<i4"),
            )
        if len(self._ranks) > 1 or self._tail_fill != len(self._ranks[0]):
            fill = self._tail_fill
            self._token_ids[-1] = self._token_ids[-1][:fill]
            self._logprobs[-1] = self._logprobs[-1][:fill]
            self._ranks[-1] = self._ranks[-1][:fill]
            self._token_ids = [np.concatenate(self._token_ids)]
            self._logprobs = [np.concatenate(self._logprobs)]
            self._ranks = [np.concatenate(self._ranks)]
            self._tail_fill = self._num_rows
        return self._token_ids[0], self._logprobs[0], self._ranks[0]

    def __len__(self) -> int:
        """Gets number of positions stored in the container."""
        if self._irregular is None:
            return self._num_rows
        return self._num_rows + len(self._irregular)

    @overload
    def __getitem__(self, position: int) -> LogprobsOnePosition: ...

    @overload
    def __getitem__(self, s: slice, /) -> list[LogprobsOnePosition]: ...

    def __getitem__(self, index: int | slice):
        """Extracts logprobs of a given position or slice."""
        if self.broken:
            raise ValueError("Sample logprobs are unavailable: storing them failed")
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        position = range(len(self))[index]
        if position >= self._num_rows:
            assert self._irregular is not None
            return self._irregular[position - self._num_rows]
        for t, lp, r in zip(self._token_ids, self._logprobs, self._ranks):
            if position < len(r):
                return _row_dict(t[position], lp[position], r[position])
            position -= len(r)
        raise AssertionError("unreachable")

    def __iter__(self) -> Iterator[LogprobsOnePosition]:
        """Iterates the positions in order."""
        if self.broken:
            raise ValueError("Sample logprobs are unavailable: storing them failed")
        remaining = self._num_rows
        for t, lp, r in zip(self._token_ids, self._logprobs, self._ranks):
            for j in range(min(len(r), remaining)):
                yield _row_dict(t[j], lp[j], r[j])
            remaining -= len(r)
        if self._irregular is not None:
            yield from self._irregular


def _row_dict(
    token_ids: np.ndarray, logprobs: np.ndarray, rank: np.integer
) -> LogprobsOnePosition:
    """An engine row as ``append_logprobs_for_next_position`` stores it."""
    ids = token_ids.tolist()
    ranks = itertools.chain((int(rank),), range(1, len(ids)))
    return {
        token_id: Logprob(logprob=value, rank=r)
        for token_id, value, r in zip(ids, logprobs.tolist(), ranks)
    }


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


def create_sample_logprobs(
    flat_logprobs: bool, array_logprobs: bool = False
) -> SampleLogprobs | ArrayLogprobs:
    """Creates a container to store decode logprobs for a request."""
    if array_logprobs:
        return ArrayLogprobs()
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
