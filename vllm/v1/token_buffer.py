# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from array import array
from collections.abc import Collection, Iterable, Iterator, Sequence
from pickle import PickleBuffer
from typing import TYPE_CHECKING, overload

import numpy as np

from vllm.utils.collection_utils import as_list

if TYPE_CHECKING:
    # Type-checked as the list it reads like.
    class TokenBuffer(list[int]):
        def __init__(self, tokens: Iterable[int] = ()) -> None: ...

else:

    class TokenBuffer(array):
        """Mutable int32 token ids with no per-token Python objects.

        A 170K-token prompt held as a list makes every cyclic-GC pass walk 170K
        references per live request; this buffer has none. Reads stay list-like:
        indexing returns ``int``, slices and ``copy()`` return lists, and it compares
        equal to a list or tuple holding the same tokens.
        """

        __slots__ = ()

        def __new__(cls, tokens: Iterable[int] = ()):
            if isinstance(tokens, (bytes, bytearray, memoryview)):
                # Packed int32 ids. array("i", memoryview) would instead take
                # each byte as a token.
                buf = super().__new__(cls, "i")
                buf.frombytes(tokens)
                return buf
            # From another array of the same type this is a memcpy.
            return super().__new__(cls, "i", tokens)

        def __getitem__(self, index):
            item = array.__getitem__(self, index)
            return item.tolist() if isinstance(index, slice) else item

        def __eq__(self, other):
            if isinstance(other, (list, tuple)):
                return len(self) == len(other) and self.tolist() == as_list(other)
            return array.__eq__(self, other)

        def __ne__(self, other):
            eq = self.__eq__(other)
            return eq if eq is NotImplemented else not eq

        __hash__ = None  # type: ignore[assignment]

        def __add__(self, other):
            return self.tolist() + as_list(other)

        def __radd__(self, other):
            return as_list(other) + self.tolist()

        def __iadd__(self, other):
            self.extend(other)
            return self

        def copy(self) -> list[int]:
            return self.tolist()

        def __reduce_ex__(self, protocol):
            # A PickleBuffer lets MessageQueue send large ids out of band. It wraps
            # a copy since zmq may hold it past enqueue, blocking resizes of self.
            data = self.tobytes()
            return _rebuild_token_buffer, (
                PickleBuffer(data) if protocol >= 5 else data,
            )

    def _rebuild_token_buffer(data) -> TokenBuffer:
        # data is whatever object pickle was given for the buffer, e.g. bytes,
        # a bytearray or a zmq.Frame, so copy out of it rather than keep it.
        tokens = TokenBuffer()
        tokens.frombytes(data)
        return tokens


def extend_from_slice(
    dst: list[int], src: Sequence[int], start: int, stop: int
) -> None:
    """``dst.extend(src[start:stop])``, copied without Python ints when both are
    TokenBuffers."""
    if isinstance(dst, array) and isinstance(src, array):
        dst.extend(array.__getitem__(src, slice(start, stop)))
    else:
        dst.extend(src[start:stop])


def rfind_tokens(
    token_ids: Sequence[int],
    targets: Collection[int | None],
    start: int = 0,
    stop: int | None = None,
) -> int:
    """Return the index of the last token in ``token_ids[start:stop]`` that is
    one of ``targets`` (``None`` entries, for absent tokens, are ignored), or -1.

    An int32 buffer (a TokenBuffer) is searched with numpy, in chunks growing
    back from ``stop`` so a recent match is found quickly, without creating a
    Python int per token. Other sequences are scanned backwards in Python.
    """
    if stop is None:
        stop = len(token_ids)
    if isinstance(token_ids, TokenIdsView):
        return token_ids.rfind(targets, start, stop)
    if isinstance(token_ids, array) and token_ids.typecode == "i":
        ids = np.frombuffer(token_ids, dtype=np.int32)
        wanted = np.array([t for t in targets if t is not None], dtype=np.int32)
        chunk = 256
        while stop > start:
            lo = max(start, stop - chunk)
            hits = np.flatnonzero(np.isin(ids[lo:stop], wanted))
            if hits.size:
                return lo + int(hits[-1])
            stop, chunk = lo, chunk * 8
        return -1
    for i in range(stop - 1, start - 1, -1):
        if token_ids[i] in targets:
            return i
    return -1


class TokenIdsView(Sequence[int]):
    """``tokens`` followed by ``suffix``, shortened from the end by pop().

    Copied into a list only when read element-wise, so a caller that ignores
    it, or searches it with rfind_tokens, doesn't copy ``tokens``.
    """

    def __init__(self, tokens: Sequence[int], suffix: Iterable[int] = ()) -> None:
        self._tokens = tokens
        self._suffix = list(suffix)
        self._len = len(tokens) + len(self._suffix)
        self._list: list[int] | None = None

    def __len__(self) -> int:
        return self._len

    def _as_list(self) -> list[int]:
        if self._list is None:
            num_tokens = min(len(self._tokens), self._len)
            self._list = list(self._tokens[:num_tokens])
            self._list.extend(self._suffix[: self._len - num_tokens])
        return self._list

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> list[int]: ...

    def __getitem__(self, index: int | slice) -> int | list[int]:
        return self._as_list()[index]

    def __iter__(self) -> Iterator[int]:
        return iter(self._as_list())

    def __contains__(self, value: object) -> bool:
        return value in self._as_list()

    def pop(self) -> None:
        if self._list is not None:
            self._list.pop()
        self._len -= 1

    def rfind(self, targets: Collection[int | None], start: int, stop: int) -> int:
        if self._list is not None:
            return rfind_tokens(self._list, targets, start, stop)
        num_tokens = len(self._tokens)
        stop = min(stop, self._len)
        for i in range(stop - 1, max(start, num_tokens) - 1, -1):
            if self._suffix[i - num_tokens] in targets:
                return i
        return rfind_tokens(self._tokens, targets, start, min(stop, num_tokens))
