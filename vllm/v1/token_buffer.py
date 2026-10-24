# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from array import array
from collections.abc import Iterable
from typing import TYPE_CHECKING

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
            # From another array of the same type this is a memcpy.
            return super().__new__(cls, "i", tokens)

        def __getitem__(self, index):
            item = array.__getitem__(self, index)
            return item.tolist() if isinstance(index, slice) else item

        def __eq__(self, other):
            if isinstance(other, (list, tuple)):
                return len(self) == len(other) and self.tolist() == list(other)
            return array.__eq__(self, other)

        def __ne__(self, other):
            eq = self.__eq__(other)
            return eq if eq is NotImplemented else not eq

        __hash__ = None  # type: ignore[assignment]

        def __add__(self, other):
            return self.tolist() + list(other)

        def __radd__(self, other):
            return list(other) + self.tolist()

        def __iadd__(self, other):
            self.extend(other)
            return self

        def copy(self) -> list[int]:
            return self.tolist()
