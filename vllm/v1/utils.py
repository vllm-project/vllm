from collections import OrderedDict
from collections.abc import Sequence
from contextlib import contextmanager
from typing import (Any, Generic, Iterator, List, Optional, TypeVar, Union,
                    overload)

import zmq

from vllm.logger import init_logger

logger = init_logger(__name__)

T = TypeVar("T")


class ConstantList(Generic[T], Sequence):

    def __init__(self, x: List[T]) -> None:
        self._x = x

    def append(self, item):
        raise Exception("Cannot append to a constant list")

    def extend(self, item):
        raise Exception("Cannot extend a constant list")

    def insert(self, item):
        raise Exception("Cannot insert into a constant list")

    def pop(self, item):
        raise Exception("Cannot pop from a constant list")

    def remove(self, item):
        raise Exception("Cannot remove from a constant list")

    def clear(self):
        raise Exception("Cannot clear a constant list")

    def index(self,
              item: T,
              start: int = 0,
              stop: Optional[int] = None) -> int:
        return self._x.index(item, start,
                             stop if stop is not None else len(self._x))

    @overload
    def __getitem__(self, item: int) -> T:
        ...

    @overload
    def __getitem__(self, s: slice, /) -> List[T]:
        ...

    def __getitem__(self, item: Union[int, slice]) -> Union[T, List[T]]:
        return self._x[item]

    @overload
    def __setitem__(self, item: int, value: T):
        ...

    @overload
    def __setitem__(self, s: slice, value: T, /):
        ...

    def __setitem__(self, item: Union[int, slice], value: Union[T, List[T]]):
        raise Exception("Cannot set item in a constant list")

    def __delitem__(self, item):
        raise Exception("Cannot delete item from a constant list")

    def __iter__(self):
        return iter(self._x)

    def __contains__(self, item):
        return item in self._x

    def __len__(self):
        return len(self._x)


@contextmanager
def make_zmq_socket(
        path: str,
        type: Any) -> Iterator[zmq.Socket]:  # type: ignore[name-defined]
    """Context manager for a ZMQ socket"""

    ctx = zmq.Context()  # type: ignore[attr-defined]
    try:
        socket = ctx.socket(type)

        if type == zmq.constants.PULL:
            socket.connect(path)
        elif type == zmq.constants.PUSH:
            socket.bind(path)
        else:
            raise ValueError(f"Unknown Socket Type: {type}")

        yield socket

    except KeyboardInterrupt:
        logger.debug("Worker had Keyboard Interrupt.")

    finally:
        ctx.destroy(linger=0)


K = TypeVar('K')
V = TypeVar('V')


class LRUDictCache(Generic[K, V]):

    def __init__(self, size: int):
        self.cache: OrderedDict[K, V] = OrderedDict()
        self.size = size

    def get(self, key: K, default=None) -> V:
        if key not in self.cache:
            return default

        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: K, value: V):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.size:
            self.cache.popitem(last=False)
