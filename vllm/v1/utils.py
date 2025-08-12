from contextlib import contextmanager
from typing import Any, Generic, Iterator, List, TypeVar, overload

import zmq

from vllm.logger import init_logger

logger = init_logger(__name__)

T = TypeVar("T")


class ConstantList(Generic[T]):

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

    def index(self, item):
        return self._x.index(item)

    @overload
    def __getitem__(self, item) -> T:
        ...

    @overload
    def __getitem__(self, s: slice, /) -> List[T]:
        ...

    def __getitem__(self, item):
        return self._x[item]

    @overload
    def __setitem__(self, item, value):
        ...

    @overload
    def __setitem__(self, s: slice, value, /):
        ...

    def __setitem__(self, item, value):
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
def make_zmq_socket(path: str, type: Any) -> Iterator[zmq.Socket]:
    """Context manager for a ZMQ socket"""

    ctx = zmq.Context()
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
