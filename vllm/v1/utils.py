from multiprocessing.process import BaseProcess
from multiprocessing.connection import Connection

from collections.abc import Sequence
from contextlib import contextmanager
from typing import (Any, Generic, Iterator, List, Optional, TypeVar, Union,
                    overload, Callable)

import zmq
import zmq.asyncio

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


def make_zmq_socket(ctx: Union[zmq.asyncio.Context, zmq.Context], path: str,
                    type: Any) -> Union[zmq.Socket, zmq.asyncio.Socket]:
    """Make a ZMQ socket with the proper bind/connext semantics."""

    import psutil
    mem = psutil.virtual_memory()

    socket = ctx.socket(type)

    total_mem = mem.total / 1024**3
    available_mem = mem.available / 1024**3
    if total_mem > 32 and available_mem > 16:
        buf_size = int(0.5 * 1024**3)
    else:
        buf_size = -1

    if type == zmq.PULL:
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)
        socket.connect(path)
    elif type == zmq.PUSH:
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, buf_size)
        socket.bind(path)
    else:
        raise ValueError(f"Unknown Socket Type: {type}")

    return socket


@contextmanager
def zmq_socket_ctx(
        path: str,
        type: Any) -> Iterator[zmq.Socket]:  # type: ignore[name-defined]
    """Context manager for a ZMQ socket"""

    ctx = zmq.Context(io_threads=2)  # type: ignore[attr-defined]
    try:
        yield make_zmq_socket(ctx, path, type)

    except KeyboardInterrupt:
        logger.debug("Worker had Keyboard Interrupt.")

    finally:
        ctx.destroy(linger=0)

from multiprocessing.process import BaseProcess
from vllm.utils import kill_process_tree
import os
import weakref
from dataclasses import dataclass

@dataclass
class BackgroundProcHandle:
    proc: BaseProcess
    input_path: str
    output_path: str

    def shutdown(self):
        # Shutdown the process if needed.
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(5)

            if self.proc.is_alive():
                kill_process_tree(self.proc.pid)

        # Remove zmq ipc socket files
        ipc_sockets = [self.output_path, self.input_path]
        for ipc_socket in ipc_sockets:
            socket_file = ipc_socket.replace("ipc://", "")
            if os and os.path.exists(socket_file):
                os.remove(socket_file)


class MPBackgroundProcess:
    
    def __init__(self, *args, fn: Callable, input_path: str, output_path: str, **kwargs):
        # Start EngineCore in background process.
        self.proc_handle: Optional[BackgroundProcHandle]
        self.proc_handle = fn(*args, input_path, output_path, kwargs)
        self._finalizer = weakref.finalize(self, self.shutdown)

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        if hasattr(self, "proc_handle") and self.proc_handle:
            self.proc_handle.shutdown()
            self.proc_handle = None