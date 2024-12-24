import os
import weakref
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from typing import (Any, Callable, Dict, Generic, Iterator, List, Optional,
                    TypeVar, Union, overload)

import zmq
import zmq.asyncio

from vllm.executor.multiproc_worker_utils import get_mp_context
from vllm.logger import init_logger
from vllm.utils import kill_process_tree

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


def make_zmq_socket(
    ctx: Union[zmq.asyncio.Context, zmq.Context],  # type: ignore[name-defined]
    path: str,
    type: Any,
) -> Union[zmq.Socket, zmq.asyncio.Socket]:  # type: ignore[name-defined]
    """Make a ZMQ socket with the proper bind/connect semantics."""

    import psutil
    mem = psutil.virtual_memory()

    socket = ctx.socket(type)

    total_mem = mem.total / 1024**3
    available_mem = mem.available / 1024**3
    if total_mem > 32 and available_mem > 16:
        buf_size = int(0.5 * 1024**3)
    else:
        buf_size = -1

    if type == zmq.constants.PULL:
        socket.setsockopt(zmq.constants.RCVHWM, 0)
        socket.setsockopt(zmq.constants.RCVBUF, buf_size)
        socket.bind(path)
    elif type == zmq.constants.PUSH:
        socket.setsockopt(zmq.constants.SNDHWM, 0)
        socket.setsockopt(zmq.constants.SNDBUF, buf_size)
        socket.connect(path)
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
        logger.debug("Got Keyboard Interrupt.")

    finally:
        ctx.destroy(linger=0)


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

    def __init__(self):
        self.proc_handle: Optional[BackgroundProcHandle]
        self._finalizer = weakref.finalize(self, self.shutdown)

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        if hasattr(self, "proc_handle") and self.proc_handle:
            self.proc_handle.shutdown()
            self.proc_handle = None

    @staticmethod
    def wait_for_startup(
        input_path: str,
        output_path: str,
        process_name: str,
        target_fn: Callable,
        process_kwargs: Dict[Any, Any],
    ) -> BackgroundProcHandle:
        context = get_mp_context()
        reader, writer = context.Pipe(duplex=False)

        assert ("ready_pipe" not in process_kwargs
                and "input_path" not in process_kwargs
                and "output_path" not in process_kwargs)
        process_kwargs["ready_pipe"] = writer
        process_kwargs["input_path"] = input_path
        process_kwargs["output_path"] = output_path

        # Run Detokenizer busy loop in background process.
        proc = context.Process(target=target_fn, kwargs=process_kwargs)
        proc.start()

        # Wait for startup.
        if reader.recv()["status"] != "READY":
            raise RuntimeError(f"{process_name} initialization failed. "
                               "See root cause above.")

        return BackgroundProcHandle(proc, input_path, output_path)
