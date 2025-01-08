import multiprocessing
import os
import weakref
from collections.abc import Sequence
from typing import (Any, Callable, Dict, Generic, List, Optional, TypeVar,
                    Union, overload)

from vllm.logger import init_logger
from vllm.utils import get_mp_context, kill_process_tree

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


class BackgroundProcHandle:
    """
    Utility class to handle creation, readiness, and shutdown
    of background processes used by the AsyncLLM and LLMEngine.
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        process_name: str,
        target_fn: Callable,
        process_kwargs: Dict[Any, Any],
    ):
        context = get_mp_context()
        reader, writer = context.Pipe(duplex=False)

        assert ("ready_pipe" not in process_kwargs
                and "input_path" not in process_kwargs
                and "output_path" not in process_kwargs)
        process_kwargs["ready_pipe"] = writer
        process_kwargs["input_path"] = input_path
        process_kwargs["output_path"] = output_path

        # Run busy loop in background process.
        self.proc = context.Process(target=target_fn, kwargs=process_kwargs)
        self._finalizer = weakref.finalize(self, shutdown, self.proc,
                                           input_path, output_path)
        self.proc.start()

        # Wait for startup.
        if reader.recv()["status"] != "READY":
            raise RuntimeError(f"{process_name} initialization failed. "
                               "See root cause above.")

    def shutdown(self):
        self._finalizer()


# Note(rob): shutdown function cannot be a bound method,
# else the gc cannot collect the object.
def shutdown(proc: multiprocessing.Process, input_path: str, output_path: str):
    # Shutdown the process.
    if proc.is_alive():
        proc.terminate()
        proc.join(5)

        if proc.is_alive():
            kill_process_tree(proc.pid)

    # Remove zmq ipc socket files.
    ipc_sockets = [output_path, input_path]
    for ipc_socket in ipc_sockets:
        socket_file = ipc_socket.replace("ipc://", "")
        if os and os.path.exists(socket_file):
            os.remove(socket_file)
