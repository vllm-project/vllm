import asyncio
from typing import Any, AsyncGenerator, Callable, Optional, Type, Union

from vllm.outputs import PoolingRequestOutput, RequestOutput


class AsyncStream:
    """A stream of RequestOutputs or PoolingRequestOutputs for a request
    that can be iterated over asynchronously via an async generator."""

    STOP_ITERATION = Exception()  # Sentinel

    def __init__(self, request_id: str, cancel: Callable[[str], None]) -> None:
        self.request_id = request_id
        self._cancel = cancel
        self._queue: asyncio.Queue = asyncio.Queue()
        self._finished = False

    def put(self, item: Union[RequestOutput, PoolingRequestOutput,
                              Exception]) -> None:
        if not self._finished:
            self._queue.put_nowait(item)

    def finish(
        self,
        exception: Optional[Union[BaseException, Type[BaseException]]] = None,
    ) -> None:
        if not self._finished:
            self._finished = True
            self._queue.put_nowait(exception if self._is_raisable(exception)
                                   else AsyncStream.STOP_ITERATION)

    async def generator(
        self
    ) -> AsyncGenerator[Union[RequestOutput, PoolingRequestOutput], None]:
        finished = False
        try:
            while True:
                result = await self._queue.get()
                if self._is_raisable(result):
                    finished = True
                    if result == AsyncStream.STOP_ITERATION:
                        return
                    raise result
                yield result
        finally:
            self._finished = True
            if not finished:
                self._cancel(self.request_id)

    @staticmethod
    def _is_raisable(value: Any):
        return isinstance(value, BaseException) or \
                (isinstance(value, type) and \
                 issubclass(value, BaseException))
