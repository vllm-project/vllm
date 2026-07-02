# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any


class SingleMethodAsyncRunner:
    def __init__(self):
        self._future: Future[Any] | None = None

    def start(self, target: Callable[..., Any], *args, **kwargs) -> Future[Any]:
        if self._future is not None:
            raise RuntimeError("Another Elastic EP async method is active")

        future: Future[Any] = Future()
        self._future = future
        threading.Thread(
            target=self._run,
            args=(target, future, args, kwargs),
            daemon=True,
            name="ElasticEPAsync",
        ).start()
        return future

    def clear(self) -> Any:
        future = self._future
        if future is None:
            raise RuntimeError("No Elastic EP async method is active")
        if not future.done():
            raise RuntimeError("Elastic EP async method is not done")
        self._future = None
        return future.result()

    def _run(
        self,
        target: Callable[..., Any],
        future: Future[Any],
        args: tuple,
        kwargs: dict,
    ) -> None:
        try:
            result = target(*args, **kwargs)
        except BaseException as e:
            future.set_exception(e)
            return

        future.set_result(result)
