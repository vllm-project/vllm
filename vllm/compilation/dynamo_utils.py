# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from types import TracebackType
from typing import Any

from torch._dynamo.utils import dynamo_timed as _torch_dynamo_timed


class _DynamoTimedCompat:
    def __call__(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        return _torch_dynamo_timed(fn)

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        return False


def dynamo_timed(label_or_fn: str | Callable[..., Any]) -> Any:
    if isinstance(label_or_fn, str):
        return _DynamoTimedCompat()
    try:
        return _torch_dynamo_timed(label_or_fn)
    except AttributeError:
        raise
