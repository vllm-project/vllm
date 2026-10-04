# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeVar, overload

from tqdm.auto import tqdm

_T = TypeVar("_T", bound=Iterable)

# One self-contained log line per refresh instead of a redrawn bar, so the
# output stays readable in non-TTY sinks (kubectl logs, structured loggers).
_NON_TTY_BAR_FORMAT = (
    "{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} "
    "[{elapsed}<{remaining}, {rate_fmt}]\n"
)
_NON_TTY_MININTERVAL = 5.0


def vllm_tqdm(*args: Any, **kwargs: Any) -> tqdm:
    """Return a tqdm bar that emits one clean log line per refresh on non-TTYs.

    In a terminal this behaves exactly like tqdm. On non-TTYs (containers,
    kubectl logs) the carriage-return animation would produce a single
    ever-growing line, so the bar is replaced with a newline-terminated
    format refreshed at a bounded interval.
    """
    fp = kwargs.get("file") or sys.stderr
    if not (hasattr(fp, "isatty") and fp.isatty()):
        kwargs.setdefault("bar_format", _NON_TTY_BAR_FORMAT)
        kwargs.setdefault("mininterval", _NON_TTY_MININTERVAL)
        kwargs.pop("dynamic_ncols", None)
    return tqdm(*args, **kwargs)


@overload
def maybe_tqdm(
    it: Sequence[_T],
    *,
    use_tqdm: bool | Callable[..., tqdm],
    **tqdm_kwargs: Any,
) -> Sequence[_T]: ...


@overload
def maybe_tqdm(
    it: Iterable[_T],
    *,
    use_tqdm: bool | Callable[..., tqdm],
    **tqdm_kwargs: Any,
) -> Iterable[_T]: ...


def maybe_tqdm(
    it: Iterable[_T],
    *,
    use_tqdm: bool | Callable[..., tqdm],
    **tqdm_kwargs: Any,
) -> Iterable[_T]:
    if not use_tqdm:
        return it

    tqdm_func = use_tqdm if callable(use_tqdm) else tqdm
    return tqdm_func(it, **tqdm_kwargs)
