# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeVar, overload

from tqdm.auto import tqdm

_T = TypeVar("_T", bound=Iterable)


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
