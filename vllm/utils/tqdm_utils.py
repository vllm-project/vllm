# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Iterable, Sequence
from typing import TypeVar, overload

from tqdm.auto import tqdm

_T = TypeVar("_T", bound=Iterable)


@overload
def maybe_tqdm(
    it: Sequence[_T],
    *,
    use_tqdm: bool | Callable[..., tqdm],
    desc: str,
) -> Sequence[_T]: ...


@overload
def maybe_tqdm(
    it: Iterable[_T],
    *,
    use_tqdm: bool | Callable[..., tqdm],
    desc: str,
) -> Iterable[_T]: ...


def maybe_tqdm(
    it: Iterable[_T],
    *,
    use_tqdm: bool | Callable[..., tqdm],
    desc: str,
) -> Iterable[_T]:
    if not use_tqdm:
        return it

    tqdm_func = use_tqdm if callable(use_tqdm) else tqdm
    return tqdm_func(it, desc=desc)
