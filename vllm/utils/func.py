# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Contains helpers that are applied to functions.

This is similar in concept to the `functools` module.
"""

import inspect
import threading
import warnings
from collections.abc import Callable, Mapping
from functools import lru_cache, partial, wraps
from typing import Any, TypeVar

from typing_extensions import ParamSpec

from vllm.logger import init_logger

logger = init_logger(__name__)


P = ParamSpec("P")
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


def identity(value: T, **kwargs) -> T:
    """Returns the first provided value."""
    return value


def run_once(f: Callable[P, None]) -> Callable[P, None]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
        if wrapper.has_run:  # type: ignore[attr-defined]
            return

        with wrapper.lock:  # type: ignore[attr-defined]
            if not wrapper.has_run:  # type: ignore[attr-defined]
                wrapper.has_run = True  # type: ignore[attr-defined]
                return f(*args, **kwargs)

    wrapper.has_run = False  # type: ignore[attr-defined]
    wrapper.lock = threading.Lock()  # type: ignore[attr-defined]
    return wrapper


def deprecate_args(
    start_index: int,
    is_deprecated: bool | Callable[[], bool] = True,
    additional_message: str | None = None,
) -> Callable[[F], F]:
    if not callable(is_deprecated):
        is_deprecated = partial(identity, is_deprecated)

    def wrapper(fn: F) -> F:
        params = inspect.signature(fn).parameters
        pos_types = (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        pos_kws = [kw for kw, param in params.items() if param.kind in pos_types]

        @wraps(fn)
        def inner(*args, **kwargs):
            if is_deprecated():
                deprecated_args = pos_kws[start_index : len(args)]
                if deprecated_args:
                    msg = (
                        f"The positional arguments {deprecated_args} are "
                        "deprecated and will be removed in a future update."
                    )
                    if additional_message is not None:
                        msg += f" {additional_message}"

                    warnings.warn(
                        DeprecationWarning(msg),
                        stacklevel=3,  # The inner function takes up one level
                    )

            return fn(*args, **kwargs)

        return inner  # type: ignore

    return wrapper


def deprecate_kwargs(
    *kws: str,
    is_deprecated: bool | Callable[[], bool] = True,
    additional_message: str | None = None,
) -> Callable[[F], F]:
    deprecated_kws = set(kws)

    if not callable(is_deprecated):
        is_deprecated = partial(identity, is_deprecated)

    def wrapper(fn: F) -> F:
        @wraps(fn)
        def inner(*args, **kwargs):
            if is_deprecated():
                deprecated_kwargs = kwargs.keys() & deprecated_kws
                if deprecated_kwargs:
                    msg = (
                        f"The keyword arguments {deprecated_kwargs} are "
                        "deprecated and will be removed in a future update."
                    )
                    if additional_message is not None:
                        msg += f" {additional_message}"

                    warnings.warn(
                        DeprecationWarning(msg),
                        stacklevel=3,  # The inner function takes up one level
                    )

            return fn(*args, **kwargs)

        return inner  # type: ignore

    return wrapper


@lru_cache
def supports_kw(
    callable: Callable[..., object],
    kw_name: str,
    *,
    requires_kw_only: bool = False,
    allow_var_kwargs: bool = True,
) -> bool:
    """Check if a keyword is a valid kwarg for a callable; if requires_kw_only
    disallows kwargs names that can also be positional arguments.
    """
    params = inspect.signature(callable).parameters
    if not params:
        return False

    param_val = params.get(kw_name)

    # Types where the it may be valid, i.e., explicitly defined & nonvariadic
    passable_kw_types = set(
        (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    )

    if param_val:
        is_sig_param = param_val.kind in passable_kw_types
        # We want kwargs only, but this is passable as a positional arg
        if (
            requires_kw_only
            and is_sig_param
            and param_val.kind != inspect.Parameter.KEYWORD_ONLY
        ):
            return False
        if (requires_kw_only and param_val.kind == inspect.Parameter.KEYWORD_ONLY) or (
            not requires_kw_only and is_sig_param
        ):
            return True

    # If we're okay with var-kwargs, it's supported as long as
    # the kw_name isn't something like *args, **kwargs
    if allow_var_kwargs:
        # Get the last param; type is ignored here because params is a proxy
        # mapping, but it wraps an ordered dict, and they appear in order.
        # Ref: https://docs.python.org/3/library/inspect.html#inspect.Signature.parameters
        last_param = params[next(reversed(params))]  # type: ignore
        return (
            last_param.kind == inspect.Parameter.VAR_KEYWORD
            and last_param.name != kw_name
        )

    return False


def get_allowed_kwarg_only_overrides(
    callable: Callable[..., object],
    overrides: Mapping[str, object] | None,
    *,
    requires_kw_only: bool = True,
    allow_var_kwargs: bool = False,
) -> dict[str, Any]:
    """
    Given a callable which has one or more keyword only params and a dict
    mapping param names to values, drop values that can be not be kwarg
    expanded to overwrite one or more keyword-only args. This is used in a
    few places to handle custom processor overrides for multimodal models,
    e.g., for profiling when processor options provided by the user
    may affect the number of mm tokens per instance.

    Args:
        callable: Callable which takes 0 or more keyword only arguments.
                  If None is provided, all overrides names are allowed.
        overrides: Potential overrides to be used when invoking the callable.
        allow_var_kwargs: Allows overrides that are expandable for var kwargs.

    Returns:
        Dictionary containing the kwargs to be leveraged which may be used
        to overwrite one or more keyword only arguments when invoking the
        callable.
    """
    if not overrides:
        return {}

    # Drop any mm_processor_kwargs provided by the user that
    # are not kwargs, unless it can fit it var_kwargs param
    filtered_overrides = {
        kwarg_name: val
        for kwarg_name, val in overrides.items()
        if supports_kw(
            callable,
            kwarg_name,
            requires_kw_only=requires_kw_only,
            allow_var_kwargs=allow_var_kwargs,
        )
    }

    # If anything is dropped, log a warning
    dropped_keys = overrides.keys() - filtered_overrides.keys()
    if dropped_keys:
        if requires_kw_only:
            logger.warning(
                "The following intended overrides are not keyword-only args "
                "and will be dropped: %s",
                dropped_keys,
            )
        else:
            logger.warning(
                "The following intended overrides are not keyword args "
                "and will be dropped: %s",
                dropped_keys,
            )

    return filtered_overrides
