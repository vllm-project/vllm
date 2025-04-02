# SPDX-License-Identifier: Apache-2.0
"""
Decorators used to prevent invoking into triton when HAS_TRITON is False.
Nothing should be done in all dummy decorator.
"""

from typing import Callable, Iterable, Optional, TypeVar

from vllm.logger import init_logger
from vllm.triton_utils.importing import HAS_TRITON

logger = init_logger(__name__)

if HAS_TRITON:
    import triton
else:
    logger.warning_once("Triton is not found in current env. Decorators like"
                        "@triton.jit will be replaced as dummy funcs. "
                        "`pip install triton` if you want to enable triton")

T = TypeVar("T")


def dummy_triton_jit(func):
    """dummy decorator for triton.jit"""

    def dummy_func(
        fn: Optional[T] = None,
        *,
        version=None,
        repr: Optional[Callable] = None,
        launch_metadata: Optional[Callable] = None,
        do_not_specialize: Optional[Iterable[int]] = None,
        debug: Optional[bool] = None,
        noinline: Optional[bool] = None,
    ):
        return func

    return dummy_func


def dummy_triton_autotune(func):
    """dummy decorator for triton.autotune"""

    def dummy_func(configs,
                   key,
                   prune_configs_by=None,
                   reset_to_zero=None,
                   restore_value=None,
                   pre_hook=None,
                   post_hook=None,
                   warmup=25,
                   rep=100,
                   use_cuda_graph=False):
        return func

    return dummy_func


def dummy_triton_heuristics(func):
    """dummy decorator for triton.heuristics"""

    def dummy_func(values):
        return func

    return dummy_func


triton_jit_decorator = \
    triton.jit if HAS_TRITON else dummy_triton_jit
triton_autotune_decorator = \
    triton.autotune if HAS_TRITON else dummy_triton_autotune
triton_heuristics_decorator = \
    triton.heuristics if HAS_TRITON else dummy_triton_heuristics
