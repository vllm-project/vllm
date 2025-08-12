import functools
import gc
from typing import Callable, TypeVar

import pytest
import torch
from typing_extensions import ParamSpec

from vllm.distributed import cleanup_dist_env_and_memory
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig


@pytest.fixture(autouse=True)
def cleanup():
    cleanup_dist_env_and_memory(shutdown_ray=True)


_P = ParamSpec("_P")
_R = TypeVar("_R")


def retry_until_skip(n: int):

    def decorator_retry(func: Callable[_P, _R]) -> Callable[_P, _R]:

        @functools.wraps(func)
        def wrapper_retry(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            for i in range(n):
                try:
                    return func(*args, **kwargs)
                except AssertionError:
                    gc.collect()
                    torch.cuda.empty_cache()
                    if i == n - 1:
                        pytest.skip(f"Skipping test after {n} attempts.")

            raise AssertionError("Code should not be reached")

        return wrapper_retry

    return decorator_retry


@pytest.fixture(autouse=True)
def tensorizer_config():
    config = TensorizerConfig(tensorizer_uri="vllm")
    return config
