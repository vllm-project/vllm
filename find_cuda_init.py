import importlib
import traceback
from functools import wraps
from typing import Callable, TypeVar
from unittest.mock import patch

from typing_extensions import ParamSpec


_P, _R_co = ParamSpec("_P"), TypeVar("_R_co", covariant=True)


def print_stack(f: Callable[_P, _R_co]) -> Callable[_P, _R_co]:
    @wraps(f)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs):
        traceback.print_stack()
        return f(*args, **kwargs)

    return wrapper


def find_cuda_init(fn: Callable[[], object]) -> None:
    """
    Helper function to debug CUDA re-initialization errors.

    If `fn` initializes CUDA, prints the stack trace of how this happens.
    """
    from torch.cuda import _lazy_init

    with patch("torch.cuda._lazy_init", print_stack(_lazy_init)):
        fn()


if __name__ == "__main__":
    find_cuda_init(lambda: importlib.import_module("vllm.model_executor.models.llava"))  # noqa: E501
