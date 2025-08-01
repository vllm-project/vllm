# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import traceback
from typing import Callable
from unittest.mock import patch


def find_cuda_init(fn: Callable[[], object]) -> None:
    """
    Helper function to debug CUDA re-initialization errors.

    If `fn` initializes CUDA, prints the stack trace of how this happens.
    """
    from torch.cuda import _lazy_init

    stack = None

    def wrapper():
        nonlocal stack
        stack = traceback.extract_stack()
        return _lazy_init()

    with patch("torch.cuda._lazy_init", wrapper):
        fn()

    if stack is not None:
        print("==== CUDA Initialized ====")
        print("".join(traceback.format_list(stack)).strip())
        print("==========================")


if __name__ == "__main__":
    find_cuda_init(
        lambda: importlib.import_module("vllm.model_executor.models.llava"))
