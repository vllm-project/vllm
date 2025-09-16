# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
These envs only work for a small part of the tests, fix what you need!
"""

import os
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    VLLM_CI_NO_SKIP: bool = False
    VLLM_CI_DTYPE: Optional[str] = None
    VLLM_CI_HEAD_DTYPE: Optional[str] = None
    VLLM_CI_HF_DTYPE: Optional[str] = None

environment_variables: dict[str, Callable[[], Any]] = {
    # A model family has many models with the same architecture.
    # By default, a model family tests only one model.
    # Through this flag, all models can be tested.
    "VLLM_CI_NO_SKIP": lambda: bool(int(os.getenv("VLLM_CI_NO_SKIP", "0"))),
    # Allow changing the dtype used by vllm in tests
    "VLLM_CI_DTYPE": lambda: os.getenv("VLLM_CI_DTYPE", None),
    # Allow changing the head dtype used by vllm in tests
    "VLLM_CI_HEAD_DTYPE": lambda: os.getenv("VLLM_CI_HEAD_DTYPE", None),
    # Allow changing the head dtype used by transformers in tests
    "VLLM_CI_HF_DTYPE": lambda: os.getenv("VLLM_CI_HF_DTYPE", None),
}


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())


def is_set(name: str):
    """Check if an environment variable is explicitly set."""
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
