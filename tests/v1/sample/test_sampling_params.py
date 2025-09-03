# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
from typing import Optional, Union

from vllm import SamplingParams


def _is_param_optional(param: inspect.Parameter) -> bool:
    return (hasattr(param.annotation, "__origin__")
            and (param.annotation.__origin__ is Optional or
                 (param.annotation.__origin__ is Union
                  and type(None) in param.annotation.__args__)))


def test_from_optional():
    # Ensure passing nothing returns the default values
    assert SamplingParams.from_optional() == SamplingParams()
    # Ensure every arg is optional
    signature = inspect.signature(SamplingParams.from_optional)
    for name, param in signature.parameters.items():
        assert _is_param_optional(
            param), f"{name} is not optional. Got {param}"
