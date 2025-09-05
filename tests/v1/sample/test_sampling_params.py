# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
from typing import Optional, Union

from vllm import SamplingParams


def _is_annotation_optional(annotation) -> bool:
    return (hasattr(annotation, "__origin__")
            and (annotation.__origin__ is Optional or
                 (annotation.__origin__ is Union
                  and type(None) in annotation.__args__)))


def test_from_optional():
    # Ensure passing nothing returns the default values
    default_sampling_params = SamplingParams()
    assert SamplingParams.from_optional() == default_sampling_params

    # Ensure every arg is optional
    signature = inspect.signature(SamplingParams.from_optional)
    for name, param in signature.parameters.items():
        assert _is_annotation_optional(
            param.annotation), f"{name} is not optional. Got {param}"

    # Ensure every public field is supported by from_optional
    func_args = set(signature.parameters.keys())
    public_fields = set(field for field in SamplingParams.__struct_fields__
                        if not field.startswith("_"))
    assert func_args == public_fields, \
        f"Missing fields: {public_fields - func_args}"

    # Ensure explicitly setting None
    #   1/ Returns the default value if the field is not optional.
    #   2/ Returns None if the field is optional. Note some optional fields
    #      have a default value other than None and explicitly setting None
    #      should override the default value
    explicit_none = SamplingParams.from_optional(
        **{field: None
           for field in public_fields})
    for field in public_fields:
        value = getattr(explicit_none, field)
        if _is_annotation_optional(signature.parameters[field].annotation):
            assert value is None, f"`{field}` is optional but got '{value}'"
        else:
            default_value = getattr(default_sampling_params, field)
            assert value == default_value, \
                f"`{field}` defaults to '{default_value}' but got '{value}'"
