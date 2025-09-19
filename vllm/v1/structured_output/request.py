# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import dataclasses
import functools
import json
from typing import Callable, Optional

from vllm.sampling_params import SamplingParams
from vllm.v1.structured_output.backend_types import (StructuredOutputKey,
                                                     StructuredOutputOptions)


class FutureGrammar:

    def __init__(self, call_back: Callable[[str], bool], request_id: str):
        self.call_back = call_back
        self.request_id = request_id

    def done(self):
        if self.call_back is None:
            return False
        return self.call_back(self.request_id)


@dataclasses.dataclass
class StructuredOutputRequest:
    sampling_params: SamplingParams
    compiled_grammar: Optional[FutureGrammar] = None
    reasoning_ended: Optional[bool] = None

    @property
    def is_grammar_ready(self) -> bool:
        if self.compiled_grammar is None:
            return False
        return self.compiled_grammar.done()

    @functools.cached_property
    def structured_output_key(self) -> StructuredOutputKey:
        return get_structured_output_key(self.sampling_params)


def get_structured_output_key(
        sampling_params: SamplingParams) -> StructuredOutputKey:
    params = sampling_params.structured_outputs
    assert params is not None, "params can't be None."
    if params.json is not None:
        if not isinstance(params.json, str):
            json_str = json.dumps(params.json)
        else:
            json_str = params.json
        return (StructuredOutputOptions.JSON, json_str)
    elif params.json_object:
        return (StructuredOutputOptions.JSON_OBJECT, "")
    elif params.regex is not None:
        return (StructuredOutputOptions.REGEX, params.regex)
    elif params.choice is not None:
        if not isinstance(params.choice, str):
            json_str = json.dumps(params.choice)
        else:
            json_str = params.choice
        return (StructuredOutputOptions.CHOICE, json_str)
    elif params.grammar is not None:
        return (StructuredOutputOptions.GRAMMAR, params.grammar)
    elif params.structural_tag is not None:
        return (StructuredOutputOptions.STRUCTURAL_TAG, params.structural_tag)
    else:
        raise ValueError("No valid structured output parameter found")
