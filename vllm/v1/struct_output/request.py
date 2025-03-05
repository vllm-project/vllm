# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import dataclasses
import functools
import json
from concurrent.futures import Future
from concurrent.futures._base import TimeoutError
from typing import Optional, Union, cast

from vllm.sampling_params import SamplingParams
from vllm.v1.struct_output.grammar import (Grammar, StructOutputKey,
                                           StructOutputOptions)


@dataclasses.dataclass
class StructOutputRequest:

    sampling_params: SamplingParams
    _grammar: Optional[Union[Future[Grammar],
                             Grammar]] = dataclasses.field(default=None)

    def _check_grammar_completion(self) -> bool:
        # NOTE: We have to lazy import to gate circular imports
        from vllm.v1.request import RequestStatus

        if isinstance(self._grammar, Future):
            try:
                # We will check whether the future is ready within 100 us
                self._grammar = self._grammar.result(timeout=0.0001)
                self.status = RequestStatus.WAITING
            except TimeoutError:
                return False
        return True

    @property
    def is_grammar_ready(self) -> bool:
        return self._check_grammar_completion()

    @property
    def grammar(self) -> Optional[Grammar]:
        completed = self._check_grammar_completion()
        return cast(Optional[Grammar], self._grammar) if completed else None

    @grammar.setter
    def grammar(self, grammar: Union[Grammar, Future[Grammar]]) -> None:
        self._grammar = grammar

    @functools.cached_property
    def struct_output_key(self) -> StructOutputKey:
        params = self.sampling_params.guided_decoding
        assert params is not None, "params can't be None."
        if params.json is not None:
            if not isinstance(params.json, str):
                json_str = json.dumps(params.json)
            else:
                json_str = params.json
            return (StructOutputOptions.JSON, json_str)
        elif params.json_object:
            return (StructOutputOptions.JSON_OBJECT, "")
        elif params.regex is not None:
            return (StructOutputOptions.REGEX, params.regex)
        elif params.choice is not None:
            if not isinstance(params.choice, str):
                json_str = json.dumps(params.choice)
            else:
                json_str = params.choice
            return (StructOutputOptions.CHOICE, json_str)
        elif params.grammar is not None:
            return (StructOutputOptions.GRAMMAR, params.grammar)
        else:
            raise ValueError("No valid structured output parameter found")
