# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
import functools
import json
from concurrent.futures import Future
from concurrent.futures._base import TimeoutError
from typing import cast

from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.structured_output.backend_types import (
    StructuredOutputGrammar,
    StructuredOutputKey,
    StructuredOutputOptions,
)


@dataclasses.dataclass
class StructuredOutputRequest:
    params: StructuredOutputsParams
    _grammar: Future[StructuredOutputGrammar] | StructuredOutputGrammar | None = None
    reasoning_ended: bool | None = None

    @staticmethod
    def from_sampling_params(
        sampling_params: SamplingParams | None,
    ) -> "StructuredOutputRequest | None":
        if sampling_params is None:
            return None
        params = sampling_params.structured_outputs
        if not params or params.all_constraints_none():
            return None
        return StructuredOutputRequest(params=params)

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
    def grammar(self) -> StructuredOutputGrammar | None:
        completed = self._check_grammar_completion()
        return (
            cast(StructuredOutputGrammar | None, self._grammar) if completed else None
        )

    @grammar.setter
    def grammar(
        self, grammar: StructuredOutputGrammar | Future[StructuredOutputGrammar]
    ) -> None:
        self._grammar = grammar

    @functools.cached_property
    def structured_output_key(self) -> StructuredOutputKey:
        return get_structured_output_key(self.params)


def get_structured_output_key(params: StructuredOutputsParams) -> StructuredOutputKey:
    if params.json is not None:
        if not isinstance(params.json, str):
            json_str = json.dumps(params.json)
        else:
            json_str = params.json
        return StructuredOutputOptions.JSON, json_str
    if params.json_object:
        return StructuredOutputOptions.JSON_OBJECT, ""
    if params.regex is not None:
        return StructuredOutputOptions.REGEX, params.regex
    if params.choice is not None:
        if not isinstance(params.choice, str):
            json_str = json.dumps(params.choice)
        else:
            json_str = params.choice
        return StructuredOutputOptions.CHOICE, json_str
    if params.grammar is not None:
        return StructuredOutputOptions.GRAMMAR, params.grammar
    if params.structural_tag is not None:
        return StructuredOutputOptions.STRUCTURAL_TAG, params.structural_tag
    raise ValueError("No valid structured output parameter found")
