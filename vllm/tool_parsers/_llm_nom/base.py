# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
# ruff: noqa

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Callable, Dict, Generic, List, Literal, Optional, TypeVar

from typing_extensions import Self

ParserInputs = TypeVar("ParserInputs", bound=Sequence)
ParserOutput = TypeVar("ParserOutput")
TextFn = Callable[[], str]
FunctionCallDict = Dict[Literal["name", "arguments"], str]


class Parser(ABC, Generic[ParserInputs, ParserOutput]):
    @abstractmethod
    def update(self, inputs: ParserInputs):
        pass

    def parse(self, inputs: ParserInputs) -> ParserOutput:
        self.update(inputs)
        final = self.get_final()
        assert final is not None
        return final

    @abstractmethod
    def get_delta(self) -> Optional[ParserOutput]:
        pass

    @abstractmethod
    def get_final(self) -> Optional[ParserOutput]:
        pass

    @abstractmethod
    def reset(self):
        pass

    def stop(self):
        pass

    def _get_sub_parsers(self) -> Optional[Dict[str, Self]]:
        """should only be used for debugging."""
        pass

    def stringify_function_calls(self, function_calls: List[FunctionCallDict]) -> str:
        raise NotImplementedError()


class PatternMismatched(Exception):
    def __init__(
        self,
        *,
        offset: int,
        expected: Any,
        actual: Any,
        reason: str,
        context_fn: Optional[TextFn] = None,
    ):
        self.offset = offset
        self.expected = expected
        self.actual = actual
        self.reason = reason
        self.context_fn = context_fn

    def __str__(self):
        main_message = f"at offset {self.offset}, {self.reason}, expected {self.expected!r} but got {self.actual!r}"
        if self.context_fn is None:
            return main_message
        else:
            return f"around context:\n{self.context_fn()}\n\n{main_message}"
