import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, cast

from vllm.inputs.prefill_only.data import (Params, PromptInput, Request,
                                           SchedulableRequest, TextOnlyInputs,
                                           TextPrompt, TextRequest,
                                           TextSchedulableRequest,
                                           TokensPrompt, ValidationError)
from vllm.inputs.prefill_only.tokenizer import Tokenizer


class InputProcessor(ABC):
    """
    Input(request_id, inputs, params, arrival_time) -> InputProcessor -> Request
    """

    @abstractmethod
    def __call__(self,
                 request_id: str,
                 inputs: Optional[Any] = None,
                 params: Optional[Params] = None,
                 arrival_time: Optional[float] = None) -> Request:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_engine(cls, engine):
        raise NotImplementedError


class TextInputProcessor(InputProcessor):

    def __call__(self,
                 request_id: str,
                 inputs: Optional[PromptInput] = None,
                 params: Optional[Params] = None,
                 arrival_time: Optional[float] = None) -> TextRequest:

        if isinstance(inputs, str):
            inputs = {"prompt": inputs}
        elif isinstance(inputs, TextPrompt):
            inputs = {"prompt": inputs.prompt}
        elif isinstance(inputs, TokensPrompt):
            inputs = {"prompt_token_ids": inputs.prompt_token_ids}
        elif isinstance(inputs, TextOnlyInputs):
            _inputs: Dict[str, Any] = {
                "prompt_token_ids": inputs.prompt_token_ids
            }

            if inputs.prompt is not None:
                _inputs["prompt"] = inputs.prompt

            inputs = _inputs

        elif isinstance(inputs, dict):
            if "prompt" not in inputs and "prompt_token_ids" not in inputs:
                raise ValidationError('"prompt" and "prompt_token_ids" '
                                      'have at least one in inputs.')
            inputs = {
                k: v
                for k, v in inputs.items()
                if k in {"prompt", "prompt_token_ids"}
            }
        else:
            raise ValidationError(
                f"Input does not support {type(inputs)} data type")

        if not arrival_time:
            arrival_time = time.time()
        request = TextRequest(request_id=str(request_id),
                              inputs=inputs,
                              arrival_time=arrival_time)
        return request

    @classmethod
    def from_engine(cls, engine):
        return cls()


class RequestProcessor(ABC):
    """
    Request -> RequestProcessor -> SchedulableRequest
    """

    @abstractmethod
    def __call__(self, request: Request) -> SchedulableRequest:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_engine(cls, engine):
        raise NotImplementedError


class TextRequestProcessor(RequestProcessor):

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, request: Request) -> TextSchedulableRequest:
        assert isinstance(request, TextRequest)

        request = cast(TextRequest, request)

        inputs = request.inputs

        if "prompt_token_ids" not in inputs:
            tokenizer = self.tokenizer

            prompt_token_ids = tokenizer.encode(inputs["prompt"])
        else:
            prompt_token_ids = inputs["prompt_token_ids"]

        schedulable_request = TextSchedulableRequest(
            request_id=request.request_id,
            inputs=TextOnlyInputs(prompt_token_ids=prompt_token_ids,
                                  prompt=inputs.get("prompt")),
            arrival_time=request.arrival_time)

        return schedulable_request

    @classmethod
    def from_engine(cls, engine):
        return cls(engine.tokenizer)
