from abc import ABC, abstractmethod
from typing import Optional, Union

from vllm.wde.core.llm_engine import LLMEngine
from vllm.wde.core.schema.engine_io import (Inputs, Params, Request,
                                            SchedulableRequest)


class InputProcessor(ABC):
    """
    Input(request_id, inputs, params, arrival_time) -> InputProcessor -> Request
    """

    @abstractmethod
    def __call__(self,
                 request_id: str,
                 inputs: Optional[Union[str, Inputs]] = None,
                 params: Optional[Params] = None,
                 arrival_time: Optional[float] = None) -> Request:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_engine(cls, engine: LLMEngine):
        raise NotImplementedError


class RequestProcessor(ABC):
    """
    Request -> RequestProcessor -> SchedulableRequest
    """

    @abstractmethod
    def __call__(self, request: Request) -> SchedulableRequest:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_engine(cls, engine: LLMEngine):
        raise NotImplementedError
