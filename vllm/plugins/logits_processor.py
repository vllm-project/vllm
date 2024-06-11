from typing import Callable, Type

from pydantic import BaseModel
from transformers import PreTrainedTokenizer

from vllm.sampling_params import LogitsProcessor


class LogitsProcessorPlugin:

    def __init__(
            self,
            logits_processor_class: Callable[[PreTrainedTokenizer, BaseModel],
                                             LogitsProcessor],
            parameters_model: Type[BaseModel]):
        self.logits_processor_class = logits_processor_class
        self.parameters_model = parameters_model
