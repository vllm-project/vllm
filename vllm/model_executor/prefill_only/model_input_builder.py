from abc import ABC, abstractmethod
from typing import cast

import torch

from vllm.attention.prefill_only.abstract import (
    PrefillOnlyAttentionMetadataBuilder)
from vllm.model_executor.prefill_only.engine_io import (
    PrefillOnlySchedulerOutput, SchedulerOutput)
from vllm.model_executor.prefill_only.execute_io import (ExecuteInput,
                                                         ModelInputForGPU)
from vllm.utils import is_pin_memory_available

pin_memory = is_pin_memory_available()


class ModelInputBuilder(ABC):
    """
    scheduler_output = scheduler.schedule()
    SchedulerOutput  -> ModelInputBuilder -> ExecuteInput
    """

    @abstractmethod
    def __call__(self, scheduler_output: SchedulerOutput) -> ExecuteInput:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_engine(cls, engine):
        raise NotImplementedError


class PrefillOnlyModelInputBuilder(ModelInputBuilder):

    def __init__(
            self,
            attention_metadata_builder: PrefillOnlyAttentionMetadataBuilder):
        self.attention_metadata_builder = attention_metadata_builder

    @classmethod
    def from_engine(cls, engine):
        return cls(engine.attn_backend.get_builder_cls()())

    def __call__(self, scheduler_output: SchedulerOutput) -> ExecuteInput:
        assert isinstance(scheduler_output, PrefillOnlySchedulerOutput)
        scheduler_output = cast(PrefillOnlySchedulerOutput, scheduler_output)

        input_tokens = []
        input_positions = []
        seq_lens = []
        for request in scheduler_output.scheduled_requests:
            prompt_token_ids = request.inputs.prompt_token_ids
            n_tokens = len(prompt_token_ids)
            input_tokens.extend(prompt_token_ids)
            input_positions.extend(list(range(0, n_tokens)))
            seq_lens.append(n_tokens)

        input_ids = torch.tensor(input_tokens,
                                 dtype=torch.long,
                                 pin_memory=pin_memory,
                                 device="cpu")
        positions = torch.tensor(input_positions,
                                 dtype=torch.long,
                                 pin_memory=pin_memory,
                                 device="cpu")
        attn_metadata = self.attention_metadata_builder(seq_lens)

        model_input = ModelInputForGPU(input_ids=input_ids,
                                       positions=positions,
                                       attn_metadata=attn_metadata)

        return ExecuteInput(worker_input=None, model_input=model_input)
