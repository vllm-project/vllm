import torch

from vllm.utils import is_pin_memory_available
from vllm.wde.core.llm_engine import LLMEngine
from vllm.wde.core.processor.model_input_builder import ModelInputBuilder
from vllm.wde.core.schema.execute_io import ExecuteInput
from vllm.wde.prefill_only.layers.attention.backends.abstract import (
    PrefillOnlyAttentionMetadataBuilder)
from vllm.wde.prefill_only.schema.engine_io import PrefillOnlySchedulerOutput
from vllm.wde.prefill_only.schema.execute_io import ModelInputForGPU

pin_memory = is_pin_memory_available()


class PrefillOnlyModelInputBuilder(ModelInputBuilder):

    def __init__(
            self,
            attention_metadata_builder: PrefillOnlyAttentionMetadataBuilder):
        self.attention_metadata_builder = attention_metadata_builder

    @classmethod
    def from_engine(cls, engine: LLMEngine):
        return cls(engine.attn_backend.get_builder_cls()())

    def __call__(self,
                 scheduler_output: PrefillOnlySchedulerOutput) -> ExecuteInput:
        input_tokens = []
        input_positions = []
        seq_lens = []
        for request in scheduler_output.requests:
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
