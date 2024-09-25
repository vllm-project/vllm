from vllm.utils import is_pin_memory_available
from vllm.wde.core.llm_engine import LLMEngine
from vllm.wde.core.schema.execute_io import ExecuteInput
from vllm.wde.prefill_only.layers.attention.backends.abstract import (
    PrefillOnlyAttentionMetadataBuilder)
from vllm.wde.prefill_only.processor.model_input_builder import (
    PrefillOnlyModelInputBuilder)
from vllm.wde.prefill_only.schema.engine_io import PrefillOnlySchedulerOutput

pin_memory = is_pin_memory_available()


class GTEQwenModelInputBuilder(PrefillOnlyModelInputBuilder):

    def __init__(
            self, eos_token_id: int,
            attention_metadata_builder: PrefillOnlyAttentionMetadataBuilder):
        super().__init__(attention_metadata_builder)
        self.eos_token_id = eos_token_id

    @classmethod
    def from_engine(cls, engine: LLMEngine):
        return cls(engine.tokenizer.eos_token_id,
                   engine.attn_backend.get_builder_cls()())

    def __call__(self,
                 scheduler_output: PrefillOnlySchedulerOutput) -> ExecuteInput:
        # gte-Qwen2 adds eos_token_id to the end of the sentence
        for request in scheduler_output.requests:
            request.inputs.prompt_token_ids = (
                request.inputs.prompt_token_ids + [self.eos_token_id])
        return super().__call__(scheduler_output)
