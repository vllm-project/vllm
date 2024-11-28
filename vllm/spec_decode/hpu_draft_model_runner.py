from typing import List, Optional

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import IntermediateTensors
from vllm.worker.hpu_model_runner import HPUModelRunner as ModelRunnerBaseCls
from vllm.worker.hpu_model_runner import ModelInputForHPUWithSamplingMetadata

logger = init_logger(__name__)

# A flag to enable debug prints for the updated input tensors
# before each step.
debug_advance_input = False
# A flag to allow GPU advance step for draft model runner.
# Set to False for debugging.
allow_gpu_advance_step = True


class HPUTP1DraftModelRunner(ModelRunnerBaseCls):
    """Specialized model runner for speculative decoding draft model.
    Since the draft model always execute k forward passes consecutively to
    generate k speculative tokens in a single speculative decoding step,
    we could get rid of most CPU-GPU synchronization and data transfer
    overheads by keeping model input and output tensors on GPU all the time.

    TODOs:
    1. Support TP > 1 (this requires some designs because we do not expect
       any broadcasting inside execute_model).
    """

    def __init__(self, *args, **kwargs):
        if kwargs.get("return_hidden_states"):
            raise ValueError(
                "return_hidden_states is not supported for TP1DraftModelRunner."
            )

        super().__init__(*args, **kwargs)

        self.indices_of_seq_with_bonus_tokens = None

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForHPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor],
        previous_hidden_states: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:
        if previous_hidden_states is not None:
            batch_size, block_size = model_input.input_tokens.shape
            previous_hidden_states = previous_hidden_states.unsqueeze(
                dim=1).expand(-1, block_size, -1)
            # because HPU will pad batch_size,
            # we need to pad previous_hidden_states as well
            batch_size_padding = batch_size - previous_hidden_states.shape[0]
            if batch_size_padding > 0:
                dummy_previous_hidden_states = torch.zeros_like(
                    previous_hidden_states[1:2]).expand(
                        batch_size_padding, -1, -1)
                previous_hidden_states = torch.cat(
                    [previous_hidden_states, dummy_previous_hidden_states],
                    dim=0)
        return super().execute_model(
            model_input=model_input,
            kv_caches=kv_caches,
            previous_hidden_states=previous_hidden_states,
            intermediate_tensors=intermediate_tensors,
            num_steps=num_steps,
        )
