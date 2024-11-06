from typing import List, Optional

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import IntermediateTensors
from vllm.worker.cpu_model_runner import CPUModelRunner as ModelRunnerBaseCls
from vllm.worker.cpu_model_runner import ModelInputForCPUWithSamplingMetadata

logger = init_logger(__name__)


class CPUTP1DraftModelRunner(ModelRunnerBaseCls):
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
        model_input: ModelInputForCPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor],
        previous_hidden_states: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:
        return super().execute_model(
            model_input=model_input,
            kv_caches=kv_caches,
            previous_hidden_states=previous_hidden_states,
            intermediate_tensors=intermediate_tensors,
            num_steps=num_steps,
        )
