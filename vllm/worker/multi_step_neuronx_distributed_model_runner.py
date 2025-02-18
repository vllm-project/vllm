from importlib.util import find_spec
import torch
from typing import List, Optional
from vllm.config import (DeviceConfig, ModelConfig, ParallelConfig,
                        SchedulerConfig, SpeculativeConfig)
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.multimodal import MultiModalInputs
from vllm.sequence import IntermediateTensors
from vllm.utils import is_neuronx_distributed_inference
from vllm.worker.neuronx_distributed_model_runner import NeuronxDistributedModelRunner

class MultiStepNeuronModelRunner(NeuronxDistributedModelRunner):

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        speculation_config: SpeculativeConfig,
    ):
        super().__init__(model_config, parallel_config, scheduler_config, device_config)
        self.speculation_config = speculation_config

    def load_model(self) -> None:
        from vllm.model_executor.model_loader.neuronx_distributed import get_neuron_speculation_model
        assert self.scheduler_config.max_num_seqs == 1, "Only batch size 1 is currently supported for speculation using neuronx-distributed-inference." 
        self.model = get_neuron_speculation_model(
            self.model_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            speculation_config=self.speculation_config)

    @torch.inference_mode()
    def execute_model(
        self,
        model_input,
        kv_caches: Optional[List[torch.Tensor]] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:
        if is_neuronx_distributed_inference():
            sampling_params = torch.tensor([[
                seq_group.sampling_params.top_k,
                seq_group.sampling_params.top_p,
                seq_group.sampling_params.temperature,
            ] for seq_group in model_input.sampling_metadata.seq_groups])

            logits = self.model(
                input_ids=model_input.input_tokens,
                positions=model_input.input_positions,
                input_block_ids=model_input.input_block_ids,
                sampling_params=sampling_params,
                **MultiModalInputs.as_kwargs(model_input.multi_modal_kwargs or {},
                                            device=self.device),
            )
        else:
            logits = self.model(
                input_ids=model_input.input_tokens,
                positions=model_input.input_positions,
                input_block_ids=model_input.input_block_ids,
                **MultiModalInputs.as_kwargs(model_input.multi_modal_kwargs or {},
                                            device=self.device),
            )

        output = self.model.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )
        return output

