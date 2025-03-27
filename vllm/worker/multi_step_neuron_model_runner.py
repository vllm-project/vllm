# SPDX-License-Identifier: Apache-2.0

from importlib.util import find_spec
from typing import List, Optional

import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.multimodal import MultiModalKwargs
from vllm.sequence import IntermediateTensors
from vllm.worker.neuron_model_runner import (ModelInputForNeuron,
                                             NeuronModelRunner)


class MultiStepNeuronModelRunner(NeuronModelRunner):
    """A model runner for multi step decoding using the transformers_neuronx
    framework"""

    def __init__(
        self,
        vllm_config: VllmConfig,
    ):
        super().__init__(vllm_config)
        self.speculation_config = self.speculative_config
        from transformers_neuronx.config import GenerationConfig
        self.speculation_config.draft_model_config.neuron_sampling_params = (
            GenerationConfig(
            max_length=self.scheduler_config.max_model_len,
            do_sample=True,
            per_batch_line=True,
            top_k=[self._MAX_NEURON_SAMPLING_TOP_K] \
                  * self.scheduler_config.max_num_seqs,
            top_p=[1.0] * self.scheduler_config.max_num_seqs,
            temperature=[1.0] * self.scheduler_config.max_num_seqs,
            dynamic=True,
            global_top_k=self._MAX_NEURON_SAMPLING_TOP_K
        ))

    def load_model(self) -> None:
        if find_spec("transformers_neuronx") is not None:
            from vllm.model_executor.model_loader.neuron import (
                get_neuron_eagle_speculation_model,
                get_neuron_speculation_model)
            if self.speculation_config.speculative_token_tree is not None:
                self.model = get_neuron_eagle_speculation_model(
                    self.model_config,
                    parallel_config=self.parallel_config,
                    scheduler_config=self.scheduler_config,
                    speculation_config=self.speculation_config)
            else:
                self.model = get_neuron_speculation_model(
                    self.model_config,
                    parallel_config=self.parallel_config,
                    scheduler_config=self.scheduler_config,
                    speculation_config=self.speculation_config)
        else:
            raise NotImplementedError(
                "Supports only Transformer-NeuronX based models.")

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForNeuron,
        kv_caches: Optional[List[torch.Tensor]] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:
        logits = self.model(
            input_ids=model_input.input_tokens,
            positions=model_input.input_positions,
            input_block_ids=model_input.input_block_ids,
            **MultiModalKwargs.as_kwargs(model_input.multi_modal_kwargs or {},
                                         device=self.device),
        )

        output = self.model.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )
        return output
