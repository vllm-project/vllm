# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

import torch
from neuronx_distributed_inference.modules.generation.sampling import (
    prepare_sampling_params)

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader.neuronx_distributed import (
    _get_model_architecture, get_neuron_model)
from vllm.sequence import IntermediateTensors
from vllm.worker.neuron_model_runner import (ModelInputForNeuron,
                                             NeuronModelRunner)

logger = init_logger(__name__)


class NeuronxDistributedModelRunner(NeuronModelRunner):

    def __init__(
        self,
        vllm_config: VllmConfig,
    ):
        super().__init__(vllm_config)

    def load_model(self) -> None:
        self.model = get_neuron_model(self.model_config,
                                      parallel_config=self.parallel_config,
                                      scheduler_config=self.scheduler_config)

    def get_nxd_sampling_params(self, sampling_metadata):
        if self.model.config.neuron_config.on_device_sampling_config:
            max_topk = (self.model.config.neuron_config.
                        on_device_sampling_config.global_topk)
        else:
            max_topk = self.model.config.vocab_size

        top_k = [1] * self.scheduler_config.max_num_seqs
        top_p = [1.0] * self.scheduler_config.max_num_seqs
        temperature = [1.0] * self.scheduler_config.max_num_seqs

        for index, sequenceGroupToSample in enumerate(
                sampling_metadata.seq_groups):
            top_k[index] = (sequenceGroupToSample.sampling_params.top_k
                            if sequenceGroupToSample.sampling_params.top_k > 0
                            else max_topk)
            top_p[index] = sequenceGroupToSample.sampling_params.top_p
            temperature[index] = (
                sequenceGroupToSample.sampling_params.temperature)

        sampling_params = prepare_sampling_params(
            batch_size=self.scheduler_config.max_num_seqs,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature)
        return sampling_params

    def get_multi_modal_data_neuron(self, input_images):
        raise NotImplementedError("need to restore multi-modal support")

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForNeuron,
        kv_caches: Optional[List[torch.Tensor]] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:
        if num_steps > 1:
            raise ValueError(
                "NeuronModelRunner does not support multi-step execution.")

        if _get_model_architecture(
                self.model.config) != "MllamaForConditionalGeneration":
            return super().execute_model(model_input, kv_caches,
                                         intermediate_tensors, num_steps)

        sampling_params = self.get_nxd_sampling_params(
            model_input.sampling_metadata)

        if model_input.multi_modal_kwargs.get('image') is not None:
            pixel_values = []
            aspect_ratios = []
            num_chunks = []
            has_image = []
            for multi_modal_input in model_input.multi_modal_kwargs.get(
                    'image'):
                image_tensors = self.get_multi_modal_data_neuron(
                    multi_modal_input.squeeze(0))
                pixel_values.append(image_tensors[0])
                aspect_ratios.append(image_tensors[1])
                num_chunks.append(image_tensors[2])
                has_image.append(image_tensors[3])

            pixel_values = torch.cat(pixel_values, dim=0)
            aspect_ratios = torch.cat(aspect_ratios, dim=0)
            num_chunks = torch.cat(num_chunks, dim=0)
            has_image = torch.cat(has_image, dim=0)

            hidden_states = self.model(
                input_ids=model_input.input_tokens,
                positions=model_input.input_positions,
                seq_ids=model_input.input_block_ids,
                pixel_values=pixel_values,
                aspect_ratios=aspect_ratios,
                sampling_params=sampling_params,
                num_chunks=num_chunks,
                has_image=has_image,
            )
        else:
            empty_pixel_values = torch.zeros([1, 1, 4, 3, 560, 560],
                                             dtype=torch.bfloat16)
            empty_aspect_ratios = torch.ones([1, 1, 2], dtype=torch.int64)
            num_chunks = torch.tensor([[1]
                                       ])  # dummy num_chunks, will not be used
            has_image = torch.tensor([0])
            hidden_states = self.model(
                input_ids=model_input.input_tokens,
                positions=model_input.input_positions,
                seq_ids=model_input.input_block_ids,
                pixel_values=empty_pixel_values,
                aspect_ratios=empty_aspect_ratios,
                sampling_params=sampling_params,
                num_chunks=num_chunks,
                has_image=has_image,
            )

        output = self.model.sample(
            hidden_states=hidden_states,
            sampling_metadata=model_input.sampling_metadata,
        )

        return [output]
