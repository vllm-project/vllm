# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
from optimum.neuron.models.inference.nxd.backend.modules.generation.sampling import (  # noqa 501
    prepare_sampling_params)

from vllm.attention.backends.abstract import AttentionBackend
from vllm.config import DeviceConfig, VllmConfig
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader.optimum_neuron import (
    OptimumNeuronModelForCausalLM, get_optimum_neuron_model)
from vllm.sampling_params import SamplingParams
from vllm.sequence import IntermediateTensors, SequenceGroupMetadata
from vllm.utils import is_pin_memory_available, make_tensor_with_pad
from vllm.worker.model_runner_base import ModelRunnerBase, ModelRunnerInputBase

logger = init_logger(__name__)


@dataclass(frozen=True)
class ModelInputForOptimumNeuron(ModelRunnerInputBase):
    """
    Used by the OptimumNeuronModelRunner.
    """
    input_ids: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    seq_ids: Optional[torch.Tensor] = None
    sampling_metadata: SamplingMetadata = None

    def as_broadcastable_tensor_dict(
            self) -> Dict[str, Union[int, torch.Tensor]]:
        return {
            "input_ids": self.input_ids,
            "position_ids": self.position_ids,
            "seq_ids": self.seq_ids,
            "sampling_metadata": self.sampling_metadata,
        }

    @classmethod
    def from_broadcasted_tensor_dict(
        cls,
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> "ModelInputForOptimumNeuron":
        return ModelInputForOptimumNeuron(
            input_ids=tensor_dict["input_ids"],
            position_ids=tensor_dict["position_ids"],
            seq_ids=tensor_dict["seq_ids"],
            sampling_metadata=tensor_dict["sampling_metadata"],
        )


class OptimumNeuronModelRunner(ModelRunnerBase[ModelInputForOptimumNeuron]):

    # NEURON has an upper limit on the top_k
    _MAX_NEURON_SAMPLING_TOP_K = 256

    def __init__(
        self,
        vllm_config: VllmConfig,
    ):
        super().__init__(vllm_config)
        self.model: OptimumNeuronModelForCausalLM = None
        device_config = (self.device_config
                         if self.device_config is not None else DeviceConfig())
        self.device = device_config.device
        self.pin_memory = is_pin_memory_available()

    def load_model(self) -> None:
        self.model = get_optimum_neuron_model(
            self.model_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config)

    def get_model(self) -> torch.nn.Module:
        return self.model

    def get_nxd_sampling_params(self, sampling_metadata):
        if self.model.model.neuron_config.on_device_sampling:
            max_topk = self.model.model.neuron_config.max_topk
        else:
            max_topk = self.model.model.config.vocab_size

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

    def _convert_to_neuron_sampling_params(
            self, sampling_params: SamplingParams) -> Tuple[int, float, float]:
        # Returns the top_k, top_p and temperature parameters for neuron.
        top_k = sampling_params.top_k
        top_p = sampling_params.top_p
        temperature = sampling_params.temperature

        if temperature == 0.0:
            # Enable greedy sampling on zero temperature
            return (1, 1.0, 1.0)
        if top_k < 1 or top_k > self._MAX_NEURON_SAMPLING_TOP_K:
            top_k = self._MAX_NEURON_SAMPLING_TOP_K

        return (top_k, top_p, temperature)

    @torch.inference_mode()
    def execute_model(
            self,
            model_input: ModelInputForOptimumNeuron,
            kv_caches: Optional[List[torch.Tensor]] = None,
            intermediate_tensors: Optional[IntermediateTensors] = None,
            num_steps: int = 1,
            **kwargs) -> Optional[List[SamplerOutput]]:
        if num_steps > 1:
            raise ValueError("OptimumNeuronModelRunner does not support "
                             "multi-step execution.")

        sampling_params = self.get_nxd_sampling_params(
            model_input.sampling_metadata)

        hidden_states = self.model(
            input_ids=model_input.input_ids,
            position_ids=model_input.position_ids,
            seq_ids=model_input.seq_ids,
            sampling_params=sampling_params,
        )

        output = self.model.sample(
            logits=hidden_states,
            sampling_metadata=model_input.sampling_metadata,
        )

        return [output]

    def make_model_input_from_broadcasted_tensor_dict(
            self, tensor_dict: Dict[str, Any]) -> ModelInputForOptimumNeuron:
        return ModelInputForOptimumNeuron.from_broadcasted_tensor_dict(
            tensor_dict)

    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        input_block_ids: List[int] = []

        seq_lens: List[int] = []
        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            seq_len = len(prompt_tokens)
            seq_lens.append(seq_len)

            input_tokens.append(prompt_tokens)
            input_positions.append(list(range(seq_len)))

            assert seq_group_metadata.block_tables is not None
            block_table = seq_group_metadata.block_tables[seq_id]
            assert len(block_table) == 1
            input_block_ids.append(block_table[0])

        max_seq_len = max(seq_lens)
        assert max_seq_len > 0
        input_ids = make_tensor_with_pad(input_tokens,
                                         pad=0,
                                         max_len=max_seq_len,
                                         dtype=torch.long,
                                         device=self.device)
        position_ids = make_tensor_with_pad(input_positions,
                                            pad=0,
                                            max_len=max_seq_len,
                                            dtype=torch.long,
                                            device=self.device)
        seq_ids = torch.tensor(input_block_ids,
                               dtype=torch.long,
                               device=self.device)

        return (input_ids, position_ids, seq_ids, seq_lens)

    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        input_block_ids: List[int] = []

        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt

            seq_ids = list(seq_group_metadata.seq_data.keys())

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append([generation_token])

                seq_len = seq_data.get_len()
                position = seq_len - 1
                input_positions.append([position])

                assert seq_group_metadata.block_tables is not None
                block_table = seq_group_metadata.block_tables[seq_id]
                assert len(block_table) == 1
                input_block_ids.append(block_table[0])

        input_ids = make_tensor_with_pad(input_tokens,
                                         pad=0,
                                         max_len=1,
                                         dtype=torch.long,
                                         device=self.device)
        position_ids = make_tensor_with_pad(input_positions,
                                            pad=0,
                                            max_len=1,
                                            dtype=torch.long,
                                            device=self.device)
        seq_ids = torch.tensor(input_block_ids,
                               dtype=torch.long,
                               device=self.device)

        return input_ids, position_ids, seq_ids

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None
    ) -> ModelInputForOptimumNeuron:
        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        is_prompt = seq_group_metadata_list[0].is_prompt
        # Prepare input tensors.
        if is_prompt:
            (input_ids, position_ids, seq_ids,
             seq_lens) = self._prepare_prompt(seq_group_metadata_list)
        else:
            (input_ids, position_ids,
             seq_ids) = self._prepare_decode(seq_group_metadata_list)
            seq_lens = None

        for seq_group_metadata in seq_group_metadata_list:
            sampling_params = seq_group_metadata.sampling_params
            top_k, top_p, temperature = (
                self._convert_to_neuron_sampling_params(sampling_params))
            sampling_params.top_k = top_k
            sampling_params.top_p = top_p
            sampling_params.temperature = temperature

        sampling_metadata = SamplingMetadata.prepare(
            seq_group_metadata_list,
            seq_lens,
            # query_lens is not needed if chunked prefill is not
            # supported. Since neuron worker doesn't support chunked prefill
            # just use seq_lens instead.
            seq_lens,
            self.device,
            self.pin_memory,
            generators=self.get_generators(finished_requests_ids))

        return ModelInputForOptimumNeuron(input_ids=input_ids,
                                          position_ids=position_ids,
                                          seq_ids=seq_ids,
                                          sampling_metadata=sampling_metadata)

    def remove_all_loras(self):
        raise NotImplementedError(
            "LoRAs are not supported in the optimum-neuron framewrok.")

    def set_active_loras(self, lora_requests: Set[LoRARequest],
                         lora_mapping: LoRAMapping) -> None:
        raise NotImplementedError(
            "LoRAs are not supported in the optimum-neuron framewrok.")

    def add_lora(self, lora_request: LoRARequest):
        raise NotImplementedError(
            "LoRAs are not supported in the optimum-neuron framewrok.")

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError(
            "LoRAs are not supported in the optimum-neuron framewrok.")

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError(
            "LoRAs are not supported in the optimum-neuron framewrok.")

    def list_loras(self) -> Set[int]:
        raise NotImplementedError(
            "LoRAs are not supported in the optimum-neuron framewrok.")
