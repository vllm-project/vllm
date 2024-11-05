import os
from dataclasses import dataclass
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers_neuronx.config import GenerationConfig

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader.neuron import get_neuron_model
from vllm.multimodal import (MULTIMODAL_REGISTRY, BatchedTensorInputs,
                             MultiModalInputs)
from vllm.sequence import IntermediateTensors, SequenceGroupMetadata
from vllm.utils import is_pin_memory_available, make_tensor_with_pad
from vllm.worker.model_runner_base import ModelRunnerBase, ModelRunnerInputBase

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

logger = init_logger(__name__)


@dataclass(frozen=True)
class ModelInputForNeuron(ModelRunnerInputBase):
    """
    Used by the NeuronModelRunner.
    """
    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    input_block_ids: Optional[torch.Tensor] = None
    sampling_metadata: Optional["SamplingMetadata"] = None
    multi_modal_kwargs: Optional[BatchedTensorInputs] = None

    def as_broadcastable_tensor_dict(
            self) -> Dict[str, Union[int, torch.Tensor]]:
        raise NotImplementedError("ModelInputForNeuron cannot be broadcast.")

    @classmethod
    def from_broadcasted_tensor_dict(
        cls,
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> "ModelInputForNeuron":
        assert attn_backend is None
        return cls.from_broadcasted_tensor_dict(tensor_dict)


class NeuronModelRunner(ModelRunnerBase[ModelInputForNeuron]):

    # NEURON has an upper limit on the top_k
    _MAX_NEURON_SAMPLING_TOP_K = 256

    def __init__(
        self,
        vllm_config: VllmConfig,
    ):
        ModelRunnerBase.__init__(self, vllm_config)
        model_config = self.model_config
        if model_config is not None and model_config.get_sliding_window():
            logger.warning("Sliding window is not supported on Neuron. "
                           "The model will run without sliding window.")
        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()

        # Multi-modal data support
        self.multi_modal_input_mapper = MULTIMODAL_REGISTRY \
            .create_input_mapper(self.model_config)

        # Lazy initialization.
        self.model: nn.Module  # initialize after load_model.

        # Once NEURON_ON_DEVICE_SAMPLING_DISABLED is set to a non-zero value,
        # turn off on-device sampling.
        self._on_device_sampling_disabled = int(
            os.getenv("NEURON_ON_DEVICE_SAMPLING_DISABLED", "0"))

        # NEURON needs to update sampling parameters when request IDs change
        # across batches. This variable stores the previous batch's request IDs
        # to determine if an update is needed.
        self._previous_batch_request_ids: List[str] = []

        if not self._on_device_sampling_disabled:
            logger.warning(
                "On-device sampling is turned on in Neuron by default, only "
                "top_k, top_p, and temperature are current supported sampling "
                "parameters. To turn off the on-device sampling, please set "
                "the environment variable NEURON_ON_DEVICE_SAMPLING_DISABLED=1."
            )
            self.model_config.neuron_sampling_params = GenerationConfig(
                max_length=self.scheduler_config.max_model_len,
                do_sample=True,
                per_batch_line=True,
                top_k=[self._MAX_NEURON_SAMPLING_TOP_K] \
                    * self.scheduler_config.max_num_seqs,
                top_p=[1.0] * self.scheduler_config.max_num_seqs,
                temperature=[1.0] * self.scheduler_config.max_num_seqs,
                dynamic=True,
                global_top_k=self._MAX_NEURON_SAMPLING_TOP_K)

    def load_model(self) -> None:
        if find_spec("transformers_neuronx") is not None:
            self.model = get_neuron_model(
                self.model_config,
                parallel_config=self.parallel_config,
                scheduler_config=self.scheduler_config)
        else:
            raise NotImplementedError(
                "Supports only Transformer-NeuronX based models.")

    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int],
               BatchedTensorInputs]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        input_block_ids: List[int] = []

        seq_lens: List[int] = []
        multi_modal_inputs_list: List[MultiModalInputs] = []
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

            mm_data = seq_group_metadata.multi_modal_data
            if mm_data:
                # Process multi-modal data
                mm_kwargs = self.multi_modal_input_mapper(
                    mm_data,
                    mm_processor_kwargs=seq_group_metadata.mm_processor_kwargs,
                )
                multi_modal_inputs_list.append(mm_kwargs)

        max_seq_len = max(seq_lens)
        assert max_seq_len > 0
        input_tokens = make_tensor_with_pad(input_tokens,
                                            pad=0,
                                            max_len=max_seq_len,
                                            dtype=torch.long,
                                            device=self.device)
        input_positions = make_tensor_with_pad(input_positions,
                                               pad=0,
                                               max_len=max_seq_len,
                                               dtype=torch.long,
                                               device=self.device)
        input_block_ids = torch.tensor(input_block_ids,
                                       dtype=torch.long,
                                       device=self.device)

        multi_modal_kwargs = MultiModalInputs.batch(multi_modal_inputs_list)

        return (input_tokens, input_positions, input_block_ids, seq_lens,
                multi_modal_kwargs)

    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        input_block_ids: List[int] = []
        context_lens: List[int] = []

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
                context_lens.append(seq_len)

                assert seq_group_metadata.block_tables is not None
                block_table = seq_group_metadata.block_tables[seq_id]
                assert len(block_table) == 1
                input_block_ids.append(block_table[0])

        input_tokens = make_tensor_with_pad(input_tokens,
                                            pad=0,
                                            max_len=1,
                                            dtype=torch.long,
                                            device=self.device)
        input_positions = make_tensor_with_pad(input_positions,
                                               pad=0,
                                               max_len=1,
                                               dtype=torch.long,
                                               device=self.device)
        context_lens = torch.tensor(context_lens,
                                    dtype=torch.int,
                                    device=self.device)
        input_block_ids = torch.tensor(input_block_ids,
                                       dtype=torch.long,
                                       device=self.device)

        return input_tokens, input_positions, input_block_ids

    def make_model_input_from_broadcasted_tensor_dict(
            self, tensor_dict: Dict[str, Any]) -> ModelInputForNeuron:
        return ModelInputForNeuron.from_broadcasted_tensor_dict(tensor_dict)

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None
    ) -> ModelInputForNeuron:
        multi_modal_kwargs = None
        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        is_prompt = seq_group_metadata_list[0].is_prompt
        # Prepare input tensors.
        if is_prompt:
            (input_tokens, input_positions, input_block_ids, seq_lens,
             multi_modal_kwargs
             ) = self._prepare_prompt(seq_group_metadata_list)
        else:
            (input_tokens, input_positions,
             input_block_ids) = self._prepare_decode(seq_group_metadata_list)
            seq_lens = None
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

        if not self._on_device_sampling_disabled:
            # Once the request IDs are changed in current iteration, we will
            # update the on-device sampling parameters.
            current_batch_request_ids = [
                seq_group_meta_data.request_id
                for seq_group_meta_data in seq_group_metadata_list
            ]
            if current_batch_request_ids != self._previous_batch_request_ids:
                self._update_neuron_sampling_params(sampling_metadata)
                self._previous_batch_request_ids = current_batch_request_ids

        return ModelInputForNeuron(input_tokens=input_tokens,
                                   input_positions=input_positions,
                                   input_block_ids=input_block_ids,
                                   sampling_metadata=sampling_metadata,
                                   multi_modal_kwargs=multi_modal_kwargs)

    def _update_neuron_sampling_params(self,
                                       sampling_metadata: SamplingMetadata):
        # Update Neuron sampling parameters (GenerationConfig in Neuron)
        current_sampling_params = self.model_config.neuron_sampling_params
        assert current_sampling_params is not None, (
            f"Failed to update sampling_params, "
            f"current sampling params is {current_sampling_params}")

        top_k = current_sampling_params.top_k
        top_p = current_sampling_params.top_p
        temperature = current_sampling_params.temperature
        for index, sequence_group_to_sample in enumerate(
                sampling_metadata.seq_groups):
            top_k[index] = self._convert_to_neuron_top_k(
                sequence_group_to_sample.sampling_params.top_k)
            top_p[index] = sequence_group_to_sample.sampling_params.top_p
            temperature[index] = \
                sequence_group_to_sample.sampling_params.temperature

        self.model.model.update_generation_config(current_sampling_params)

    def _convert_to_neuron_top_k(self, top_k: int) -> int:
        if top_k < 0 or top_k > self._MAX_NEURON_SAMPLING_TOP_K:
            return self._MAX_NEURON_SAMPLING_TOP_K
        return top_k

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

        hidden_states = self.model(
            input_ids=model_input.input_tokens,
            positions=model_input.input_positions,
            input_block_ids=model_input.input_block_ids,
            **MultiModalInputs.as_kwargs(model_input.multi_modal_kwargs or {},
                                         device=self.device),
        )

        # Compute the logits only if the on-device sampling is turned off as
        # on-device sampling outputs the token ids.
        if self._on_device_sampling_disabled:
            logits = self.model.compute_logits(hidden_states,
                                               model_input.sampling_metadata)
        else:
            logits = hidden_states

        # Sample the next token.
        output = self.model.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )
        return [output]

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()
