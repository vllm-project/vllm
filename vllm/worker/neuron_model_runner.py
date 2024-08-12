from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from vllm.config import (DeviceConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.model_loader.neuron import get_neuron_model
from vllm.multimodal import (MULTIMODAL_REGISTRY, BatchedTensorInputs,
                             MultiModalInputs)
from vllm.sequence import (IntermediateTensors, SamplerOutput,
                           SequenceGroupMetadata)
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

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config

        if model_config is not None and model_config.get_sliding_window():
            logger.warning("Sliding window is not supported on Neuron. "
                           "The model will run without sliding window.")
        self.device_config = (device_config
                              if device_config is not None else DeviceConfig())
        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()

        # Multi-modal data support
        self.multi_modal_input_mapper = MULTIMODAL_REGISTRY \
            .create_input_mapper(self.model_config)

        # Lazy initialization.
        self.model: nn.Module  # initialize after load_model.

    def load_model(self) -> None:
        self.model = get_neuron_model(self.model_config,
                                      parallel_config=self.parallel_config,
                                      scheduler_config=self.scheduler_config)

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
                mm_kwargs = self.multi_modal_input_mapper(mm_data)
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
            seq_lens = []
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

        return ModelInputForNeuron(input_tokens=input_tokens,
                                   input_positions=input_positions,
                                   input_block_ids=input_block_ids,
                                   sampling_metadata=sampling_metadata,
                                   multi_modal_kwargs=multi_modal_kwargs)

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

        # Compute the logits.
        logits = self.model.compute_logits(hidden_states,
                                           model_input.sampling_metadata)

        # Sample the next token.
        output = self.model.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )
        return [output]

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()
