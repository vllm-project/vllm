import dataclasses
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, cast

import torch

from vllm.attention import AttentionMetadata
from vllm.forward_context import set_forward_context
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.multimodal import MultiModalKwargs
from vllm.sequence import IntermediateTensors, SequenceGroupMetadata
from vllm.utils import make_tensor_with_pad
from vllm.worker.cpu_model_runner import (CPUModelRunnerBase,
                                          ModelInputForCPUBuilder,
                                          ModelInputForCPUWithSamplingMetadata)
from vllm.worker.model_runner_base import (
    _add_attn_metadata_broadcastable_dict,
    _add_sampling_metadata_broadcastable_dict)

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend


@dataclasses.dataclass(frozen=True)
class EncoderDecoderModelInputForCPU(ModelInputForCPUWithSamplingMetadata):
    """
    Used by the EncoderDecoderModelRunner.
    """
    encoder_input_tokens: Optional[torch.Tensor] = None
    encoder_input_positions: Optional[torch.Tensor] = None

    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
            "encoder_input_tokens": self.encoder_input_tokens,
            "encoder_input_positions": self.encoder_input_positions,
            "multi_modal_kwargs": self.multi_modal_kwargs,
        }
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)
        _add_sampling_metadata_broadcastable_dict(tensor_dict,
                                                  self.sampling_metadata)
        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
        cls,
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> "EncoderDecoderModelInputForCPU":
        return cast(
            EncoderDecoderModelInputForCPU,
            super().from_broadcasted_tensor_dict(tensor_dict, attn_backend))


class CPUEncoderDecoderModelRunner(
        CPUModelRunnerBase[EncoderDecoderModelInputForCPU]):
    _model_input_cls: Type[EncoderDecoderModelInputForCPU] = (
        EncoderDecoderModelInputForCPU)
    _builder_cls: Type[ModelInputForCPUBuilder] = ModelInputForCPUBuilder

    def _list_to_int32_tensor(
        self,
        _list: List[int],
    ) -> torch.Tensor:
        return torch.tensor(_list, dtype=torch.int32, device=self.device)

    def _list_to_long_tensor(
        self,
        _list: List[int],
    ) -> torch.Tensor:
        return torch.tensor(_list, dtype=torch.long, device=self.device)

    def _empty_int32_tensor(self) -> torch.Tensor:
        return self._list_to_int32_tensor([])

    def _empty_long_tensor(self) -> torch.Tensor:
        return self._list_to_long_tensor([])

    def make_model_input_from_broadcasted_tensor_dict(
            self, tensor_dict: Dict[str,
                                    Any]) -> EncoderDecoderModelInputForCPU:
        return EncoderDecoderModelInputForCPU.from_broadcasted_tensor_dict(
            tensor_dict,
            attn_backend=self.attn_backend,
        )

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None
    ) -> EncoderDecoderModelInputForCPU:
        model_input = self._prepare_model_input_tensors(
            seq_group_metadata_list, finished_requests_ids)
        (
            attn_metadata,
            encoder_input_tokens_tensor,
            encoder_input_positions_tensor,
        ) = self._prepare_encoder_model_input_tensors(seq_group_metadata_list,
                                                      model_input)
        # Sampling metadata is only required for the final pp group
        generators = self.get_generators(finished_requests_ids)
        sampling_metadata = SamplingMetadata.prepare(seq_group_metadata_list,
                                                     model_input.seq_lens,
                                                     model_input.query_lens,
                                                     self.device,
                                                     pin_memory=False,
                                                     generators=generators)
        return dataclasses.replace(
            model_input,
            sampling_metadata=sampling_metadata,
            attn_metadata=attn_metadata,
            encoder_input_tokens=encoder_input_tokens_tensor,
            encoder_input_positions=encoder_input_positions_tensor,
            virtual_engine=virtual_engine,
        )

    def _prepare_encoder_model_input_tensors(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        model_input: EncoderDecoderModelInputForCPU,
    ) -> Tuple[AttentionMetadata, Optional[torch.Tensor],
               Optional[torch.Tensor]]:
        """Helper method to prepare the encoder- and cross-attn-related
        model inputs based on a given sequence group. These additional inputs
        are used to augment an already-computed `EncoderDecoderModelInput`
        data structure which already has decoder-related model inputs
        populated.

        Sets the following attn_metadata fields:
        * `num_encoder_tokens`
        * `encoder_seq_lens`
        * `encoder_seq_lens_tensor`
        * `max_encoder_seq_len`
        * `cross_slot_mapping`
        * `cross_block_tables`

        Constructs a new model inputs data structure, based on
        (1) the existing fields in the `model_inputs` argument,
        and (2) the following additional fields which are
        computed (or in the case of `attn_metadata`, updated) 
        by this function:
        * attn_metadata
        * encoder_input_tokens
        * encoder_input_positions

        Arguments:

        * seq_group_metadata_list: list of sequence groups for which to
                                   compute inputs
        * model_inputs: model inputs data structure with decoder-oriented
                        fields already computed.

        Return:

        * Updated model inputs data structure
        """

        if len(seq_group_metadata_list) == 0:
            return (model_input.attn_metadata, None, None)

        # Since we are not supporting chunked prefill either the entire
        # batch is prefill or it is decode
        is_prompt = seq_group_metadata_list[0].is_prompt

        # Build encoder inputs
        encoder_seq_lens: List[int] = []
        if is_prompt:
            # Prefill phase.
            cross_block_tables = self._empty_int32_tensor().view(
                len(seq_group_metadata_list), -1)

            # Extract input tokens/positions, cross-attention slot-mapping,
            # & seq len from each sequence group metadata
            (
                encoder_input_tokens,
                encoder_input_positions,
                cross_slot_mapping,
            ) = (
                [],
                [],
                [],
            )
            for seq_group_metadata in seq_group_metadata_list:
                # Build seq lens
                seq_len = seq_group_metadata.encoder_seq_data.get_len()
                token_ids = seq_group_metadata.encoder_seq_data.get_token_ids()
                encoder_seq_lens.append(seq_len)

                # Build slot mapping
                for i in range(0, seq_len):
                    block_number = seq_group_metadata.cross_block_table[
                        i // self.block_size]
                    block_offset = i % self.block_size
                    slot = block_number * self.block_size + block_offset
                    cross_slot_mapping.append(slot)

                # Build encoder input tokens
                encoder_input_tokens.extend(token_ids)
                encoder_input_positions.extend(list(range(0, seq_len)))

            # Convert tokens/positions & cross-attention
            # slot-mapping to encoder input tensors
            encoder_input_tokens_tensor = self._list_to_long_tensor(
                encoder_input_tokens)
            encoder_input_positions_tensor = self._list_to_long_tensor(
                encoder_input_positions)
            cross_slot_mapping_tensor = self._list_to_long_tensor(
                cross_slot_mapping)

        else:
            # Decode phase.
            encoder_input_tokens_tensor = self._empty_long_tensor()
            encoder_input_positions_tensor = self._empty_long_tensor()
            cross_slot_mapping_tensor = self._empty_long_tensor()
            # Extract cross-attention block tables &
            # seq len from each sequence group metadata.
            # Cross-attention block tables are empty
            # during vLLM memory profiling.
            cross_block_tables = []
            for seq_group_metadata in seq_group_metadata_list:
                for _ in range(len(seq_group_metadata.seq_data)):
                    encoder_seq_lens.append(
                        seq_group_metadata.encoder_seq_data.get_len())
                    cross_block_table = seq_group_metadata.cross_block_table
                    cross_block_tables.append([] if (
                        cross_block_table is None) else cross_block_table)

            max_len_of_block_table = max(
                len(block_table) for block_table in cross_block_tables)

            cross_block_tables = make_tensor_with_pad(
                cross_block_tables,
                max_len=max_len_of_block_table,
                pad=0,
                dtype=torch.int32,
                device=self.device,
            )

        # Compute encoder sequence lengths & encoder
        # sequence starting offset tensors
        max_encoder_seq_len = max(encoder_seq_lens, default=0)
        encoder_seq_lens_tensor = self._list_to_int32_tensor(encoder_seq_lens)
        encoder_seq_start_loc = torch.zeros(encoder_seq_lens_tensor.shape[0] +
                                            1,
                                            dtype=torch.int32,
                                            device=self.device)
        torch.cumsum(encoder_seq_lens_tensor,
                     dim=0,
                     dtype=encoder_seq_start_loc.dtype,
                     out=encoder_seq_start_loc[1:])

        # Update attention metadata with encoder-oriented attributes
        attn_metadata = model_input.attn_metadata
        assert attn_metadata is not None
        (
            attn_metadata.num_encoder_tokens,
            attn_metadata.encoder_seq_lens,
            attn_metadata.encoder_seq_lens_tensor,
            attn_metadata.max_encoder_seq_len,
            attn_metadata.cross_slot_mapping,
            attn_metadata.cross_block_tables,
        ) = (
            sum(encoder_seq_lens),
            encoder_seq_lens,
            encoder_seq_lens_tensor,
            max_encoder_seq_len,
            cross_slot_mapping_tensor,
            cross_block_tables,
        )

        return (attn_metadata, encoder_input_tokens_tensor,
                encoder_input_positions_tensor)

    @torch.no_grad()
    def execute_model(
        self,
        model_input: EncoderDecoderModelInputForCPU,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:
        if num_steps > 1:
            raise ValueError(
                "CPU worker does not support multi-step execution.")

        model_executable = self.model
        execute_model_kwargs = {
            "input_ids":
            model_input.input_tokens,
            "positions":
            model_input.input_positions,
            "encoder_input_ids":
            model_input.encoder_input_tokens,
            "encoder_positions":
            model_input.encoder_input_positions,
            "kv_caches":
            kv_caches,
            "attn_metadata":
            model_input.attn_metadata,
            **MultiModalKwargs.as_kwargs(model_input.multi_modal_kwargs or {},
                                         device=self.device),
            "intermediate_tensors":
            intermediate_tensors,
        }

        with set_forward_context(model_input.attn_metadata, self.vllm_config,
                                 model_input.virtual_engine):
            hidden_states = model_executable(**execute_model_kwargs)

        # Compute the logits.
        logits = self.model.compute_logits(hidden_states,
                                           model_input.sampling_metadata)

        # Only perform sampling in the driver worker.
        if not self.is_driver_worker:
            return []

        # Sample the next token.
        output = self.model.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )
        return [output]
