# import dataclasses
from typing import List, Optional, Union

import torch

from vllm.distributed import get_pp_group
from vllm.multimodal import MultiModalInputs
# from vllm.utils import make_tensor_with_pad
# from vllm.model_executor import SamplingMetadata
from vllm.sequence import IntermediateTensors, SamplerOutput #, SequenceGroupMetadata
from vllm.worker.model_runner import (ModelRunner, ModelInputForGPUWithSamplingMetadata, #GPUModelRunnerBase, ModelInputForGPUBuilder,
                                      FLASHINFER_WORKSPACE_BUFFER_SIZE, BatchDecodeWithPagedKVCacheWrapper, # _PAD_SLOT_ID,
                                      BatchPrefillWithPagedKVCacheWrapper)
# from vllm.worker.model_runner_base import (
#     _add_attn_metadata_broadcastable_dict,
#     _add_sampling_metadata_broadcastable_dict,
#     _init_attn_metadata_from_tensor_dict,
#     _init_sampling_metadata_from_tensor_dict)

# if TYPE_CHECKING:
#     from vllm.attention.backends.abstract import AttentionBackend


class CFGModelRunner(ModelRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.inference_mode()
    def model_execute(
        self,
        model_input: ModelInputForGPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> torch.Tensor:
        if num_steps > 1:
            raise ValueError("num_steps > 1 is not supported in ModelRunner")

        if self.lora_config:
            assert model_input.lora_requests is not None
            assert model_input.lora_mapping is not None
            self.set_active_loras(model_input.lora_requests,
                                  model_input.lora_mapping)

        if self.prompt_adapter_config:
            assert model_input.prompt_adapter_requests is not None
            assert model_input.prompt_adapter_mapping is not None
            self.set_active_prompt_adapters(
                model_input.prompt_adapter_requests,
                model_input.prompt_adapter_mapping)

        if self.attn_backend.get_name() == "flashinfer":
            assert model_input.attn_metadata is not None
            assert model_input.input_tokens is not None
            if self.flashinfer_decode_workspace_buffer is None:
                self.flashinfer_decode_workspace_buffer = torch.empty(
                    FLASHINFER_WORKSPACE_BUFFER_SIZE,
                    dtype=torch.uint8,
                    device=self.device)
                self.flashinfer_decode_wrapper = \
                    BatchDecodeWithPagedKVCacheWrapper(
                    self.flashinfer_decode_workspace_buffer, "NHD")
                self.flashinfer_prefill_workspace_buffer = torch.empty(
                    FLASHINFER_WORKSPACE_BUFFER_SIZE,
                    dtype=torch.uint8,
                    device=self.device)
                self.flashinfer_prefill_wrapper = \
                    BatchPrefillWithPagedKVCacheWrapper(
                    self.flashinfer_prefill_workspace_buffer, "NHD")

            model_input.attn_metadata.prefill_wrapper = \
                self.flashinfer_prefill_wrapper
            if model_input.attn_metadata.use_cuda_graph:
                batch_size = model_input.input_tokens.shape[0]
                model_input.attn_metadata.decode_wrapper = self.graph_runners[
                    model_input.
                    virtual_engine][batch_size].flashinfer_decode_wrapper
            else:
                model_input.attn_metadata.decode_wrapper = \
                    self.flashinfer_decode_wrapper
            model_input.attn_metadata.begin_forward()

        # Currently cuda graph is only supported by the decode phase.
        assert model_input.attn_metadata is not None
        prefill_meta = model_input.attn_metadata.prefill_metadata
        decode_meta = model_input.attn_metadata.decode_metadata
        # TODO(andoorve): We can remove this once all
        # virtual engines share the same kv cache.
        virtual_engine = model_input.virtual_engine
        if prefill_meta is None and decode_meta.use_cuda_graph:
            assert model_input.input_tokens is not None
            graph_batch_size = model_input.input_tokens.shape[0]
            model_executable = self.graph_runners[virtual_engine][
                graph_batch_size]
        else:
            model_executable = self.model

        multi_modal_kwargs = model_input.multi_modal_kwargs or {}
        seqlen_agnostic_kwargs = {
            "finished_requests_ids": model_input.finished_requests_ids,
            "request_ids_to_seq_ids": model_input.request_ids_to_seq_ids,
        } if self.has_seqlen_agnostic else {}
        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time):
            raise NotImplementedError("")

        hidden_or_intermediate_states = model_executable(
            input_ids=model_input.input_tokens,
            positions=model_input.input_positions,
            kv_caches=kv_caches,
            attn_metadata=model_input.attn_metadata,
            intermediate_tensors=intermediate_tensors,
            **MultiModalInputs.as_kwargs(multi_modal_kwargs,
                                         device=self.device),
            **seqlen_agnostic_kwargs)

        return hidden_or_intermediate_states

    @torch.inference_mode()
    def get_logits(
        self,
        hidden_or_intermediate_states: torch.Tensor,
        model_input: ModelInputForGPUWithSamplingMetadata,
    ) -> torch.Tensor:
        return self.model._get_logits(hidden_or_intermediate_states, 
                                      model_input.sampling_metadata)

    @torch.inference_mode()
    def compute_logits(
        self,
        logits: torch.Tensor,
        model_input: ModelInputForGPUWithSamplingMetadata,
    ) -> torch.Tensor:
        return self.model.compute_logits(logits,
                                         model_input.sampling_metadata)

    @torch.inference_mode()
    def do_sample(
        self,
        logits: torch.Tensor,
        model_input: ModelInputForGPUWithSamplingMetadata,
    ):
        if not self.is_driver_worker:
            return []

        # Sample the next token.
        output: SamplerOutput = self.model.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )

        if self.return_hidden_states:
            raise NotImplementedError("return_hidden_states is not supported in CFGModelRunner")

        return [output]

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForGPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[Union[List[SamplerOutput], IntermediateTensors]]:

        hidden_or_intermediate_states = self.model_execute(model_input, kv_caches, intermediate_tensors, num_steps)

        if not get_pp_group().is_last_rank:
            return hidden_or_intermediate_states

        hidden_or_intermediate_states = self.get_logits(hidden_or_intermediate_states, model_input)
        logits = self.compute_logits(hidden_or_intermediate_states, model_input)

        return self.do_sample(logits, model_input)


# @dataclasses.dataclass(frozen=True)
# class PositiveNegativeModelInput(ModelInputForGPUWithSamplingMetadata):
#     """
#     Used by the ClassifierFreeGuidanceModelRunner.
#     """
#     negative_input_tokens: Optional[torch.Tensor] = None
#     negative_input_positions: Optional[torch.Tensor] = None

#     def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
#         tensor_dict = {
#             "input_tokens": self.input_tokens,
#             "input_positions": self.input_positions,
#             "negative_input_tokens": self.negative_input_tokens,
#             "negative_input_positions": self.negative_input_positions,
#             "virtual_engine": self.virtual_engine,
#             "request_ids_to_seq_ids": self.request_ids_to_seq_ids,
#             "finished_requests_ids": self.finished_requests_ids,
#         }
#         _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)
#         _add_sampling_metadata_broadcastable_dict(tensor_dict,
#                                                   self.sampling_metadata)
#         return tensor_dict

#     @classmethod
#     def from_broadcasted_tensor_dict(
#         cls,
#         tensor_dict: Dict[str, Any],
#         attn_backend: Optional["AttentionBackend"] = None,
#     ) -> "ModelInputForGPUWithSamplingMetadata":
#         return cast(
#             ModelInputForGPUWithSamplingMetadata,
#             super().from_broadcasted_tensor_dict(tensor_dict, attn_backend))


# class ClassifierFreeGuidanceModelRunner(GPUModelRunnerBase[PositiveNegativeModelInput]):
#     _model_input_cls: Type[PositiveNegativeModelInput] = (
#         PositiveNegativeModelInput)
#     _builder_cls: Type[ModelInputForGPUBuilder] = (ModelInputForGPUBuilder)

#     @torch.inference_mode()
#     def execute_model(
#         self,
#         model_input: PositiveNegativeModelInput,
#         kv_caches: List[torch.Tensor],
#         intermediate_tensors: Optional[IntermediateTensors] = None,
#         num_steps: int = 1,
#     ) -> Optional[Union[List[SamplerOutput], IntermediateTensors]]:

#         if num_steps > 1:
#             raise ValueError("num_steps > 1 is not supported in ModelRunner")




#     def prepare_model_input(
#         self,
#         seq_group_metadata_list: List[SequenceGroupMetadata],
#         virtual_engine: int = 0,
#         finished_requests_ids: Optional[List[str]] = None
#     ) -> PositiveNegativeModelInput:

#         model_input = self._prepare_model_input_tensors(
#             seq_group_metadata_list, finished_requests_ids)

#         (
#             attn_metadata,
#             negative_input_tokens_tensor,
#             negative_input_positions_tensor,
#         ) = (self._prepare_model_negative_input_tensors(seq_group_metadata_list,
#                                                         model_input))

#         model_input = dataclasses.replace(
#             model_input,
#             attn_metadata=attn_metadata,
#             negative_input_tokens=negative_input_tokens_tensor,
#             negative_input_positions=negative_input_positions_tensor,
#         )

#         sampling_metadata = SamplingMetadata.prepare(seq_group_metadata_list,
#                                                      model_input.seq_lens,
#                                                      model_input.query_lens,
#                                                      self.device,
#                                                      self.pin_memory)
#         is_prompt = (seq_group_metadata_list[0].is_prompt
#                      if seq_group_metadata_list else None)
#         return dataclasses.replace(model_input,
#                                    sampling_metadata=sampling_metadata,
#                                    is_prompt=is_prompt,
#                                    virtual_engine=virtual_engine)

#     def _prepare_model_negative_input_tensors(
#         self,
#         seq_group_metadata_list: List[SequenceGroupMetadata],
#         model_input: PositiveNegativeModelInput,
#     ):
#         if len(seq_group_metadata_list) == 0:
#             return (model_input.attn_metadata, None, None)
        
#         is_prompt = seq_group_metadata_list[0].is_prompt

#         negative_seq_lens: List[int] = []
#         if is_prompt:
#             # Prefill phase.
#             negative_block_tables = self._empty_int32_tensor().view(
#                 len(seq_group_metadata_list), -1)

#             (
#                 negative_input_tokens,
#                 negative_input_positions,
#                 negative_slot_mapping,
#             ) = (
#                 [],
#                 [],
#                 [],
#             )

#             for seq_group_metadata in seq_group_metadata_list:
#                 seq_len = seq_group_metadata.negative_seq_data.get_len()
#                 token_ids = seq_group_metadata.negative_seq_data.get_token_ids()
#                 negative_seq_lens.append(seq_len)

#                 is_profile_run = (seq_group_metadata.block_tables is None)
#                 if is_profile_run:
#                     negative_slot_mapping.extend([_PAD_SLOT_ID] * seq_len)
#                 else:
#                     for i in range(0, seq_len):
#                         block_number = seq_group_metadata.negative_block_table[
#                             i // self.block_size]
#                         block_offset = i % self.block_size
#                         slot = block_number * self.block_size + block_offset
#                         negative_slot_mapping.append(slot)

#                 negative_input_tokens.extend(token_ids)
#                 negative_input_positions.extend(list(range(0, seq_len)))

#             negative_input_tokens_tensor = self._list_to_long_tensor(
#                 negative_input_tokens)
#             negative_input_positions_tensor = self._list_to_long_tensor(
#                 negative_input_positions)
#             negative_slot_mapping_tensor = self._list_to_long_tensor(
#                 negative_slot_mapping)
#         else:
#             # Decode phase.
#             negative_input_tokens_tensor = self._empty_long_tensor()
#             negative_input_positions_tensor = self._empty_long_tensor()
#             negative_slot_mapping_tensor = self._empty_long_tensor()

#             negative_block_tables = []
#             for seq_group_metadata in seq_group_metadata_list:
#                 negative_seq_lens.append(
#                     seq_group_metadata.negative_seq_data.get_len())
#                 negative_block_table = seq_group_metadata.negative_block_table
#                 negative_block_tables.append([] if (
#                     negative_block_table is None) else negative_block_table)

#             negative_block_tables = make_tensor_with_pad(
#                 negative_block_tables,
#                 max_len=max(
#                     len(block_table) for block_table in negative_block_tables),
#                 pad=0,
#                 dtype=torch.int32,
#                 device=self.device,
#             )

#         max_negative_seq_len = max(negative_seq_lens, default=0)
#         negative_seq_lens_tensor = self._list_to_int32_tensor(negative_seq_lens)
#         negative_seq_start_loc = torch.zeros(negative_seq_lens_tensor.shape[0] +
#                                             1,
#                                             dtype=torch.int32,
#                                             device=self.device)
#         torch.cumsum(negative_seq_lens_tensor,
#                      dim=0,
#                      dtype=negative_seq_start_loc.dtype,
#                      out=negative_seq_start_loc[1:])

#         attn_metadata = model_input.attn_metadata
#         assert attn_metadata is not None
#         (
#             attn_metadata.num_negative_tokens,
#             attn_metadata.negative_seq_lens,
#             attn_metadata.negative_seq_lens_tensor,
#             attn_metadata.max_negative_seq_len,
#             attn_metadata.negative_slot_mapping,
#             attn_metadata.negative_block_tables,
#         ) = (
#             sum(negative_seq_lens),
#             negative_seq_lens,
#             negative_seq_lens_tensor,
#             max_negative_seq_len,
#             negative_slot_mapping_tensor,
#             negative_block_tables,
#         )

#         return (attn_metadata, negative_input_tokens_tensor,
#                 negative_input_positions_tensor)
