import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Type

import torch

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, MultiModalConfig, ParallelConfig,
                         PromptAdapterConfig, SchedulerConfig)
from vllm.logger import init_logger
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.pooling_params import PoolingParams
from vllm.sequence import (IntermediateTensors, PoolerOutput, SequenceData,
                           SequenceGroupMetadata)
from vllm.worker.model_runner import GPUModelRunnerBase, ModelInputForGPU

logger = init_logger(__name__)

@dataclasses.dataclass(frozen=True)
class EncoderDecoderModelInput(ModelInputForGPU):
    """
    Used by the EncoderDecoderModelRunner.
    """
    encoder_input_tokens: Optional[torch.Tensor] = None
    encoder_input_positions: Optional[torch.Tensor] = None

class EncoderDecoderModelRunner(
        GPUModelRunnerBase[EncoderDecoderModelInput]):
    _model_input_cls: Type[EncoderDecoderModelInput] = (
        EncoderDecoderModelInput)

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        prompt_adapter_config: Optional[PromptAdapterConfig] = None,
        multimodal_config: Optional[MultiModalConfig] = None,
    ):
        super().__init__(model_config,
                         parallel_config,
                         scheduler_config,
                         device_config,
                         cache_config,
                         load_config,
                         lora_config=lora_config,
                         kv_cache_dtype=kv_cache_dtype,
                         is_driver_worker=is_driver_worker,
                         prompt_adapter_config=prompt_adapter_config,
                         multimodal_config=multimodal_config)

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: EncoderDecoderModelInput,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[PoolerOutput]]:
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
            raise NotImplementedError("FlashInfer is currently not supported "
                                      "for encoder/decoder models.")

        # Currently cuda graph is not supported for encoder/decoder models
        assert model_input.attn_metadata is not None
        prefill_meta = model_input.attn_metadata.prefill_metadata
        decode_meta = model_input.attn_metadata.decode_metadata
        if prefill_meta is None and decode_meta.use_cuda_graph:
            raise NotImplementedError("CUDAGraph is currently not supported "
                                      "for encoder/decoder models.")
        # TODO(andoorve): We can remove this once all
        # virtual engines share the same kv cache.
        # virtual_engine = model_input.virtual_engine
        # if prefill_meta is None and decode_meta.use_cuda_graph:
        #     assert model_input.input_tokens is not None
        #     graph_batch_size = model_input.input_tokens.shape[0]
        #     model_executable = self.graph_runners[virtual_engine][
        #         graph_batch_size]
        # else:
        model_executable = self.model

        multi_modal_kwargs = model_input.multi_modal_kwargs or {}
        seqlen_agnostic_kwargs = {
            "finished_requests_ids": model_input.finished_requests_ids,
            "request_ids_to_seq_ids": model_input.request_ids_to_seq_ids,
        } if self.has_seqlen_agnostic else {}
        hidden_or_intermediate_states = model_executable(
            input_ids=model_input.input_tokens,
            positions=model_input.input_positions,
            encoder_input_ids=model_input.encoder_input_tokens,
            encoder_positions=model_input.encoder_input_positions,
            kv_caches=kv_caches,
            attn_metadata=model_input.attn_metadata,
            intermediate_tensors=intermediate_tensors,
            **multi_modal_kwargs,
            **seqlen_agnostic_kwargs)

        # Compute the logits in the last pipeline stage.
        if not get_pp_group().is_last_rank:
            return hidden_or_intermediate_states

        logits = self.model.compute_logits(hidden_or_intermediate_states,
                                           model_input.sampling_metadata)

        if not self.is_driver_worker:
            return []

        # Sample the next token.
        output: SamplerOutput = self.model.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )

        if self.return_hidden_states:
            # we only need to pass hidden states of most recent token
            assert model_input.sampling_metadata is not None
            indices = model_input.sampling_metadata.selected_token_indices
            if model_input.is_prompt:
                hidden_states = hidden_or_intermediate_states.index_select(
                    0, indices)
            elif decode_meta.use_cuda_graph:
                hidden_states = hidden_or_intermediate_states[:len(indices)]
            else:
                hidden_states = hidden_or_intermediate_states

            output.hidden_states = hidden_states

        return [output]

    def make_model_input_from_broadcasted_tensor_dict(
            self,
            tensor_dict: Dict[str,
                              Any]) -> ModelInputForGPUWithPoolingMetadata:
        return ModelInputForGPUWithPoolingMetadata.from_broadcasted_tensor_dict(
            tensor_dict,
            attn_backend=self.attn_backend,
        )

    def prepare_model_input(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None
    ) -> ModelInputForGPUWithPoolingMetadata:
        assert seq_group_metadata_list is not None
        model_input = self._prepare_model_input_tensors(
            seq_group_metadata_list, finished_requests_ids)
        # Prepare PoolingMetadata.
        assert model_input.seq_lens is not None
        pooling_metadata = self._prepare_pooling(seq_group_metadata_list,
                                                 model_input.seq_lens)

        return dataclasses.replace(model_input,
                                   pooling_metadata=pooling_metadata)

    def _prepare_pooling(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        prompt_lens: List[int],
    ) -> PoolingMetadata:
        """Prepare PoolingMetadata for the sequence group metadata list."""
        seq_groups: List[Tuple[List[int], PoolingParams]] = []
        for i, seq_group_metadata in enumerate(seq_group_metadata_list):
            seq_ids = list(seq_group_metadata.seq_data.keys())
            pooling_params = seq_group_metadata.pooling_params
            seq_groups.append((seq_ids, pooling_params))

        seq_data: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data.update(seq_group_metadata.seq_data)

        pooling_metadata = PoolingMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
        )

        return pooling_metadata
