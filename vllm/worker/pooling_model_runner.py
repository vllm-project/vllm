# SPDX-License-Identifier: Apache-2.0

import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.multimodal import MultiModalKwargs
from vllm.pooling_params import PoolingParams
from vllm.sequence import (IntermediateTensors, PoolerOutput, SequenceData,
                           SequenceGroupMetadata)
from vllm.worker.model_runner import (GPUModelRunnerBase, ModelInputForGPU,
                                      ModelInputForGPUBuilder)

logger = init_logger(__name__)


@dataclasses.dataclass(frozen=True)
class ModelInputForGPUWithPoolingMetadata(ModelInputForGPU):
    """
    Used by the PoolingModelRunner.
    """
    pooling_metadata: Optional["PoolingMetadata"] = None


class PoolingModelRunner(
        GPUModelRunnerBase[ModelInputForGPUWithPoolingMetadata]):
    _model_input_cls: Type[ModelInputForGPUWithPoolingMetadata] = (
        ModelInputForGPUWithPoolingMetadata)
    _builder_cls: Type[ModelInputForGPUBuilder] = ModelInputForGPUBuilder

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
    ):
        super().__init__(vllm_config=vllm_config,
                         kv_cache_dtype=kv_cache_dtype,
                         is_driver_worker=is_driver_worker)

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForGPUWithPoolingMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[Union[List[PoolerOutput], IntermediateTensors]]:
        if num_steps > 1:
            raise ValueError(
                "PoolingModelRunner does not support multi-step execution.")

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

        # Currently cuda graph is only supported by the decode phase.
        assert model_input.attn_metadata is not None
        prefill_meta = model_input.attn_metadata.prefill_metadata
        decode_meta = model_input.attn_metadata.decode_metadata
        virtual_engine = model_input.virtual_engine
        # Pooling models are (ab-)used also to integrate non text models that
        # are not autoregressive (PrithviGeosaptialMAE).
        # These model might not use attention and do not really have a prefill
        # and decode phase. The model input is processed in one shot and both
        # decode_metadata and prefill_metadata would be None for such models.
        # See the PlaceholderAttentionMetadata class.
        # TODO: Figure out if cuda_graph is of any use for these models and
        #  explore how to leverage it.
        if (prefill_meta is None and decode_meta is not None
                and decode_meta.use_cuda_graph):
            if model_input.inputs_embeds is None:
                assert model_input.input_tokens is not None
                graph_batch_size = model_input.input_tokens.shape[0]
                model_executable = (
                    self.graph_runners[model_input.virtual_engine][(
                        graph_batch_size, False)])
            else:
                graph_batch_size = model_input.inputs_embeds.shape[0]
                model_executable = (
                    self.graph_runners[model_input.virtual_engine][(
                        graph_batch_size, True)])
        else:
            model_executable = self.model

        multi_modal_kwargs = model_input.multi_modal_kwargs or {}
        seqlen_agnostic_kwargs = {
            "finished_requests_ids": model_input.finished_requests_ids,
            "request_ids_to_seq_ids": model_input.request_ids_to_seq_ids,
        } if self.has_inner_state else {}
        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time):
            model_forward_start = torch.cuda.Event(enable_timing=True)
            model_forward_end = torch.cuda.Event(enable_timing=True)
            model_forward_start.record()

        cross_enc_kwargs = {}
        if model_input.token_types is not None:
            cross_enc_kwargs["token_type_ids"] = model_input.token_types

        with set_forward_context(model_input.attn_metadata, self.vllm_config,
                                 virtual_engine):
            hidden_or_intermediate_states = model_executable(
                input_ids=model_input.input_tokens,
                positions=model_input.input_positions,
                intermediate_tensors=intermediate_tensors,
                **MultiModalKwargs.as_kwargs(multi_modal_kwargs,
                                             device=self.device),
                **cross_enc_kwargs,
                **seqlen_agnostic_kwargs)

        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time):
            model_forward_end.record()

        # Only perform pooling in the last pipeline stage.
        if not get_pp_group().is_last_rank:
            if (self.is_driver_worker
                    and hidden_or_intermediate_states is not None
                    and isinstance(hidden_or_intermediate_states,
                                   IntermediateTensors)
                    and self.observability_config is not None
                    and self.observability_config.collect_model_forward_time):
                model_forward_end.synchronize()
                model_forward_time = model_forward_start.elapsed_time(
                    model_forward_end)
                orig_model_forward_time = 0.0
                if intermediate_tensors is not None:
                    orig_model_forward_time = intermediate_tensors.tensors.get(
                        "model_forward_time", torch.tensor(0.0)).item()
                hidden_or_intermediate_states.tensors["model_forward_time"] = (
                    torch.tensor(model_forward_time + orig_model_forward_time))
            return hidden_or_intermediate_states

        # Only perform pooling in the driver worker.
        if not self.is_driver_worker:
            return []

        return [
            self.model.pooler(hidden_states=hidden_or_intermediate_states,
                              pooling_metadata=model_input.pooling_metadata)
        ]

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
