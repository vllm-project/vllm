from typing import Any, Dict, List, Optional, Type

import torch

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ObservabilityConfig, ParallelConfig,
                         PromptAdapterConfig, SchedulerConfig)
from vllm.logger import init_logger
from vllm.sequence import (IntermediateTensors, SequenceGroupMetadata,
                           SimpleOutput)
from vllm.worker.model_runner import (GPUModelRunnerBase, ModelInputForGPU,
                                      ModelInputForGPUBuilder)

logger = init_logger(__name__)


class SimpleModelRunner(GPUModelRunnerBase[ModelInputForGPU]):
    _model_input_cls: Type[ModelInputForGPU] = (ModelInputForGPU)
    _builder_cls: Type[ModelInputForGPUBuilder] = ModelInputForGPUBuilder

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
        observability_config: Optional[ObservabilityConfig] = None,
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
                         observability_config=observability_config)

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForGPU,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SimpleOutput]]:
        if num_steps > 1:
            raise ValueError(
                "SimpleModelRunner does not support multi-step execution.")

        if self.lora_config:
            assert model_input.lora_requests is not None
            assert model_input.lora_mapping is not None
            self.set_active_loras(model_input.lora_requests,
                                  model_input.lora_mapping)

        # Currently cuda graph is only supported by the decode phase.
        assert model_input.attn_metadata is not None
        prefill_meta = model_input.attn_metadata.prefill_metadata
        decode_meta = model_input.attn_metadata.decode_metadata
        virtual_engine = model_input.virtual_engine
        if prefill_meta is None and decode_meta.use_cuda_graph:
            assert model_input.input_tokens is not None
            graph_batch_size = model_input.input_tokens.shape[0]
            model_executable = self.graph_runners[virtual_engine][
                graph_batch_size]
        else:
            model_executable = self.model

        num_layers = self.model_config.get_num_layers(self.parallel_config)
        kv_caches = [None] * num_layers

        execute_model_kwargs = {
            "input_ids": model_input.input_tokens,
            "positions": model_input.input_positions,
            "kv_caches": kv_caches,
            "attn_metadata": model_input.attn_metadata,
            **(model_input.multi_modal_kwargs or {}),
        }

        hidden_states = model_executable(**execute_model_kwargs)
        if not self.is_driver_worker:
            return []

        return [hidden_states]

    def make_model_input_from_broadcasted_tensor_dict(
            self, tensor_dict: Dict[str, Any]) -> ModelInputForGPU:
        return ModelInputForGPU.from_broadcasted_tensor_dict(
            tensor_dict,
            attn_backend=self.attn_backend,
        )

    def prepare_model_input(
            self,
            seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
            virtual_engine: int = 0,
            finished_requests_ids: Optional[List[str]] = None
    ) -> ModelInputForGPU:
        assert seq_group_metadata_list is not None
        model_input = self._prepare_model_input_tensors(
            seq_group_metadata_list, finished_requests_ids)
        assert model_input.seq_lens is not None
        return model_input
