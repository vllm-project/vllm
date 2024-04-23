from typing import Dict, List, Optional, Set, Tuple

import torch

from vllm.attention import AttentionMetadata
from vllm.config import (DeviceConfig, LoRAConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, VisionLanguageConfig)
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.model_executor.parallel_utils.communication_op import (
    broadcast_tensor_dict)
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.pooling_params import PoolingParams
from vllm.sequence import PoolerOutput, SequenceData, SequenceGroupMetadata
from vllm.worker.model_runner import ModelRunner

logger = init_logger(__name__)


class EmbeddingModelRunner(ModelRunner):

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        lora_config: Optional[LoRAConfig],
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        vision_language_config: Optional[VisionLanguageConfig] = None,
    ):
        super().__init__(model_config,
                         parallel_config,
                         scheduler_config,
                         device_config,
                         lora_config=lora_config,
                         kv_cache_dtype=kv_cache_dtype,
                         is_driver_worker=is_driver_worker,
                         vision_language_config=vision_language_config)

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        kv_caches: List[torch.Tensor],
    ) -> Optional[PoolerOutput]:
        (input_tokens, input_positions, attn_metadata, pooling_metadata,
         lora_requests, lora_mapping, multi_modal_input
         ) = self.prepare_input_tensors(seq_group_metadata_list)

        if self.lora_config:
            self.set_active_loras(lora_requests, lora_mapping)

        # Execute the model.
        if attn_metadata.use_cuda_graph:
            graph_batch_size = input_tokens.shape[0]
            model_executable = self.graph_runners[graph_batch_size]
        else:
            model_executable = self.model

        num_layers = self.model_config.get_num_layers(self.parallel_config)
        kv_caches = [None] * num_layers

        execute_model_kwargs = {
            "input_ids": input_tokens,
            "positions": input_positions,
            "kv_caches": kv_caches,
            "attn_metadata": attn_metadata,
        }
        if self.vision_language_config:
            execute_model_kwargs.update({"image_input": multi_modal_input})
        hidden_states = model_executable(**execute_model_kwargs)

        return self.model.pooler(hidden_states=hidden_states,
                                 attention_metadata=attn_metadata)

    def prepare_input_tensors(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
    ) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata, PoolingMetadata,
               Set[int], LoRAMapping, torch.Tensor]:
        if self.is_driver_worker:
            # NOTE: We assume that all sequences in the group are all prompts or
            # all decodes.
            is_prompt = seq_group_metadata_list[0].is_prompt
            # Prepare input tensors.
            if is_prompt:
                (input_tokens, input_positions, attn_metadata, prompt_lens,
                 subquery_lens, lora_index_mapping, lora_prompt_mapping,
                 lora_requests, multi_modal_input
                 ) = self._prepare_prompt(seq_group_metadata_list)
            else:
                logger.warning(
                    "Embedding model should not have non-prompt inputs.")
                (input_tokens, input_positions, attn_metadata,
                 lora_index_mapping, lora_prompt_mapping,
                 lora_requests) = self._prepare_decode(seq_group_metadata_list)
                prompt_lens = []
                multi_modal_input = None
            pooling_metadata = self._prepare_pooling(seq_group_metadata_list,
                                                     prompt_lens)

            if self.lora_config:
                lora_mapping = LoRAMapping(
                    lora_index_mapping,
                    lora_prompt_mapping,
                )
            else:
                lora_mapping = None

            # Broadcast the metadata.
            metadata_dict = {
                "input_tokens": input_tokens,
                "input_positions": input_positions,
                "lora_requests": lora_requests,
                "lora_mapping": lora_mapping,
                "multi_modal_input": multi_modal_input,
                "prompt_lens": pooling_metadata.prompt_lens,
            }
            metadata_dict.update(attn_metadata.asdict_zerocopy())
            broadcast_tensor_dict(metadata_dict, src=0)
        else:
            metadata_dict = broadcast_tensor_dict(src=0)
            print(metadata_dict)
            input_tokens = metadata_dict.pop("input_tokens")
            input_positions = metadata_dict.pop("input_positions")
            lora_mapping = metadata_dict.pop("lora_mapping")
            lora_requests = metadata_dict.pop("lora_requests")
            multi_modal_input = metadata_dict.pop("multi_modal_input")
            attn_metadata = self.attn_backend.make_metadata(**metadata_dict)

            prompt_lens = metadata_dict.pop("prompt_lens")

            pooling_metadata = PoolingMetadata(seq_groups=None,
                                               seq_data=None,
                                               prompt_lens=None)

        return (input_tokens, input_positions, attn_metadata, pooling_metadata,
                lora_requests, lora_mapping, multi_modal_input)

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
