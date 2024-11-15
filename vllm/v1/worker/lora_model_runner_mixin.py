"""
Define LoRA adapter for model runner.
"""

from typing import List

from vllm.v1.worker.lora_request_batch import LoRARequestBatch

from vllm.model_executor.models import supports_lora, supports_multimodal
from vllm.logger import init_logger

from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.v1.core.scheduler import SchedulerOutput

import torch
import numpy as np
import torch.nn as nn

logger = init_logger(__name__)

# Defined as a mixin for GPUModelRunner
class LoRAModelRunnerMixin:

    # TODO (varun) : self is untyped. This has ide code completion issues and
    # could potentially lead to bugs. 
    def load_lora_model(self) -> nn.Module:

        assert supports_lora(
            self.model
        ), f"{self.model.__class__.__name__} does not support LoRA yet."

        if supports_multimodal(self.model):
            logger.warning("Regarding multimodal models, vLLM currently "
                           "only supports adding LoRA to language model.")
        # It's necessary to distinguish between the max_position_embeddings
        # of VLMs and LLMs.
        if hasattr(self.model.config, "max_position_embeddings"):
            max_pos_embeddings = self.model.config.max_position_embeddings
        else:
            max_pos_embeddings = (
                self.model.config.text_config.max_position_embeddings)

        self.lora_manager = LRUCacheWorkerLoRAManager(
            self.scheduler_config.max_num_seqs,
            self.scheduler_config.max_num_batched_tokens,
            self.model_config.get_vocab_size(),
            self.lora_config,
            self.device,
            self.model.embedding_modules,
            self.model.embedding_padding_modules,
            max_position_embeddings=max_pos_embeddings,
        )
        return self.lora_manager.create_lora_manager(self.model)

    def set_activte_loras(self,
                          request_batch: LoRARequestBatch,
                          num_scheduled_tokens: np.array) -> None:

        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")

        lora_mapping, lora_request_ids = request_batch.prepare_lora_inputs(num_scheduled_tokens)

        # Get lora requests
        # TODO (varun) : LoRA request batch assumes that the adapters are uniquely identifiable based
        # on the lora ID --- Use that here for better or worse !
        lora_requests = map(lambda req_id: self.requests[req_id].lora_request ,lora_request_ids) 
        lora_requests = filter(lambda x: x is not None, lora_requests)
        lora_requests: set[LoRARequest] = set(lora_requests)

        self.lora_manager.set_active_adapters(lora_requests, lora_mapping)
