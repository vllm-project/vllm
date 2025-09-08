# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Define LoRA functionality mixin for model runners.
"""

from contextlib import contextmanager
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from vllm.config import LoRAConfig, ModelConfig, SchedulerConfig
from vllm.forward_context import ALoRAMetadata
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.model_executor.models import supports_lora, supports_multimodal
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.gpu_input_batch import InputBatch as GPUInputBatch
from vllm.v1.worker.tpu_input_batch import InputBatch as TPUInputBatch

InputBatch = Union[TPUInputBatch, GPUInputBatch]

logger = init_logger(__name__)


# Defined as a mixin for GPUModelRunner
class LoRAModelRunnerMixin:

    LORA_WARMUP_RANK = 8

    def load_lora_model(self, model: nn.Module, model_config: ModelConfig,
                        scheduler_config: SchedulerConfig,
                        lora_config: LoRAConfig,
                        device: torch.device) -> nn.Module:

        if not supports_lora(model):
            raise ValueError(
                f"{model.__class__.__name__} does not support LoRA yet.")

        if supports_multimodal(model):
            logger.warning("Regarding multimodal models, vLLM currently "
                           "only supports adding LoRA to language model.")

        # Use get_text_config() in case of multimodal models
        text_config = model_config.hf_config.get_text_config()

        # Add LoRA Manager to the Model Runner
        self.lora_manager = LRUCacheWorkerLoRAManager(
            scheduler_config.max_num_seqs,
            scheduler_config.max_num_batched_tokens,
            model_config.get_vocab_size(),
            lora_config,
            device,
            model.embedding_modules,
            model.embedding_padding_modules,
            max_position_embeddings=text_config.max_position_embeddings,
        )
        return self.lora_manager.create_lora_manager(model)

    def _set_active_loras(self, prompt_lora_mapping: tuple[int, ...],
                          token_lora_mapping: tuple[int, ...],
                          lora_requests: set[LoRARequest]) -> None:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")

        # Set is_prefill to True, so we always use the SGMV kernels on
        # non-cuda platforms.
        # On cuda platforms we use the same kernels for prefill and
        # decode and this flag is generally ignored.
        lora_mapping = LoRAMapping(token_lora_mapping,
                                   prompt_lora_mapping,
                                   is_prefill=True)
        self.lora_manager.set_active_adapters(lora_requests, lora_mapping)

    def set_active_loras(self, input_batch: InputBatch,
                         num_scheduled_tokens: np.ndarray) -> None:

        prompt_lora_mapping: tuple[int, ...]  # of size input_batch.num_reqs
        token_lora_mapping: tuple[int,
                                  ...]  # of size np.sum(num_scheduled_tokens)
        lora_requests: set[LoRARequest]
        prompt_lora_mapping, token_lora_mapping, lora_requests = \
                            input_batch.make_lora_inputs(num_scheduled_tokens)
        return self._set_active_loras(prompt_lora_mapping, token_lora_mapping,
                                      lora_requests)

    def build_alora_metadata(self, num_reqs: int, positions_np: np.ndarray,
                             req_indices: np.ndarray,
                             total_num_scheduled_tokens: int,
                             input_batch: InputBatch,
                             requests: dict[str, CachedRequestState],
                             mask1d: torch.Tensor) -> ALoRAMetadata:
        invocation_start = np.empty(shape=(num_reqs, ), dtype=int)
        for req_id in input_batch.req_ids:
            req_index = input_batch.req_id_to_index[req_id]
            cached_lora_request = requests[req_id].lora_request
            if (cached_lora_request is not None
                    and cached_lora_request.invocation_start is not None):
                invocation_start[
                    req_index] = cached_lora_request.invocation_start
            else:
                invocation_start[req_index] = len(
                    requests[req_id].prompt_token_ids)
        mask1d_cpu = torch.tensor(positions_np < invocation_start[req_indices],
                                  dtype=torch.bool,
                                  device="cpu")
        mask1d = mask1d[:total_num_scheduled_tokens]
        mask1d.copy_(mask1d_cpu, non_blocking=True)
        return ALoRAMetadata(mask1d=mask1d)

    def build_dummy_alora_metadata(self, num_tokens: int,
                                   mask1d: torch.tensor):
        alora_metadata = ALoRAMetadata(mask1d=mask1d[:num_tokens])
        # needed to avoid guard failures
        torch._dynamo.mark_dynamic(alora_metadata.mask1d, 0)
        return alora_metadata

    @contextmanager
    def maybe_setup_dummy_loras(self,
                                lora_config: Optional[LoRAConfig],
                                remove_lora: bool = True):
        if lora_config is None:
            yield
        else:
            # __enter__ code
            assert self.lora_manager is not None, "LoRA is not enabled"

            num_loras = lora_config.max_loras

            # Make dummy lora requests
            lora_requests: set[LoRARequest] = {
                LoRARequest(lora_name=f"warmup_{lora_id}",
                            lora_int_id=lora_id,
                            lora_path="/not/a/real/path")
                for lora_id in range(1, num_loras + 1)
            }

            with self.lora_manager.dummy_lora_cache():
                # Add the dummy LoRAs here so _set_active_loras doesn't try to
                # load from disk.
                for lr in lora_requests:
                    self.lora_manager.add_dummy_lora(
                        lr, rank=self.LORA_WARMUP_RANK)

                yield

            # __exit__ code
            if remove_lora:
                self.lora_manager.remove_all_adapters()

    @contextmanager
    def maybe_select_dummy_loras(self, lora_config: Optional[LoRAConfig],
                                 num_scheduled_tokens: np.ndarray):
        if lora_config is None:
            yield
        else:
            # __enter__ code
            assert self.lora_manager is not None, "LoRA is not enabled"

            num_reqs = len(num_scheduled_tokens)
            num_loras = lora_config.max_loras

            # Make prompt lora mapping
            # Assign LoRA IDs cyclically to simulate a worst-case scenario.
            prompt_lora_mapping = (np.arange(num_reqs, dtype=np.int32) %
                                   num_loras) + 1

            # Make token lora mapping
            token_lora_mapping = np.repeat(prompt_lora_mapping,
                                           num_scheduled_tokens)

            # Make dummy lora requests
            lora_requests: set[LoRARequest] = {
                LoRARequest(lora_name=f"warmup_{lora_id}",
                            lora_int_id=lora_id,
                            lora_path="/not/a/real/path")
                for lora_id in range(1, num_loras + 1)
            }

            self._set_active_loras(tuple(prompt_lora_mapping),
                                   tuple(token_lora_mapping), lora_requests)

            yield

    @contextmanager
    def maybe_dummy_run_with_lora(self,
                                  lora_config: Optional[LoRAConfig],
                                  num_scheduled_tokens: np.ndarray,
                                  remove_lora: bool = True):
        with (
                self.maybe_setup_dummy_loras(lora_config, remove_lora),
                self.maybe_select_dummy_loras(lora_config,
                                              num_scheduled_tokens),
        ):
            yield

    def maybe_remove_all_loras(self, lora_config: Optional[LoRAConfig]):
        if lora_config is None:
            return
        self.lora_manager.remove_all_adapters()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.add_adapter(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.remove_adapter(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.pin_adapter(lora_id)

    def list_loras(self) -> set[int]:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.list_adapters()
