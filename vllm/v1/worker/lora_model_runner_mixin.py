# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Define LoRA functionality mixin for model runners.
"""

from contextlib import contextmanager
from typing import TypeAlias

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.lora import LoRAConfig
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping, LoRAMappingType
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.model_executor.models import supports_lora
from vllm.tokenformer.tokenformer_model_manager import TokenformerModelManager
from vllm.v1.worker.gpu_input_batch import InputBatch as GPUInputBatch
from vllm.v1.worker.tpu_input_batch import InputBatch as TPUInputBatch

InputBatch: TypeAlias = TPUInputBatch | GPUInputBatch

logger = init_logger(__name__)


# Defined as a mixin for GPUModelRunner
class LoRAModelRunnerMixin:
    def load_lora_model(
        self,
        model: nn.Module,
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> nn.Module:
        if not supports_lora(model):
            raise ValueError(
                f"{model.__class__.__name__} does not support LoRA yet.")

        self.lora_manager = TokenformerModelManager(model=model,
                                                    device=device)

        logger.info("Created LoRA manager for model "
                    f"{model.__class__.__name__} with device {device}.")

        return self.lora_manager.model


    def _set_active_loras(
        self,
        prompt_lora_mapping: tuple[int, ...],
        token_lora_mapping: tuple[int, ...],
        lora_requests: set[LoRARequest],
        mapping_type: LoRAMappingType = LoRAMappingType.LANGUAGE,
    ) -> None:
        self._ensure_lora_enabled()
        if self.lora_manager is None:
            raise RuntimeError("LoRA is not enabled.")

        # Set is_prefill to True, so we always use the SGMV kernels on
        # non-cuda platforms.
        # On cuda platforms we use the same kernels for prefill and
        # decode and this flag is generally ignored.
        lora_mapping = LoRAMapping(
            token_lora_mapping,
            prompt_lora_mapping,
            is_prefill=True,
            type=mapping_type,
        )
        self.lora_manager.set_active_adapters(lora_requests, lora_mapping)

    def _ensure_lora_enabled(self) -> None:
        if not hasattr(self, "lora_manager"):
            raise RuntimeError("LoRA is not enabled. Use --enable-lora to enable LoRA.")

    def set_active_loras(
        self,
        input_batch: InputBatch,
        num_scheduled_tokens: np.ndarray,
        num_sampled_tokens: np.ndarray | None = None,
        mapping_type: LoRAMappingType = LoRAMappingType.LANGUAGE,
    ) -> None:
        if num_sampled_tokens is None:
            num_sampled_tokens = np.ones_like(num_scheduled_tokens, dtype=np.int32)

        prompt_lora_mapping: tuple[int, ...]  # of size np.sum(num_sampled_tokens)
        token_lora_mapping: tuple[int, ...]  # of size np.sum(num_scheduled_tokens)
        lora_requests: set[LoRARequest]
        prompt_lora_mapping, token_lora_mapping, lora_requests = (
            input_batch.make_lora_inputs(num_scheduled_tokens, num_sampled_tokens)
        )
        return self._set_active_loras(
            prompt_lora_mapping, token_lora_mapping, lora_requests, mapping_type
        )

    @contextmanager
    def maybe_setup_dummy_loras(
        self, lora_config: LoRAConfig | None, remove_lora: bool = True
    ):
        if lora_config is None:
            yield
        else:
            # __enter__ code
            assert self.lora_manager is not None, "LoRA is not enabled"

            num_loras = lora_config.max_loras
            lora_warmup_rank = (
                lora_config.max_lora_rank if lora_config.max_lora_rank < 8 else 8
            )
            # Make dummy lora requests
            lora_requests: set[LoRARequest] = {
                LoRARequest(
                    lora_name=f"warmup_{lora_id}",
                    lora_int_id=lora_id,
                    lora_path="/not/a/real/path",
                )
                for lora_id in range(1, num_loras + 1)
            }

            with self.lora_manager.dummy_lora_cache():
                # Add the dummy LoRAs here so _set_active_loras doesn't try to
                # load from disk.
                for lr in lora_requests:
                    self.lora_manager.add_dummy_lora(lr, rank=lora_warmup_rank)

                yield

            # __exit__ code
            if remove_lora:
                self.lora_manager.remove_all_adapters()

    @contextmanager
    def maybe_select_dummy_loras(
        self,
        lora_config: LoRAConfig | None,
        num_scheduled_tokens: np.ndarray,
        mapping_type: LoRAMappingType = LoRAMappingType.LANGUAGE,
        num_sampled_tokens: np.ndarray | None = None,
        activate_lora: bool = True,
    ):
        if num_sampled_tokens is None:
            num_sampled_tokens = np.ones_like(num_scheduled_tokens, dtype=np.int32)

        if lora_config is None:
            yield
        else:
            # __enter__ code
            assert self.lora_manager is not None, "LoRA is not enabled"

            num_reqs = len(num_scheduled_tokens)
            num_loras = lora_config.max_loras

            # Make prompt lora mapping
            # Assign LoRA IDs cyclically to simulate a worst-case scenario.
            if activate_lora:
                prompt_lora_mapping = (
                    np.arange(num_reqs, dtype=np.int32) % num_loras
                ) + 1
            else:
                prompt_lora_mapping = np.zeros(num_reqs, dtype=np.int32)

            # Make sample lora mapping
            sample_lora_mapping = np.repeat(prompt_lora_mapping, num_sampled_tokens)

            # Make token lora mapping
            token_lora_mapping = np.repeat(prompt_lora_mapping, num_scheduled_tokens)

            # Make dummy lora requests
            lora_requests: set[LoRARequest] = {
                LoRARequest(
                    lora_name=f"warmup_{lora_id}",
                    lora_int_id=lora_id,
                    lora_path="/not/a/real/path",
                )
                for lora_id in range(1, num_loras + 1)
            }

            self._set_active_loras(
                tuple(sample_lora_mapping),
                tuple(token_lora_mapping),
                lora_requests,
                mapping_type,
            )

            yield

    @contextmanager
    def maybe_dummy_run_with_lora(
        self,
        lora_config: LoRAConfig | None,
        num_scheduled_tokens: np.ndarray,
        num_sampled_tokens: np.ndarray,
        activate_lora: bool = True,
        remove_lora: bool = True,
        mapping_type: LoRAMappingType = LoRAMappingType.LANGUAGE,
    ):
        with (
            self.maybe_setup_dummy_loras(lora_config, remove_lora),
            self.maybe_select_dummy_loras(
                lora_config,
                num_scheduled_tokens,
                mapping_type,
                num_sampled_tokens,
                activate_lora,
            ),
        ):
            yield

    def maybe_remove_all_loras(self, lora_config: LoRAConfig | None):
        if lora_config is None:
            return
        self.lora_manager.remove_all_adapters()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        self._ensure_lora_enabled()
        return self.lora_manager.add_adapter(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        self._ensure_lora_enabled()
        return self.lora_manager.remove_adapter(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        self._ensure_lora_enabled()
        return self.lora_manager.pin_adapter(lora_id)

    def list_loras(self) -> set[int]:
        self._ensure_lora_enabled()
        return self.lora_manager.list_adapters()
