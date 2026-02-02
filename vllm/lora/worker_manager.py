# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from typing import Any, Literal

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.lora.lora_model import LoRAModel
from vllm.lora.model_manager import (
    LoRAModelManager,
    LRUCacheLoRAModelManager,
    create_lora_manager,
)
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path

logger = init_logger(__name__)


class WorkerLoRAManager:
    """WorkerLoRAManager that manages LoRA models on the worker side.

    Every request, the requested LoRAs will be loaded (unless they are already
    loaded), and every other LoRA will be unloaded."""

    _manager_cls: type[LoRAModelManager] = LoRAModelManager

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        embedding_modules: dict[str, str],
        lora_model_cls: type[LoRAModel] = LoRAModel,
    ):
        self._lora_model_cls = lora_model_cls
        self.embedding_modules = embedding_modules
        self._cached_dummy_lora: None | Literal[False] | LoRAModel = False
        self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.vocab_size = vllm_config.model_config.get_vocab_size()
        self.lora_config = vllm_config.lora_config

        # Use get_text_config() in case of multimodal models
        text_config = vllm_config.model_config.hf_config.get_text_config()

        self.max_position_embeddings = text_config.max_position_embeddings
        self.device = device
        # Lazily initialized by create_lora_manager.
        self._adapter_manager: LoRAModelManager

    @contextmanager
    def dummy_lora_cache(self):
        """Use this context manager to reuse the dummy lora model
        to avoid creating it repeatedly."""
        self._cached_dummy_lora = None
        yield
        self._cached_dummy_lora = False

    @property
    def is_enabled(self) -> bool:
        return True

    def create_lora_manager(
        self,
        model: torch.nn.Module,
        vllm_config: VllmConfig | None = None,
    ) -> Any:
        lora_manager = create_lora_manager(
            model,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
            vocab_size=self.vocab_size,
            lora_config=self.lora_config,
            device=self.device,
            lora_manager_cls=self._manager_cls,
            vllm_config=vllm_config,
        )
        self._adapter_manager = lora_manager
        return lora_manager.model

    def _load_adapter(self, lora_request: LoRARequest) -> LoRAModel:
        try:
            supported_lora_modules = self._adapter_manager.supported_lora_modules
            packed_modules_mapping = self._adapter_manager.packed_modules_mapping
            expected_lora_lst: list[str] = []
            for module in supported_lora_modules:
                if module in packed_modules_mapping:
                    expected_lora_lst.extend(packed_modules_mapping[module])
                else:
                    expected_lora_lst.append(module)
                if module == "experts":
                    expected_lora_lst.append(module)
            expected_lora_modules = set(expected_lora_lst)
            lora_path = get_adapter_absolute_path(lora_request.lora_path)

            peft_helper = PEFTHelper.from_local_dir(
                lora_path,
                self.max_position_embeddings,
                lora_request.tensorizer_config_dict,
            )

            # Validates the LoRA configuration against requirements before
            # loading weights, throwing an exception if validation fails.
            peft_helper.validate_legal(self.lora_config)

            # For some models like Qwen2VL, we need to use hf_to_vllm_mapper
            # to ensure correct loading of lora weights.
            model = self._adapter_manager.model
            hf_to_vllm_mapper = getattr(model, "hf_to_vllm_mapper", None)

            # Get model-defined prefixes to skip during LoRA loading.
            lora_skip_prefixes = getattr(model, "lora_skip_prefixes", None)

            lora = self._lora_model_cls.from_local_checkpoint(
                lora_path,
                expected_lora_modules,
                peft_helper=peft_helper,
                lora_model_id=lora_request.lora_int_id,
                device="cpu",
                dtype=self.lora_config.lora_dtype,
                model_vocab_size=self.vocab_size,
                tensorizer_config_dict=lora_request.tensorizer_config_dict,
                weights_mapper=hf_to_vllm_mapper,
                skip_prefixes=lora_skip_prefixes,
            )

        except FileNotFoundError as e:
            # FileNotFoundError should be raised if both
            # - No adapter found to download from huggingface (or in
            #       offline mode)
            # - No local adapter files found at `lora_request.lora_path`
            # For NotFoundError
            raise ValueError(
                f"Loading lora {lora_request.lora_name} failed: No adapter "
                f"found for {lora_request.lora_path}"
            ) from e
        except Exception as e:
            # For BadRequestError
            raise e

        if self.lora_config.lora_target_modules:
            for module_name in lora.loras:
                if not self._adapter_manager._check_target_module_exists(
                    self.lora_config, module_name
                ):
                    logger.warning(
                        "LoRA module '%s' in adapter '%s' is not targeted by "
                        "the current configuration (lora_target_modules). "
                        "These parameters will be ignored, which may cause "
                        "abnormal model behavior.",
                        module_name,
                        lora_request.lora_path,
                    )

        return lora

    def add_dummy_lora(self, lora_request: LoRARequest, rank: int) -> bool:
        if lora_request.lora_int_id in self.list_adapters():
            return False
        if isinstance(self._cached_dummy_lora, LoRAModel):
            dummy_lora = self._cached_dummy_lora.clone(lora_request.lora_int_id)
        else:
            dummy_lora = self._adapter_manager.create_dummy_lora(
                lora_request.lora_int_id, rank, self.embedding_modules
            )
            if self._cached_dummy_lora is None:
                self._cached_dummy_lora = dummy_lora
        return self._adapter_manager.add_adapter(dummy_lora)

    def pin_adapter(self, adapter_id: int) -> bool:
        return self._adapter_manager.pin_adapter(adapter_id)

    def set_active_adapters(self, requests: set[Any], mapping: Any | None) -> None:
        self._apply_adapters(requests)
        if mapping is not None:
            self._adapter_manager.set_adapter_mapping(mapping)

    def supports_tower_connector_lora(self) -> bool:
        return (
            self._adapter_manager.supports_mm
            and self._adapter_manager.supports_tower_connector_lora
        )

    def _apply_adapters(self, adapter_requests: set[Any]) -> None:
        existing_adapters = self.list_adapters()
        models_map = {
            adapter_request.adapter_id: adapter_request
            for adapter_request in adapter_requests
            if adapter_request
        }
        if len(models_map) > self._adapter_manager.adapter_slots:
            raise RuntimeError(
                f"Number of requested models ({len(models_map)}) is greater "
                "than the number of GPU model slots "
                f"({self._adapter_manager.adapter_slots})."
            )
        requested_ids = set(models_map)
        for adapter_id in existing_adapters - requested_ids:
            self.remove_adapter(adapter_id)
        for adapter_id in requested_ids - existing_adapters:
            self.add_adapter(models_map[adapter_id])

    def add_adapter(self, adapter_request: Any) -> bool:
        if adapter_request.adapter_id in self.list_adapters():
            return False
        loaded_adapter = self._load_adapter(adapter_request)
        loaded = self._adapter_manager.add_adapter(loaded_adapter)
        self._adapter_manager.activate_adapter(loaded_adapter.id)
        return loaded

    def remove_adapter(self, adapter_id: int) -> bool:
        return self._adapter_manager.remove_adapter(adapter_id)

    def remove_all_adapters(self):
        self._adapter_manager.remove_all_adapters()

    def list_adapters(self) -> set[int]:
        return set(self._adapter_manager.list_adapters())


class LRUCacheWorkerLoRAManager(WorkerLoRAManager):
    """WorkerLoRAManager that manages LoRA models on the worker side.

    Uses an LRU Cache. Every request, the requested LoRAs will be loaded
    (unless they are already loaded) and least recently used LoRAs will
    be unloaded if the cache is above capacity."""

    _manager_cls: type[LRUCacheLoRAModelManager] = LRUCacheLoRAModelManager

    def create_lora_manager(
        self,
        model: torch.nn.Module,
        vllm_config: VllmConfig | None = None,
    ) -> Any:
        lora_manager = create_lora_manager(
            model,
            lora_manager_cls=self._manager_cls,
            max_num_seqs=self.max_num_seqs,
            vocab_size=self.vocab_size,
            lora_config=self.lora_config,
            device=self.device,
            max_num_batched_tokens=self.max_num_batched_tokens,
            vllm_config=vllm_config,
        )
        self._adapter_manager = lora_manager
        return lora_manager.model

    def _apply_adapters(self, lora_requests: set[LoRARequest]) -> None:
        loras_map = {
            lora_request.lora_int_id: lora_request
            for lora_request in lora_requests
            if lora_request
        }
        if len(loras_map) > self._adapter_manager.lora_slots:
            raise RuntimeError(
                f"Number of requested LoRAs ({len(loras_map)}) is greater "
                "than the number of GPU LoRA slots "
                f"({self._adapter_manager.lora_slots})."
            )
        for lora in loras_map.values():
            self.add_adapter(lora)

    def add_adapter(self, lora_request: LoRARequest) -> bool:
        # Note that this method is not thread-safe. It may be invoked multiple
        # times for the same adapter when using multiple API servers.
        # This is ok because it's currently only called from
        # the single-threaded core engine loop.

        if (
            lora_request.lora_int_id not in self.list_adapters()
            or lora_request.load_inplace
        ):
            # Load the new adapter first to ensure it is actually valid, before
            # evicting any existing adapters.
            # This may cause the # of loaded lora adapters to very temporarily
            # exceed `--max-cpu-loras`.
            lora = self._load_adapter(lora_request)

            # Remove the existing adapter if it exists
            # Use case for LoRA inplace
            self._adapter_manager.remove_adapter(lora.id)

            # Loading succeeded, now check if we will exceed cache capacity and
            # evict if the oldest adapter if so
            if len(self._adapter_manager) + 1 > self._adapter_manager.capacity:
                assert isinstance(self._adapter_manager, LRUCacheLoRAModelManager)
                self._adapter_manager.remove_oldest_adapter()
            # Then add the new adapter to the cache
            loaded = self._adapter_manager.add_adapter(lora)
        else:
            # If the lora is already loaded, just touch it to
            # update its position in the caches
            loaded = (
                self._adapter_manager.get_adapter(lora_request.lora_int_id) is not None
            )
        self._adapter_manager.activate_adapter(lora_request.lora_int_id)
        return loaded
