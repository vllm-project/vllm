# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any

import torch

from vllm.adapter_commons.utils import (add_adapter_worker,
                                        apply_adapters_worker,
                                        list_adapters_worker,
                                        set_active_adapters_worker)
from vllm.adapter_commons.worker_manager import AbstractWorkerManager
from vllm.config import ControlVectorConfig
from vllm.control_vectors.models import (ControlVectorModel,
                                         ControlVectorModelManager,
                                         LRUCacheControlVectorModelManager,
                                         create_cv_manager)
from vllm.control_vectors.request import ControlVectorRequest

logger = logging.getLogger(__name__)


class WorkerControlVectorManager(AbstractWorkerManager):
    """WorkerControlVectorManager that manages 
    control vector models on the worker side.

    Every request, the requested control vectors will be 
    loaded (unless they are already loaded), 
    and every other control vector will be unloaded."""

    _manager_cls: type[ControlVectorModelManager] = ControlVectorModelManager

    def __init__(
        self,
        device: torch.device,
        control_vector_config: ControlVectorConfig,
        control_vector_model_cls: type[ControlVectorModel] = ControlVectorModel
    ):
        self._adapter_manager: ControlVectorModelManager
        self._control_vector_model_cls = control_vector_model_cls
        self.control_vector_config = control_vector_config
        super().__init__(device)

    @property
    def is_enabled(self) -> bool:
        return True

    def create_control_vector_manager(
        self,
        model: torch.nn.Module,
    ) -> Any:
        control_vector_manager = create_cv_manager(
            model,
            control_vector_config=self.control_vector_config,
            control_vector_manager_cls=self._manager_cls,
        )
        self._adapter_manager = control_vector_manager
        return control_vector_manager.model

    def _load_adapter(
            self, control_vector_request: ControlVectorRequest
    ) -> ControlVectorModel:
        try:
            control_vector = (
                self._control_vector_model_cls.from_local_checkpoint(
                    control_vector_request.control_vector_local_path,
                    control_vector_id=control_vector_request.control_vector_id,
                    config=self.control_vector_config,
                    device=str(self.device),
                    scale_factor=control_vector_request.scale_factor))
        except Exception as e:
            raise RuntimeError(
                f"Loading control vector "
                f"{control_vector_request.control_vector_local_path}"
                f" failed") from e
        return control_vector

    def add_dummy_control_vector(
            self, control_vector_request: ControlVectorRequest) -> bool:
        return True

    def pin_adapter(self, adapter_id: int) -> bool:
        return self._adapter_manager.pin_adapter(adapter_id)

    def set_active_adapters(self, requests: set[Any]) -> None:
        assert len(requests) <= 1, "No more than 1 control vector at a time"
        mapping = next((request.adapter_id for request in requests), None)
        set_active_adapters_worker(requests, mapping, self._apply_adapters,
                                   self._adapter_manager.set_adapter_mapping)

    def add_adapter(self, adapter_request: Any) -> bool:
        return add_adapter_worker(adapter_request, self.list_adapters,
                                  self._load_adapter,
                                  self._adapter_manager.add_adapter,
                                  self._adapter_manager.activate_adapter)

    def _apply_adapters(self, adapter_requests: set[Any]) -> None:
        apply_adapters_worker(adapter_requests, self.list_adapters,
                              self._adapter_manager.adapter_slots,
                              self.remove_adapter, self.add_adapter)

    def remove_adapter(self, adapter_id: int) -> bool:
        return self._adapter_manager.remove_adapter(adapter_id)

    def remove_all_adapters(self):
        self._adapter_manager.remove_all_adapters()

    def list_adapters(self) -> set[int]:
        return list_adapters_worker(self._adapter_manager.list_adapters)


class LRUCacheWorkerControlVectorManager(WorkerControlVectorManager):
    """WorkerControlVectorManager that manages 
    control vector models on the worker side.

    Uses an LRU Cache. Every request, the requested 
    control vectors will be loaded (unless they are already loaded) 
    and least recently used control vectors will
    be unloaded if the cache is above capacity."""

    _control_vector_manager_cls: type[
        LRUCacheControlVectorModelManager] = LRUCacheControlVectorModelManager

    def create_control_vector_manager(
        self,
        model: torch.nn.Module,
    ) -> Any:
        control_vector_manager = create_cv_manager(
            model,
            control_vector_config=self.control_vector_config,
            control_vector_manager_cls=self._control_vector_manager_cls)
        self._adapter_manager: LRUCacheControlVectorModelManager = (
            control_vector_manager)
        return control_vector_manager.model

    def _apply_adapters(
            self, control_vector_requests: set[ControlVectorRequest]) -> None:
        control_vectors_map = {
            control_vector_request.control_vector_id: control_vector_request
            for control_vector_request in control_vector_requests
            if control_vector_request
        }
        if len(control_vectors_map) > self._adapter_manager.adapter_slots:
            raise RuntimeError(f"Number of requested control vectors "
                               f"({len(control_vectors_map)}) is greater "
                               "than the number of GPU control vector slots "
                               f"({self._adapter_manager.adapter_slots}).")
        for control_vector in control_vectors_map.values():
            self.add_adapter(control_vector)

    def add_adapter(self,
                    control_vector_request: ControlVectorRequest) -> bool:
        if control_vector_request.control_vector_id not in self.list_adapters(
        ):
            # Remove before we load the new control vector to save memory
            if len(self._adapter_manager) + 1 > self._adapter_manager.capacity:
                self._adapter_manager.remove_oldest_adapter()
            control_vector = self._load_adapter(control_vector_request)
            loaded = self._adapter_manager.add_adapter(control_vector)
        else:
            loaded = self._adapter_manager.get_adapter(
                control_vector_request.adapter_id)
        self._adapter_manager.activate_adapter(
            control_vector_request.control_vector_id)
        return loaded
