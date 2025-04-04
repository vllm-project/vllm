# SPDX-License-Identifier: Apache-2.0
import logging
from pathlib import Path
from typing import Any, Callable, Optional

import gguf
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch import nn

from vllm.adapter_commons.models import (AdapterLRUCache, AdapterModel,
                                         AdapterModelManager)
from vllm.adapter_commons.utils import (add_adapter, deactivate_adapter,
                                        get_adapter, list_adapters,
                                        remove_adapter, set_adapter_mapping)
from vllm.config import ControlVectorConfig
from vllm.control_vectors.layers import (ControlVectorMapping,
                                         MLPWithControlVector)

logger = logging.getLogger(__name__)

_GLOBAL_CONTROL_VECTOR_ID = 0


def get_control_vector_id():
    global _GLOBAL_CONTROL_VECTOR_ID
    _GLOBAL_CONTROL_VECTOR_ID += 1
    return _GLOBAL_CONTROL_VECTOR_ID


_all_cv_classes = {"mlp": MLPWithControlVector}


def parse_number_from_string(s: str) -> int:
    parts = s.split('.')
    for part in parts:
        if part.isdigit():
            return int(part)
    return None


def load_control_vector_file(file_path, revision="main"):
    try:
        if Path(file_path).exists():
            return str(Path(file_path).resolve())
        parts = file_path.split("/")
        repo_id = "/".join(parts[:2])
        file_name = "/".join(parts[2:])

        return hf_hub_download(repo_id=repo_id,
                               filename=file_name,
                               revision=revision)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_path}") from e
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}") from e


class ControlVectorModel(AdapterModel):

    def __init__(self,
                 control_vector_id=None,
                 control_vector_weights=None,
                 scale_factor=1.0) -> None:
        self.id = control_vector_id
        self.control_vector_weights = control_vector_weights
        self.scale_factor = scale_factor

    @classmethod
    def from_local_checkpoint(
        cls,
        control_vector_model_path: str,
        control_vector_id: int,
        config: ControlVectorConfig,
        device: str = "cuda",
        scale_factor: float = 1.0,
    ) -> "ControlVectorModel":

        try:
            file_path = load_control_vector_file(control_vector_model_path)
            reader = gguf.GGUFReader(file_path)
            archf = reader.get_field("general.architecture")

            if not archf or not len(archf.parts):
                logger.error(".gguf file missing architecture field")
            else:
                arch = str(bytes(archf.parts[-1]),
                           encoding="utf-8",
                           errors="replace")
                if arch != "controlvector":
                    logger.error(
                        ".gguf file with arch %s is not a control vector",
                        arch)
            modelf = reader.get_field('controlvector.model_hint')

            if not modelf or not len(modelf.parts):
                raise ValueError(
                    ".gguf file missing controlvector.model_hint field")

            model_hint = str(bytes(modelf.parts[-1]), encoding="utf-8")
            logger.info("Control Vector for %s loaded.", model_hint)
            cv_weights = {}
            for tensor in reader.tensors:
                if not tensor.name.startswith("direction."):
                    continue
                try:
                    layer = int(tensor.name.split(".")[1])
                except ValueError as e:
                    raise ValueError(
                        ".gguf file has invalid direction field name: %s",
                        tensor.name) from e
                np_copy = np.array(tensor.data, copy=True)
                cv_weights[layer] = torch.from_numpy(np_copy).to(device).to(
                    config.adapter_dtype)

            return cls(control_vector_id, cv_weights, scale_factor)

        except Exception as e:
            raise e


class ControlVectorModelManager(AdapterModelManager):

    def __init__(self, model: nn.Module,
                 control_vector_config: ControlVectorConfig):
        self.model = model
        self._registered_adapters = {}
        self._active_adapters = {}
        self.control_vector_config = control_vector_config
        self._last_mapping = None
        self.model.control_vector_manager = self
        self.control_vector_index_to_id: list[
            Optional[int]] = [None] * self.adapter_slots
        self.modules: dict[str, nn.Module] = {}
        self._create_cv_modules()

    @property
    def adapter_slots(self) -> int:
        return self.capacity

    @property
    def capacity(self) -> int:
        return self.control_vector_config.max_control_vectors

    def activate_adapter(
        self,
        control_vector_id: int,
    ) -> bool:
        if control_vector_id in self._active_adapters:
            return False
        first_free_slot = next(
            ((i, control_vector_id) for i, control_vector_id in enumerate(
                self.control_vector_index_to_id) if control_vector_id is None),
            None)
        if first_free_slot is None:
            raise ValueError("No free control vector slots")
        index, _ = first_free_slot
        self._active_adapters[control_vector_id] = None
        control_vector_model = (self._registered_adapters[control_vector_id])
        logger.debug("Activating control vector. int id: %d, slot index: %d",
                     control_vector_model.id, index)
        self.control_vector_index_to_id[index] = control_vector_model.id
        for k, v in self.modules.items():
            layer_index = parse_number_from_string(k)
            if layer_index in control_vector_model.control_vector_weights:
                v.set_control_vector(
                    index,
                    control_vector_model.control_vector_weights[layer_index] *
                    control_vector_model.scale_factor)
        return True

    def _deactivate_adapter(self, control_vector_id: int):
        try:
            index = self.control_vector_index_to_id.index(control_vector_id)
            self.control_vector_index_to_id[index] = None
            for _, module in self.modules.items():
                module.reset_control_vector(index)
        except ValueError:
            pass

    def _add_adapter(self, control_vector: ControlVectorModel):
        self._registered_adapters[control_vector.id] = control_vector

    def get_index_from_id(self, id):
        for i in range(len(self.control_vector_index_to_id)):
            if self.control_vector_index_to_id[i] == id:
                return i
        return None

    def _set_adapter_mapping(self, id: int) -> None:
        index = self.get_index_from_id(id)

        for k, v in self.modules.items():
            v.set_active_tensor(index)

    def _create_cv_modules(self):
        for module_name, module in self.model.named_modules():
            for key in _all_cv_classes:
                if not module_name.endswith(key):
                    continue
                if isinstance(module, _all_cv_classes[key]):
                    continue
                new_module = self.replace_submodule(
                    self.model, module_name, _all_cv_classes[key](module))
                new_module.set_layer_id(parse_number_from_string(module_name))
                self.register_module(module_name, new_module)
                new_module.set_normalization(
                    self.control_vector_config.normalize)

    def replace_submodule(self, model: nn.Module, module_name: str,
                          new_module: nn.Module) -> nn.Module:
        """Replace a submodule in a model with a new module."""
        parent = model.get_submodule(".".join(module_name.split(".")[:-1]))
        target_name = module_name.split(".")[-1]
        setattr(parent, target_name, new_module)
        return new_module

    def register_module(self, module_name: str, module: nn.Module):
        self.modules[module_name] = module

    def remove_all_adapters(self):
        """Remove all PromptAdapterModel from the manager."""
        self._registered_adapters.clear()
        self.control_vector_index_to_id = [None] * self.adapter_slots
        self._active_adapters.clear()

    def deactivate_adapter(self, adapter_id: int) -> bool:
        return deactivate_adapter(adapter_id, self._active_adapters,
                                  self._deactivate_adapter)

    def add_adapter(self, adapter: ControlVectorModel) -> bool:
        return add_adapter(adapter, self._registered_adapters, self.capacity,
                           self._add_adapter)

    def set_adapter_mapping(self, mapping: ControlVectorMapping) -> None:
        self._last_mapping = set_adapter_mapping(mapping, self._last_mapping,
                                                 self._set_adapter_mapping)

    def remove_adapter(self, adapter_id: int) -> bool:
        return remove_adapter(adapter_id, self._registered_adapters,
                              self.deactivate_adapter)

    def list_adapters(self) -> dict[int, Any]:
        return list_adapters(self._registered_adapters)

    def get_adapter(self, adapter_id: int) -> Optional[Any]:
        return get_adapter(adapter_id, self._registered_adapters)

    def pin_adapter(self, adapter_id: int) -> bool:
        raise NotImplementedError


class ControlVectorLRUCache(AdapterLRUCache[ControlVectorModel]):

    def __init__(self, capacity: int, deactivate_cv_fn: Callable[[int], bool]):
        super().__init__(capacity, deactivate_cv_fn)


class LRUCacheControlVectorModelManager(ControlVectorModelManager):

    def __init__(self, model: nn.Module,
                 control_vector_config: ControlVectorConfig):
        self.control_vector_config = control_vector_config
        super().__init__(model, control_vector_config)
        self._registered_adapters = ControlVectorLRUCache(
            self.capacity, self.deactivate_adapter)
        self._active_adapters = ControlVectorLRUCache(self.adapter_slots,
                                                      self._deactivate_adapter)

    def list_adapters(self) -> dict[int, ControlVectorModel]:
        """List all registered ControlVectorModel."""
        return dict(self._registered_adapters.cache)

    def activate_adapter(
        self,
        control_vector_id: int,
    ) -> bool:
        if control_vector_id not in self._active_adapters and len(
                self._active_adapters) >= self.adapter_slots:
            self._active_adapters.remove_oldest()
        result = super().activate_adapter(control_vector_id)
        # We always touch to update the LRU cache order
        self._active_adapters.touch(control_vector_id)
        return result

    def remove_oldest_adapter(self) -> bool:
        if len(self._registered_adapters) > 0:
            self._registered_adapters.remove_oldest()
            return True
        return False


def create_cv_manager(
    model: nn.Module,
    control_vector_config: ControlVectorConfig,
    control_vector_manager_cls: type[
        ControlVectorModelManager] = ControlVectorModelManager,
) -> ControlVectorModelManager:
    control_vector_manager = control_vector_manager_cls(
        model=model, control_vector_config=control_vector_config)

    return control_vector_manager
