import torch
from contextlib import contextmanager
from torch import nn
from safetensors.torch import safe_open
from pathlib import Path
from typing import Optional, Any, Dict, List
from collections import OrderedDict
import copy
from vllm.tokenformer.tokenformer_surgeon import (
    TokenformerSurgeon,
)
from vllm.model_executor.models import SupportsLoRA, supports_tokenformer
from vllm.lora.utils import get_adapter_absolute_path, get_lora_id
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import process_weights_after_loading

import os

logger = init_logger(__name__)


class TokenformerModel:
    """A tokenformer pre-trained model."""

    def __init__(self, tokenformers: Dict[str, torch.Tensor]) -> None:
        self.id = get_lora_id()
        self.tokenformers = tokenformers

    @classmethod
    def from_local_checkpoint(
        cls, model_dir: str, device: torch.device
    ) -> "TokenformerModel":
        # Find all .pt files in the directory
        files = list(Path(model_dir).glob("*.pt"))

        if len(files) == 0:
            raise FileNotFoundError(f"No .pt file found in {model_dir}")

        checkpoint_file = files[0]

        tokenformers = {}
        state_dict = torch.load(checkpoint_file, map_location=device)
        module_state_dict = state_dict['model_state_dict']
        for module, tensor in module_state_dict.items():
            logger.info(f"Loading {module} from {checkpoint_file}")
            tokenformers[module] = tensor.to(device)

        return cls(tokenformers)

class TokenformerModelManager:
    """A manager that manages tokenformer models."""

    def __init__(
        self,
        model: SupportsLoRA,
        device: torch.device,
    ):
        if supports_tokenformer(model):
            self.model = TokenformerSurgeon(model, device).insert_adapter_modules()
        else:
            self.model = model

        self._registered_adapters: Dict[int, Any] = {}
        self._active_adapter: Any = None
        self.tokenformer_model_cls = TokenformerModel
        self.dtype = next(self.model.parameters()).dtype
        self.device = device
        self.original_tensors = {}
        self._lru_adaptor_ids = []

    def activate_adapter(self, adapter_id: int) -> bool:
        assert adapter_id in self._registered_adapters, f"Adapter {adapter_id} not found"

        if adapter_id == self._active_adapter:
            logger.info(f"Tokenformer {adapter_id} is already active")
            return False

        self.update_lru_position(adapter_id)

        logger.info(f"Activating Tokenformer - {adapter_id}")

        model_state_dict = self.model.state_dict()

        tokenformers = self._registered_adapters[adapter_id].tokenformers

        # Save original tensors if not already saved
        for key in tokenformers:
            if key not in self.original_tensors:
                logger.info(f"Saving original tensor {key} before loading adapter {adapter_id}")
                if key in model_state_dict:
                    self.original_tensors[key] = copy.deepcopy(model_state_dict[key])

        for key, value in self.original_tensors.items():
            logger.info(f"Loading original tensor {key} from adapter {adapter_id}")
            model_state_dict[key] = value

        for key, value in tokenformers.items():
            logger.info(f"Loading {key} from adapter {adapter_id}")
            model_state_dict[key] = value

        self.model.load_weights(model_state_dict.items())
        process_weights_after_loading(self.model, self.model.model_config, self.device)

        self._active_adapter = adapter_id

        return True

    def update_lru_position(self, adapter_id: int) -> None:
        if adapter_id in self._lru_adaptor_ids:
            self._lru_adaptor_ids.remove(adapter_id)
        self._lru_adaptor_ids.append(adapter_id)

    def deactivate_adapter(self, adapter_id: int) -> bool:
        return self._deactivate_adapter(adapter_id)

    def _deactivate_adapter(self, adapter_id: int):
        logger.info(f"Deactivating Tokenformer - {adapter_id}")
        model_state_dict = self.model.state_dict()
        tokenformers = self._registered_adapters[adapter_id].tokenformers

        for key in tokenformers:
            if "tokenformer_p" in key:
                nn.init.zeros_(model_state_dict[key])
            elif key in self.original_tensors:
                model_state_dict[key] = self.original_tensors[key]

        self.model.load_state_dict(model_state_dict, strict=False)

    def add_adapter(self, request) -> bool:
        lora_path = get_adapter_absolute_path(request.lora_path)
        tokenformer = self.tokenformer_model_cls.from_local_checkpoint(
            lora_path, device=self.device
        )

        if len(self._registered_adapters) >= self.capacity:
            # Remove the least recently used adapter
            lru_adapter_id = self._lru_adaptor_ids.pop(0)
            self.remove_adapter(lru_adapter_id)

        self._registered_adapters[request.adapter_id] = tokenformer
        self._lru_adaptor_ids.append(request.adapter_id)

        logger.info(f"Adapter {request.adapter_id} added")

        return True

    def set_active_adapters(self, lora_requests, lora_mapping):
        if len(lora_requests) == 0:
            self.deactivate_all_adapters()
        else:
            for request in lora_requests:
                self.activate_adapter(request.adapter_id)

    def set_adapter_mapping(self, mapping: Any) -> None:
        pass

    def remove_adapter(self, adapter_id: int) -> bool:
        return remove_adapter(
            adapter_id, self._registered_adapters, self._remove_adapter
        )

    def _remove_adapter(self, adapter_id: int) -> None:
        if adapter_id not in self._registered_adapters:
            logger.warning(f"Adapter {adapter_id} not found")
            return

        if adapter_id == self._active_adapter:
            self.deactivate_adapter(adapter_id)

        del self._registered_adapters[adapter_id]
        logger.info(f"Adapter {adapter_id} removed")

    def deactivate_all_adapters(self) -> None:
        if self._active_adapter is not None:
            self.deactivate_adapter(self._active_adapter)
        self._active_adapter = None

    def remove_all_adapters(self) -> None:
        for id in self._registered_adapters:
            self.deactivate_adapter(id)
        self._registered_adapters.clear()
        self._active_adapter = None

    def get_adapter(self, adapter_id: int) -> Optional[Any]:
        get_adapter(adapter_id, self._registered_adapters)

    def list_adapters(self) -> Dict[int, Any]:
        return list_adapters(self._registered_adapters)

    def pin_adapter(self, adapter_id: int) -> bool:
        pass

    @property
    def capacity(self) -> int:
        return int(os.getenv("TOKENFORMER_CACHE_CAPACITY", "4"))

    @property
    def adapter_slots(self) -> int:
        pass

    @contextmanager
    def dummy_lora_cache(self):
        """Context manager for dummy LoRA cache during warmup."""
        # Simple pass-through context manager since tokenformer doesn't need special cache handling
        yield

    def add_dummy_lora(self, lora_request, rank: int = 8):
        """Add a dummy LoRA for warmup purposes.

        Args:
            lora_request: The LoRA request object
            rank: The rank for the dummy LoRA (default 8)
        """
        # Tokenformer doesn't need to actually add dummy LoRAs, just accept the call
        logger.debug(f"Adding dummy LoRA {lora_request.lora_name} with rank {rank} (no-op for tokenformer)")
        pass

def add_adapter(adapter: Any, registered_adapters: dict[int, Any],
                capacity: int, add_func: callable) -> bool:
    if adapter.id not in registered_adapters:
        if len(registered_adapters) >= capacity:
            raise RuntimeError('No free adapter slots.')
        add_func(adapter)
        registered_adapters[adapter.id] = adapter
        return True
    return False

def deactivate_adapter(adapter_id: int, active_adapters: dict[int, None],
                       deactivate_func: callable) -> bool:
    if adapter_id in active_adapters:
        deactivate_func(adapter_id)
        active_adapters.pop(adapter_id)
        return True
    return False


def remove_adapter(adapter_id: int, registered_adapters: dict[int, Any],
                   deactivate_func: callable) -> bool:
    deactivate_func(adapter_id)
    return bool(registered_adapters.pop(adapter_id, None))


def list_adapters(registered_adapters: dict[int, Any]) -> dict[int, Any]:
    return dict(registered_adapters)
