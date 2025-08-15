# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from safetensors.torch import safe_open
from pathlib import Path
from typing import Optional, Any, Dict, List
from collections import OrderedDict
import copy
from vllm.model_executor.models import SupportsLoRA, supports_tokenformer
from vllm.lora.models import get_lora_id
from vllm.logger import init_logger

from vllm.adapter_commons.models import AdapterModel, AdapterModelManager
from vllm.attention import AttentionMetadata, AttentionType

logger = init_logger(__name__)


class vLLMTokenformerSurgeon:
    """A class that modifies a vLLM model to support tokenformer adapters."""

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def insert_adapter_modules(self) -> torch.nn.Module:
        """Insert tokenformer adapter modules into the model."""
        logger.info("Inserting tokenformer adapter modules")
        
        # For now, return the model as-is since we're using the protocol approach
        # In a full implementation, this would modify attention layers
        return self.model

    def _is_attn_layer(self, name: str) -> bool:
        """Check if a layer is an attention layer that needs tokenformer support."""
        return "attn" in name.lower() or "attention" in name.lower()


class TokenformerModel(AdapterModel):
    """A tokenformer pre-trained model."""

    def __init__(self, tokenformers: Dict[str, torch.Tensor]) -> None:
        super().__init__(get_lora_id())
        self.tokenformers = tokenformers

    @classmethod
    def from_local_checkpoint(
        cls, model_dir: str, device: torch.device
    ) -> "TokenformerModel":
        """Load tokenformer model from local checkpoint."""
        model_path = Path(model_dir)
        files = list(model_path.glob("*.pt"))
        if len(files) == 0:
            raise FileNotFoundError(f"No .pt file found in {model_dir}")

        checkpoint_file = files[0]

        tokenformers = {}
        state_dict = torch.load(checkpoint_file, map_location=device)
        module_state_dict = state_dict['model_state_dict']
        for module, tensor in module_state_dict.items():
            if any(key in module for key in ("tokenformer", "lm_head")):
                logger.info(f"Loading {module} from {checkpoint_file}")
                tokenformers[module] = tensor.to(device)

        return cls(tokenformers)


class TokenformerModelManager(AdapterModelManager):
    """A manager that manages tokenformer models."""

    def __init__(
        self,
        model: SupportsLoRA,
        device: torch.device,
    ):
        if supports_tokenformer(model):
            self.model = vLLMTokenformerSurgeon(model, device).insert_adapter_modules()
        else:
            self.model = model
        self._registered_adapters: Dict[int, Any] = {}
        self._active_adapter: Any = None
        self.tokenformer_model_cls = TokenformerModel
        self.dtype = next(self.model.parameters()).dtype
        self.orig_lm_head = copy.deepcopy(
            {
                k: v.to(self.dtype)
                for k, v in self.model.state_dict().items()
                if "lm_head" in k
            }
        )

    def _activate_adapter(self, adapter_id: int):
        """Activate a tokenformer adapter."""
        logger.info(f"Activating Tokenformer - {adapter_id}")

        model_state_dict = self.model.state_dict()
        tokenformers = self._registered_adapters[adapter_id].tokenformers

        for key, value in self.orig_lm_head.items():
            logger.info(f"Loading original lm head {key} from adapter {adapter_id}")
            model_state_dict[key] = value

        for key, value in tokenformers.items():
            logger.info(f"Loading {key} from adapter {adapter_id}")
            model_state_dict[key] = value

        load_result = self.model.load_state_dict(model_state_dict, strict=False)
        logger.info(f"Load result: {load_result}")

        self._active_adapter = adapter_id

    def _deactivate_adapter(self, adapter_id: int):
        """Deactivate a tokenformer adapter."""
        logger.info(f"Deactivating Tokenformer - {adapter_id}")
        model_state_dict = self.model.state_dict()
        
        for key, value in self.orig_lm_head.items():
            model_state_dict[key] = value

        load_result = self.model.load_state_dict(model_state_dict, strict=False)
        logger.info(f"Load result: {load_result}")
        
        self._active_adapter = None

    def _create_merged_adapter(self, adapter_ids: List[int]) -> AdapterModel:
        """Create a merged adapter from multiple adapters."""
        # Simple implementation - just use the first adapter
        if adapter_ids:
            return self._registered_adapters[adapter_ids[0]]
        raise ValueError("No adapter IDs provided")


class WorkerTokenformerManager:
    """Worker-level tokenformer manager."""

    def __init__(self, device: torch.device):
        self.device = device
        self._adapter_manager: Optional[TokenformerModelManager] = None

    def create_tokenformer_manager(self, model: SupportsLoRA) -> torch.nn.Module:
        """Create and return a tokenformer-enabled model."""
        self._adapter_manager = TokenformerModelManager(model, self.device)
        return self._adapter_manager.model

    def add_adapter(self, lora_request) -> bool:
        """Add a tokenformer adapter."""
        if self._adapter_manager is None:
            return False
            
        try:
            adapter_path = lora_request.lora_path
            tokenformer_model = TokenformerModel.from_local_checkpoint(
                adapter_path, self.device
            )
            self._adapter_manager.add_adapter(tokenformer_model)
            return True
        except Exception as e:
            logger.error(f"Failed to add tokenformer adapter: {e}")
            return False

    def activate_adapter(self, lora_request) -> bool:
        """Activate a tokenformer adapter."""
        if self._adapter_manager is None:
            return False
            
        try:
            self._adapter_manager.activate_adapter(lora_request.adapter_id)
            return True
        except Exception as e:
            logger.error(f"Failed to activate tokenformer adapter: {e}")
            return False

    def deactivate_all_adapters(self) -> bool:
        """Deactivate all tokenformer adapters."""
        if self._adapter_manager is None:
            return False
            
        try:
            self._adapter_manager.deactivate_all_adapters()
            return True
        except Exception as e:
            logger.error(f"Failed to deactivate tokenformer adapters: {e}")
            return False