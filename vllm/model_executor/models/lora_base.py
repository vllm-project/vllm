from typing import TYPE_CHECKING, ClassVar, Dict, List, Optional

from torch import nn
from transformers import PretrainedConfig

from vllm.config import LoRAConfig

if TYPE_CHECKING:
    from vllm.lora.models import LoRAModelManager


class LoRASupportedModelBase(nn.Module):
    """Base class for all models that support LoRA."""

    packed_modules_mapping: ClassVar[Dict[str, List[str]]] = {}
    supported_lora_modules: ClassVar[List[str]] = []
    embedding_modules: ClassVar[Dict[str, str]] = {}
    embedding_padding_modules: ClassVar[List[str]] = []

    # Assigned by LoRAModelManager at runtime
    lora_manager: "LoRAModelManager"

    def __init__(
        self,
        config: PretrainedConfig,
        # This is None when LoRA is not enabled
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.lora_config = lora_config
