from typing import Optional, Iterable, Tuple, List, Type

from transformers import PretrainedConfig
from importlib.metadata import entry_points

from vllm import ModelRegistry
from dataclasses import dataclass
import torch

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationConfig
from abc import ABC, abstractmethod

from vllm.sequence import IntermediateTensors


logger = init_logger(__name__)


class ModelArchitectureBase(torch.nn.Module, ABC):
    def __init__(
            self,
            config: PretrainedConfig,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
    ):
        self.config = config
        self.cache_config = cache_config
        self.quant_config = quant_config

    @abstractmethod
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        pass

    @abstractmethod
    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        pass


@dataclass
class ModelPlugin:
    architecture_name: str
    implementation_cls: Type[ModelArchitectureBase]


def load_model_plugins():
    for entry_point in entry_points().select(group="vllm.model_architectures"):
        logger.debug(f"Loading model architecture plugin {entry_point.name}")
        model_architecture_plugin = entry_point.load()
        if not isinstance(model_architecture_plugin, ModelPlugin):
            raise ValueError(
                f"Model architecture plugin must be an instance of ModelPlugin, got {model_architecture_plugin}"
            )

        ModelRegistry.register_model(
            model_architecture_plugin.architecture_name,
            model_architecture_plugin.implementation_cls,
        )
