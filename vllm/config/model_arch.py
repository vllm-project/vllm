# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import json
from dataclasses import field
from typing import Any

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from torch import nn

from vllm.config.utils import config
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, extra="allow"))
class ModelArchitectureTextConfig:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    use_deepseek_mla: bool
    head_dim: int
    vocab_size: int
    num_key_value_heads: int
    num_experts: int

    def __init__(
        self,
        model_type: str,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        use_deepseek_mla: bool,
        head_dim: int,
        vocab_size: int,
        num_key_value_heads: int,
        num_experts: int,
        **kwargs,
    ):
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.num_key_value_heads = num_key_value_heads
        self.num_experts = num_experts
        self.use_deepseek_mla = use_deepseek_mla

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        config_dict_json = json.dumps(self.__dict__, indent=2, sort_keys=True) + "\n"
        return f"{self.__class__.__name__} {config_dict_json}"


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, extra="allow"))
class ModelArchitectureVisionConfig:
    def __init__(
        self,
        **kwargs,
    ):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        config_dict_json = json.dumps(self.__dict__, indent=2, sort_keys=True) + "\n"
        return f"{self.__class__.__name__} {config_dict_json}"


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, extra="allow"))
class ModelArchitectureAudioConfig:
    def __init__(
        self,
        **kwargs,
    ):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        config_dict_json = json.dumps(self.__dict__, indent=2, sort_keys=True) + "\n"
        return f"{self.__class__.__name__} {config_dict_json}"


@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ModelArchitectureConfig:
    """
    Configuration for model architecture
    """

    text_config: ModelArchitectureTextConfig = field(init=True)
    """Text model configuration containing text-specific architecture details."""

    architectures: list[str] = field(default_factory=list)
    """List of model architecture class names (e.g., ['LlamaForCausalLM'])."""

    model_type: str = ""
    """Model type identifier (e.g., 'llama', 'gpt2')."""

    # TODO: Formalize quantization_config in parser
    quantization_config: dict[str, Any] = field(default_factory=dict)
    """Quantization configuration dictionary containing quantization parameters."""

    torch_dtype: str = ""
    """PyTorch data type for model weights (e.g., 'float16', 'bfloat16')."""

    per_layer_attention_cls: list[type[nn.Module]] = field(default_factory=list)
    """Per-layer attention class of the model."""

    vision_config: ModelArchitectureVisionConfig | None = None
    """Vision model configuration for multimodal models (optional)."""

    audio_config: ModelArchitectureAudioConfig | None = None
    """Audio model configuration for multimodal models (optional)."""

    def __init__(
        self,
        architectures: list[str],
        model_type: str,
        quantization_config: dict[str, Any],
        torch_dtype: str,
        text_config: ModelArchitectureTextConfig,
        per_layer_attention_cls: list[type[nn.Module]] | None = None,
        vision: ModelArchitectureVisionConfig | None = None,
        audio: ModelArchitectureAudioConfig | None = None,
    ):
        self.architectures = architectures
        self.model_type = model_type
        self.quantization_config = quantization_config
        self.torch_dtype = torch_dtype
        self.text_config = text_config
        self.per_layer_attention_cls = (
            per_layer_attention_cls if per_layer_attention_cls is not None else []
        )
        self.vision = vision
        self.audio = audio

    @functools.cached_property
    def support_multimodal(self) -> bool:
        raise NotImplementedError
