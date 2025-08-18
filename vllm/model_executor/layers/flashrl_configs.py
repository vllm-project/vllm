"""
Flash RL source code


Configs modified from: https://github.com/yaof20/Flash-RL  
"""

from dataclasses import dataclass, field
from typing import List


def get_default_config(config_name: str):
    if config_name == "fp8_channel":
        return FP8ChannelConfig()
    elif config_name in ["fp8_vllm", "fp8"]:
        return FP8vLLMConfig()
    elif config_name == "bf16":
        return BF16Config()
    elif config_name == "int8":
        return Int8Config()
    elif config_name == "int8_prune":
        return Int8PruneConfig()
    raise ValueError(f"Invalid config name: {config_name}")


@dataclass
class FP8TensorConfig:
    fn: str = "fp8_tensor"
    load_format: str = "dummy"
    module_attribute_to_preserve: List[str] = field(default_factory=lambda: ["workspace"])


@dataclass
class FP8ChannelConfig:
    fn: str = "fp8_channel"
    load_format: str = "dummy"
    module_attribute_to_preserve: List[str] = field(default_factory=lambda: ["workspace"])


@dataclass
class FP8vLLMConfig:
    fn: str = "fp8_vllm"
    load_format: str = "auto"
    module_attribute_to_preserve: List[str] = field(default_factory=lambda: ["workspace"])
    quantization: str = "fp8"


@dataclass
class BF16Config:
    fn: str = "bf16"
    load_format: str = "dummy"


@dataclass
class Int8Config:
    fn: str = "int8"
    load_format: str = "auto"


@dataclass
class Int8PruneConfig:
    fn: str = "int8_prune"
    load_format: str = "auto"
