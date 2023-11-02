from typing import Any, Dict, List

import torch

from vllm.model_executor.quantization_utils.base import QuantizationConfig


class SqueezeLLMConfig(QuantizationConfig):
    """Config class for SqueezeLLM.

    Reference: https://arxiv.org/pdf/2306.07629
    """

    def __init__(
        self,
        weight_bits: int,
    ) -> None:
        self.weight_bits = weight_bits

        if self.weight_bits != 4:
            raise ValueError(
                "Currently, only 4-bit weight quantization is supported for "
                f"SqueezeLLM, but got {self.weight_bits} bits.")

        self.pack_factor = 32 // self.weight_bits

    def __repr__(self) -> str:
        return f"SqueezeLLMConfig(weight_bits={self.weight_bits})"

    @classmethod
    def get_name(cls) -> str:
        return "squeezellm"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quant_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SqueezeLLMConfig":
        weight_bits = cls.get_from_keys(config, ["wbits"])
        return cls(weight_bits)

    @classmethod
    def get_packed_tensors(cls) -> Dict[str, int]:
        return {"qweight": 0}

    @classmethod
    def get_transposed_tensor_names(cls) -> List[str]:
        return ["qweight"]

    @classmethod
    def get_col_parallel_tensor_names(cls) -> List[str]:
        return ["qweight", "lookup_table"]

    @classmethod
    def get_row_parallel_tensor_names(cls) -> List[str]:
        return ["qweight"]
