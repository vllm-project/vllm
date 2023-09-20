from typing import Any, Dict, List

import torch

from vllm.model_executor.quantization_utils.base import QuantizationConfig


class SmoothQuantConfig(QuantizationConfig):
    """Config class for SmoothQuant

    Reference: https://github.com/mit-han-lab/smoothquant
    """

    def __init__(
        self,
        weight_bits: int = 8,
        quant_type: str = "tensor"
    ) -> None:
        self.weight_bits = weight_bits
        self.quant_type = quant_type

        if self.weight_bits != 8:
            raise ValueError(
                "Currently, only w8a8 quantization is supported for "
                f"SmoothQuant, but got {self.weight_bits} bits.")
        if self.quant_type != "tensor":
            raise ValueError(
                "Currently, only tensor wise quantization is supported for "
                f"SmoothQuant, but got {self.quant_type} type quantization.")

    def __repr__(self) -> str:
        return (f"SmoothQuantConfig(weight_bits={self.weight_bits}, "
                f"quant_type={self.quant_type})")

    @classmethod
    def get_name(cls) -> str:
        return "smoothquant"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.float]

    @classmethod
    def get_min_capability(cls) -> int:
        # The smoothquant kernel only supports Ampere or newer GPUs.
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        """List of filenames to search for in the model directory."""
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SmoothQuantConfig":
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        quant_type = cls.get_from_keys(config, ["quant_type", "q_type"])
        return cls(weight_bits, quant_type)

    @classmethod
    def get_packed_tensor_names(cls) -> List[str]:
        return []

    @classmethod
    def get_transposed_tensor_names(cls) -> List[str]:
        return ["weight", "bias"]

    @classmethod
    def get_tp_tensor_names(cls) -> List[str]:
        return ["weight", "bias"]
    
