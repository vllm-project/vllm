from typing import List


class QuantizationConfig:

    @property
    def name(self) -> str:
        """Name of the quantization method."""
        raise NotImplementedError

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        """List of filenames to search for in the model directory."""
        raise NotImplementedError


class AWQConfig(QuantizationConfig):
    """AWQ: Activation-aware Weight Quantization.

    Reference: https://arxiv.org/abs/2306.00978
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.pack_factor = 32 // self.weight_bits

    @property
    def name(self) -> str:
        return "awq"

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quant_config.json", "quantization_config.json"]


_QUANTIZATION_REGISTRY = {
    "awq": AWQConfig,
}


def get_quant_config(quantization: str) -> QuantizationConfig:
    if quantization not in _QUANTIZATION_REGISTRY:
        raise ValueError(f"Invalid quantization method: {quantization}")
    return _QUANTIZATION_REGISTRY[quantization]
