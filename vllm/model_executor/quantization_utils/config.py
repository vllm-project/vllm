from typing import Any, Dict, List


class QuantizationConfig:

    @property
    def name(self) -> str:
        """Name of the quantization method."""
        raise NotImplementedError

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        """List of filenames to search for in the model directory."""
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "QuantizationConfig":
        """Create a config class from the model's quantization config."""
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

        if 32 % self.weight_bits != 0:
            raise ValueError("Weight bits must be a factor of 32, but got "
                             f"{self.weight_bits}")
        self.pack_factor = 32 // self.weight_bits

    @property
    def name(self) -> str:
        return "awq"

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return [
            "quant_config.json",  # E.g., casperhansen/vicuna-7b-v1.5-awq
            "quantization_config.json",  # E.g., abhinavkulkarni/mosaicml-mpt-7b-instruct-w4-g128-awq  # pylint: disable=line-too-long
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AWQConfig":
        weight_bits = _get_config_value(config, ["w_bit", "bits"])
        group_size = _get_config_value(config, ["q_group_size", "group_size"])
        zero_point = _get_config_value(config, ["zero_point"])
        return cls(weight_bits, group_size, zero_point)


_QUANTIZATION_REGISTRY = {
    "awq": AWQConfig,
}


def get_quant_config(quantization: str) -> QuantizationConfig:
    if quantization not in _QUANTIZATION_REGISTRY:
        raise ValueError(f"Invalid quantization method: {quantization}")
    return _QUANTIZATION_REGISTRY[quantization]


def _get_config_value(cls, config: Dict[str, Any], keys: List[str]) -> Any:
    """Get a value from the model's quantization config."""
    for key in keys:
        if key in config:
            return config[key]
    raise ValueError(f"Could not find any of {keys} in the model's "
                     "quantization config.")
