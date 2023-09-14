from typing import Any, Dict, List, Type


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

    @staticmethod
    def get_from_keys(config: Dict[str, Any], keys: List[str]) -> Any:
        """Get a value from the model's quantization config."""
        for key in keys:
            if key in config:
                return config[key]
        raise ValueError(f"Cannot find any of {keys} in the model's "
                         "quantization config.")


class AWQConfig(QuantizationConfig):
    """Config class for AWQ.

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

        if self.weight_bits != 4:
            raise ValueError(
                "Currently, only 4-bit weight quantization is supported for "
                f"AWQ, but got {self.weight_bits} bits.")
        self.pack_factor = 32 // self.weight_bits

    @property
    def name(self) -> str:
        return "awq"

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return [
            "quant_config.json",  # E.g., casperhansen/vicuna-7b-v1.5-awq
            "quantize_config.json",  # E.g., abhinavkulkarni/mosaicml-mpt-7b-instruct-w4-g128-awq  # pylint: disable=line-too-long
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AWQConfig":
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        return cls(weight_bits, group_size, zero_point)

    def __repr__(self) -> str:
        return (f"AWQConfig(weight_bits={self.weight_bits}, "
                f"group_size={self.group_size}, "
                f"zero_point={self.zero_point})")


_QUANTIZATION_REGISTRY = {
    "awq": AWQConfig,
}


def get_quant_class(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in _QUANTIZATION_REGISTRY:
        raise ValueError(f"Invalid quantization method: {quantization}")
    return _QUANTIZATION_REGISTRY[quantization]
