from enum import Enum

__all__ = ["QuantizationFields"]


class QuantizationFields(Enum):
    num_bits = "num_bits"
    strategy = "strategy"
    symmetric = "symmetric"
    dynamic = "dynamic"