import importlib.util
from enum import Enum
from typing import TYPE_CHECKING, Optional, Union

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)
core_C_available = importlib.util.find_spec('._core_C', 'vllm') is not None


# Mirrors enum in `core/scalar_type.hpp`
class NanRepr(Enum):
    NONE = 0  # nans are not supported
    IEEE_754 = 1  # nans are: Exp all 1s, mantissa not all 0s
    EXTD_RANGE_MAX_MIN = 2  # nans are: Exp all 1s, mantissa all 1s


if TYPE_CHECKING or not core_C_available:
    # On platforms were we cannot use/build the C++ core extension (i.e. namely
    # neuron and tpu), we define the mock ScalarType class here that partially
    # mimics the C++ ScalarType class.
    #
    # We also use this provide type signatures to the Python LSP for the methods
    # in the C++ ScalarType class. So these type signatures should be kept
    # in sync with csrc/core/scalar_type.hpp

    from dataclasses import dataclass

    @dataclass(frozen=True)
    class ScalarType:
        """
        ScalarType can represent a wide range of floating point and integer 
        types, in particular it can be used to represent sub-byte data types 
        (something that torch.dtype currently does not support). It is also 
        capable of  representing types with a bias, i.e.:
          `stored_value = value + bias`, 
        this is useful for quantized types (e.g. standard GPTQ 4bit uses a bias 
        of 8). The implementation for this class can be found in 
        csrc/core/scalar_type.hpp, these type signatures should be kept in sync 
        with that file.
        """

        exponent: int
        """
        Number of bits in the exponent if this is a floating point type
        (zero if this an integer type)
        """

        mantissa: int
        """
        Number of bits in the mantissa if this is a floating point type,
        or the number bits representing an integer excluding the sign bit if 
        this an integer type.
        """

        bias: int
        """
        bias used to encode the values in this scalar type 
        (value = stored_value - bias, default 0) for example if we store the 
        type as an unsigned integer with a bias of 128 then the value 0 will be 
        stored as 128 and -1 will be stored as 127 and 1 will be stored as 129.
        """

        signed: bool
        "If the type is signed (i.e. has a sign bit)"

        _finite_values_only: bool = False
        """
        Private: if NANs are supported, used `has_infs()` instead.
        """

        nan_repr: int = NanRepr.IEEE_754.value
        """
        How NaNs are represent in this scalar type, returns NanRepr value. 
        (not applicable for integer types)
        """

        @property
        def size_bits(self):
            return self.exponent + self.mantissa + int(self.signed)

        def min(self) -> Union[int, float]:
            """
            Min representable value for this scalar type. 
            (accounting for bias if there is one)
            """
            raise NotImplementedError

        def max(self) -> Union[int, float]:
            """
            Max representable value for this scalar type. 
            (accounting for bias if there is one)
            """
            raise NotImplementedError

        def is_signed(self) -> bool:
            """
            If the type is signed (i.e. has a sign bit), same as `signed`
            added for consistency with:
            https://pytorch.org/docs/stable/generated/torch.Tensor.is_signed.html
            """
            ...

        def is_floating_point(self):
            "If the type is a floating point type"
            return self.exponent != 0

        def is_integer(self):
            "If the type is an integer type"
            return self.exponent == 0

        def has_bias(self):
            "If the type has a non-zero bias"
            return self.bias != 0

        def has_infs(self):
            "If the type is floating point and supports infinity"
            return not self._finite_values_only

        def has_nans(self):
            return self.nan_repr != NanRepr.NONE.value

        def is_ieee_754(self) -> bool:
            """
            If the type is a floating point type that follows IEEE 754 
            conventions
            """
            return self.nan_repr == NanRepr.IEEE_754.value and \
                not self._finite_values_only

        def __str__(self) -> str:
            raise NotImplementedError

        def __repr__(self) -> str:
            raise NotImplementedError

        #
        # Convenience Constructors
        #

        @classmethod
        def int_(cls, size_bits: int, bias: Optional[int]) -> 'ScalarType':
            "Create a signed integer scalar type (size_bits includes sign-bit)."
            return cls(size_bits - 1, size_bits, bias if bias else 0, True)

        @classmethod
        def uint(cls, size_bits: int, bias: Optional[int]) -> 'ScalarType':
            """Create a unsigned integer scalar type."""
            return cls(size_bits, size_bits, bias if bias else 0, False)

        @classmethod
        def float_IEEE754(cls, exponent: int, mantissa: int) -> 'ScalarType':
            """
            Create a standard floating point type 
            (i.e. follows IEEE 754 conventions).
            """
            return cls(exponent, mantissa, 0, True)

        @classmethod
        def float_(cls, exponent: int, mantissa: int, finite_values_only: bool,
                   nan_repr: int):
            """
            Create a non-standard floating point type 
            (i.e. does not follow IEEE 754 conventions).
            """
            return cls(exponent, mantissa, 0, True, finite_values_only,
                       nan_repr)

elif core_C_available:
    try:
        import vllm._core_C  # noqa: F401
    except ImportError as e:
        logger.warning("Failed to import from vllm._core_C with %r", e)

    ScalarType = torch.classes._core_C.ScalarType
