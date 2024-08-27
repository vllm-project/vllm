import importlib.util
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

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

        def is_floating_point(self) -> bool:
            "If the type is a floating point type"
            return self.exponent != 0

        def is_integer(self) -> bool:
            "If the type is an integer type"
            return self.exponent == 0

        def has_bias(self) -> bool:
            "If the type has a non-zero bias"
            return self.bias != 0

        def has_infs(self) -> bool:
            "If the type is floating point and supports infinity"
            return not self._finite_values_only

        def has_nans(self) -> bool:
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

        # __len__ needs to be defined (and has to throw TypeError) for pytorch's
        # opcheck to work.
        def __len__(self) -> int:
            raise TypeError

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
                   nan_repr: int) -> 'ScalarType':
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

    if (hasattr(torch, "_library")
            and hasattr(torch._library, "register_fake_class")):
        # Needed for dynamo support of ScalarType.
        @torch._library.register_fake_class("_core_C::ScalarType")
        class FakeScalarType:

            def __init__(self, scalar_type):
                self.ScalarType = scalar_type

            def bias_getter(self) -> int:
                return self.ScalarType.bias

            def exponent_getter(self) -> int:
                return self.ScalarType.exponent

            def mantissa_getter(self) -> int:
                return self.ScalarType.mantissa

            def signed_getter(self) -> bool:
                return self.ScalarType.signed

            def size_bits_getter(self) -> int:
                return self.ScalarType.size_bits

            @property
            def size_bits(self) -> int:
                return self.ScalarType.size_bits

            def min(self) -> Union[int, float]:
                return self.ScalarType.min()

            def max(self) -> Union[int, float]:
                return self.ScalarType.max()

            def is_signed(self) -> bool:
                return self.ScalarType.is_signed()

            def is_floating_point(self) -> bool:
                return self.ScalarType.is_floating_point()

            def is_integer(self) -> bool:
                return self.ScalarType.is_integer()

            def has_bias(self) -> bool:
                return self.ScalarType.has_bias()

            def has_infs(self) -> bool:
                return self.ScalarType.has_infs()

            def has_nans(self) -> bool:
                return self.ScalarType.has_nans()

            def is_ieee_754(self) -> bool:
                return self.ScalarType.is_ieee_754()

            def __str__(self) -> str:
                return self.ScalarType.__str__()

            def __repr__(self) -> str:
                return self.ScalarType.__repr__()

            def __len__(self) -> int:
                return self.ScalarType.__len__()

            def __obj_flatten__(self) -> Tuple[Tuple[str, Any], ...]:
                return torch.classes._core_C.ScalarType.__obj_flatten__(
                    self.ScalarType)

            @classmethod
            def __obj_unflatten__(
                    cls, flat_type: Tuple[Tuple[str, Any],
                                          ...]) -> 'ScalarType':
                return cls(
                    torch.classes._core_C.ScalarType.__obj_unflatten__(
                        flat_type))

            @classmethod
            def int_(cls, size_bits: int, bias: Optional[int]) -> 'ScalarType':
                return ScalarType.int_(size_bits, bias)

            @classmethod
            def uint(cls, size_bits: int, bias: Optional[int]) -> 'ScalarType':
                return ScalarType.uint(size_bits, bias)

            @classmethod
            def float_IEEE754(cls, exponent: int,
                              mantissa: int) -> 'ScalarType':
                return ScalarType.float_IEEE754(exponent, mantissa)

            @classmethod
            def float_(cls, exponent: int, mantissa: int,
                       finite_values_only: bool,
                       nan_repr: int) -> 'ScalarType':
                return ScalarType.float_(exponent, mantissa,
                                         finite_values_only, nan_repr)
