from typing import Optional, Union

class ScalarType:
    """
    ScalarType can represent a wide range of floating point and integer types,
    in particular it can be used to represent sub-byte data types (something
    that torch.dtype currently does not support). It is also cabaable of 
    representing types with a bias, i.e. the stored_value = value + bias, this
    is useful for quantized types (e.g. standard GPTQ 4bit uses a bias of 8).
    
    The implementation for this class can be found in csrc/core/scalar_type.hpp,
    these type definitions should be kept in sync with that file.
    """

    def __init__(self, exponent: int, mantissa: int, bias: int,
                 signed: bool) -> None:
        ...

    @classmethod
    def int_(cls, size_bits: int, bias: Optional[int]) -> ScalarType:
        "Create a signed integer scalar type (size_bits includes the sign-bit)."
        ...

    @classmethod
    def uint(cls, size_bits: int, bias: Optional[int]) -> ScalarType:
        """Create a signed integer scalar type."""
        ...

    @classmethod
    def float_IEEE754(cls, exponent: int, mantissa: int) -> ScalarType:
        """
        Create a standard floating point type 
        (i.e. follows IEEE 754 conventions).
        """
        ...

    @classmethod
    def float_(cls, exponent: int, mantissa: int, finite_values_only: bool,
               nan_repr: int) -> ScalarType:
        """
        Create a non-standard floating point type 
        (i.e. does not follow IEEE 754 conventions).
        """
        ...

    @property
    def mantissa(self) -> int:
        """
        Number of bits in the mantissa if this is a floating point type,
        or the number bits representing an integer excluding the sign bit if 
        this an integer type.
        """
        ...

    @property
    def exponent(self) -> int:
        """
        Number of bits in the exponent if this is a floating point type
        (zero if this an integer type)
        """
        ...

    @property
    def bias(self) -> int:
        """
        bias used to encode the values in this scalar type 
        (value = stored_value - bias, default 0) for example if we store the 
        type as an unsigned integer with a bias of 128 then the value 0 will be 
        stored as 128 and -1 will be stored as 127 and 1 will be stored as 129.
        """
        ...

    @property
    def size_bits(self) -> int:
        "Total size of the scalar type in bits."
        ...

    @property
    def nan_repr(self) -> int:
        """
        How NaNs are represent in this scalar type, returns NanRepr value. 
        (not applicable for integer types)
        """
        ...

    def max(self) -> Union[int, float]:
        """
        Max representable value for this scalar type. 
        (accounting for bias if there is one)
        """
        ...

    def min(self) -> Union[int, float]:
        """
        Min representable value for this scalar type. 
        (accounting for bias if there is one)
        """
        ...

    def is_signed(self) -> bool:
        "If the type is signed (i.e. has a sign bit)"
        ...

    def is_integer(self) -> bool:
        "If the type is an integer type"
        ...

    def is_floating_point(self) -> bool:
        "If the type is a floating point type"
        ...

    def is_ieee_754(self) -> bool:
        "If the type is a floating point type that follows IEEE 754 conventions"
        ...

    def has_nans(self) -> bool:
        "If the type is floating point and supports NaN(s)"
        ...

    def has_infs(self) -> bool:
        "If the type is floating point and supports infinity"
        ...

    def has_bias(self) -> bool:
        "If the type has a non-zero bias"
        ...

    def __eq__(self, value: object) -> bool:
        ...

    def __str__(self) -> str:
        ...

    def __repr__(self) -> str:
        ...
