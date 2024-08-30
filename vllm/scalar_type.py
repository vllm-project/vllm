from ._core_ext import NanRepr, ScalarType

# naming generally follows: https://github.com/jax-ml/ml_dtypes
# for floating point types (leading f) the scheme is:
#  `float<size_bits>_e<exponent_bits>m<mantissa_bits>[flags]`
#  flags:
#  - no-flags: means it follows IEEE 754 conventions
#  - f: means finite values only (no infinities)
#  - n: means nans are supported (non-standard encoding)
# for integer types the scheme is:
#  `[u]int<size_bits>[b<bias>]`
#  - if bias is not present it means its zero


class scalar_types:
    int4 = ScalarType.int_(4, None)
    uint4 = ScalarType.uint(4, None)
    int8 = ScalarType.int_(8, None)
    uint8 = ScalarType.uint(8, None)
    float8_e4m3fn = ScalarType.float_(4, 3, True,
                                      NanRepr.EXTD_RANGE_MAX_MIN.value)
    float8_e5m2 = ScalarType.float_IEEE754(5, 2)
    float16_e8m7 = ScalarType.float_IEEE754(8, 7)
    float16_e5m10 = ScalarType.float_IEEE754(5, 10)

    # fp6, https://github.com/usyd-fsalab/fp6_llm/tree/main
    float6_e3m2f = ScalarType.float_(3, 2, True, NanRepr.NONE.value)

    # "gptq" types
    uint4b8 = ScalarType.uint(4, 8)
    uint8b128 = ScalarType.uint(8, 128)

    # colloquial names
    bfloat16 = float16_e8m7
    float16 = float16_e5m10
