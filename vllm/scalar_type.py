from ._core_ext import NanRepr, ScalarType

# naming generally follows:
#   https://github.com/jax-ml/ml_dtypes#specifications-of-implemented-floating-point-formats
#   https://onnx.ai/onnx/technical/float8.html
#   https://github.com/openxla/xla/blob/38c9f654dc308c1247374199a272ac636a5fb79e/xla/xla_data.proto#L85-L100
# for floating point types (leading f) the scheme is:
#  `float<size_bits>_e<exponent_bits>m<mantissa_bits>[flags]`
#  flags:
#  - no-flags: means it follows IEEE 754 conventions
#  - f: means finite values only (no infinities)
#  - n: means nans are supported (non-standard encoding)
#  - uz: means NAN is represented using `-0.0`, (freeing up all 1s for an actual
#        value), uz also implies the exponent_bias is increased by 1, this is
#        the convention used by the references above
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
    float8_e4m3fnuz = ScalarType.float_(4, 3, True,
                                        NanRepr.EXTD_RANGE_NEG_ZERO.value)
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
