#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <variant>

// For STD_TORCH_CHECK (stable ABI equivalent of TORCH_CHECK)
#include <torch/headeronly/util/shim_utils.h>

namespace vllm {

//
//  ScalarType can represent a wide range of floating point and integer types,
//  in particular it can be used to represent sub-byte data types (something
//  that torch.dtype currently does not support).
//
//  The type definitions on the Python side can be found in: vllm/scalar_type.py
//  these type definitions should be kept up to date with any Python API changes
//  here.
//
class ScalarType {
 public:
  enum NanRepr : uint8_t {
    NAN_NONE = 0,                // nans are not supported
    NAN_IEEE_754 = 1,            // nans are: exp all 1s, mantissa not all 0s
    NAN_EXTD_RANGE_MAX_MIN = 2,  // nans are: exp all 1s, mantissa all 1s

    NAN_REPR_ID_MAX
  };

  constexpr ScalarType(uint8_t exponent, uint8_t mantissa, bool signed_,
                       int32_t bias, bool finite_values_only = false,
                       NanRepr nan_repr = NAN_IEEE_754)
      : exponent(exponent),
        mantissa(mantissa),
        signed_(signed_),
        bias(bias),
        finite_values_only(finite_values_only),
        nan_repr(nan_repr) {};

  static constexpr ScalarType int_(uint8_t size_bits, int32_t bias = 0) {
    return ScalarType(0, size_bits - 1, true, bias);
  }

  static constexpr ScalarType uint(uint8_t size_bits, int32_t bias = 0) {
    return ScalarType(0, size_bits, false, bias);
  }

  // IEEE 754 compliant floating point type
  static constexpr ScalarType float_IEEE754(uint8_t exponent,
                                            uint8_t mantissa) {
    STD_TORCH_CHECK(mantissa > 0 && exponent > 0);
    return ScalarType(exponent, mantissa, true, 0, false, NAN_IEEE_754);
  }

  // IEEE 754 non-compliant floating point type
  static constexpr ScalarType float_(uint8_t exponent, uint8_t mantissa,
                                     bool finite_values_only,
                                     NanRepr nan_repr) {
    STD_TORCH_CHECK(nan_repr < NAN_REPR_ID_MAX, "Invalid NanRepr");
    STD_TORCH_CHECK(mantissa > 0 && exponent > 0);
    STD_TORCH_CHECK(
        nan_repr != NAN_IEEE_754,
        "use `float_IEEE754` constructor for floating point types that "
        "follow IEEE 754 conventions");
    return ScalarType(exponent, mantissa, true, 0, finite_values_only,
                      nan_repr);
  }

  uint8_t const exponent;  // size of the exponent field (0 for integer types)
  uint8_t const mantissa;  // size of the mantissa field (size of the integer
                           // excluding the sign bit for integer types)
  bool const signed_;  // flag if the type supports negative numbers (i.e. has a
                       // sign bit)
  int32_t const bias;  // stored values equal value + bias,
                       // used for quantized type

  // Extra Floating point info
  bool const finite_values_only;  // i.e. no +/-inf if true
  NanRepr const nan_repr;         // how NaNs are represented
                                  // (not applicable for integer types)

  using Id = int64_t;

 private:
  // Field size in id
  template <typename T_>
  static constexpr size_t member_id_field_width() {
    using T = std::decay_t<T_>;
    return std::is_same_v<T, bool> ? 1 : sizeof(T) * 8;
  }

  template <typename Fn, typename Init, typename Member, typename... Rest>
  static constexpr auto reduce_members_helper(Fn f, Init val, Member member,
                                              Rest... rest) {
    auto new_val = f(val, member);
    if constexpr (sizeof...(rest) > 0) {
      return reduce_members_helper(f, new_val, rest...);
    } else {
      return new_val;
    };
  }

  template <typename Fn, typename Init>
  constexpr auto reduce_members(Fn f, Init init) const {
    // Should be in constructor order for `from_id`
    return reduce_members_helper(f, init, exponent, mantissa, signed_, bias,
                                 finite_values_only, nan_repr);
  };

  template <typename Fn, typename Init>
  static constexpr auto reduce_member_types(Fn f, Init init) {
    constexpr auto dummy_type = ScalarType(0, 0, false, 0, false, NAN_NONE);
    return dummy_type.reduce_members(f, init);
  };

  static constexpr auto id_size_bits() {
    return reduce_member_types(
        [](int acc, auto member) -> int {
          return acc + member_id_field_width<decltype(member)>();
        },
        0);
  }

 public:
  // unique id for this scalar type that can be computed at compile time for
  //  c++17 template specialization this is not needed once we migrate to
  //  c++20 and can pass literal classes as template parameters
  constexpr Id id() const {
    static_assert(id_size_bits() <= sizeof(Id) * 8,
                  "ScalarType id is too large to be stored");

    auto or_and_advance = [](std::pair<Id, uint32_t> result,
                             auto member) -> std::pair<Id, uint32_t> {
      auto [id, bit_offset] = result;
      auto constexpr bits = member_id_field_width<decltype(member)>();
      return {id | (int64_t(member) & ((uint64_t(1) << bits) - 1))
                       << bit_offset,
              bit_offset + bits};
    };
    return reduce_members(or_and_advance, std::pair<Id, uint32_t>{}).first;
  }

  // create a ScalarType from an id, for c++17 template specialization,
  //  this is not needed once we migrate to c++20 and can pass literal
  //  classes as template parameters
  static constexpr ScalarType from_id(Id id) {
    auto extract_and_advance = [id](auto result, auto member) {
      using T = decltype(member);
      auto [tuple, bit_offset] = result;
      auto constexpr bits = member_id_field_width<T>();
      auto extracted_val = static_cast<T>((int64_t(id) >> bit_offset) &
                                          ((uint64_t(1) << bits) - 1));
      auto new_tuple = std::tuple_cat(tuple, std::make_tuple(extracted_val));
      return std::pair<decltype(new_tuple), int>{new_tuple, bit_offset + bits};
    };

    auto [tuple_args, _] = reduce_member_types(extract_and_advance,
                                               std::pair<std::tuple<>, int>{});
    return std::apply([](auto... args) { return ScalarType(args...); },
                      tuple_args);
  }

  constexpr int64_t size_bits() const {
    return mantissa + exponent + is_signed();
  }
  constexpr bool is_signed() const { return signed_; }
  constexpr bool is_integer() const { return exponent == 0; }
  constexpr bool is_floating_point() const { return exponent > 0; }
  constexpr bool is_ieee_754() const {
    return is_floating_point() && finite_values_only == false &&
           nan_repr == NAN_IEEE_754;
  }
  constexpr bool has_nans() const {
    return is_floating_point() && nan_repr != NAN_NONE;
  }
  constexpr bool has_infs() const {
    return is_floating_point() && finite_values_only == false;
  }
  constexpr bool has_bias() const { return bias != 0; }

 private:
  double _floating_point_max() const {
    STD_TORCH_CHECK(mantissa <= 52 && exponent <= 11,
                    "Cannot represent max/min as a double for type ", str());

    uint64_t max_mantissa = (uint64_t(1) << mantissa) - 1;
    if (nan_repr == NAN_EXTD_RANGE_MAX_MIN) {
      max_mantissa -= 1;
    }

    uint64_t max_exponent = (uint64_t(1) << exponent) - 2;
    if (nan_repr == NAN_EXTD_RANGE_MAX_MIN || nan_repr == NAN_NONE) {
      STD_TORCH_CHECK(exponent < 11,
                      "Cannot represent max/min as a double for type ", str());
      max_exponent += 1;
    }

    // adjust the exponent to match that of a double
    //  for now we assume the exponent bias is the standard 2^(e-1) -1, (where e
    //  is the exponent bits), there is some precedent for non-standard biases,
    //  example `float8_e4m3b11fnuz` here: https://github.com/jax-ml/ml_dtypes
    //  but to avoid premature over complication we are just assuming the
    //  standard exponent bias until there is a need to support non-standard
    //  biases
    uint64_t exponent_bias = (uint64_t(1) << (exponent - 1)) - 1;
    uint64_t exponent_bias_double = (uint64_t(1) << 10) - 1;  // double e = 11

    uint64_t max_exponent_double =
        max_exponent - exponent_bias + exponent_bias_double;

    // shift the mantissa into the position for a double and
    // the exponent
    uint64_t double_raw =
        (max_mantissa << (52 - mantissa)) | (max_exponent_double << 52);

    return *reinterpret_cast<double*>(&double_raw);
  }

  constexpr std::variant<int64_t, double> _raw_max() const {
    if (is_floating_point()) {
      return {_floating_point_max()};
    } else {
      STD_TORCH_CHECK(size_bits() < 64 || size_bits() == 64 && is_signed(),
                      "Cannot represent max as a int64_t");
      return {(int64_t(1) << mantissa) - 1};
    }
  }

  constexpr std::variant<int64_t, double> _raw_min() const {
    if (is_floating_point()) {
      STD_TORCH_CHECK(
          is_signed(),
          "We currently assume all floating point types are signed");
      constexpr uint64_t sign_bit_double = (uint64_t(1) << 63);

      double max = _floating_point_max();
      uint64_t max_raw = *reinterpret_cast<uint64_t*>(&max);
      uint64_t min_raw = max_raw | sign_bit_double;
      return {*reinterpret_cast<double*>(&min_raw)};
    } else {
      STD_TORCH_CHECK(!is_signed() || size_bits() <= 64,
                      "Cannot represent min as a int64_t");
      if (is_signed()) {
        // set the top bit to 1 (i.e. INT64_MIN) and the rest to 0
        // then perform an arithmetic shift right to set all the bits above
        // (size_bits() - 1) to 1
        return {INT64_MIN >> (64 - size_bits())};
      } else {
        return {int64_t(0)};
      }
    }
  }

 public:
  // Max representable value for this scalar type.
  // (accounting for bias if there is one)
  constexpr std::variant<int64_t, double> max() const {
    return std::visit(
        [this](auto x) -> std::variant<int64_t, double> { return {x - bias}; },
        _raw_max());
  }

  // Min representable value for this scalar type.
  // (accounting for bias if there is one)
  constexpr std::variant<int64_t, double> min() const {
    return std::visit(
        [this](auto x) -> std::variant<int64_t, double> { return {x - bias}; },
        _raw_min());
  }

  std::string str() const {
    /* naming generally follows: https://github.com/jax-ml/ml_dtypes
     * for floating point types (leading f) the scheme is:
     *  `float<size_bits>_e<exponent_bits>m<mantissa_bits>[flags]`
     *  flags:
     *  - no-flags: means it follows IEEE 754 conventions
     *  - f: means finite values only (no infinities)
     *  - n: means nans are supported (non-standard encoding)
     * for integer types the scheme is:
     *  `[u]int<size_bits>[b<bias>]`
     *  - if bias is not present it means its zero
     */
    if (is_floating_point()) {
      auto ret = "float" + std::to_string(size_bits()) + "_e" +
                 std::to_string(exponent) + "m" + std::to_string(mantissa);
      if (!is_ieee_754()) {
        if (finite_values_only) {
          ret += "f";
        }
        if (nan_repr != NAN_NONE) {
          ret += "n";
        }
      }
      return ret;
    } else {
      auto ret = ((is_signed()) ? "int" : "uint") + std::to_string(size_bits());
      if (has_bias()) {
        ret += "b" + std::to_string(bias);
      }
      return ret;
    }
  }

  constexpr bool operator==(ScalarType const& other) const {
    return mantissa == other.mantissa && exponent == other.exponent &&
           bias == other.bias && signed_ == other.signed_ &&
           finite_values_only == other.finite_values_only &&
           nan_repr == other.nan_repr;
  }
};

using ScalarTypeId = ScalarType::Id;

// "rust style" names generally following:
//   https://github.com/pytorch/pytorch/blob/6d9f74f0af54751311f0dd71f7e5c01a93260ab3/torch/csrc/api/include/torch/types.h#L60-L70
static inline constexpr auto kS4 = ScalarType::int_(4);
static inline constexpr auto kU4 = ScalarType::uint(4);
static inline constexpr auto kU4B8 = ScalarType::uint(4, 8);
static inline constexpr auto kS8 = ScalarType::int_(8);
static inline constexpr auto kU8 = ScalarType::uint(8);
static inline constexpr auto kU8B128 = ScalarType::uint(8, 128);

static inline constexpr auto kFE2M1f =
    ScalarType::float_(2, 1, true, ScalarType::NAN_NONE);
static inline constexpr auto kFE3M2f =
    ScalarType::float_(3, 2, true, ScalarType::NAN_NONE);
static inline constexpr auto kFE4M3fn =
    ScalarType::float_(4, 3, true, ScalarType::NAN_EXTD_RANGE_MAX_MIN);
static inline constexpr auto kFE8M0fnu =
    ScalarType(8, 0, false, 0, true, ScalarType::NAN_EXTD_RANGE_MAX_MIN);
static inline constexpr auto kFE5M2 = ScalarType::float_IEEE754(5, 2);
static inline constexpr auto kFE8M7 = ScalarType::float_IEEE754(8, 7);
static inline constexpr auto kFE5M10 = ScalarType::float_IEEE754(5, 10);

// Fixed width style names, generally following:
//  https://github.com/pytorch/pytorch/blob/6d9f74f0af54751311f0dd71f7e5c01a93260ab3/torch/csrc/api/include/torch/types.h#L47-L57
static inline constexpr auto kInt4 = kS4;
static inline constexpr auto kUint4 = kU4;
static inline constexpr auto kUint4b8 = kU4B8;
static inline constexpr auto kInt8 = kS8;
static inline constexpr auto kUint8 = kU8;
static inline constexpr auto kUint8b128 = kU8B128;

static inline constexpr auto kFloat4_e2m1f = kFE2M1f;
static inline constexpr auto kFloat6_e3m2f = kFE3M2f;
static inline constexpr auto kFloat8_e4m3fn = kFE4M3fn;
static inline constexpr auto kFloat8_e5m2 = kFE5M2;
static inline constexpr auto kFloat16_e8m7 = kFE8M7;
static inline constexpr auto kFloat16_e5m10 = kFE5M10;

// colloquial names
static inline constexpr auto kHalf = kFE5M10;
static inline constexpr auto kFloat16 = kHalf;
static inline constexpr auto kBFloat16 = kFE8M7;

static inline constexpr auto kFloat16Id = kFloat16.id();
};  // namespace vllm
