/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/json_schema_converter.cc
 * \brief Implementation of JSONSchemaConverter and related utilities.
 */
#include "json_schema_converter.h"

#include <picojson.h>

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "fsm_builder.h"
#include "grammar_builder.h"
#include "grammar_functor.h"
#include "json_schema_converter_ext.h"
#include "regex_converter.h"
#include "support/json_parse.h"
#include "support/logging.h"

namespace xgrammar {

// ==================== Spec ToString implementations ====================

std::string IntegerSpec::ToString() const {
  return "IntegerSpec{minimum=" + (minimum.has_value() ? std::to_string(*minimum) : "null") +
         ", maximum=" + (maximum.has_value() ? std::to_string(*maximum) : "null") +
         ", exclusive_minimum=" +
         (exclusive_minimum.has_value() ? std::to_string(*exclusive_minimum) : "null") +
         ", exclusive_maximum=" +
         (exclusive_maximum.has_value() ? std::to_string(*exclusive_maximum) : "null") +
         ", multiple_of=" + (multiple_of.has_value() ? std::to_string(*multiple_of) : "null") + "}";
}

std::string NumberSpec::ToString() const {
  return "NumberSpec{minimum=" + (minimum.has_value() ? std::to_string(*minimum) : "null") +
         ", maximum=" + (maximum.has_value() ? std::to_string(*maximum) : "null") +
         ", exclusive_minimum=" +
         (exclusive_minimum.has_value() ? std::to_string(*exclusive_minimum) : "null") +
         ", exclusive_maximum=" +
         (exclusive_maximum.has_value() ? std::to_string(*exclusive_maximum) : "null") + "}";
}

std::string StringSpec::ToString() const {
  return "StringSpec{pattern=" + (pattern.has_value() ? "\"" + *pattern + "\"" : "null") +
         ", format=" + (format.has_value() ? "\"" + *format + "\"" : "null") +
         ", min_length=" + std::to_string(min_length) +
         ", max_length=" + std::to_string(max_length) + "}";
}

std::string BooleanSpec::ToString() const { return "BooleanSpec{}"; }

std::string NullSpec::ToString() const { return "NullSpec{}"; }

std::string AnySpec::ToString() const { return "AnySpec{}"; }

std::string ArraySpec::ToString() const {
  return "ArraySpec{prefix_items.size()=" + std::to_string(prefix_items.size()) +
         ", allow_additional_items=" + (allow_additional_items ? "true" : "false") +
         ", additional_items=" + (additional_items ? "SchemaSpec" : "null") +
         ", min_items=" + std::to_string(min_items) + ", max_items=" + std::to_string(max_items) +
         "}";
}

std::string ObjectSpec::ToString() const {
  std::string s =
      "ObjectSpec{properties.size()=" + std::to_string(properties.size()) + ", properties=[";
  for (size_t i = 0; i < properties.size(); ++i) {
    if (i != 0) s += ", ";
    s += properties[i].name;
  }
  s += "], pattern_properties.size()=" + std::to_string(pattern_properties.size()) + ", required=[";
  bool first = true;
  for (const auto& r : required) {
    if (!first) s += ", ";
    s += r;
    first = false;
  }
  s +=
      std::string("], allow_additional_properties=") +
      (allow_additional_properties ? "true" : "false") +
      ", additional_properties_schema=" + (additional_properties_schema ? "SchemaSpec" : "null") +
      ", allow_unevaluated_properties=" + (allow_unevaluated_properties ? "true" : "false") +
      ", unevaluated_properties_schema=" + (unevaluated_properties_schema ? "SchemaSpec" : "null") +
      ", property_names=" + (property_names ? "SchemaSpec" : "null") +
      ", min_properties=" + std::to_string(min_properties) +
      ", max_properties=" + std::to_string(max_properties) + "}";
  return s;
}

std::string ConstSpec::ToString() const { return "ConstSpec{json_value=\"" + json_value + "\"}"; }

std::string EnumSpec::ToString() const {
  std::string s =
      "EnumSpec{json_values.size()=" + std::to_string(json_values.size()) + ", json_values=[";
  for (size_t i = 0; i < json_values.size(); ++i) {
    if (i != 0) s += ", ";
    s += "\"" + json_values[i] + "\"";
  }
  s += "]}";
  return s;
}

std::string RefSpec::ToString() const { return "RefSpec{uri=\"" + uri + "\"}"; }

std::string AnyOfSpec::ToString() const {
  return "AnyOfSpec{options.size()=" + std::to_string(options.size()) + "}";
}

std::string OneOfSpec::ToString() const {
  return "OneOfSpec{options.size()=" + std::to_string(options.size()) + "}";
}

std::string AllOfSpec::ToString() const {
  return "AllOfSpec{schemas.size()=" + std::to_string(schemas.size()) + "}";
}

std::string TypeArraySpec::ToString() const {
  return "TypeArraySpec{type_schemas.size()=" + std::to_string(type_schemas.size()) + "}";
}

std::string SchemaSpec::ToString() const {
  std::string spec_str;
  std::visit([&spec_str](const auto& s) { spec_str = s.ToString(); }, spec);
  return "SchemaSpec{spec=" + spec_str + ", cache_key=\"" + cache_key + "\", rule_name_hint=\"" +
         rule_name_hint + "\"}";
}

// ==================== SchemaParser (Internal) ====================

namespace {

enum class SchemaErrorType : int {
  kInvalidSchema = 0,
  kUnsatisfiableSchema = 1,
  kUnsupportedSchema = 2,
};

using SchemaError = TypedError<SchemaErrorType>;

// Unbounded integer multipleOf emits a modulo DFA: states ~= N, transitions ~= 10N.
// Fail closed above the cap to keep generated grammars bounded.
constexpr int64_t kIntegerMultipleOfMax = 1024;
constexpr int64_t kIntegerMultipleOfRangeWidthMax = 10000;

bool IsMultipleOf(int64_t value, int64_t multiple_of) { return (value % multiple_of) == 0; }

bool HasMultipleInRange(int64_t start, int64_t end, int64_t multiple_of) {
  for (int64_t value = start; value <= end; ++value) {
    if (IsMultipleOf(value, multiple_of)) return true;
    if (value == std::numeric_limits<int64_t>::max()) break;
  }
  return false;
}

constexpr const char* kUnsupportedOneOfMessage =
    "oneOf with overlapping or non-provably-disjoint branches cannot be represented exactly; "
    "falling back to anyOf semantics";

bool IsSchemaAnnotationKey(const std::string& key) {
  static const std::unordered_set<std::string> kAnnotationKeys = {
      "title",
      "default",
      "description",
      "examples",
      "deprecated",
      "readOnly",
      "writeOnly",
      "$comment",
      "$schema",
  };
  return kAnnotationKeys.count(key) != 0;
}

bool HasOnlyKeys(
    const picojson::object& schema, const std::unordered_set<std::string>& allowed_keys
) {
  for (const auto& [key, _] : schema) {
    if (allowed_keys.count(key) == 0 && !IsSchemaAnnotationKey(key)) {
      return false;
    }
  }
  return true;
}

bool IsSupportedJSONType(const std::string& type) {
  static const std::unordered_set<std::string> kTypes = {
      "null",
      "boolean",
      "object",
      "array",
      "number",
      "string",
      "integer",
  };
  return kTypes.count(type) != 0;
}

bool NormalizeTypeSet(
    const picojson::value& type_value, std::unordered_set<std::string>* type_set
) {
  if (type_value.is<std::string>()) {
    const auto& type = type_value.get<std::string>();
    if (!IsSupportedJSONType(type)) {
      return false;
    }
    type_set->insert(type);
    return true;
  }
  if (!type_value.is<picojson::array>()) {
    return false;
  }

  const auto& type_array = type_value.get<picojson::array>();
  if (type_array.empty()) {
    return false;
  }
  for (const auto& item : type_array) {
    if (!item.is<std::string>()) {
      return false;
    }
    const auto& type = item.get<std::string>();
    if (!IsSupportedJSONType(type)) {
      return false;
    }
    type_set->insert(type);
  }
  return true;
}

bool IsNumericValue(const picojson::value& value) {
  return value.is<int64_t>() || value.is<double>();
}

bool IsIntegerValue(const picojson::value& value) {
  if (value.is<int64_t>()) {
    return true;
  }
  if (!value.is<double>()) {
    return false;
  }
  double number = value.get<double>();
  return std::isfinite(number) && std::floor(number) == number;
}

bool JSONValuesMayOverlap(const picojson::value& lhs, const picojson::value& rhs) {
  if (IsNumericValue(lhs) || IsNumericValue(rhs)) {
    if (!IsNumericValue(lhs) || !IsNumericValue(rhs)) {
      return false;
    }
    if (lhs.is<int64_t>() && rhs.is<int64_t>()) {
      return lhs.get<int64_t>() == rhs.get<int64_t>();
    }
    return true;
  }
  if (lhs.is<picojson::null>() || rhs.is<picojson::null>()) {
    return lhs.is<picojson::null>() && rhs.is<picojson::null>();
  }
  if (lhs.is<bool>() || rhs.is<bool>()) {
    return lhs.is<bool>() && rhs.is<bool>() && lhs.get<bool>() == rhs.get<bool>();
  }
  if (lhs.is<std::string>() || rhs.is<std::string>()) {
    return lhs.is<std::string>() && rhs.is<std::string>() &&
           lhs.get<std::string>() == rhs.get<std::string>();
  }
  if (lhs.is<picojson::array>() || rhs.is<picojson::array>()) {
    if (!lhs.is<picojson::array>() || !rhs.is<picojson::array>()) {
      return false;
    }
    const auto& lhs_array = lhs.get<picojson::array>();
    const auto& rhs_array = rhs.get<picojson::array>();
    if (lhs_array.size() != rhs_array.size()) {
      return false;
    }
    for (size_t i = 0; i < lhs_array.size(); ++i) {
      if (!JSONValuesMayOverlap(lhs_array[i], rhs_array[i])) {
        return false;
      }
    }
    return true;
  }
  if (lhs.is<picojson::object>() || rhs.is<picojson::object>()) {
    if (!lhs.is<picojson::object>() || !rhs.is<picojson::object>()) {
      return false;
    }
    const auto& lhs_object = lhs.get<picojson::object>();
    const auto& rhs_object = rhs.get<picojson::object>();
    if (lhs_object.size() != rhs_object.size()) {
      return false;
    }
    for (const auto& [key, lhs_value] : lhs_object) {
      auto rhs_it = rhs_object.find(key);
      if (rhs_it == rhs_object.end() || !JSONValuesMayOverlap(lhs_value, rhs_it->second)) {
        return false;
      }
    }
    return true;
  }
  return lhs.serialize() == rhs.serialize();
}

bool ValueMatchesType(const picojson::value& value, const std::string& type) {
  if (type == "null") {
    return value.is<picojson::null>();
  }
  if (type == "boolean") {
    return value.is<bool>();
  }
  if (type == "string") {
    return value.is<std::string>();
  }
  if (type == "integer") {
    return IsIntegerValue(value);
  }
  if (type == "number") {
    return IsNumericValue(value);
  }
  if (type == "array") {
    return value.is<picojson::array>();
  }
  if (type == "object") {
    return value.is<picojson::object>();
  }
  return false;
}

bool IsRangeWidthOverCap(int64_t start, int64_t end, int64_t cap) {
  uint64_t cap_u = static_cast<uint64_t>(cap);
  if (start <= 0 && end >= 0) {
    // Count [start, end] inclusively without evaluating -INT64_MIN or overflowing the sum.
    uint64_t negative_count = start < 0 ? static_cast<uint64_t>(-(start + 1)) + 1 : 0;
    if (negative_count > cap_u) return true;
    uint64_t remaining = cap_u - negative_count;
    if (remaining == 0) return true;
    --remaining;  // zero
    uint64_t positive_count = end > 0 ? static_cast<uint64_t>(end) : 0;
    return positive_count > remaining;
  }

  uint64_t value_count = static_cast<uint64_t>(end - start) + 1;
  return value_count > cap_u;
}

// Effective inclusive integer range after folding exclusive bounds into minimum/maximum. A nullopt
// side means that side is unbounded.
struct EffectiveIntegerRange {
  std::optional<int64_t> start;
  std::optional<int64_t> end;
};

// Fold the inclusive [minimum, maximum] bounds together with any exclusive bounds so the stricter
// bound wins on each side. Shared by ParseInteger (range validation) and GenerateInteger (grammar
// emission) so the two can never disagree about the effective range. Precondition:
// exclusive_minimum != INT64_MAX and exclusive_maximum != INT64_MIN (ParseInteger rejects those
// before building the spec), so the +1/-1 below cannot overflow.
EffectiveIntegerRange ComputeEffectiveIntegerRange(const IntegerSpec& spec) {
  EffectiveIntegerRange range;
  if (spec.minimum.has_value()) {
    range.start = spec.minimum;
  }
  if (spec.exclusive_minimum.has_value()) {
    // Smallest integer strictly greater than exclusive_minimum; the larger lower bound wins.
    int64_t excl_start = *spec.exclusive_minimum + 1;
    range.start = range.start.has_value() ? std::max(*range.start, excl_start) : excl_start;
  }
  if (spec.maximum.has_value()) {
    range.end = spec.maximum;
  }
  if (spec.exclusive_maximum.has_value()) {
    // Largest integer strictly less than exclusive_maximum; the smaller upper bound wins.
    int64_t excl_end = *spec.exclusive_maximum - 1;
    range.end = range.end.has_value() ? std::min(*range.end, excl_end) : excl_end;
  }
  return range;
}

bool TypeSetsOverlap(
    const std::unordered_set<std::string>& lhs, const std::unordered_set<std::string>& rhs
) {
  for (const auto& lhs_type : lhs) {
    for (const auto& rhs_type : rhs) {
      if (lhs_type == rhs_type) {
        return true;
      }
      if ((lhs_type == "integer" || lhs_type == "number") &&
          (rhs_type == "integer" || rhs_type == "number")) {
        return true;
      }
    }
  }
  return false;
}

bool FiniteValuesOverlap(
    const std::vector<picojson::value>& lhs, const std::vector<picojson::value>& rhs
) {
  for (const auto& lhs_value : lhs) {
    for (const auto& rhs_value : rhs) {
      if (JSONValuesMayOverlap(lhs_value, rhs_value)) {
        return true;
      }
    }
  }
  return false;
}

bool FiniteValuesOverlapTypeSet(
    const std::vector<picojson::value>& values, const std::unordered_set<std::string>& type_set
) {
  for (const auto& value : values) {
    if (IsNumericValue(value) && (type_set.count("integer") || type_set.count("number"))) {
      return true;
    }
    for (const auto& type : type_set) {
      if (ValueMatchesType(value, type)) {
        return true;
      }
    }
  }
  return false;
}

bool TryGetFiniteValues(const picojson::object& schema, std::vector<picojson::value>* values) {
  if (schema.count("const")) {
    values->push_back(schema.at("const"));
    return true;
  }
  if (schema.count("enum")) {
    if (!schema.at("enum").is<picojson::array>()) {
      return false;
    }
    const auto& enum_values = schema.at("enum").get<picojson::array>();
    if (enum_values.empty()) {
      return false;
    }
    values->insert(values->end(), enum_values.begin(), enum_values.end());
    return true;
  }
  return false;
}

struct OneOfArmProof {
  enum class Kind { kTypeSet, kFiniteValues };

  Kind kind;
  std::unordered_set<std::string> type_set;
  std::vector<picojson::value> finite_values;
};

std::optional<OneOfArmProof> ClassifyTypeOrFiniteOneOfArm(const picojson::value& option) {
  if (!option.is<picojson::object>()) {
    return std::nullopt;
  }
  const auto& schema = option.get<picojson::object>();

  if (schema.count("$ref") || schema.count("anyOf") || schema.count("allOf") ||
      schema.count("oneOf")) {
    return std::nullopt;
  }

  std::vector<picojson::value> finite_values;
  if (TryGetFiniteValues(schema, &finite_values)) {
    OneOfArmProof proof;
    proof.kind = OneOfArmProof::Kind::kFiniteValues;
    proof.finite_values = std::move(finite_values);
    return proof;
  }

  if (!schema.count("type") || !HasOnlyKeys(schema, {"type"})) {
    return std::nullopt;
  }

  std::unordered_set<std::string> type_set;
  if (!NormalizeTypeSet(schema.at("type"), &type_set)) {
    return std::nullopt;
  }
  if (type_set.count("object")) {
    return std::nullopt;
  }

  OneOfArmProof proof;
  proof.kind = OneOfArmProof::Kind::kTypeSet;
  proof.type_set = std::move(type_set);
  return proof;
}

bool OneOfArmProofsAreDisjoint(const OneOfArmProof& lhs, const OneOfArmProof& rhs) {
  if (lhs.kind == OneOfArmProof::Kind::kTypeSet && rhs.kind == OneOfArmProof::Kind::kTypeSet) {
    return !TypeSetsOverlap(lhs.type_set, rhs.type_set);
  }
  if (lhs.kind == OneOfArmProof::Kind::kFiniteValues &&
      rhs.kind == OneOfArmProof::Kind::kFiniteValues) {
    return !FiniteValuesOverlap(lhs.finite_values, rhs.finite_values);
  }
  if (lhs.kind == OneOfArmProof::Kind::kFiniteValues && rhs.kind == OneOfArmProof::Kind::kTypeSet) {
    return !FiniteValuesOverlapTypeSet(lhs.finite_values, rhs.type_set);
  }
  return !FiniteValuesOverlapTypeSet(rhs.finite_values, lhs.type_set);
}

std::optional<std::vector<picojson::value>> GetDiscriminatorValues(
    const picojson::value& option, const std::string& discriminator_key
) {
  if (!option.is<picojson::object>()) {
    return std::nullopt;
  }
  const auto& schema = option.get<picojson::object>();
  if (schema.count("$ref") || schema.count("anyOf") || schema.count("allOf") ||
      schema.count("oneOf")) {
    return std::nullopt;
  }
  if (!schema.count("type") || !schema.at("type").is<std::string>() ||
      schema.at("type").get<std::string>() != "object") {
    return std::nullopt;
  }
  if (!schema.count("required") || !schema.at("required").is<picojson::array>()) {
    return std::nullopt;
  }

  bool requires_discriminator = false;
  for (const auto& required_key : schema.at("required").get<picojson::array>()) {
    if (!required_key.is<std::string>()) {
      return std::nullopt;
    }
    if (required_key.get<std::string>() == discriminator_key) {
      requires_discriminator = true;
    }
  }
  if (!requires_discriminator) {
    return std::nullopt;
  }

  if (!schema.count("properties") || !schema.at("properties").is<picojson::object>()) {
    return std::nullopt;
  }
  const auto& properties = schema.at("properties").get<picojson::object>();
  auto property_it = properties.find(discriminator_key);
  if (property_it == properties.end() || !property_it->second.is<picojson::object>()) {
    return std::nullopt;
  }

  std::vector<picojson::value> values;
  if (!TryGetFiniteValues(property_it->second.get<picojson::object>(), &values)) {
    return std::nullopt;
  }
  return values;
}

std::vector<std::string> GetDiscriminatorCandidates(const picojson::value& option) {
  std::vector<std::string> candidates;
  if (!option.is<picojson::object>()) {
    return candidates;
  }
  const auto& schema = option.get<picojson::object>();
  if (!schema.count("required") || !schema.at("required").is<picojson::array>() ||
      !schema.count("properties") || !schema.at("properties").is<picojson::object>()) {
    return candidates;
  }
  const auto& properties = schema.at("properties").get<picojson::object>();
  for (const auto& required_key : schema.at("required").get<picojson::array>()) {
    if (!required_key.is<std::string>()) {
      continue;
    }
    const auto& key = required_key.get<std::string>();
    auto property_it = properties.find(key);
    if (property_it == properties.end() || !property_it->second.is<picojson::object>()) {
      continue;
    }
    std::vector<picojson::value> values;
    if (TryGetFiniteValues(property_it->second.get<picojson::object>(), &values)) {
      candidates.push_back(key);
    }
  }
  return candidates;
}

bool TryProveStrictDiscriminatorOneOf(const picojson::array& options) {
  if (options.empty()) {
    return false;
  }

  for (const auto& discriminator_key : GetDiscriminatorCandidates(options.front())) {
    std::vector<std::vector<picojson::value>> branch_values;
    bool all_branches_have_key = true;
    for (const auto& option : options) {
      auto values = GetDiscriminatorValues(option, discriminator_key);
      if (!values.has_value()) {
        all_branches_have_key = false;
        break;
      }
      branch_values.push_back(std::move(values.value()));
    }
    if (!all_branches_have_key) {
      continue;
    }

    bool pairwise_disjoint = true;
    for (size_t i = 0; i < branch_values.size() && pairwise_disjoint; ++i) {
      for (size_t j = i + 1; j < branch_values.size(); ++j) {
        if (FiniteValuesOverlap(branch_values[i], branch_values[j])) {
          pairwise_disjoint = false;
          break;
        }
      }
    }
    if (pairwise_disjoint) {
      return true;
    }
  }
  return false;
}

bool TryProveTypeOrFiniteOneOf(const picojson::array& options) {
  std::vector<OneOfArmProof> proofs;
  proofs.reserve(options.size());
  for (const auto& option : options) {
    auto proof = ClassifyTypeOrFiniteOneOfArm(option);
    if (!proof.has_value()) {
      return false;
    }
    proofs.push_back(std::move(proof.value()));
  }

  for (size_t i = 0; i < proofs.size(); ++i) {
    for (size_t j = i + 1; j < proofs.size(); ++j) {
      if (!OneOfArmProofsAreDisjoint(proofs[i], proofs[j])) {
        return false;
      }
    }
  }
  return true;
}

bool TryProvePairwiseDisjointOneOf(const picojson::array& options) {
  return TryProveStrictDiscriminatorOneOf(options) || TryProveTypeOrFiniteOneOf(options);
}

/*!
 * \brief Parser for JSON Schema, converts JSON Schema to SchemaSpec intermediate representation.
 */
class SchemaParser {
 public:
  struct Config {
    bool strict_mode = false;
    JSONFormat json_format;
  };

  explicit SchemaParser(const picojson::value& root_schema, const Config& config)
      : config_(config), root_schema_(root_schema) {}

  Result<SchemaSpecPtr, SchemaError> Parse(
      const picojson::value& schema,
      const std::string& rule_name_hint = "root",
      std::optional<std::string> default_type = std::nullopt
  );

  const picojson::value& GetRootSchema() const { return root_schema_; }
  bool IsStrictMode() const { return config_.strict_mode; }

  Result<SchemaSpecPtr, SchemaError> ResolveRef(
      const std::string& uri, const std::string& rule_name_hint
  );

 private:
  Result<IntegerSpec, SchemaError> ParseInteger(const picojson::object& schema);
  Result<NumberSpec, SchemaError> ParseNumber(const picojson::object& schema);
  Result<StringSpec, SchemaError> ParseString(const picojson::object& schema);
  Result<BooleanSpec, SchemaError> ParseBoolean(const picojson::object& schema);
  Result<NullSpec, SchemaError> ParseNull(const picojson::object& schema);
  Result<ArraySpec, SchemaError> ParseArray(const picojson::object& schema);
  Result<ObjectSpec, SchemaError> ParseObject(const picojson::object& schema);
  Result<ConstSpec, SchemaError> ParseConst(const picojson::object& schema);
  Result<EnumSpec, SchemaError> ParseEnum(const picojson::object& schema);
  Result<RefSpec, SchemaError> ParseRef(const picojson::object& schema);
  Result<AnyOfSpec, SchemaError> ParseAnyOf(
      const picojson::object& schema, const std::string& keyword
  );
  Result<OneOfSpec, SchemaError> ParseOneOf(const picojson::object& schema);
  Result<AllOfSpec, SchemaError> ParseAllOf(const picojson::object& schema);
  Result<TypeArraySpec, SchemaError> ParseTypeArray(
      const picojson::object& schema, const std::string& rule_name_hint
  );

  std::string ComputeCacheKey(const picojson::value& schema);

  static void WarnUnsupportedKeywords(
      const picojson::object& schema, const std::vector<std::string>& keywords, bool verbose = false
  );

  Config config_;
  picojson::value root_schema_;
  std::unordered_map<std::string, SchemaSpecPtr> ref_cache_;
  std::unordered_map<std::string, SchemaSpecPtr> schema_cache_;
};

std::string SchemaParser::ComputeCacheKey(const picojson::value& schema) {
  static const std::unordered_set<std::string> kSkippedKeys = {
      "title",
      "default",
      "description",
      "examples",
      "deprecated",
      "readOnly",
      "writeOnly",
      "$comment",
      "$schema",
  };

  if (schema.is<picojson::object>()) {
    std::string result = "{";
    std::vector<std::pair<std::string, picojson::value>> sorted_kv;
    for (const auto& kv : schema.get<picojson::object>()) {
      if (kSkippedKeys.count(kv.first) == 0) {
        sorted_kv.push_back(kv);
      }
    }
    std::sort(sorted_kv.begin(), sorted_kv.end(), [](const auto& lhs, const auto& rhs) {
      return lhs.first < rhs.first;
    });
    int64_t idx = 0;
    for (const auto& [key, value] : sorted_kv) {
      if (idx != 0) {
        result += ",";
      }
      ++idx;
      result += "\"" + key + "\":" + ComputeCacheKey(value);
    }
    return result + "}";
  } else if (schema.is<picojson::array>()) {
    std::string result = "[";
    int64_t idx = 0;
    for (const auto& item : schema.get<picojson::array>()) {
      if (idx != 0) {
        result += ",";
      }
      ++idx;
      result += ComputeCacheKey(item);
    }
    return result + "]";
  }
  return schema.serialize(false);
}

void SchemaParser::WarnUnsupportedKeywords(
    const picojson::object& schema, const std::vector<std::string>& keywords, bool verbose
) {
  if (!verbose) {
    return;
  }
  for (const auto& keyword : keywords) {
    if (schema.find(keyword) != schema.end()) {
      XGRAMMAR_LOG(WARNING) << "Keyword " << keyword << " is not supported";
    }
  }
}

Result<SchemaSpecPtr, SchemaError> SchemaParser::Parse(
    const picojson::value& schema,
    const std::string& rule_name_hint,
    std::optional<std::string> default_type
) {
  std::string cache_key = ComputeCacheKey(schema);
  if (schema_cache_.count(cache_key)) {
    return ResultOk(schema_cache_[cache_key]);
  }

  if (schema.is<bool>()) {
    if (!schema.get<bool>()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kUnsatisfiableSchema, "Schema 'false' cannot accept any value"
      );
    }
    auto spec = SchemaSpec::Make(AnySpec{}, cache_key, rule_name_hint);
    schema_cache_[cache_key] = spec;
    return ResultOk(spec);
  }

  if (!schema.is<picojson::object>()) {
    return ResultErr<SchemaError>(
        SchemaErrorType::kInvalidSchema,
        "Schema should be an object or bool, but got " + schema.serialize(false)
    );
  }

  const auto& schema_obj = schema.get<picojson::object>();
  WarnUnsupportedKeywords(
      schema_obj, {"not", "if", "then", "else", "dependentRequired", "dependentSchemas"}
  );

  SchemaSpecPtr result;

  if (schema_obj.count("$ref")) {
    auto ref_result = ParseRef(schema_obj);
    if (ref_result.IsErr()) return ResultErr(std::move(ref_result).UnwrapErr());
    auto ref_spec = std::move(ref_result).Unwrap();
    result = SchemaSpec::Make(std::move(ref_spec), cache_key, rule_name_hint);
  } else if (schema_obj.count("const")) {
    auto const_result = ParseConst(schema_obj);
    if (const_result.IsErr()) return ResultErr(std::move(const_result).UnwrapErr());
    result = SchemaSpec::Make(std::move(const_result).Unwrap(), cache_key, rule_name_hint);
  } else if (schema_obj.count("enum")) {
    auto enum_result = ParseEnum(schema_obj);
    if (enum_result.IsErr()) return ResultErr(std::move(enum_result).UnwrapErr());
    result = SchemaSpec::Make(std::move(enum_result).Unwrap(), cache_key, rule_name_hint);
  } else if (schema_obj.count("anyOf")) {
    auto anyof_result = ParseAnyOf(schema_obj, "anyOf");
    if (anyof_result.IsErr()) return ResultErr(std::move(anyof_result).UnwrapErr());
    result = SchemaSpec::Make(std::move(anyof_result).Unwrap(), cache_key, rule_name_hint);
  } else if (schema_obj.count("oneOf")) {
    auto oneof_result = ParseOneOf(schema_obj);
    if (oneof_result.IsErr()) {
      if (oneof_result.ErrRef().Type() != SchemaErrorType::kUnsupportedSchema) {
        return ResultErr(std::move(oneof_result).UnwrapErr());
      }
      XGRAMMAR_LOG(WARNING) << oneof_result.ErrRef().what();
      auto anyof_result = ParseAnyOf(schema_obj, "oneOf");
      if (anyof_result.IsErr()) return ResultErr(std::move(anyof_result).UnwrapErr());
      result = SchemaSpec::Make(std::move(anyof_result).Unwrap(), cache_key, rule_name_hint);
    } else {
      result = SchemaSpec::Make(std::move(oneof_result).Unwrap(), cache_key, rule_name_hint);
    }
  } else if (schema_obj.count("allOf")) {
    auto allof_result = ParseAllOf(schema_obj);
    if (allof_result.IsErr()) return ResultErr(std::move(allof_result).UnwrapErr());
    result = SchemaSpec::Make(std::move(allof_result).Unwrap(), cache_key, rule_name_hint);
  } else if (schema_obj.count("type") || default_type.has_value()) {
    if (schema_obj.count("type") && schema_obj.at("type").is<picojson::array>()) {
      auto type_array_result = ParseTypeArray(schema_obj, rule_name_hint);
      if (type_array_result.IsErr()) return ResultErr(std::move(type_array_result).UnwrapErr());
      result = SchemaSpec::Make(std::move(type_array_result).Unwrap(), cache_key, rule_name_hint);
    } else {
      if (schema_obj.count("type") && !schema_obj.at("type").is<std::string>()) {
        return ResultErr<SchemaError>(SchemaErrorType::kInvalidSchema, "Type should be a string");
      }
      const std::string& type = schema_obj.count("type") ? schema_obj.at("type").get<std::string>()
                                                         : default_type.value();
      if (type == "integer") {
        auto int_result = ParseInteger(schema_obj);
        if (int_result.IsErr()) return ResultErr(std::move(int_result).UnwrapErr());
        result = SchemaSpec::Make(std::move(int_result).Unwrap(), cache_key, rule_name_hint);
      } else if (type == "number") {
        auto num_result = ParseNumber(schema_obj);
        if (num_result.IsErr()) return ResultErr(std::move(num_result).UnwrapErr());
        result = SchemaSpec::Make(std::move(num_result).Unwrap(), cache_key, rule_name_hint);
      } else if (type == "string") {
        auto str_result = ParseString(schema_obj);
        if (str_result.IsErr()) return ResultErr(std::move(str_result).UnwrapErr());
        result = SchemaSpec::Make(std::move(str_result).Unwrap(), cache_key, rule_name_hint);
      } else if (type == "boolean") {
        auto bool_result = ParseBoolean(schema_obj);
        if (bool_result.IsErr()) return ResultErr(std::move(bool_result).UnwrapErr());
        result = SchemaSpec::Make(std::move(bool_result).Unwrap(), cache_key, rule_name_hint);
      } else if (type == "null") {
        auto null_result = ParseNull(schema_obj);
        if (null_result.IsErr()) return ResultErr(std::move(null_result).UnwrapErr());
        result = SchemaSpec::Make(std::move(null_result).Unwrap(), cache_key, rule_name_hint);
      } else if (type == "array") {
        auto array_result = ParseArray(schema_obj);
        if (array_result.IsErr()) return ResultErr(std::move(array_result).UnwrapErr());
        result = SchemaSpec::Make(std::move(array_result).Unwrap(), cache_key, rule_name_hint);
      } else if (type == "object") {
        auto obj_result = ParseObject(schema_obj);
        if (obj_result.IsErr()) return ResultErr(std::move(obj_result).UnwrapErr());
        result = SchemaSpec::Make(std::move(obj_result).Unwrap(), cache_key, rule_name_hint);
      } else {
        return ResultErr<SchemaError>(
            SchemaErrorType::kInvalidSchema, "Unsupported type \"" + type + "\""
        );
      }
    }
  } else if (schema_obj.count("properties") || schema_obj.count("additionalProperties") ||
             schema_obj.count("unevaluatedProperties")) {
    auto obj_result = ParseObject(schema_obj);
    if (obj_result.IsErr()) return ResultErr(std::move(obj_result).UnwrapErr());
    result = SchemaSpec::Make(std::move(obj_result).Unwrap(), cache_key, rule_name_hint);
  } else if (schema_obj.count("items") || schema_obj.count("prefixItems") ||
             schema_obj.count("unevaluatedItems")) {
    auto array_result = ParseArray(schema_obj);
    if (array_result.IsErr()) return ResultErr(std::move(array_result).UnwrapErr());
    result = SchemaSpec::Make(std::move(array_result).Unwrap(), cache_key, rule_name_hint);
  } else {
    result = SchemaSpec::Make(AnySpec{}, cache_key, rule_name_hint);
  }

  schema_cache_[cache_key] = result;
  return ResultOk(result);
}

Result<IntegerSpec, SchemaError> SchemaParser::ParseInteger(const picojson::object& schema) {
  IntegerSpec spec;

  auto checkAndConvertIntegerBound = [](const picojson::value& value
                                     ) -> Result<int64_t, SchemaError> {
    if (!value.is<int64_t>() && !value.is<double>()) {
      return ResultErr<SchemaError>(SchemaErrorType::kInvalidSchema, "Value must be a number");
    }
    if (value.is<int64_t>()) return ResultOk<int64_t>(value.get<int64_t>());
    double val = value.get<double>();
    if (val != std::floor(val)) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "Integer constraint must be a whole number"
      );
    }
    static const double PROBLEMATIC_MIN = -9223372036854776000.0;
    static const double PROBLEMATIC_MAX = 9223372036854776000.0;
    if (val == PROBLEMATIC_MIN) {
      XGRAMMAR_CHECK(false
      ) << "Integer exceeds minimum limit due to precision loss at 64-bit boundary";
    }

    if (val == PROBLEMATIC_MAX) {
      XGRAMMAR_CHECK(false
      ) << "Integer exceeds maximum limit due to precision loss at 64-bit boundary";
    }
    static const double MAX_INT64_AS_DOUBLE =
        static_cast<double>(std::numeric_limits<int64_t>::max());
    static const double MIN_INT64_AS_DOUBLE =
        static_cast<double>(std::numeric_limits<int64_t>::min());
    XGRAMMAR_CHECK(val <= MAX_INT64_AS_DOUBLE) << "Integer exceeds maximum limit";
    XGRAMMAR_CHECK(val >= MIN_INT64_AS_DOUBLE) << "Integer exceeds minimum limit";
    return ResultOk<int64_t>(static_cast<int64_t>(val));
  };

  auto checkAndConvertMultipleOf = [](const picojson::value& value
                                   ) -> Result<int64_t, SchemaError> {
    double val;
    if (value.is<int64_t>()) {
      val = static_cast<double>(value.get<int64_t>());
    } else if (value.is<double>()) {
      val = value.get<double>();
    } else {
      return ResultErr<SchemaError>(SchemaErrorType::kInvalidSchema, "Value must be a number");
    }
    if (val <= 0) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "multipleOf must be greater than 0"
      );
    }
    if (val != std::floor(val)) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kUnsupportedSchema, "multipleOf for type:integer must be an integer"
      );
    }
    if (val > static_cast<double>(kIntegerMultipleOfMax)) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kUnsupportedSchema,
          "multipleOf for type:integer must be > 0 and <= " + std::to_string(kIntegerMultipleOfMax)
      );
    }
    return ResultOk<int64_t>(static_cast<int64_t>(val));
  };

  if (schema.count("multipleOf")) {
    auto result = checkAndConvertMultipleOf(schema.at("multipleOf"));
    if (result.IsErr()) {
      if (result.ErrRef().Type() != SchemaErrorType::kUnsupportedSchema) {
        return ResultErr(std::move(result).UnwrapErr());
      }
      XGRAMMAR_LOG(WARNING) << result.ErrRef().what() << "; ignoring multipleOf";
    } else {
      spec.multiple_of = std::move(result).Unwrap();
    }
  }
  if (schema.count("minimum")) {
    auto result = checkAndConvertIntegerBound(schema.at("minimum"));
    if (result.IsErr()) return ResultErr(std::move(result).UnwrapErr());
    spec.minimum = std::move(result).Unwrap();
  }
  if (schema.count("maximum")) {
    auto result = checkAndConvertIntegerBound(schema.at("maximum"));
    if (result.IsErr()) return ResultErr(std::move(result).UnwrapErr());
    spec.maximum = std::move(result).Unwrap();
  }
  if (schema.count("exclusiveMinimum")) {
    auto result = checkAndConvertIntegerBound(schema.at("exclusiveMinimum"));
    if (result.IsErr()) return ResultErr(std::move(result).UnwrapErr());
    int64_t val = std::move(result).Unwrap();
    if (val == std::numeric_limits<int64_t>::max()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kUnsatisfiableSchema, "exclusiveMinimum would cause integer overflow"
      );
    }
    spec.exclusive_minimum = val;
  }
  if (schema.count("exclusiveMaximum")) {
    auto result = checkAndConvertIntegerBound(schema.at("exclusiveMaximum"));
    if (result.IsErr()) return ResultErr(std::move(result).UnwrapErr());
    int64_t val = std::move(result).Unwrap();
    if (val == std::numeric_limits<int64_t>::min()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kUnsatisfiableSchema, "exclusiveMaximum would cause integer underflow"
      );
    }
    spec.exclusive_maximum = val;
  }

  EffectiveIntegerRange effective_range = ComputeEffectiveIntegerRange(spec);
  int64_t effective_min = effective_range.start.value_or(std::numeric_limits<int64_t>::min());
  int64_t effective_max = effective_range.end.value_or(std::numeric_limits<int64_t>::max());
  if (effective_min > effective_max) {
    return ResultErr<SchemaError>(
        SchemaErrorType::kUnsatisfiableSchema, "Invalid range: minimum greater than maximum"
    );
  }
  if (spec.multiple_of.has_value()) {
    bool has_lower_bound = spec.minimum.has_value() || spec.exclusive_minimum.has_value();
    bool has_upper_bound = spec.maximum.has_value() || spec.exclusive_maximum.has_value();
    if (has_lower_bound || has_upper_bound) {
      if (!has_lower_bound || !has_upper_bound ||
          IsRangeWidthOverCap(effective_min, effective_max, kIntegerMultipleOfRangeWidthMax)) {
        XGRAMMAR_LOG(WARNING
        ) << "range + multipleOf combination not yet supported; ignoring multipleOf";
        spec.multiple_of.reset();
        return ResultOk(std::move(spec));
      }
      if (!HasMultipleInRange(effective_min, effective_max, *spec.multiple_of)) {
        return ResultErr<SchemaError>(
            SchemaErrorType::kUnsatisfiableSchema, "range contains no multipleOf value"
        );
      }
    }
  }
  return ResultOk(std::move(spec));
}

Result<NumberSpec, SchemaError> SchemaParser::ParseNumber(const picojson::object& schema) {
  if (schema.count("multipleOf")) {
    const auto& value = schema.at("multipleOf");
    if (!value.is<int64_t>() && !value.is<double>()) {
      return ResultErr<SchemaError>(SchemaErrorType::kInvalidSchema, "Value must be a number");
    }
    double multiple_of =
        value.is<int64_t>() ? static_cast<double>(value.get<int64_t>()) : value.get<double>();
    if (multiple_of <= 0) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "multipleOf must be greater than 0"
      );
    }
    XGRAMMAR_LOG(WARNING) << "multipleOf is not supported for type:number; ignoring multipleOf";
  }
  NumberSpec spec;

  auto getDouble = [](const picojson::value& value) -> Result<double, SchemaError> {
    if (!value.is<double>() && !value.is<int64_t>()) {
      return ResultErr<SchemaError>(SchemaErrorType::kInvalidSchema, "Value must be a number");
    }
    return ResultOk<double>(value.get<double>());
  };

  if (schema.count("minimum")) {
    auto result = getDouble(schema.at("minimum"));
    if (result.IsErr()) return ResultErr(std::move(result).UnwrapErr());
    spec.minimum = std::move(result).Unwrap();
  }
  if (schema.count("maximum")) {
    auto result = getDouble(schema.at("maximum"));
    if (result.IsErr()) return ResultErr(std::move(result).UnwrapErr());
    spec.maximum = std::move(result).Unwrap();
  }
  if (schema.count("exclusiveMinimum")) {
    auto result = getDouble(schema.at("exclusiveMinimum"));
    if (result.IsErr()) return ResultErr(std::move(result).UnwrapErr());
    spec.exclusive_minimum = std::move(result).Unwrap();
  }
  if (schema.count("exclusiveMaximum")) {
    auto result = getDouble(schema.at("exclusiveMaximum"));
    if (result.IsErr()) return ResultErr(std::move(result).UnwrapErr());
    spec.exclusive_maximum = std::move(result).Unwrap();
  }

  // The range is empty if any lower bound conflicts with any upper bound. An
  // exclusive bound also rules out equality, so it uses ">=" instead of ">".
  auto empty = []() {
    return ResultErr<SchemaError>(
        SchemaErrorType::kUnsatisfiableSchema, "Invalid range: empty range"
    );
  };

  // minimum (x >= min) vs maximum (x <= max).
  if (spec.minimum && spec.maximum && *spec.minimum > *spec.maximum) {
    return empty();
  }
  // minimum (x >= min) vs exclusiveMaximum (x < exclMax).
  if (spec.minimum && spec.exclusive_maximum && *spec.minimum >= *spec.exclusive_maximum) {
    return empty();
  }
  // exclusiveMinimum (x > exclMin) vs maximum (x <= max).
  if (spec.exclusive_minimum && spec.maximum && *spec.exclusive_minimum >= *spec.maximum) {
    return empty();
  }
  // exclusiveMinimum (x > exclMin) vs exclusiveMaximum (x < exclMax).
  if (spec.exclusive_minimum && spec.exclusive_maximum &&
      *spec.exclusive_minimum >= *spec.exclusive_maximum) {
    return empty();
  }
  return ResultOk(std::move(spec));
}

Result<StringSpec, SchemaError> SchemaParser::ParseString(const picojson::object& schema) {
  StringSpec spec;
  if (schema.count("format")) spec.format = schema.at("format").get<std::string>();
  if (schema.count("pattern")) spec.pattern = schema.at("pattern").get<std::string>();
  // Lengths become int32 repetition bounds. A minimum beyond int32 can never be satisfied; a
  // maximum beyond it is unbounded in practice. Neither may wrap around when converted.
  constexpr int64_t kMaxBound = std::numeric_limits<int32_t>::max();
  if (schema.count("minLength")) {
    if (!schema.at("minLength").is<int64_t>() ||
        schema.at("minLength").get<int64_t>() > kMaxBound) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema,
          "minLength must be an integer not exceeding " + std::to_string(kMaxBound)
      );
    }
    spec.min_length = static_cast<int>(schema.at("minLength").get<int64_t>());
  }
  if (schema.count("maxLength")) {
    if (!schema.at("maxLength").is<int64_t>()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "maxLength must be an integer"
      );
    }
    if (schema.at("maxLength").get<int64_t>() <= kMaxBound) {
      spec.max_length = static_cast<int>(schema.at("maxLength").get<int64_t>());
    }
  }
  if (spec.max_length != -1 && spec.min_length > spec.max_length) {
    return ResultErr<SchemaError>(
        SchemaErrorType::kUnsatisfiableSchema,
        "minLength " + std::to_string(spec.min_length) + " is greater than maxLength " +
            std::to_string(spec.max_length)
    );
  }
  // A pattern or built-in format takes the whole GenerateString branch, so minLength/maxLength
  // would be dropped silently (issue #749). Warn here rather than in GenerateString: the XML
  // tool-calling converter overrides that method, but every converter goes through ParseString.
  if (spec.min_length != 0 || spec.max_length != -1) {
    const char* generative = nullptr;
    if (spec.pattern.has_value()) {
      generative = "pattern";
    } else if (spec.format.has_value() && JSONSchemaConverter::IsBuiltinFormat(*spec.format)) {
      generative = "format";
    }
    if (generative != nullptr) {
      XGRAMMAR_LOG(WARNING) << generative
                            << " combined with minLength/maxLength is not supported; ignoring "
                               "minLength/maxLength";
    }
  }
  return ResultOk(std::move(spec));
}

Result<BooleanSpec, SchemaError> SchemaParser::ParseBoolean(const picojson::object&) {
  return ResultOk(BooleanSpec{});
}

Result<NullSpec, SchemaError> SchemaParser::ParseNull(const picojson::object&) {
  return ResultOk(NullSpec{});
}

Result<ArraySpec, SchemaError> SchemaParser::ParseArray(const picojson::object& schema) {
  WarnUnsupportedKeywords(schema, {"uniqueItems", "contains", "minContains", "maxContains"});
  ArraySpec spec;

  if (schema.count("prefixItems")) {
    if (!schema.at("prefixItems").is<picojson::array>()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "prefixItems must be an array"
      );
    }
    for (const auto& item : schema.at("prefixItems").get<picojson::array>()) {
      if (item.is<bool>() && !item.get<bool>()) {
        return ResultErr<SchemaError>(
            SchemaErrorType::kUnsatisfiableSchema, "prefixItems contains false"
        );
      } else if (!item.is<picojson::object>()) {
        return ResultErr<SchemaError>(
            SchemaErrorType::kInvalidSchema, "prefixItems must be an array of objects or booleans"
        );
      }
      auto item_result = Parse(item, "prefix_item");
      if (item_result.IsErr()) return ResultErr(std::move(item_result).UnwrapErr());
      spec.prefix_items.push_back(std::move(item_result).Unwrap());
    }
  }

  if (schema.count("items")) {
    auto items_value = schema.at("items");
    if (!items_value.is<bool>() && !items_value.is<picojson::object>()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "items must be a boolean or an object"
      );
    }
    if (items_value.is<bool>() && !items_value.get<bool>()) {
      spec.allow_additional_items = false;
    } else {
      spec.allow_additional_items = true;
      auto items_result = Parse(items_value, "item");
      if (items_result.IsErr()) return ResultErr(std::move(items_result).UnwrapErr());
      spec.additional_items = std::move(items_result).Unwrap();
    }
  } else if (schema.count("unevaluatedItems")) {
    auto unevaluated_items_value = schema.at("unevaluatedItems");
    if (!unevaluated_items_value.is<bool>() && !unevaluated_items_value.is<picojson::object>()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "unevaluatedItems must be a boolean or an object"
      );
    }
    if (unevaluated_items_value.is<bool>() && !unevaluated_items_value.get<bool>()) {
      spec.allow_additional_items = false;
    } else {
      spec.allow_additional_items = true;
      auto items_result = Parse(unevaluated_items_value, "unevaluated_item");
      if (items_result.IsErr()) return ResultErr(std::move(items_result).UnwrapErr());
      spec.additional_items = std::move(items_result).Unwrap();
    }
  } else if (!config_.strict_mode) {
    spec.allow_additional_items = true;
    spec.additional_items = SchemaSpec::Make(AnySpec{}, "", "any");
  } else {
    spec.allow_additional_items = false;
  }

  if (schema.count("minItems")) {
    if (!schema.at("minItems").is<int64_t>()) {
      return ResultErr<SchemaError>(SchemaErrorType::kInvalidSchema, "minItems must be an integer");
    }
    spec.min_items = std::max(static_cast<int64_t>(0), schema.at("minItems").get<int64_t>());
  }
  if (schema.count("minContains")) {
    if (!schema.at("minContains").is<int64_t>()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "minContains must be an integer"
      );
    }
    spec.min_items = std::max(spec.min_items, schema.at("minContains").get<int64_t>());
  }
  if (schema.count("maxItems")) {
    if (!schema.at("maxItems").is<int64_t>() || schema.at("maxItems").get<int64_t>() < 0) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "maxItems must be a non-negative integer"
      );
    }
    spec.max_items = schema.at("maxItems").get<int64_t>();
  }
  // Item counts become int32 repetition bounds, see ParseString for the rationale.
  constexpr int64_t kMaxBound = std::numeric_limits<int32_t>::max();
  if (spec.min_items > kMaxBound) {
    return ResultErr<SchemaError>(
        SchemaErrorType::kInvalidSchema,
        "minItems and minContains must not exceed " + std::to_string(kMaxBound)
    );
  }
  if (spec.max_items > kMaxBound) {
    spec.max_items = -1;
  }

  if (spec.max_items != -1 && spec.min_items > spec.max_items) {
    return ResultErr<SchemaError>(
        SchemaErrorType::kUnsatisfiableSchema,
        "minItems is greater than maxItems: " + std::to_string(spec.min_items) + " > " +
            std::to_string(spec.max_items)
    );
  }
  if (spec.max_items != -1 && spec.max_items < static_cast<int64_t>(spec.prefix_items.size())) {
    return ResultErr<SchemaError>(
        SchemaErrorType::kUnsatisfiableSchema,
        "maxItems is less than the number of prefixItems: " + std::to_string(spec.max_items) +
            " < " + std::to_string(spec.prefix_items.size())
    );
  }
  if (!spec.allow_additional_items) {
    int64_t prefix_size = static_cast<int64_t>(spec.prefix_items.size());
    if (prefix_size < spec.min_items) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kUnsatisfiableSchema,
          "minItems is greater than the number of prefixItems, but additional items are not "
          "allowed: " +
              std::to_string(spec.min_items) + " > " + std::to_string(prefix_size)
      );
    }
    if (spec.max_items != -1 && prefix_size > spec.max_items) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kUnsatisfiableSchema,
          "maxItems is less than the number of prefixItems, but additional items are not "
          "allowed: " +
              std::to_string(spec.max_items) + " < " + std::to_string(prefix_size)
      );
    }
  }
  return ResultOk(std::move(spec));
}

Result<ObjectSpec, SchemaError> SchemaParser::ParseObject(const picojson::object& schema) {
  ObjectSpec spec;

  if (schema.count("properties")) {
    if (!schema.at("properties").is<picojson::object>()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "properties must be an object"
      );
    }
    auto properties_obj = schema.at("properties").get<picojson::object>();
    for (const auto& key : properties_obj.ordered_keys()) {
      auto prop_result = Parse(properties_obj.at(key), key);
      if (prop_result.IsErr()) return ResultErr(std::move(prop_result).UnwrapErr());
      spec.properties.push_back({key, std::move(prop_result).Unwrap()});
    }
  }

  if (schema.count("required")) {
    if (!schema.at("required").is<picojson::array>()) {
      return ResultErr<SchemaError>(SchemaErrorType::kInvalidSchema, "required must be an array");
    }
    for (const auto& req : schema.at("required").get<picojson::array>()) {
      spec.required.insert(req.get<std::string>());
    }
  }

  if (schema.count("patternProperties")) {
    if (!schema.at("patternProperties").is<picojson::object>()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "patternProperties must be an object"
      );
    }
    auto pattern_props = schema.at("patternProperties").get<picojson::object>();
    for (const auto& key : pattern_props.ordered_keys()) {
      auto prop_result = Parse(pattern_props.at(key), "pattern_prop");
      if (prop_result.IsErr()) return ResultErr(std::move(prop_result).UnwrapErr());
      spec.pattern_properties.push_back({key, std::move(prop_result).Unwrap()});
    }
  }

  if (schema.count("propertyNames")) {
    if (!schema.at("propertyNames").is<picojson::object>()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "propertyNames must be an object"
      );
    }
    auto property_names_obj = schema.at("propertyNames").get<picojson::object>();
    if (property_names_obj.count("type") && property_names_obj.at("type").is<std::string>() &&
        property_names_obj.at("type").get<std::string>() != "string") {
      return ResultErr<SchemaError>(
          SchemaErrorType::kUnsatisfiableSchema,
          "propertyNames must be an object that validates string"
      );
    }
    auto prop_names_result = Parse(schema.at("propertyNames"), "property_name", "string");
    if (prop_names_result.IsErr()) return ResultErr(std::move(prop_names_result).UnwrapErr());
    spec.property_names = std::move(prop_names_result).Unwrap();
  }

  spec.allow_additional_properties = !config_.strict_mode;
  if (schema.count("additionalProperties")) {
    auto add_props = schema.at("additionalProperties");
    if (add_props.is<bool>()) {
      spec.allow_additional_properties = add_props.get<bool>();
    } else {
      spec.allow_additional_properties = true;
      auto add_props_result = Parse(add_props, "additional");
      if (add_props_result.IsErr()) return ResultErr(std::move(add_props_result).UnwrapErr());
      spec.additional_properties_schema = std::move(add_props_result).Unwrap();
    }
  }

  spec.allow_unevaluated_properties = true;
  if (schema.count("additionalProperties")) {
    spec.allow_unevaluated_properties = spec.allow_additional_properties;
  } else if (schema.count("unevaluatedProperties")) {
    auto uneval_props = schema.at("unevaluatedProperties");
    if (uneval_props.is<bool>()) {
      spec.allow_unevaluated_properties = uneval_props.get<bool>();
    } else {
      spec.allow_unevaluated_properties = true;
      auto uneval_result = Parse(uneval_props, "unevaluated");
      if (uneval_result.IsErr()) return ResultErr(std::move(uneval_result).UnwrapErr());
      spec.unevaluated_properties_schema = std::move(uneval_result).Unwrap();
    }
  } else if (config_.strict_mode) {
    spec.allow_unevaluated_properties = false;
  }

  if (schema.count("minProperties")) {
    if (!schema.at("minProperties").is<int64_t>()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "minProperties must be an integer"
      );
    }
    spec.min_properties = static_cast<int>(schema.at("minProperties").get<int64_t>());
    if (spec.min_properties < 0) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kUnsatisfiableSchema, "minProperties must be a non-negative integer"
      );
    }
  }
  if (schema.count("maxProperties")) {
    if (!schema.at("maxProperties").is<int64_t>()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "maxProperties must be an integer"
      );
    }
    spec.max_properties = static_cast<int>(schema.at("maxProperties").get<int64_t>());
    if (spec.max_properties < 0) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kUnsatisfiableSchema, "maxProperties must be a non-negative integer"
      );
    }
  }

  if (spec.max_properties != -1 && spec.min_properties > spec.max_properties) {
    return ResultErr<SchemaError>(
        SchemaErrorType::kUnsatisfiableSchema,
        "minProperties is greater than maxProperties: " + std::to_string(spec.min_properties) +
            " > " + std::to_string(spec.max_properties)
    );
  }
  if (spec.max_properties != -1 && static_cast<int>(spec.required.size()) > spec.max_properties) {
    return ResultErr<SchemaError>(
        SchemaErrorType::kUnsatisfiableSchema,
        "maxProperties is less than the number of required properties: " +
            std::to_string(spec.max_properties) + " < " + std::to_string(spec.required.size())
    );
  }
  if (spec.pattern_properties.empty() && !spec.property_names &&
      !spec.allow_additional_properties && !spec.allow_unevaluated_properties &&
      spec.min_properties > static_cast<int>(spec.properties.size())) {
    return ResultErr<SchemaError>(
        SchemaErrorType::kUnsatisfiableSchema,
        "minProperties is greater than the number of properties, but additional properties aren't "
        "allowed: " +
            std::to_string(spec.min_properties) + " > " + std::to_string(spec.properties.size())
    );
  }
  return ResultOk(std::move(spec));
}

Result<ConstSpec, SchemaError> SchemaParser::ParseConst(const picojson::object& schema) {
  ConstSpec spec;
  spec.json_value = schema.at("const").serialize();
  return ResultOk(std::move(spec));
}

Result<EnumSpec, SchemaError> SchemaParser::ParseEnum(const picojson::object& schema) {
  EnumSpec spec;
  if (!schema.at("enum").is<picojson::array>()) {
    return ResultErr<SchemaError>(SchemaErrorType::kInvalidSchema, "enum must be an array");
  }
  const auto& enum_array = schema.at("enum").get<picojson::array>();
  if (enum_array.empty()) {
    return ResultErr<SchemaError>(SchemaErrorType::kInvalidSchema, "enum array must not be empty");
  }
  for (const auto& value : enum_array) {
    spec.json_values.push_back(value.serialize());
  }
  return ResultOk(std::move(spec));
}

Result<RefSpec, SchemaError> SchemaParser::ParseRef(const picojson::object& schema) {
  if (!schema.at("$ref").is<std::string>()) {
    return ResultErr<SchemaError>(SchemaErrorType::kInvalidSchema, "$ref must be a string");
  }
  RefSpec spec;
  spec.uri = schema.at("$ref").get<std::string>();
  return ResultOk(std::move(spec));
}

Result<SchemaSpecPtr, SchemaError> SchemaParser::ResolveRef(
    const std::string& uri, const std::string& rule_name_hint
) {
  if (ref_cache_.count(uri)) return ResultOk(ref_cache_[uri]);

  if (uri == "#") {
    auto placeholder = SchemaSpec::Make(AnySpec{}, "", "root");
    ref_cache_[uri] = placeholder;
    auto result = Parse(root_schema_, "root");
    if (result.IsErr()) return ResultErr(std::move(result).UnwrapErr());
    auto resolved = std::move(result).Unwrap();
    ref_cache_[uri] = resolved;
    return ResultOk(resolved);
  }

  if (uri.size() < 2 || uri[0] != '#' || uri[1] != '/') {
    XGRAMMAR_LOG(WARNING) << "URI should either be '#' or start with '#/' but got " << uri;
    return ResultOk(SchemaSpec::Make(AnySpec{}, "", "any"));
  }

  std::vector<std::string> parts;
  std::stringstream ss(uri.substr(2));
  std::string part;
  std::string new_rule_name_prefix;
  while (std::getline(ss, part, '/')) {
    if (!part.empty()) parts.push_back(part);
    if (!new_rule_name_prefix.empty()) new_rule_name_prefix += "_";
    for (const auto& c : part) {
      if (std::isalpha(c) || c == '_' || c == '-' || c == '.') new_rule_name_prefix += c;
    }
  }

  auto current = std::cref(root_schema_);
  for (const auto& p : parts) {
    if (!current.get().is<picojson::object>() || !current.get().contains(p)) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "Cannot find field " + p + " in " + uri
      );
    }
    current = current.get().get(p);
  }

  auto result = Parse(current, new_rule_name_prefix);
  if (result.IsErr()) return ResultErr(std::move(result).UnwrapErr());
  auto resolved = std::move(result).Unwrap();
  ref_cache_[uri] = resolved;
  return ResultOk(resolved);
}

Result<AnyOfSpec, SchemaError> SchemaParser::ParseAnyOf(
    const picojson::object& schema, const std::string& keyword
) {
  AnyOfSpec spec;
  if (!schema.at(keyword).is<picojson::array>()) {
    return ResultErr<SchemaError>(SchemaErrorType::kInvalidSchema, keyword + " must be an array");
  }
  int idx = 0;
  for (const auto& option : schema.at(keyword).get<picojson::array>()) {
    auto option_result = Parse(option, "case_" + std::to_string(idx));
    if (option_result.IsErr()) return ResultErr(std::move(option_result).UnwrapErr());
    spec.options.push_back(std::move(option_result).Unwrap());
    ++idx;
  }
  return ResultOk(std::move(spec));
}

Result<OneOfSpec, SchemaError> SchemaParser::ParseOneOf(const picojson::object& schema) {
  OneOfSpec spec;
  if (!schema.at("oneOf").is<picojson::array>()) {
    return ResultErr<SchemaError>(SchemaErrorType::kInvalidSchema, "oneOf must be an array");
  }

  const auto& options = schema.at("oneOf").get<picojson::array>();
  if (options.empty()) {
    return ResultErr<SchemaError>(SchemaErrorType::kUnsupportedSchema, kUnsupportedOneOfMessage);
  }

  int idx = 0;
  for (const auto& option : options) {
    auto option_result = Parse(option, "case_" + std::to_string(idx));
    if (option_result.IsErr()) return ResultErr(std::move(option_result).UnwrapErr());
    spec.options.push_back(std::move(option_result).Unwrap());
    ++idx;
  }

  if (!TryProvePairwiseDisjointOneOf(options)) {
    return ResultErr<SchemaError>(SchemaErrorType::kUnsupportedSchema, kUnsupportedOneOfMessage);
  }

  return ResultOk(std::move(spec));
}

Result<AllOfSpec, SchemaError> SchemaParser::ParseAllOf(const picojson::object& schema) {
  AllOfSpec spec;
  if (!schema.at("allOf").is<picojson::array>()) {
    return ResultErr<SchemaError>(SchemaErrorType::kInvalidSchema, "allOf must be an array");
  }
  int idx = 0;
  for (const auto& sub_schema : schema.at("allOf").get<picojson::array>()) {
    auto sub_result = Parse(sub_schema, "all_" + std::to_string(idx));
    if (sub_result.IsErr()) return ResultErr(std::move(sub_result).UnwrapErr());
    spec.schemas.push_back(std::move(sub_result).Unwrap());
    ++idx;
  }
  return ResultOk(std::move(spec));
}

Result<TypeArraySpec, SchemaError> SchemaParser::ParseTypeArray(
    const picojson::object& schema, const std::string& rule_name_hint
) {
  TypeArraySpec spec;
  auto type_array = schema.at("type").get<picojson::array>();
  picojson::object schema_copy = schema;
  if (type_array.empty()) {
    schema_copy.erase("type");
    auto any_result = Parse(picojson::value(schema_copy), rule_name_hint);
    if (any_result.IsErr()) return ResultErr(std::move(any_result).UnwrapErr());
    spec.type_schemas.push_back(std::move(any_result).Unwrap());
    return ResultOk(std::move(spec));
  }
  for (const auto& type : type_array) {
    if (!type.is<std::string>()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "type must be a string or an array of strings"
      );
    }
    schema_copy["type"] = type;
    auto type_result =
        Parse(picojson::value(schema_copy), rule_name_hint + "_" + type.get<std::string>());
    if (type_result.IsErr()) return ResultErr(std::move(type_result).UnwrapErr());
    spec.type_schemas.push_back(std::move(type_result).Unwrap());
  }
  return ResultOk(std::move(spec));
}

}  // namespace

// ==================== IndentManager Implementation ====================

IndentManager::IndentManager(
    std::optional<int> indent,
    const std::string& separator,
    bool any_whitespace,
    std::optional<int> max_whitespace_cnt
)
    : any_whitespace_(any_whitespace),
      enable_newline_(indent.has_value()),
      indent_(indent.value_or(0)),
      separator_(separator),
      total_indent_(0),
      is_first_({true}),
      max_whitespace_cnt_(max_whitespace_cnt) {
  if (max_whitespace_cnt.has_value() && max_whitespace_cnt.value() <= 0) {
    XGRAMMAR_LOG(FATAL) << "max_whitespace_cnt must be positive.";
  }
}

void IndentManager::StartIndent() {
  total_indent_ += indent_;
  is_first_.push_back(true);
}

void IndentManager::EndIndent() {
  total_indent_ -= indent_;
  is_first_.pop_back();
}

std::string IndentManager::StartSeparator() {
  if (any_whitespace_) {
    if (!max_whitespace_cnt_.has_value()) {
      return "[ \\n\\r\\t]*";
    } else {
      return "[ \\n\\r\\t]{0," + std::to_string(max_whitespace_cnt_.value()) + "}";
    }
  }
  if (!enable_newline_) {
    return "\"\"";
  }
  return "\"\\n" + std::string(total_indent_, ' ') + "\"";
}

std::string IndentManager::MiddleSeparator() {
  if (any_whitespace_) {
    std::string whitespace_part;
    if (!max_whitespace_cnt_.has_value()) {
      whitespace_part = "[ \\n\\r\\t]*";
    } else {
      whitespace_part = "[ \\n\\r\\t]{0," + std::to_string(max_whitespace_cnt_.value()) + "}";
    }
    return whitespace_part + " \"" + separator_ + "\" " + whitespace_part;
  }
  if (!enable_newline_) {
    return "\"" + separator_ + "\"";
  }
  return "\"" + separator_ + "\\n" + std::string(total_indent_, ' ') + "\"";
}

std::string IndentManager::EndSeparator() {
  if (any_whitespace_) {
    if (!max_whitespace_cnt_.has_value()) {
      return "[ \\n\\r\\t]*";
    } else {
      return "[ \\n\\r\\t]{0," + std::to_string(max_whitespace_cnt_.value()) + "}";
    }
  }
  if (!enable_newline_) {
    return "\"\"";
  }
  return "\"\\n" + std::string(total_indent_ - indent_, ' ') + "\"";
}

std::string IndentManager::EmptySeparator() {
  if (any_whitespace_) {
    if (!max_whitespace_cnt_.has_value()) {
      return "[ \\n\\r\\t]*";
    } else {
      return "[ \\n\\r\\t]{0," + std::to_string(max_whitespace_cnt_.value()) + "}";
    }
  }
  return "\"\"";
}

std::string IndentManager::NextSeparator(bool is_end) {
  if (any_whitespace_) {
    if (is_first_.back() || is_end) {
      is_first_.back() = false;
      if (!max_whitespace_cnt_.has_value()) {
        return "[ \\n\\r\\t]*";
      } else {
        return "[ \\n\\r\\t]{0," + std::to_string(max_whitespace_cnt_.value()) + "}";
      }
    } else {
      std::string whitespace_part;
      if (!max_whitespace_cnt_.has_value()) {
        whitespace_part = "[ \\n\\r\\t]*";
      } else {
        whitespace_part = "[ \\n\\r\\t]{0," + std::to_string(max_whitespace_cnt_.value()) + "}";
      }
      return whitespace_part + " \"" + separator_ + "\" " + whitespace_part;
    }
  }

  std::string res = "";
  if (!is_first_.back() && !is_end) {
    res += separator_;
  }
  is_first_.back() = false;

  if (enable_newline_) {
    res += "\\n";
  }

  if (!is_end) {
    res += std::string(total_indent_, ' ');
  } else {
    res += std::string(total_indent_ - indent_, ' ');
  }

  return "\"" + res + "\"";
}

// ==================== Static Constants ====================

const std::string JSONSchemaConverter::kBasicAny = "basic_any";
const std::string JSONSchemaConverter::kBasicInteger = "basic_integer";
const std::string JSONSchemaConverter::kBasicNumber = "basic_number";
const std::string JSONSchemaConverter::kBasicString = "basic_string";
const std::string JSONSchemaConverter::kBasicBoolean = "basic_boolean";
const std::string JSONSchemaConverter::kBasicNull = "basic_null";
const std::string JSONSchemaConverter::kBasicArray = "basic_array";
const std::string JSONSchemaConverter::kBasicObject = "basic_object";
const std::string JSONSchemaConverter::kBasicEscape = "basic_escape";
const std::string JSONSchemaConverter::kBasicStringSub = "basic_string_sub";

// ==================== JSONSchemaConverter Implementation ====================

JSONSchemaConverter::JSONSchemaConverter(
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool any_whitespace,
    std::optional<int> max_whitespace_cnt,
    RefResolver ref_resolver,
    bool any_order,
    std::vector<std::string> excludes
)
    : indent_manager_(
          indent,
          separators.has_value() ? separators->first
                                 : (any_whitespace ? "," : (indent.has_value() ? "," : ", ")),
          any_whitespace,
          max_whitespace_cnt
      ),
      any_whitespace_(any_whitespace),
      max_whitespace_cnt_(max_whitespace_cnt),
      any_order_(any_order),
      excludes_(std::move(excludes)),
      ref_resolver_(std::move(ref_resolver)) {
  std::string colon_sep =
      separators.has_value() ? separators->second : (any_whitespace ? ":" : ": ");
  std::string whitespace = GetWhitespacePattern();
  colon_expr_id_ = FormattingExpression(
      any_whitespace ? whitespace + " \"" + colon_sep + "\" " + whitespace : "\"" + colon_sep + "\""
  );
}

Grammar JSONSchemaConverter::Convert(const SchemaSpecPtr& spec) {
  AddBasicRules();

  // Register the root rule for circular reference handling
  // This allows $ref: "#" to resolve to "root"
  int32_t root_rule_id = builder_.AddEmptyRuleWithHint("root");
  std::string root_rule_name = builder_.GetRule(root_rule_id).name;
  uri_to_rule_id_[RefCacheKey("#")] = root_rule_id;

  // Check if the spec can be directly mapped to an existing rule
  auto cached_rule = GetCache(spec->cache_key);
  if (cached_rule.has_value()) {
    // Root schema matches a basic type, just reference it
    builder_.UpdateRuleBody(root_rule_id, RuleRef(*cached_rule));
  } else {
    // Generate the rule body
    if (!spec->cache_key.empty()) {
      AddCache(spec->cache_key, root_rule_id);
    }
    builder_.UpdateRuleBody(root_rule_id, GenerateFromSpec(spec, root_rule_name));
  }
  return builder_.Get(root_rule_id);
}

void JSONSchemaConverter::AddBasicRules() { AddBasicRules({}); }

void JSONSchemaConverter::AddBasicRules(const std::vector<std::string>& additional_rule_names) {
  std::vector<std::string> basic_rule_names = {
      kBasicEscape,
      kBasicStringSub,
      kBasicAny,
      kBasicInteger,
      kBasicNumber,
      kBasicString,
      kBasicBoolean,
      kBasicNull,
      kBasicArray,
      kBasicObject,
  };
  basic_rule_names.insert(
      basic_rule_names.end(), additional_rule_names.begin(), additional_rule_names.end()
  );
  for (const auto& name : basic_rule_names) {
    builder_.AddEmptyRule(name);
  }
  AddHelperRules();

  // Create basic rules with a temporary indent manager for compact format
  auto saved_indent_manager = indent_manager_;
  indent_manager_ = IndentManager(
      std::nullopt,
      any_whitespace_ ? "," : ", ",
      any_whitespace_,
      any_whitespace_ ? max_whitespace_cnt_ : std::nullopt
  );

  // basic_any - use "{}" as the cache key for empty schema
  auto any_spec = SchemaSpec::Make(AnySpec{}, "{}", kBasicAny);
  builder_.UpdateRuleBody(kBasicAny, GenerateAny(std::get<AnySpec>(any_spec->spec), kBasicAny));
  AddCache("{}", builder_.GetRuleId(kBasicAny));

  // basic_integer - cache_key matches SchemaParser::ComputeCacheKey for {"type": "integer"}
  constexpr const char* kIntegerCacheKey = "{\"type\":\"integer\"}";
  builder_.UpdateRuleBody(kBasicInteger, GenerateInteger(IntegerSpec{}, kBasicInteger));
  AddCache(kIntegerCacheKey, builder_.GetRuleId(kBasicInteger));

  // basic_number - cache_key matches SchemaParser::ComputeCacheKey for {"type": "number"}
  constexpr const char* kNumberCacheKey = "{\"type\":\"number\"}";
  builder_.UpdateRuleBody(kBasicNumber, GenerateNumber(NumberSpec{}, kBasicNumber));
  AddCache(kNumberCacheKey, builder_.GetRuleId(kBasicNumber));

  constexpr const char* kStringCacheKey = "{\"type\":\"string\"}";
  builder_.UpdateRuleBody(kBasicString, Sequence({ByteString("\""), RuleRef(kBasicStringSub)}));
  AddCache(kStringCacheKey, builder_.GetRuleId(kBasicString));

  // basic_boolean - cache_key matches SchemaParser::ComputeCacheKey for {"type": "boolean"}
  constexpr const char* kBooleanCacheKey = "{\"type\":\"boolean\"}";
  builder_.UpdateRuleBody(kBasicBoolean, GenerateBoolean(BooleanSpec{}, kBasicBoolean));
  AddCache(kBooleanCacheKey, builder_.GetRuleId(kBasicBoolean));

  // basic_null - cache_key matches SchemaParser::ComputeCacheKey for {"type": "null"}
  constexpr const char* kNullCacheKey = "{\"type\":\"null\"}";
  builder_.UpdateRuleBody(kBasicNull, GenerateNull(NullSpec{}, kBasicNull));
  AddCache(kNullCacheKey, builder_.GetRuleId(kBasicNull));

  // basic_array - cache_key matches SchemaParser::ComputeCacheKey for {"type": "array"}
  constexpr const char* kArrayCacheKey = "{\"type\":\"array\"}";
  ArraySpec array_spec_val;
  array_spec_val.allow_additional_items = true;
  array_spec_val.additional_items = any_spec;
  builder_.UpdateRuleBody(kBasicArray, GenerateArray(array_spec_val, kBasicArray));
  AddCache(kArrayCacheKey, builder_.GetRuleId(kBasicArray));

  // basic_object - cache_key matches SchemaParser::ComputeCacheKey for {"type": "object"}
  constexpr const char* kObjectCacheKey = "{\"type\":\"object\"}";
  ObjectSpec obj_spec_val;
  obj_spec_val.allow_additional_properties = true;
  obj_spec_val.additional_properties_schema = any_spec;
  builder_.UpdateRuleBody(kBasicObject, GenerateObject(obj_spec_val, kBasicObject));
  AddCache(kObjectCacheKey, builder_.GetRuleId(kBasicObject));

  indent_manager_ = saved_indent_manager;
}

// The content of a JSON string, escapes included, as RegexExpression parses it.
static constexpr const char kJSONStringBodyRegex[] =
    R"(([^"\\\x00-\x1f]|\\(["\\/bfnrt]|u[0-9a-fA-F]{4}))*)";

void JSONSchemaConverter::AddHelperRules() {
  if (max_whitespace_cnt_.has_value()) {
    // Preserve historical helper-rule numbering after grammar optimization. The text parser
    // allocated one initial bounded-repetition helper that dead-code elimination later removed.
    builder_.AddRuleWithHint(kBasicStringSub, Empty());
  }
  int32_t escaped_character = builder_.AddCharacterClass(
      {{'"', '"'},
       {'\\', '\\'},
       {'/', '/'},
       {'b', 'b'},
       {'f', 'f'},
       {'n', 'n'},
       {'r', 'r'},
       {'t', 't'}}
  );
  int32_t hexadecimal_character = builder_.AddCharacterClass({{'A', 'F'}, {'a', 'f'}, {'0', '9'}});
  int32_t unicode_escape = Sequence(
      {ByteString("u"),
       hexadecimal_character,
       hexadecimal_character,
       hexadecimal_character,
       hexadecimal_character}
  );
  builder_.UpdateRuleBody(kBasicEscape, Choice({escaped_character, unicode_escape}));

  int32_t normal_character = builder_.AddCharacterClass(
      {{0, 0x1f}, {'"', '"'}, {'\\', '\\'}, {'\r', '\r'}, {'\n', '\n'}}, true
  );
  int32_t string_sub_ref = RuleRef(kBasicStringSub);
  int32_t string_sub_body = Choice(
      {ByteString("\""),
       Sequence({normal_character, string_sub_ref}),
       Sequence({ByteString("\\"), RuleRef(kBasicEscape), string_sub_ref})}
  );
  builder_.UpdateRuleBody(kBasicStringSub, string_sub_body);
  if (!excludes_.empty()) {
    builder_.UpdateRuleBody(
        kBasicStringSub, ExcludingString(kJSONStringBodyRegex, kBasicStringSub, true)
    );
  }
  int32_t closing_context =
      builder_.AddCharacterClass({{',', ','}, {'}', '}'}, {']', ']'}, {':', ':'}});
  builder_.UpdateLookaheadAssertion(
      kBasicStringSub, Sequence({WhitespaceExpression(), closing_context})
  );
}

// Keep converter-specific node reuse local; GrammarBuilder creates all AST nodes.
int32_t JSONSchemaConverter::Empty() {
  if (!empty_expr_id_.has_value()) {
    empty_expr_id_ = builder_.AddEmptyStr();
  }
  return *empty_expr_id_;
}

int32_t JSONSchemaConverter::Unsatisfiable() {
  if (!unsatisfiable_expr_id_.has_value()) {
    unsatisfiable_expr_id_ = builder_.AddCharacterClass({{0, 0x10ffff}}, true);
  }
  return *unsatisfiable_expr_id_;
}

int32_t JSONSchemaConverter::ByteString(const std::string& value) {
  auto it = byte_string_expr_ids_.find(value);
  if (it != byte_string_expr_ids_.end()) {
    return it->second;
  }
  int32_t expr_id = value.empty() ? Empty() : builder_.AddByteString(value);
  byte_string_expr_ids_[value] = expr_id;
  return expr_id;
}

int32_t JSONSchemaConverter::TagDispatch(
    bool loop_after_dispatch, std::vector<std::string> excludes
) {
  return builder_.AddTagDispatch(
      Grammar::Impl::TagDispatch{{}, loop_after_dispatch, std::move(excludes)}
  );
}

int32_t JSONSchemaConverter::RuleRef(int32_t rule_id) {
  auto it = rule_ref_expr_ids_.find(rule_id);
  if (it != rule_ref_expr_ids_.end()) {
    return it->second;
  }
  int32_t expr_id = builder_.AddRuleRef(rule_id);
  rule_ref_expr_ids_[rule_id] = expr_id;
  return expr_id;
}

int32_t JSONSchemaConverter::RuleRef(const std::string& rule_name) {
  int32_t rule_id = builder_.GetRuleId(rule_name);
  XGRAMMAR_CHECK(rule_id != -1) << "Rule " << rule_name << " is not allocated";
  return RuleRef(rule_id);
}

int32_t JSONSchemaConverter::Sequence(const std::vector<int32_t>& elements) {
  if (elements.empty()) {
    return Empty();
  }
  if (elements.size() == 1) {
    return elements[0];
  }
  return builder_.AddSequence(elements);
}

int32_t JSONSchemaConverter::Choice(const std::vector<int32_t>& choices) {
  if (choices.empty()) {
    return Empty();
  }
  if (choices.size() == 1) {
    return choices[0];
  }
  return builder_.AddChoices(choices);
}

int32_t JSONSchemaConverter::Repeat(
    const std::string& rule_name_hint, int32_t expr_id, int32_t min_count, int32_t max_count
) {
  if (min_count == 0 && max_count == 0) {
    return Empty();
  }
  if (min_count == 1 && max_count == 1) {
    return expr_id;
  }
  if (min_count == 0 && max_count == 1) {
    return Choice({Empty(), expr_id});
  }
  if (min_count == 0 && max_count == -1) {
    auto expr = builder_.GetGrammarExpr(expr_id);
    if (expr.type == GrammarBuilder::GrammarExprType::kCharacterClass) {
      std::vector<int32_t> data(expr.begin(), expr.end());
      return builder_.AddGrammarExpr(
          {GrammarBuilder::GrammarExprType::kCharacterClassStar,
           data.data(),
           static_cast<int32_t>(data.size())}
      );
    }
  }
  return builder_.AddRepeatFromExpr(rule_name_hint, expr_id, min_count, max_count);
}

int32_t JSONSchemaConverter::AddSubGrammar(const Grammar& grammar) {
  int32_t rule_id = SubGrammarAdder::Apply(&builder_, grammar);
  return RuleRef(rule_id);
}

std::string JSONSchemaConverter::GetWhitespacePattern() const {
  if (!max_whitespace_cnt_.has_value()) {
    return "[ \\n\\r\\t]*";
  }
  return "[ \\n\\r\\t]{0," + std::to_string(*max_whitespace_cnt_) + "}";
}

int32_t JSONSchemaConverter::WhitespaceExpression() {
  std::vector<CharacterClassElement> elements = {
      {' ', ' '}, {'\n', '\n'}, {'\r', '\r'}, {'\t', '\t'}
  };
  if (!max_whitespace_cnt_.has_value()) {
    if (!whitespace_expr_id_.has_value()) {
      whitespace_expr_id_ = builder_.AddCharacterClassStar(elements);
    }
    return *whitespace_expr_id_;
  }
  // Bounded whitespace occurrences intentionally remain distinct, matching the historical
  // parser-produced rule shape after normalization.
  return Repeat(
      "whitespace",
      builder_.AddCharacterClass(elements),
      0,
      static_cast<int32_t>(*max_whitespace_cnt_)
  );
}

int32_t JSONSchemaConverter::FormattingExpression(const std::string& expression) {
  const std::string whitespace = GetWhitespacePattern();
  if (expression == whitespace) {
    return WhitespaceExpression();
  }

  const std::string prefix = whitespace + " ";
  const std::string suffix = " " + whitespace;
  if (expression.size() >= prefix.size() + suffix.size() &&
      expression.compare(0, prefix.size(), prefix) == 0 &&
      expression.compare(expression.size() - suffix.size(), suffix.size(), suffix) == 0) {
    return Sequence(
        {WhitespaceExpression(),
         FormattingExpression(
             expression.substr(prefix.size(), expression.size() - prefix.size() - suffix.size())
         ),
         WhitespaceExpression()}
    );
  }

  picojson::value value;
  std::string error = ParseJSON(value, expression);
  XGRAMMAR_CHECK(error.empty() && value.is<std::string>())
      << "Unsupported indentation expression: " << expression;
  return ByteString(value.get<std::string>());
}

std::string JSONSchemaConverter::NextSeparator(bool is_end) {
  return indent_manager_.NextSeparator(is_end);
}

int32_t JSONSchemaConverter::NextSeparatorExpression(bool is_end) {
  return FormattingExpression(NextSeparator(is_end));
}

std::string JSONSchemaConverter::GetKeyPattern() const { return kBasicString; }

int32_t JSONSchemaConverter::KeyPatternExpression() { return RuleRef(GetKeyPattern()); }

int32_t JSONSchemaConverter::BuildTrieBody(const TrieNode& node, const std::string& rule_name) {
  std::vector<int32_t> choices;
  if (!node.is_terminal) {
    choices.push_back(ByteString("\""));
  }

  std::vector<CharacterClassElement> excluded = {
      {0, 0x1f}, {'"', '"'}, {'\\', '\\'}, {'\r', '\r'}, {'\n', '\n'}
  };
  for (const auto& [character, child] : node.children) {
    static_cast<void>(child);
    excluded.push_back({character, character});
  }
  choices.push_back(Sequence({builder_.AddCharacterClass(excluded, true), RuleRef(kBasicStringSub)})
  );
  choices.push_back(Sequence({ByteString("\\"), RuleRef(kBasicEscape), RuleRef(kBasicStringSub)}));
  for (const auto& [character, child] : node.children) {
    choices.push_back(Sequence(
        {ByteString(std::string(1, static_cast<char>(character))), BuildTrieBody(child, rule_name)}
    ));
  }
  return Choice(choices);
}

int32_t JSONSchemaConverter::GetKeyPatternExcluding(
    const std::vector<ObjectSpec::Property>& properties, const std::string& rule_name
) {
  if (properties.empty()) {
    return KeyPatternExpression();
  }
  if (!excludes_.empty()) {
    std::vector<std::string> keys;
    for (const auto& property : properties) {
      auto encoded = picojson::value(property.name).serialize(false);
      keys.push_back(encoded.substr(1, encoded.size() - 2));
    }
    return Sequence(
        {ByteString("\""),
         ExcludingString(kJSONStringBodyRegex, rule_name + "_addl_key", true, keys)}
    );
  }

  // Build trie from property names
  // TODO(linzhang): The trie only excludes the literal unescaped spelling of each property name.
  TrieNode root;
  for (const auto& prop : properties) {
    TrieNode* cur = &root;
    for (unsigned char c : prop.name) {
      cur = &cur->children[c];
    }
    cur->is_terminal = true;
  }

  int32_t key_rule_id = builder_.AddEmptyRuleWithHint(rule_name + "_addl_key");
  std::string key_rule_name = builder_.GetRule(key_rule_id).name;
  builder_.UpdateRuleBody(
      key_rule_id, Sequence({ByteString("\""), BuildTrieBody(root, key_rule_name)})
  );
  builder_.UpdateLookaheadAssertion(
      key_rule_id,
      Sequence(
          {WhitespaceExpression(),
           builder_.AddCharacterClass({{',', ','}, {'}', '}'}, {']', ']'}, {':', ':'}})}
      )
  );
  return RuleRef(key_rule_id);
}

std::string JSONSchemaConverter::GetBasicAnyRuleName() const { return kBasicAny; }

void JSONSchemaConverter::AddCache(const std::string& key, int32_t rule_id) {
  if (!key.empty()) {
    rule_cache_manager_.AddCache(key, true, rule_id);
  }
}

std::optional<int32_t> JSONSchemaConverter::GetCache(const std::string& key) const {
  if (key.empty()) {
    return std::nullopt;
  }
  return rule_cache_manager_.GetCache(key, true);
}

int32_t JSONSchemaConverter::CreateRule(
    const SchemaSpecPtr& spec, const std::string& rule_name_hint
) {
  // Only check cache for basic rules (pre-populated in AddBasicRules)
  // Don't cache other rules to match original behavior
  auto cached = GetCache(spec->cache_key);
  if (cached.has_value()) {
    return cached.value();
  }
  int32_t rule_id = builder_.AddEmptyRuleWithHint(rule_name_hint);
  // Copy the name before generating: GenerateFromSpec may add rules and reallocate the
  // builder's rule storage, invalidating references into it.
  std::string rule_name = builder_.GetRule(rule_id).name;
  builder_.UpdateRuleBody(rule_id, GenerateFromSpec(spec, rule_name));
  return rule_id;
}

int32_t JSONSchemaConverter::GenerateFromSpec(
    const SchemaSpecPtr& spec, const std::string& rule_name_hint
) {
  return std::visit(
      [this, &rule_name_hint](const auto& s) -> int32_t {
        using T = std::decay_t<decltype(s)>;
        if constexpr (std::is_same_v<T, IntegerSpec>) {
          return GenerateInteger(s, rule_name_hint);
        } else if constexpr (std::is_same_v<T, NumberSpec>) {
          return GenerateNumber(s, rule_name_hint);
        } else if constexpr (std::is_same_v<T, StringSpec>) {
          return GenerateString(s, rule_name_hint);
        } else if constexpr (std::is_same_v<T, BooleanSpec>) {
          return GenerateBoolean(s, rule_name_hint);
        } else if constexpr (std::is_same_v<T, NullSpec>) {
          return GenerateNull(s, rule_name_hint);
        } else if constexpr (std::is_same_v<T, ArraySpec>) {
          return GenerateArray(s, rule_name_hint);
        } else if constexpr (std::is_same_v<T, ObjectSpec>) {
          return GenerateObject(s, rule_name_hint);
        } else if constexpr (std::is_same_v<T, AnySpec>) {
          return GenerateAny(s, rule_name_hint);
        } else if constexpr (std::is_same_v<T, ConstSpec>) {
          return GenerateConst(s, rule_name_hint);
        } else if constexpr (std::is_same_v<T, EnumSpec>) {
          return GenerateEnum(s, rule_name_hint);
        } else if constexpr (std::is_same_v<T, RefSpec>) {
          return GenerateRef(s, rule_name_hint);
        } else if constexpr (std::is_same_v<T, AnyOfSpec>) {
          return GenerateAnyOf(s, rule_name_hint);
        } else if constexpr (std::is_same_v<T, OneOfSpec>) {
          return GenerateOneOf(s, rule_name_hint);
        } else if constexpr (std::is_same_v<T, AllOfSpec>) {
          return GenerateAllOf(s, rule_name_hint);
        } else if constexpr (std::is_same_v<T, TypeArraySpec>) {
          return GenerateTypeArray(s, rule_name_hint);
        } else {
          XGRAMMAR_LOG(FATAL) << "Unknown spec type";
        }
      },
      spec->spec
  );
}

/*!
 * \brief Emit the grammar expression matching a regex. Prefer the Regex node with
 * json_string=true so the pattern is compiled into a single automaton by GrammarFSMBuilder;
 * json_string=true excludes the characters that must be escaped in a JSON string ('"', '\\'
 * and the control characters) from every character match, so classes like \S cannot emit an
 * unescaped quote. Fall back to the CFG expansion when the FSM regex engine does not support
 * the pattern, or when the exclusion makes the pattern unmatchable (e.g. a pattern requiring
 * a literal '"').
 */
int32_t JSONSchemaConverter::RegexExpression(
    const std::string& regex, bool json_string, bool force_cfg_expansion
) {
  bool can_use_fsm = !force_cfg_expansion;
  if (json_string) {
    can_use_fsm =
        can_use_fsm && std::all_of(regex.begin(), regex.end(), [](unsigned char character) {
          return character >= 0x20 && character <= 0x7e;
        });
  }
  if (can_use_fsm) {
    auto fsm_result = GrammarFSMBuilder::Regex(regex, json_string);
    if (fsm_result.IsOk()) {
      auto fsm = std::move(fsm_result).Unwrap();
      std::unordered_set<int> reachable_states;
      fsm.GetReachableStates(&reachable_states);
      bool language_is_empty =
          std::none_of(reachable_states.begin(), reachable_states.end(), [&](int state) {
            return fsm.IsEndState(state);
          });
      if (!language_is_empty) {
        return builder_.AddRegex(regex, json_string);
      }
    }
  }

  // Keep regex conversion independent. Only the uncommon fallback path converts its existing
  // EBNF result to a subgrammar; the JSON Schema rule graph itself is still built directly.
  return AddSubGrammar(Grammar::FromEBNF(RegexToEBNF(regex)));
}

// ==================== Generate Methods ====================

void JSONSchemaConverter::WarnDroppedLengthConstraints(
    const StringSpec& spec, const std::string& rule_name
) const {
  XGRAMMAR_LOG(WARNING) << "Ignoring the length constraints of string " << rule_name
                        << " (minLength=" << spec.min_length << ", maxLength=" << spec.max_length
                        << "): they are not applied together with JSONSchemaFormat.excludes";
}

bool JSONSchemaConverter::IsAllowedString(const std::string& text) const {
  return std::none_of(excludes_.begin(), excludes_.end(), [&](const auto& excluded) {
    return text.find(excluded) != std::string::npos;
  });
}

bool JSONSchemaConverter::IsAllowedLiteral(const picojson::value& value, bool raw_string) const {
  if (excludes_.empty()) return true;
  if (value.is<std::string>()) {
    if (raw_string) return IsAllowedString(value.get<std::string>());
    auto encoded = value.serialize(false);
    return IsAllowedString(encoded.substr(1, encoded.size() - 2));
  }
  if (value.is<picojson::array>()) {
    for (const auto& item : value.get<picojson::array>()) {
      if (!IsAllowedLiteral(item)) return false;
    }
  }
  if (value.is<picojson::object>()) {
    for (const auto& [key, item] : value.get<picojson::object>()) {
      if (!IsAllowedLiteral(picojson::value(key)) || !IsAllowedLiteral(item)) return false;
    }
  }
  return true;
}

bool JSONSchemaConverter::IsAllowedJSONLiteral(const std::string& json_value, bool raw_string)
    const {
  if (excludes_.empty()) return true;
  picojson::value value;
  XGRAMMAR_CHECK(picojson::parse(value, json_value).empty());
  return IsAllowedLiteral(value, raw_string);
}

int32_t JSONSchemaConverter::ExcludingString(
    const std::string& regex,
    const std::string& rule_name,
    bool close_json_string,
    const std::vector<std::string>& excluded_keys
) {
  auto parsed = GrammarFSMBuilder::Regex(regex, false);
  XGRAMMAR_CHECK(parsed.IsOk()) << "Cannot build the FSM of " << regex << ": "
                                << std::move(parsed).UnwrapErr().what();
  auto exclusion = GrammarFSMBuilder::TagDispatch({{}, false, excludes_});
  XGRAMMAR_CHECK(exclusion.has_value()) << "Invalid JSONSchemaFormat.excludes";
  auto intersected = FSMWithStartEnd::Intersect(std::move(parsed).Unwrap(), *exclusion);
  XGRAMMAR_CHECK(intersected.IsOk()) << "Cannot intersect JSONSchemaFormat.excludes: "
                                     << std::move(intersected).UnwrapErr().what();
  auto fsm = std::move(intersected).Unwrap();
  if (!excluded_keys.empty()) {
    auto keys = TrieFSMBuilder::Build(excluded_keys, {});
    XGRAMMAR_CHECK(keys.has_value());
    auto other_keys = keys->Not();
    XGRAMMAR_CHECK(other_keys.IsOk());
    auto filtered = FSMWithStartEnd::Intersect(fsm, std::move(other_keys).Unwrap());
    XGRAMMAR_CHECK(filtered.IsOk());
    fsm = std::move(filtered).Unwrap();
  }
  // Do not emit paths that can never complete after exclusions. Otherwise a matcher
  // could accept a forbidden alternative's prefix and reach an all-rejected mask later.
  std::vector<std::vector<int>> predecessors(fsm.NumStates());
  for (int state = 0; state < fsm.NumStates(); ++state) {
    for (const auto& edge : fsm.GetFsm().GetEdges(state)) {
      predecessors[edge.target].push_back(state);
    }
  }
  std::vector<bool> productive(fsm.NumStates(), false);
  std::vector<int> worklist(fsm.GetEnds().begin(), fsm.GetEnds().end());
  for (int state : worklist) productive[state] = true;
  for (size_t index = 0; index < worklist.size(); ++index) {
    for (int state : predecessors[worklist[index]]) {
      if (!productive[state]) {
        productive[state] = true;
        worklist.push_back(state);
      }
    }
  }
  if (!productive[fsm.GetStart()]) {
    return Unsatisfiable();
  }

  // Collapse UTF-8 paths into codepoint transitions before emitting character classes.
  // Emitting individual continuation bytes as string literals would not round-trip through
  // EBNF, whose escaped string literals represent Unicode codepoints rather than raw bytes.
  struct CodepointRange {
    int min, max, target;
  };
  using Ranges = std::vector<CodepointRange>;
  auto merge_ranges = [](Ranges ranges) {
    std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) {
      return std::tie(a.target, a.min, a.max) < std::tie(b.target, b.min, b.max);
    });
    Ranges merged;
    for (const auto& range : ranges) {
      if (!merged.empty() && merged.back().target == range.target &&
          range.min <= merged.back().max + 1) {
        merged.back().max = std::max(merged.back().max, range.max);
      } else {
        merged.push_back(range);
      }
    }
    return merged;
  };
  auto append_ranges = [](Ranges* ranges, int min, int max, int shift, const CodepointRange& suffix
                       ) {
    if (suffix.min == 0 && suffix.max == (1 << shift) - 1) {
      ranges->push_back({min << shift, (max << shift) | suffix.max, suffix.target});
    } else {
      for (int prefix = min; prefix <= max; ++prefix) {
        ranges->push_back(
            {(prefix << shift) | suffix.min, (prefix << shift) | suffix.max, suffix.target}
        );
      }
    }
  };
  // Memoize suffixes so wide Unicode classes do not enumerate every codepoint.
  std::map<std::pair<int, int>, Ranges> suffix_cache;
  std::function<const Ranges&(int, int)> suffix_ranges = [&](int state,
                                                             int remaining) -> const Ranges& {
    auto [it, inserted] = suffix_cache.emplace(std::make_pair(state, remaining), Ranges{});
    if (!inserted) return it->second;
    Ranges ranges;
    if (remaining == 0) {
      ranges.push_back({0, 0, state});
    } else {
      for (const auto& edge : fsm.GetFsm().GetEdges(state)) {
        if (!productive[edge.target]) continue;
        XGRAMMAR_CHECK(edge.IsCharRange());
        int min = std::max(edge.min, 0x80), max = std::min(edge.max, 0xbf);
        if (min > max) continue;
        for (const auto& suffix : suffix_ranges(edge.target, remaining - 1)) {
          append_ranges(&ranges, min & 0x3f, max & 0x3f, 6 * (remaining - 1), suffix);
        }
      }
    }
    it->second = merge_ranges(std::move(ranges));
    return it->second;
  };

  std::vector<int32_t> rules(fsm.NumStates(), -1);
  std::vector<int> pending{fsm.GetStart()};
  // Share complete fallback branches across key prefixes to reuse their token masks.
  // The key contains the target state followed by every codepoint interval.
  std::map<std::vector<int32_t>, int32_t> shared_key_continuations;
  rules[fsm.GetStart()] = builder_.AddEmptyRuleWithHint(rule_name + "_exclude");
  for (size_t index = 0; index < pending.size(); ++index) {
    int state = pending[index];
    Ranges ranges;
    for (const auto& edge : fsm.GetFsm().GetEdges(state)) {
      if (!productive[edge.target]) continue;
      XGRAMMAR_CHECK(edge.IsCharRange());
      if (edge.min < 128) {
        ranges.push_back({edge.min, std::min(edge.max, 127), edge.target});
      }
      const int leading_min[] = {0, 0xc2, 0xe0, 0xf0};
      const int leading_max[] = {0, 0xdf, 0xef, 0xf4};
      for (int remaining = 1; remaining <= 3; ++remaining) {
        int min = std::max(edge.min, leading_min[remaining]);
        int max = std::min(edge.max, leading_max[remaining]);
        if (min > max) continue;
        int mask = (1 << (6 - remaining)) - 1;
        Ranges unicode_ranges;
        for (const auto& suffix : suffix_ranges(edge.target, remaining)) {
          append_ranges(&unicode_ranges, min & mask, max & mask, 6 * remaining, suffix);
        }
        // The byte FSM can contain non-canonical UTF-8 paths. Never turn an overlong
        // encoding into a second transition for an ASCII character (bypassing exclusions).
        const int codepoint_min[] = {0, 0x80, 0x800, 0x10000};
        const int codepoint_max[] = {0, 0x7ff, 0xffff, 0x10ffff};
        for (auto range : unicode_ranges) {
          range.min = std::max(range.min, codepoint_min[remaining]);
          range.max = std::min(range.max, codepoint_max[remaining]);
          if (range.min <= range.max) ranges.push_back(range);
        }
      }
    }
    std::map<int, std::vector<GrammarBuilder::CharacterClassElement>> transitions;
    for (const auto& range : merge_ranges(std::move(ranges))) {
      transitions[range.target].push_back({range.min, range.max});
    }
    std::vector<int32_t> choices;
    // Keep the closing quote on the accepting body states. A nullable body rule
    // would otherwise finish at every character and force token masks to speculate
    // across its parent rules. The quote is appended AFTER the exclusion intersection:
    // it is JSON syntax, not string content subject to excludes.
    if (fsm.IsEndState(state)) {
      choices.push_back(close_json_string ? ByteString("\"") : Empty());
    }
    for (const auto& [target, codepoints] : transitions) {
      if (rules[target] == -1) {
        rules[target] = builder_.AddEmptyRuleWithHint(rule_name + "_exclude");
        pending.push_back(target);
      }
      bool single_ascii = codepoints.size() == 1 && codepoints[0].lower == codepoints[0].upper &&
                          codepoints[0].upper <= 0x7f;
      // Keep self-loops local for the compiler's speculative string-mask fast path.
      if (!excluded_keys.empty() && !single_ascii && target != state) {
        std::vector<int32_t> key{target};
        key.reserve(1 + codepoints.size() * 2);
        for (const auto& range : codepoints) {
          key.push_back(range.lower);
          key.push_back(range.upper);
        }
        auto [it, inserted] = shared_key_continuations.emplace(std::move(key), -1);
        if (inserted) {
          it->second = builder_.AddRuleWithHint(
              rule_name + "_exclude_continuation",
              Sequence({builder_.AddCharacterClass(codepoints), RuleRef(rules[target])})
          );
        }
        choices.push_back(RuleRef(it->second));
      } else {
        choices.push_back(Sequence({builder_.AddCharacterClass(codepoints), RuleRef(rules[target])})
        );
      }
    }
    if (choices.empty()) {
      choices.push_back(Unsatisfiable());
    }
    builder_.UpdateRuleBody(rules[state], Choice(choices));
  }
  return RuleRef(rules[fsm.GetStart()]);
}

int32_t JSONSchemaConverter::GenerateInteger(
    const IntegerSpec& spec, const std::string& rule_name
) {
  // Shared with ParseInteger's range validation so emission and validation agree on the effective
  // range; a nullopt side means that side is unbounded.
  const EffectiveIntegerRange range = ComputeEffectiveIntegerRange(spec);
  std::optional<int64_t> start = range.start;
  std::optional<int64_t> end = range.end;

  if (spec.multiple_of.has_value()) {
    // ParseInteger keeps multiple_of only when the range is fully bounded (enumerate the
    // multiples) or fully unbounded (emit a modulo DFA); the half-bounded case is dropped there.
    if (start.has_value() && end.has_value()) {
      std::vector<int32_t> multiples;
      for (int64_t value = *start; value <= *end; ++value) {
        if (IsMultipleOf(value, *spec.multiple_of)) {
          multiples.push_back(ByteString(std::to_string(value)));
        }
        if (value == std::numeric_limits<int64_t>::max()) {
          break;
        }
      }
      return Choice(multiples);
    }
    return GenerateIntegerMultipleOfDFA(*spec.multiple_of, rule_name);
  }
  if (start.has_value() || end.has_value()) {
    return RegexExpression(
        GenerateRangeRegex(start, end),
        false,
        /*force_cfg_expansion=*/true
    );
  }
  int32_t optional_minus = Choice({Empty(), ByteString("-")});
  return Choice(
      {ByteString("0"),
       Sequence(
           {optional_minus,
            builder_.AddCharacterClass({{'1', '9'}}),
            builder_.AddCharacterClassStar({{'0', '9'}})}
       )}
  );
}

int32_t JSONSchemaConverter::GenerateIntegerMultipleOfDFA(
    int64_t multiple_of, const std::string& rule_name
) {
  std::vector<int32_t> states(multiple_of);
  for (int64_t state = 0; state < multiple_of; ++state) {
    states[state] = builder_.AddEmptyRuleWithHint(
        rule_name + "_multiple_of_" + std::to_string(multiple_of) + "_mod_" + std::to_string(state)
    );
  }
  for (int64_t state = 0; state < multiple_of; ++state) {
    std::vector<int32_t> transitions;
    if (state == 0) {
      transitions.push_back(Empty());
    }
    for (int64_t digit = 0; digit <= 9; ++digit) {
      int64_t next_state = (state * 10 + digit) % multiple_of;
      transitions.push_back(
          Sequence({ByteString(std::to_string(digit)), RuleRef(states[next_state])})
      );
    }
    builder_.UpdateRuleBody(states[state], Choice(transitions));
  }

  std::vector<int32_t> non_zero_starts;
  for (int64_t digit = 1; digit <= 9; ++digit) {
    non_zero_starts.push_back(
        Sequence({ByteString(std::to_string(digit)), RuleRef(states[digit % multiple_of])})
    );
  }
  return Choice(
      {ByteString("0"), Sequence({Choice({Empty(), ByteString("-")}), Choice(non_zero_starts)})}
  );
}

int32_t JSONSchemaConverter::GenerateNumber(const NumberSpec& spec, const std::string& rule_name) {
  std::optional<double> start = spec.minimum;
  std::optional<double> end = spec.maximum;
  bool exclusive_start = false;
  bool exclusive_end = false;
  // When both bounds are present the larger lower bound wins; on a tie the
  // exclusive one is stricter.
  if (spec.exclusive_minimum.has_value() &&
      (!start.has_value() || *spec.exclusive_minimum >= *start)) {
    start = spec.exclusive_minimum;
    exclusive_start = true;
  }
  if (spec.exclusive_maximum.has_value() && (!end.has_value() || *spec.exclusive_maximum <= *end)) {
    end = spec.exclusive_maximum;
    exclusive_end = true;
  }
  if (start.has_value() || end.has_value()) {
    return RegexExpression(
        GenerateFloatRangeRegex(start, end, /*precision=*/6, exclusive_start, exclusive_end),
        false,
        /*force_cfg_expansion=*/true
    );
  }

  int32_t optional_minus = Choice({Empty(), ByteString("-")});
  int32_t integer_part = Choice(
      {ByteString("0"),
       Sequence(
           {builder_.AddCharacterClass({{'1', '9'}}), builder_.AddCharacterClassStar({{'0', '9'}})}
       )}
  );
  int32_t one_or_more_digits =
      Repeat(rule_name + "_digits", builder_.AddCharacterClass({{'0', '9'}}), 1, -1);
  int32_t fraction = Choice({Empty(), Sequence({ByteString("."), one_or_more_digits})});
  int32_t exponent = Choice(
      {Empty(),
       Sequence(
           {builder_.AddCharacterClass({{'e', 'e'}, {'E', 'E'}}),
            Choice({Empty(), builder_.AddCharacterClass({{'+', '+'}, {'-', '-'}})}),
            one_or_more_digits}
       )}
  );
  // Note: The format must be "-"? ("0" | ...) not ("0" | "-"? ...)
  // The first allows -0, -123, 0, 123
  // The second allows 0, -123, 123 but not -0
  return Sequence({optional_minus, integer_part, fraction, exponent});
}

int32_t JSONSchemaConverter::GenerateString(const StringSpec& spec, const std::string& rule_name) {
  // Check for format
  if (spec.format.has_value()) {
    auto regex = JSONFormatToRegexPattern(*spec.format);
    if (regex.has_value()) {
      // The built-in format regexes use constructs that the FSM regex engine does not fully
      // support yet (e.g. quoted email local parts), so they keep the CFG expansion.
      return Sequence({ByteString("\""), RegexExpression(*regex, false, true), ByteString("\"")});
    }
  }
  // Check for pattern
  if (spec.pattern.has_value()) {
    return Sequence(
        {ByteString("\""), RegexExpression(*spec.pattern, /*json_string=*/true), ByteString("\"")}
    );
  }
  // Check for length constraints. They are dropped when there are exclusions: intersecting the
  // unrolled bound with the exclusion automaton emits one rule per position and automaton state
  // (about 18 rules per character for three markers), so the string keeps only the exclusions.
  // Exclusions apply only to the strings without pattern and format above: those constraints
  // are the schema's own contract for the string.
  if (spec.min_length != 0 || spec.max_length != -1) {
    if (!excludes_.empty()) {
      WarnDroppedLengthConstraints(spec, rule_name);
    } else {
      int32_t character =
          builder_.AddCharacterClass({{'"', '"'}, {'\\', '\\'}, {'\r', '\r'}, {'\n', '\n'}}, true);
      int32_t body = Repeat(rule_name + "_characters", character, spec.min_length, spec.max_length);
      return Sequence({ByteString("\""), body, ByteString("\"")});
    }
  }
  // Default string
  return Sequence({ByteString("\""), RuleRef(kBasicStringSub)});
}

int32_t JSONSchemaConverter::GenerateBoolean(
    const BooleanSpec& spec, const std::string& rule_name
) {
  return Choice({ByteString("true"), ByteString("false")});
}

int32_t JSONSchemaConverter::GenerateNull(const NullSpec& spec, const std::string& rule_name) {
  return ByteString("null");
}

int32_t JSONSchemaConverter::GenerateArray(const ArraySpec& spec, const std::string& rule_name) {
  indent_manager_.StartIndent();
  int32_t start_separator = FormattingExpression(indent_manager_.StartSeparator());
  int32_t middle_separator = FormattingExpression(indent_manager_.MiddleSeparator());
  int32_t end_separator = FormattingExpression(indent_manager_.EndSeparator());
  int32_t empty_separator = FormattingExpression(indent_manager_.EmptySeparator());

  std::vector<int32_t> item_rule_ids;
  for (size_t index = 0; index < spec.prefix_items.size(); ++index) {
    item_rule_ids.push_back(
        CreateRule(spec.prefix_items[index], rule_name + "_item_" + std::to_string(index))
    );
  }
  int32_t additional_rule_id = -1;
  if (spec.allow_additional_items && spec.additional_items) {
    additional_rule_id = CreateRule(spec.additional_items, rule_name + "_additional");
  }
  indent_manager_.EndIndent();

  int32_t left_bracket = ByteString("[");
  int32_t right_bracket = ByteString("]");
  int32_t empty_array = Sequence({left_bracket, empty_separator, right_bracket});

  if (item_rule_ids.empty()) {
    if (!spec.allow_additional_items || spec.max_items == 0) {
      return empty_array;
    }
    int32_t additional = RuleRef(additional_rule_id);
    int32_t tail = Repeat(
        rule_name + "_items",
        Sequence({middle_separator, additional}),
        spec.min_items == 0 ? 0 : static_cast<int32_t>(spec.min_items - 1),
        spec.max_items == -1 ? -1 : static_cast<int32_t>(spec.max_items - 1)
    );
    int32_t nonempty =
        Sequence({left_bracket, start_separator, additional, tail, end_separator, right_bracket});
    return spec.min_items == 0 ? Choice({nonempty, empty_array}) : nonempty;
  }

  // Per Draft 2020-12, prefixItems entries are positional: the instance may
  // end after any prefix position (subject to minItems), and additional items
  // are only allowed after the full prefix (issue #824).
  size_t mandatory_count = static_cast<size_t>(std::min<int64_t>(
      std::max<int64_t>(0, spec.min_items), static_cast<int64_t>(item_rule_ids.size())
  ));

  // Mandatory head: the first min(minItems, n) items, separated.
  std::vector<int32_t> prefix_elements;
  for (size_t index = 0; index < mandatory_count; ++index) {
    if (index != 0) {
      prefix_elements.push_back(middle_separator);
    }
    prefix_elements.push_back(RuleRef(item_rule_ids[index]));
  }

  // Suffix after the mandatory head, flattened into a right-recursive chain
  // of rules   suffix_k ::= "" | sep item_k suffix_{k+1}   so each position
  // is encoded once instead of once per truncation length. The chain ends
  // with the additional-items tail. Positions from index 1 on are separated
  // by middle_separator; position 0, when it is not part of the mandatory
  // head, gets its own rule without the separator.
  int32_t suffix = Empty();
  if (spec.allow_additional_items && spec.additional_items) {
    int64_t minimum_additional =
        std::max(int64_t{0}, spec.min_items - static_cast<int64_t>(item_rule_ids.size()));
    suffix = Repeat(
        rule_name + "_additional_items",
        Sequence({middle_separator, RuleRef(additional_rule_id)}),
        static_cast<int32_t>(minimum_additional),
        spec.max_items == -1
            ? -1
            : static_cast<int32_t>(spec.max_items - static_cast<int64_t>(item_rule_ids.size()))
    );
  }
  size_t chain_start = std::max<size_t>(mandatory_count, 1);
  for (size_t k = item_rule_ids.size(); k-- > chain_start;) {
    int32_t with_item = Sequence({middle_separator, RuleRef(item_rule_ids[k]), suffix});
    int32_t suffix_rule_id = builder_.AddRuleWithHint(
        rule_name + "_suffix_" + std::to_string(k), Choice({Empty(), with_item})
    );
    suffix = RuleRef(suffix_rule_id);
  }
  if (mandatory_count == 0) {
    int32_t with_first = Sequence({RuleRef(item_rule_ids[0]), suffix});
    int32_t suffix_rule_id =
        builder_.AddRuleWithHint(rule_name + "_suffix_0", Choice({Empty(), with_first}));
    suffix = RuleRef(suffix_rule_id);
  }

  std::vector<int32_t> content_elements = prefix_elements;
  content_elements.push_back(suffix);
  int32_t prefix = Sequence(content_elements);
  return Sequence({left_bracket, start_separator, prefix, end_separator, right_bracket});
}

int32_t JSONSchemaConverter::FormatPropertyKey(
    const std::string& key, const SchemaSpecPtr& schema
) {
  if (!IsAllowedLiteral(picojson::value(key))) {
    return Unsatisfiable();
  }
  return ByteString(picojson::value(key).serialize());
}

int32_t JSONSchemaConverter::FormatProperty(
    const std::string& key,
    int32_t value_rule_id,
    const std::string& rule_name,
    int64_t idx,
    const SchemaSpecPtr& schema
) {
  return Sequence({FormatPropertyKey(key, schema), colon_expr_id_, RuleRef(value_rule_id)});
}

int32_t JSONSchemaConverter::FormatOtherProperty(
    int32_t key_pattern_expr,
    int32_t value_rule_id,
    const std::string& rule_name,
    const std::string& rule_name_suffix,
    const SchemaSpecPtr& schema
) {
  return Sequence({key_pattern_expr, colon_expr_id_, RuleRef(value_rule_id)});
}

int32_t JSONSchemaConverter::GetPropertyWithNumberConstraints(
    int32_t pattern,
    int min_properties,
    int max_properties,
    int already_repeated_times,
    const std::string& rule_name
) {
  if (max_properties != -1 && max_properties == already_repeated_times) {
    return Empty();
  }
  int lower = std::max(0, min_properties - already_repeated_times);
  int upper = max_properties == -1 ? -1 : std::max(-1, max_properties - already_repeated_times);
  return Repeat(rule_name + "_properties", pattern, lower, upper);
}

int32_t JSONSchemaConverter::GetAnyOrderRuleForProperties(
    const std::vector<ObjectSpec::Property>& properties,
    const std::unordered_set<std::string>& required,
    const SchemaSpecPtr& additional,
    const std::string& rule_name,
    const std::string& additional_suffix,
    int min_properties,
    int max_properties,
    const std::optional<int32_t>& additional_property_override
) {
  int32_t first_separator = NextSeparatorExpression();
  int32_t middle_separator = NextSeparatorExpression();
  int32_t last_separator = NextSeparatorExpression(true);

  // Build one "item" alternation over every property (any required/optional key) plus any
  // additional/pattern key; any_order does not care which key goes where.
  std::vector<int32_t> items;
  for (size_t index = 0; index < properties.size(); ++index) {
    const auto& property = properties[index];
    int32_t value_rule_id =
        CreateRule(property.schema, rule_name + "_prop_" + std::to_string(index));
    items.push_back(FormatProperty(property.name, value_rule_id, rule_name, index, property.schema)
    );
  }
  if (additional != nullptr) {
    if (additional_property_override.has_value()) {
      items.push_back(*additional_property_override);
    } else {
      int32_t value_rule_id = CreateRule(additional, rule_name + "_" + additional_suffix);
      items.push_back(FormatOtherProperty(
          GetKeyPatternExcluding(properties, rule_name),
          value_rule_id,
          rule_name,
          additional_suffix,
          additional
      ));
    }
  }

  int32_t item_rule_id = builder_.AddRuleWithHint(rule_name + "_item", Choice(items));

  // Repeat `item` between n = max(minProperties, #required) and m = maxProperties times; only the
  // count is constrained, not which keys appear.
  int minimum_count = std::max(min_properties, static_cast<int>(required.size()));
  int32_t repeated_items = GetPropertyWithNumberConstraints(
      Sequence({middle_separator, RuleRef(item_rule_id)}),
      minimum_count,
      max_properties,
      1,
      rule_name
  );
  return Sequence({first_separator, RuleRef(item_rule_id), repeated_items, last_separator});
}

int32_t JSONSchemaConverter::GetPartialRuleForProperties(
    const std::vector<ObjectSpec::Property>& properties,
    const std::unordered_set<std::string>& required,
    const SchemaSpecPtr& additional,
    const std::string& rule_name,
    const std::string& additional_suffix,
    int min_properties,
    int max_properties,
    const std::optional<int32_t>& additional_property_override
) {
  if (max_properties == 0) {
    return Empty();
  }
  if (any_order_) {
    return GetAnyOrderRuleForProperties(
        properties,
        required,
        additional,
        rule_name,
        additional_suffix,
        min_properties,
        max_properties,
        additional_property_override
    );
  }

  int32_t first_separator = NextSeparatorExpression();
  int32_t middle_separator = NextSeparatorExpression();
  int32_t last_separator = NextSeparatorExpression(true);

  std::vector<int32_t> property_patterns;
  for (size_t index = 0; index < properties.size(); ++index) {
    int32_t value_rule_id =
        CreateRule(properties[index].schema, rule_name + "_prop_" + std::to_string(index));
    property_patterns.push_back(FormatProperty(
        properties[index].name, value_rule_id, rule_name, index, properties[index].schema
    ));
  }

  bool allow_additional = additional != nullptr;
  std::optional<int32_t> additional_pattern;
  auto get_additional_pattern = [&]() -> int32_t {
    if (!additional_pattern.has_value()) {
      if (additional_property_override.has_value()) {
        additional_pattern = *additional_property_override;
      } else {
        int32_t value_rule_id = CreateRule(additional, rule_name + "_" + additional_suffix);
        additional_pattern = FormatOtherProperty(
            GetKeyPatternExcluding(properties, rule_name),
            value_rule_id,
            rule_name,
            additional_suffix,
            additional
        );
      }
    }
    return *additional_pattern;
  };

  if (min_properties == 0 && max_properties == -1) {
    // Case 1: No property number constraints
    std::vector<int32_t> tails(properties.size(), Empty());
    std::vector<uint8_t> is_required(properties.size(), false);

    if (allow_additional) {
      int32_t repeated_additional = Repeat(
          rule_name + "_additional_properties",
          Sequence({middle_separator, get_additional_pattern()}),
          0,
          -1
      );
      int32_t tail_rule_id = builder_.AddRuleWithHint(
          rule_name + "_part_" + std::to_string(static_cast<int>(properties.size()) - 1),
          repeated_additional
      );
      tails.back() = RuleRef(tail_rule_id);
    }

    for (int index = static_cast<int>(properties.size()) - 2; index >= 0; --index) {
      int32_t with_property =
          Sequence({middle_separator, property_patterns[index + 1], tails[index + 1]});
      int32_t body = with_property;
      if (!required.count(properties[index + 1].name)) {
        body = Choice({tails[index + 1], with_property});
      } else {
        is_required[index + 1] = true;
      }
      int32_t tail_rule_id =
          builder_.AddRuleWithHint(rule_name + "_part_" + std::to_string(index), body);
      tails[index] = RuleRef(tail_rule_id);
    }
    if (required.count(properties[0].name)) {
      is_required[0] = true;
    }

    std::vector<int32_t> choices;
    for (size_t index = 0; index < properties.size(); ++index) {
      choices.push_back(Sequence({property_patterns[index], tails[index]}));
      if (is_required[index]) {
        break;
      }
    }
    if (allow_additional && required.empty()) {
      choices.push_back(Sequence({get_additional_pattern(), tails.back()}));
    }
    return Sequence({first_separator, Choice(choices), last_separator});
  }

  const int property_count = static_cast<int>(properties.size());
  std::vector<uint8_t> is_required(property_count, false);
  std::vector<int> matched_min(property_count, 0);
  bool found_required = required.count(properties[0].name);
  matched_min[0] = 1;
  for (int index = 1; index < property_count; ++index) {
    if (required.count(properties[index].name)) {
      is_required[index] = true;
      matched_min[index] = matched_min[index - 1] + 1;
    } else {
      matched_min[index] = matched_min[index - 1];
    }
    if (!found_required) {
      matched_min[index] = 1;
    }
    if (is_required[index]) {
      found_required = true;
    }
  }
  if (required.count(properties[0].name)) {
    is_required[0] = true;
  }

  if (max_properties == -1) {
    // Case 2: With constraint on the lower bound of the properties number
    std::vector<std::vector<int32_t>> tails(property_count);
    matched_min.back() = allow_additional ? std::max(1, matched_min.back())
                                          : std::max(min_properties, matched_min.back());
    for (int index = property_count - 2; index >= 0; --index) {
      matched_min[index] = std::max(matched_min[index], matched_min[index + 1] - 1);
    }

    for (int matched = matched_min.back(); matched <= property_count; ++matched) {
      int32_t body = allow_additional ? GetPropertyWithNumberConstraints(
                                            Sequence({middle_separator, get_additional_pattern()}),
                                            min_properties,
                                            max_properties,
                                            matched,
                                            rule_name
                                        )
                                      : Empty();
      if (allow_additional) {
        int32_t tail_rule_id = builder_.AddRuleWithHint(
            rule_name + "_part_" + std::to_string(property_count - 1) + "_" +
                std::to_string(matched),
            body
        );
        tails.back().push_back(RuleRef(tail_rule_id));
      } else {
        tails.back().push_back(body);
      }
    }

    for (int index = property_count - 2; index >= 0; --index) {
      for (int matched = matched_min[index]; matched <= index + 1; ++matched) {
        int32_t with_property = Sequence(
            {middle_separator,
             property_patterns[index + 1],
             tails[index + 1][matched + 1 - matched_min[index + 1]]}
        );
        int32_t body =
            (is_required[index + 1] || matched == matched_min[index + 1] - 1)
                ? with_property
                : Choice({tails[index + 1][matched - matched_min[index + 1]], with_property});
        int32_t tail_rule_id = builder_.AddRuleWithHint(
            rule_name + "_part_" + std::to_string(index) + "_" + std::to_string(matched), body
        );
        tails[index].push_back(RuleRef(tail_rule_id));
      }
    }

    std::vector<int32_t> choices;
    for (int index = 0; index < property_count; ++index) {
      if (matched_min[index] > 1) {
        break;
      }
      choices.push_back(Sequence({property_patterns[index], tails[index][1 - matched_min[index]]}));
      if (is_required[index]) {
        break;
      }
    }
    if (allow_additional && required.empty()) {
      choices.push_back(Sequence(
          {get_additional_pattern(),
           GetPropertyWithNumberConstraints(
               Sequence({middle_separator, get_additional_pattern()}),
               min_properties,
               max_properties,
               1,
               rule_name
           )}
      ));
    }
    return Sequence({first_separator, Choice(choices), last_separator});
  }

  // Case 3: With constraints on both lower & upper bound of the properties number
  std::vector<std::vector<int32_t>> tails(property_count);
  std::vector<int> matched_max(property_count, property_count);
  matched_max[0] = 1;
  for (int index = 1; index < property_count; ++index) {
    matched_max[index] = matched_max[index - 1] + 1;
  }
  matched_min.back() = allow_additional ? std::max(1, matched_min.back())
                                        : std::max(min_properties, matched_min.back());
  matched_max.back() = std::min(max_properties, matched_max.back());
  for (int index = property_count - 2; index >= 0; --index) {
    matched_min[index] = std::max(matched_min[index], matched_min[index + 1] - 1);
    matched_max[index] = is_required[index + 1]
                             ? std::min(matched_max[index], matched_max[index + 1] - 1)
                             : std::min(matched_max[index], matched_max[index + 1]);
  }

  for (int matched = matched_min.back(); matched <= matched_max.back(); ++matched) {
    int32_t body = allow_additional ? GetPropertyWithNumberConstraints(
                                          Sequence({middle_separator, get_additional_pattern()}),
                                          min_properties,
                                          max_properties,
                                          matched,
                                          rule_name
                                      )
                                    : Empty();
    if (allow_additional) {
      int32_t tail_rule_id = builder_.AddRuleWithHint(
          rule_name + "_part_" + std::to_string(property_count - 1) + "_" + std::to_string(matched),
          body
      );
      tails.back().push_back(RuleRef(tail_rule_id));
    } else {
      tails.back().push_back(body);
    }
  }

  for (int index = property_count - 2; index >= 0; --index) {
    for (int matched = matched_min[index]; matched <= matched_max[index]; ++matched) {
      int32_t body;
      if (matched == matched_max[index + 1]) {
        body = tails[index + 1][matched - matched_min[index + 1]];
      } else {
        int32_t with_property = Sequence(
            {middle_separator,
             property_patterns[index + 1],
             tails[index + 1][matched + 1 - matched_min[index + 1]]}
        );
        body = (is_required[index + 1] || matched == matched_min[index + 1] - 1)
                   ? with_property
                   : Choice({tails[index + 1][matched - matched_min[index + 1]], with_property});
      }
      int32_t tail_rule_id = builder_.AddRuleWithHint(
          rule_name + "_part_" + std::to_string(index) + "_" + std::to_string(matched), body
      );
      tails[index].push_back(RuleRef(tail_rule_id));
    }
  }

  std::vector<int32_t> choices;
  for (int index = 0; index < property_count; ++index) {
    if (matched_max[index] < matched_min[index]) {
      continue;
    }
    if (matched_min[index] > 1) {
      break;
    }
    choices.push_back(Sequence({property_patterns[index], tails[index][1 - matched_min[index]]}));
    if (is_required[index]) {
      break;
    }
  }
  if (allow_additional && required.empty()) {
    choices.push_back(Sequence(
        {get_additional_pattern(),
         GetPropertyWithNumberConstraints(
             Sequence({middle_separator, get_additional_pattern()}),
             min_properties,
             max_properties,
             1,
             rule_name
         )}
    ));
  }
  return Sequence({first_separator, Choice(choices), last_separator});
}

int32_t JSONSchemaConverter::GenerateObject(
    const ObjectSpec& spec, const std::string& rule_name, bool need_braces
) {
  // Determine additional property handling
  std::string additional_suffix;
  SchemaSpecPtr additional_property;
  if (spec.allow_additional_properties && spec.additional_properties_schema) {
    additional_suffix = "addl";
    additional_property = spec.additional_properties_schema;
  } else if (spec.allow_unevaluated_properties && spec.unevaluated_properties_schema) {
    additional_suffix = "uneval";
    additional_property = spec.unevaluated_properties_schema;
  } else if (spec.allow_additional_properties || spec.allow_unevaluated_properties) {
    additional_suffix = "addl";
    additional_property = SchemaSpec::Make(AnySpec{}, "", "any");
  }

  indent_manager_.StartIndent();
  bool has_content = false;
  bool could_be_empty = false;
  int32_t content = Empty();

  // Build a key rule through GenerateString rather than spelling out a JSON string here. At the
  // JSON root this still produces `"key"`, while XML-style converters override GenerateString to
  // produce the unquoted key body expected inside their parameter wrappers.
  auto create_pattern_key_rule = [&](const std::string& pattern,
                                     const std::string& rule_name_hint) -> int32_t {
    StringSpec key_spec;
    key_spec.pattern = pattern;
    return CreateRule(
        SchemaSpec::Make(std::move(key_spec), /*cache_key=*/"", rule_name_hint), rule_name_hint
    );
  };

  if (!spec.properties.empty() && (!spec.pattern_properties.empty() || spec.property_names)) {
    // Case 1a: properties coexist with patternProperties and/or propertyNames.
    // Use GetPartialRuleForProperties for named properties, and build
    // patternProperties/propertyNames as the additional property pattern override.
    SchemaSpecPtr effective_additional = additional_property;
    std::string effective_suffix = additional_suffix;
    std::optional<int32_t> additional_override;

    if (!spec.pattern_properties.empty()) {
      // Build patternProperties as additional property alternatives
      std::vector<int32_t> patterns;
      for (size_t index = 0; index < spec.pattern_properties.size(); ++index) {
        const auto& pattern_property = spec.pattern_properties[index];
        std::string pattern_suffix = "pp_" + std::to_string(index);
        int32_t key_rule_id = create_pattern_key_rule(
            pattern_property.pattern, rule_name + "_" + pattern_suffix + "_key"
        );
        int32_t value_rule_id =
            CreateRule(pattern_property.schema, rule_name + "_" + pattern_suffix);
        patterns.push_back(FormatOtherProperty(
            RuleRef(key_rule_id), value_rule_id, rule_name, pattern_suffix, pattern_property.schema
        ));
      }
      // Merge with existing additionalProperties if present
      if (effective_additional) {
        int32_t value_rule_id =
            CreateRule(effective_additional, rule_name + "_" + effective_suffix);
        patterns.push_back(FormatOtherProperty(
            KeyPatternExpression(), value_rule_id, rule_name, effective_suffix, effective_additional
        ));
      }
      additional_override = Choice(patterns);
      if (!effective_additional) {
        effective_additional = SchemaSpec::Make(AnySpec{}, "", "any");
      }
      effective_suffix = "pp";
    } else if (spec.property_names && effective_additional) {
      // propertyNames constrains keys of additional properties.
      // Only apply when additional properties are allowed - when additionalProperties
      // is false, no extra keys beyond named properties should be permitted.
      int32_t key_rule_id = CreateRule(spec.property_names, rule_name + "_name");
      int32_t value_rule_id = CreateRule(effective_additional, rule_name + "_" + effective_suffix);
      additional_override = FormatOtherProperty(
          RuleRef(key_rule_id),
          value_rule_id,
          rule_name,
          /*rule_name_suffix=*/"pn",
          effective_additional
      );
      effective_suffix = "pn";
    }

    content = GetPartialRuleForProperties(
        spec.properties,
        spec.required,
        effective_additional,
        rule_name,
        effective_suffix,
        spec.min_properties,
        spec.max_properties,
        additional_override
    );
    has_content = spec.max_properties != 0;
    could_be_empty = spec.required.empty() && spec.min_properties == 0;
  } else if (!spec.pattern_properties.empty() || spec.property_names) {
    // Case 1b: patternProperties or propertyNames without named properties
    if (spec.max_properties != 0) {
      int32_t beginning_separator = NextSeparatorExpression();
      std::vector<int32_t> property_choices;
      if (!spec.pattern_properties.empty()) {
        for (size_t index = 0; index < spec.pattern_properties.size(); ++index) {
          const auto& pattern_property = spec.pattern_properties[index];
          std::string pattern_suffix = "prop_" + std::to_string(index);
          int32_t key_rule_id = create_pattern_key_rule(
              pattern_property.pattern, rule_name + "_" + pattern_suffix + "_key"
          );
          int32_t value_rule_id =
              CreateRule(pattern_property.schema, rule_name + "_" + pattern_suffix);
          property_choices.push_back(Sequence(
              {beginning_separator,
               FormatOtherProperty(
                   RuleRef(key_rule_id),
                   value_rule_id,
                   rule_name,
                   pattern_suffix,
                   pattern_property.schema
               )}
          ));
        }
      } else {
        int32_t key_rule_id = CreateRule(spec.property_names, rule_name + "_name");
        // propertyNames constrains only the key, so a typed additionalProperties
        // schema still applies to the value (issue #826).
        int32_t value_rule_id;
        if (additional_property) {
          value_rule_id = CreateRule(additional_property, rule_name + "_" + additional_suffix);
        } else {
          value_rule_id = builder_.GetRuleId(GetBasicAnyRuleName());
          XGRAMMAR_DCHECK(value_rule_id != -1);
        }
        property_choices.push_back(Sequence(
            {beginning_separator,
             FormatOtherProperty(
                 RuleRef(key_rule_id),
                 value_rule_id,
                 rule_name,
                 /*rule_name_suffix=*/"pn",
                 additional_property
             )}
        ));
      }

      int32_t property_rule_id =
          builder_.AddRuleWithHint(rule_name + "_prop", Choice(property_choices));
      int32_t subsequent_property =
          Sequence({NextSeparatorExpression(), RuleRef(property_rule_id)});
      content = Sequence(
          {RuleRef(property_rule_id),
           GetPropertyWithNumberConstraints(
               subsequent_property, spec.min_properties, spec.max_properties, 1, rule_name
           ),
           NextSeparatorExpression(true)}
      );
      has_content = true;
      could_be_empty = spec.min_properties == 0;
    } else {
      could_be_empty = true;
    }
  } else if (!spec.properties.empty()) {
    // Case 2: properties defined (no patternProperties/propertyNames)
    content = GetPartialRuleForProperties(
        spec.properties,
        spec.required,
        additional_property,
        rule_name,
        additional_suffix,
        spec.min_properties,
        spec.max_properties
    );
    has_content = spec.max_properties != 0;
    could_be_empty = spec.required.empty() && spec.min_properties == 0;
  } else if (additional_property) {
    // Case 3: no properties defined, additional properties allowed
    if (spec.max_properties != 0) {
      int32_t value_rule_id = CreateRule(additional_property, rule_name + "_" + additional_suffix);
      int32_t property = FormatOtherProperty(
          KeyPatternExpression(), value_rule_id, rule_name, additional_suffix, additional_property
      );
      content = Sequence(
          {NextSeparatorExpression(),
           property,
           GetPropertyWithNumberConstraints(
               Sequence({NextSeparatorExpression(), property}),
               spec.min_properties,
               spec.max_properties,
               1,
               rule_name
           ),
           NextSeparatorExpression(true)}
      );
      has_content = true;
    }
    could_be_empty = spec.min_properties == 0;
  } else {
    // Case 4: no properties, no additional properties, no pattern properties
    // The object is unconditionally empty.
    could_be_empty = true;
  }

  indent_manager_.EndIndent();

  int32_t result = need_braces ? Sequence({ByteString("{"), content, ByteString("}")}) : content;
  if (could_be_empty) {
    int32_t empty_content = any_whitespace_ ? WhitespaceExpression() : Empty();
    int32_t empty_result =
        need_braces ? Sequence({ByteString("{"), empty_content, ByteString("}")}) : empty_content;
    return has_content ? Choice({result, empty_result}) : empty_result;
  }
  return result;
}

int32_t JSONSchemaConverter::GenerateAny(const AnySpec& spec, const std::string& rule_name) {
  return Choice(
      {RuleRef(kBasicNumber),
       RuleRef(kBasicString),
       RuleRef(kBasicBoolean),
       RuleRef(kBasicNull),
       RuleRef(kBasicArray),
       RuleRef(kBasicObject)}
  );
}

int32_t JSONSchemaConverter::GenerateConst(const ConstSpec& spec, const std::string& rule_name) {
  if (!IsAllowedJSONLiteral(spec.json_value)) {
    return Unsatisfiable();
  }
  return ByteString(spec.json_value);
}

int32_t JSONSchemaConverter::GenerateEnum(const EnumSpec& spec, const std::string& rule_name) {
  XGRAMMAR_DCHECK(!spec.json_values.empty())
      << "GenerateEnum called with empty enum spec for rule: " << rule_name;
  std::vector<int32_t> values;
  values.reserve(spec.json_values.size());
  for (const auto& value : spec.json_values) {
    if (IsAllowedJSONLiteral(value)) {
      values.push_back(ByteString(value));
    }
  }
  if (values.empty()) {
    return Unsatisfiable();
  }
  return Choice(values);
}

SchemaSpecPtr JSONSchemaConverter::ResolveRefSchema(
    const RefSpec& spec, const std::string& rule_name_hint
) {
  if (!ref_resolver_) {
    XGRAMMAR_LOG(FATAL) << "Ref resolver not set; cannot resolve $ref: " << spec.uri;
  }
  return ref_resolver_(spec.uri, rule_name_hint);
}

std::string JSONSchemaConverter::RefCacheKey(const std::string& uri) const { return uri; }

int32_t JSONSchemaConverter::GenerateRef(const RefSpec& spec, const std::string& rule_name) {
  const std::string cache_key = RefCacheKey(spec.uri);
  // First check if we have a direct URI mapping (for circular references)
  if (uri_to_rule_id_.count(cache_key)) {
    return RuleRef(uri_to_rule_id_[cache_key]);
  }

  // Derive rule name from URI path (like original URIToRule) so that the same
  // $ref always gets the same rule name, and allocate before resolving to prevent
  // dead recursion when the ref target contains a ref back.
  std::string rule_name_hint = "ref";
  if (spec.uri.size() >= 2 && spec.uri[0] == '#' && spec.uri[1] == '/') {
    std::string new_rule_name_prefix;
    std::stringstream ss(spec.uri.substr(2));
    std::string part;
    while (std::getline(ss, part, '/')) {
      if (!part.empty()) {
        if (!new_rule_name_prefix.empty()) {
          new_rule_name_prefix += "_";
        }
        for (char c : part) {
          if (std::isalpha(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.') {
            new_rule_name_prefix += c;
          }
        }
      }
    }
    if (!new_rule_name_prefix.empty()) {
      rule_name_hint = std::move(new_rule_name_prefix);
    }
  }

  int32_t allocated_rule_id = builder_.AddEmptyRuleWithHint(rule_name_hint);
  std::string allocated_rule_name = builder_.GetRule(allocated_rule_id).name;
  uri_to_rule_id_[cache_key] = allocated_rule_id;
  SchemaSpecPtr resolved = ResolveRefSchema(spec, allocated_rule_name);
  builder_.UpdateRuleBody(allocated_rule_id, GenerateFromSpec(resolved, allocated_rule_name));
  if (!resolved->cache_key.empty()) {
    AddCache(resolved->cache_key, allocated_rule_id);
  }
  return RuleRef(allocated_rule_id);
}

int32_t JSONSchemaConverter::GenerateAnyOf(const AnyOfSpec& spec, const std::string& rule_name) {
  std::vector<int32_t> choices;
  for (size_t index = 0; index < spec.options.size(); ++index) {
    choices.push_back(
        RuleRef(CreateRule(spec.options[index], rule_name + "_case_" + std::to_string(index)))
    );
  }
  return Choice(choices);
}

int32_t JSONSchemaConverter::GenerateOneOf(const OneOfSpec& spec, const std::string& rule_name) {
  std::vector<int32_t> choices;
  for (size_t index = 0; index < spec.options.size(); ++index) {
    choices.push_back(
        RuleRef(CreateRule(spec.options[index], rule_name + "_case_" + std::to_string(index)))
    );
  }
  return Choice(choices);
}

int32_t JSONSchemaConverter::GenerateAllOf(const AllOfSpec& spec, const std::string& rule_name) {
  if (spec.schemas.size() == 1) {
    return GenerateFromSpec(spec.schemas[0], rule_name + "_case_0");
  }
  XGRAMMAR_LOG(WARNING) << "Support for allOf with multiple options is still ongoing";
  return GenerateFromSpec(SchemaSpec::Make(AnySpec{}, "", "any"), rule_name);
}

int32_t JSONSchemaConverter::GenerateTypeArray(
    const TypeArraySpec& spec, const std::string& rule_name
) {
  std::vector<int32_t> choices;
  for (size_t index = 0; index < spec.type_schemas.size(); ++index) {
    choices.push_back(
        RuleRef(CreateRule(spec.type_schemas[index], rule_name + "_type_" + std::to_string(index)))
    );
  }
  return Choice(choices);
}

// ==================== Static Helper Methods ====================

std::optional<std::string> JSONSchemaConverter::JSONFormatToRegexPattern(const std::string& format
) {
  static const auto regex_map = []() -> std::unordered_map<std::string, std::string> {
    std::unordered_map<std::string, std::string> m;

    std::string atext = "[\\w!#$%&'*+/=?^`{|}~-]";
    std::string dot_string = "(" + atext + "+(\\." + atext + "+)*)";
    std::string quoted_string =
        "\\\\\"(\\\\[\\x20-\\x7E]|[\\x20\\x21\\x23-\\x5B\\x5D-\\x7E])*\\\\\"";
    std::string domain =
        "([A-Za-z0-9]([\\-A-Za-z0-9]*[A-Za-z0-9])?)((\\.[A-Za-z0-9][\\-A-Za-z0-9]*[A-Za-z0-9])*"
        ")";
    m["email"] = "^(" + dot_string + "|" + quoted_string + ")@" + domain + "$";

    m["date"] = "^(\\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[1-2]\\d|3[01]))$";
    m["time"] =
        "^([01]\\d|2[0-3]):[0-5]\\d:([0-5]\\d|60)(\\.\\d+)?(Z|[+-]([01]\\d|2[0-3]):[0-5]\\d)$";
    m["date-time"] =
        "^(\\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[1-2]\\d|3[01]))T([01]\\d|2[0-3]):[0-5]\\d:([0-5]\\d|60)("
        "\\.\\d+)?(Z|[+-]([01]\\d|2[0-3]):[0-5]\\d)$";
    m["duration"] =
        "^P((\\d+D|\\d+M(\\d+D)?|\\d+Y(\\d+M(\\d+D)?)?)(T(\\d+S|\\d+M(\\d+S)?|\\d+H(\\d+M(\\d+"
        "S)?"
        ")?))?|T(\\d+S|\\d+M(\\d+S)?|\\d+H(\\d+M(\\d+S)?)?)|\\d+W)$";

    std::string decbyte = "(25[0-5]|2[0-4]\\d|[0-1]?\\d?\\d)";
    m["ipv4"] = "^(" + decbyte + "\\.){3}" + decbyte + "$";

    m["ipv6"] =
        "("
        "([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|"
        "([0-9a-fA-F]{1,4}:){1,7}:|"
        "([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|"
        "([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|"
        "([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|"
        "([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|"
        "([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|"
        "[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|"
        ":((:[0-9a-fA-F]{1,4}){1,7}|:)|"
        "::(ffff(:0{1,4}){0,1}:){0,1}"
        "((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\\.){3,3}"
        "(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|"
        "([0-9a-fA-F]{1,4}:){1,4}:"
        "((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\\.){3,3}"
        "(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])"
        ")";

    m["hostname"] = "^([a-z0-9]([a-z0-9-]*[a-z0-9])?)(\\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)*$";
    m["uuid"] = "^[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$";

    std::string schema_pat = "[a-zA-Z][a-zA-Z+\\.-]*";
    std::string pchar = "([\\w\\.~!$&'()*+,;=:@-]|%[0-9A-Fa-f][0-9A-Fa-f])";
    std::string query_fragment_char = "([\\w\\.~!$&'()*+,;=:@/\\?-]|%[0-9A-Fa-f][0-9A-Fa-f])*";
    std::string query = "(\\?" + query_fragment_char + ")?";
    std::string fragment = "(#" + query_fragment_char + ")?";
    std::string path_abempty = "(/" + pchar + "*)*";
    std::string path_absolute_rootless_empty = "/?(" + pchar + "+(/" + pchar + "*)*)?";
    std::string userinfo = "([\\w\\.~!$&'()*+,;=:-]|%[0-9A-Fa-f][0-9A-Fa-f])*";
    std::string host = "([\\w\\.~!$&'()*+,;=-]|%[0-9A-Fa-f][0-9A-Fa-f])*";
    std::string authority = "(" + userinfo + "@)?" + host + "(:\\d*)?";
    std::string hier_part =
        "(//" + authority + path_abempty + "|" + path_absolute_rootless_empty + ")";
    m["uri"] = "^" + schema_pat + ":" + hier_part + query + fragment + "$";

    pchar = "([\\w\\.~!$&'()*+,;=:@-]|%[0-9A-Fa-f][0-9A-Fa-f])";
    query_fragment_char = "([\\w\\.~!$&'()*+,;=:@/\\?-]|%[0-9A-Fa-f][0-9A-Fa-f])*";
    query = "(\\?" + query_fragment_char + ")?";
    fragment = "(#" + query_fragment_char + ")?";
    path_abempty = "(/" + pchar + "*)*";
    std::string path_absolute = "/(" + pchar + "+(/" + pchar + "*)*)?";
    std::string segment_nz_nc = "([\\w\\.~!$&'()*+,;=@-]|%[0-9A-Fa-f][0-9A-Fa-f])+";
    std::string path_noscheme = segment_nz_nc + "(/" + pchar + "*)*";
    userinfo = "([\\w\\.~!$&'()*+,;=:-]|%[0-9A-Fa-f][0-9A-Fa-f])*";
    host = "([\\w\\.~!$&'()*+,;=-]|%[0-9A-Fa-f][0-9A-Fa-f])*";
    authority = "(" + userinfo + "@)?" + host + "(:\\d*)?";
    std::string relative_part =
        "(//" + authority + path_abempty + "|" + path_absolute + "|" + path_noscheme + ")?";
    m["uri-reference"] = "^" + relative_part + query + fragment + "$";

    std::string literals =
        "([\\x21\\x23-\\x24\\x26\\x28-\\x3B\\x3D\\x3F-\\x5B\\x5D\\x5F\\x61-\\x7A\\x7E]"
        "|%[0-9A-Fa-f][0-9A-Fa-f])";
    std::string op = "[+#\\./;\\?&=,!@|]";
    std::string varchar = "(\\w|%[0-9A-Fa-f][0-9A-Fa-f])";
    std::string varname = varchar + "(\\.?" + varchar + ")*";
    std::string varspec = varname + "(:[1-9]\\d?\\d?\\d?|\\*)?";
    std::string variable_list = varspec + "(," + varspec + ")*";
    std::string expression = "\\{(" + op + ")?" + variable_list + "\\}";
    m["uri-template"] = "^(" + literals + "|" + expression + ")*$";

    m["json-pointer"] = "^(/([\\x00-\\x2E]|[\\x30-\\x7D]|[\\x7F-\\U0010FFFF]|~[01])*)*$";
    m["relative-json-pointer"] =
        "^(0|[1-9][0-9]*)(#|(/([\\x00-\\x2E]|[\\x30-\\x7D]|[\\x7F-\\U0010FFFF]|~[01])*)*)$";

    return m;
  }();

  auto it = regex_map.find(format);
  if (it == regex_map.end()) {
    return std::nullopt;
  }
  return it->second;
}

// ==================== XMLToolCallingConverter Implementation ====================

namespace {

constexpr const char* kStringCacheKey = "{\"type\":\"string\"}";
constexpr const char* kObjectCacheKey = "{\"type\":\"object\"}";

}  // namespace

const std::string XMLToolCallingConverter::kXMLString = "xml_string";
const std::string XMLToolCallingConverter::kXMLAny = "xml_any";
const std::string XMLToolCallingConverter::kXMLObject = "xml_object";
const std::string XMLToolCallingConverter::kXMLVariableName = "xml_variable_name";

const std::unordered_map<JSONFormat, XMLToolCallingConverter::XMLWrapper>
    XMLToolCallingConverter::kKeyWrapperMap = {
        {JSONFormat::kQwenXML, converter_ext::GetQwenXMLWrapper()},
        {JSONFormat::kMiniMaxXML, converter_ext::GetMiniMaxXMLWrapper()},
        {JSONFormat::kDeepSeekXML, converter_ext::GetDeepSeekXMLWrapper()},
        {JSONFormat::kDeepSeekV41XML, converter_ext::GetDeepSeekV41XMLWrapper()},
        {JSONFormat::kGlmXML, converter_ext::GetGLMXMLWrapper()},
        {JSONFormat::kCohereXML, converter_ext::GetCohereXMLWrapper()},
        {JSONFormat::kKimiK3XML, converter_ext::GetKimiK3XMLWrapper()},
};

XMLToolCallingConverter::XMLToolCallingConverter(
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool any_whitespace,
    std::optional<int> max_whitespace_cnt,
    RefResolver ref_resolver,
    JSONFormat json_format,
    bool any_order,
    std::vector<std::string> excludes
)
    : JSONSchemaConverter(
          indent,
          separators,
          any_whitespace,
          max_whitespace_cnt,
          ref_resolver,
          any_order,
          std::move(excludes)
      ),
      json_format_(json_format),
      nested_object_level_(0),
      xml_wrapper_(kKeyWrapperMap.at(json_format)) {
  // XML formatting can add whitespace outside a constrained raw string's rule.
  // Reject exclusions that could straddle that boundary instead of silently bypassing them.
  auto is_padding = [](char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
  for (const auto& excluded : excludes_) {
    XGRAMMAR_CHECK(
        excluded.empty() || (!is_padding(excluded.front()) && !is_padding(excluded.back()))
    ) << "XML JSONSchemaFormat.excludes must not start or end with formatting whitespace";
  }
}

Grammar XMLToolCallingConverter::Convert(const SchemaSpecPtr& spec) {
  nested_object_level_ = 0;
  return JSONSchemaConverter::Convert(spec);
}

std::optional<std::string> XMLToolCallingConverter::GetRenderedJSONType(const SchemaSpecPtr& spec) {
  if (spec == nullptr) {
    return std::nullopt;
  }
  auto type_of_json_value = [](const std::string& json_value) -> std::optional<std::string> {
    picojson::value value;
    if (!ParseJSON(value, json_value).empty()) {
      return std::nullopt;
    }
    if (value.is<std::string>()) return "string";
    if (value.is<bool>()) return "boolean";
    if (value.is<double>()) return "number";
    if (value.is<picojson::null>()) return "null";
    if (value.is<picojson::object>()) return "object";
    if (value.is<picojson::array>()) return "array";
    return std::nullopt;
  };

  return std::visit(
      [&](auto&& arg) -> std::optional<std::string> {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, StringSpec>) {
          return "string";
        } else if constexpr (std::is_same_v<T, IntegerSpec> || std::is_same_v<T, NumberSpec>) {
          // Both integer and floating-point values have the JSON type "number".
          return "number";
        } else if constexpr (std::is_same_v<T, BooleanSpec>) {
          return "boolean";
        } else if constexpr (std::is_same_v<T, NullSpec>) {
          return "null";
        } else if constexpr (std::is_same_v<T, ArraySpec>) {
          return "array";
        } else if constexpr (std::is_same_v<T, ObjectSpec>) {
          return "object";
        } else if constexpr (std::is_same_v<T, ConstSpec>) {
          return type_of_json_value(arg.json_value);
        } else if constexpr (std::is_same_v<T, EnumSpec>) {
          // Only pin the attribute when every alternative renders with the same type.
          std::optional<std::string> common;
          for (const auto& json_value : arg.json_values) {
            auto type_name = type_of_json_value(json_value);
            if (!type_name.has_value()) return std::nullopt;
            if (!common.has_value()) {
              common = type_name;
            } else if (*common != *type_name) {
              return std::nullopt;
            }
          }
          return common;
        } else {
          // Any, $ref and the combinators may render as more than one type; keep them open.
          return std::nullopt;
        }
      },
      spec->spec
  );
}

std::string XMLToolCallingConverter::XMLValue(const std::string& json_value) const {
  picojson::value value;
  std::string error = ParseJSON(value, json_value);
  if (error.empty() && value.is<std::string>()) {
    return value.get<std::string>();
  }
  return json_value;
}

int32_t XMLToolCallingConverter::XMLKeySuffix(const std::optional<std::string>& pinned_type) {
  auto value_choices = [this](const std::vector<const char*>& values) {
    std::vector<int32_t> choices;
    choices.reserve(values.size());
    for (const auto* value : values) {
      choices.push_back(ByteString(value));
    }
    return Choice(choices);
  };
  if (json_format_ == JSONFormat::kKimiK3XML) {
    const auto& suffix = converter_ext::GetKimiK3XMLKeySuffix();
    // A declared property carries exactly the type its value grammar is rendered with, so the
    // parser decodes the value back to the schema's type. Free-form keys have no single schema
    // type, so they keep the full set.
    int32_t type_expr =
        pinned_type.has_value() ? ByteString(*pinned_type) : value_choices(suffix.values);
    return Sequence({ByteString(suffix.prefix), type_expr, ByteString(suffix.suffix)});
  }
  return ByteString(xml_wrapper_.key_wrapper_suffix);
}

void XMLToolCallingConverter::AddBasicRules() {
  // First add JSON basic rules. These should be in the inner layer of the XML format.
  XGRAMMAR_DCHECK(nested_object_level_ == 0);
  // The nested part, true json format, is at level 2.
  nested_object_level_ = 2;
  JSONSchemaConverter::AddBasicRules({kXMLString, kXMLAny, kXMLObject, kXMLVariableName});

  auto any_spec = SchemaSpec::Make(AnySpec{}, "{}", kBasicAny);

  // The outer part, xml format, is at level 1.
  nested_object_level_ = 1;
  // Keep the unrestricted raw body as a single TagDispatch. The argument suffix is matched
  // by the enclosing property rule, outside this body's exclusion scope.
  auto string_excludes = excludes_;
  string_excludes.push_back(xml_wrapper_.parameter_suffix);
  builder_.UpdateRuleBody(kXMLString, TagDispatch(false, std::move(string_excludes)));
  AddCache(kStringCacheKey, builder_.GetRuleId(kXMLString));

  // Add XML any rule
  builder_.UpdateRuleBody(kXMLAny, GenerateAny(AnySpec{}, kXMLAny));
  AddCache("{}", builder_.GetRuleId(kXMLAny));

  // Reset the nested object level to 0, which is the root level.
  nested_object_level_ = 0;

  // Add XML object rule
  ObjectSpec xml_object_spec;
  xml_object_spec.allow_additional_properties = true;
  xml_object_spec.additional_properties_schema = any_spec;
  builder_.UpdateRuleBody(kXMLObject, GenerateObject(xml_object_spec, kXMLObject));
  AddCache(kObjectCacheKey, builder_.GetRuleId(kXMLObject));

  // Add XML variable name rule
  builder_.UpdateRuleBody(
      kXMLVariableName,
      !excludes_.empty()
          ? ExcludingString("[a-zA-Z_][a-zA-Z0-9_]*", kXMLVariableName, false)
          : Sequence(
                {builder_.AddCharacterClass({{'a', 'z'}, {'A', 'Z'}, {'_', '_'}}),
                 builder_.AddCharacterClassStar({{'a', 'z'}, {'A', 'Z'}, {'0', '9'}, {'_', '_'}})}
            )
  );
}

std::string XMLToolCallingConverter::GetKeyPattern() const {
  if (nested_object_level_ <= 1) {
    return kXMLVariableName;
  }
  return kBasicString;
}

std::string XMLToolCallingConverter::GetBasicAnyRuleName() const {
  if (nested_object_level_ <= 1) {
    return kXMLAny;
  }
  return kBasicAny;
}

int32_t XMLToolCallingConverter::GetKeyPatternExcluding(
    const std::vector<ObjectSpec::Property>& properties, const std::string& rule_name
) {
  if (nested_object_level_ <= 1) {
    return RuleRef(GetKeyPattern());
  }
  return JSONSchemaConverter::GetKeyPatternExcluding(properties, rule_name);
}

std::string XMLToolCallingConverter::NextSeparator(bool is_end) {
  if (nested_object_level_ <= 1) {
    return GetWhitespacePattern();
  }
  return JSONSchemaConverter::NextSeparator(is_end);
}

int32_t XMLToolCallingConverter::GenerateString(
    const StringSpec& spec, const std::string& rule_name
) {
  if (nested_object_level_ <= 1) {
    if (spec.format.has_value()) {
      auto regex = JSONFormatToRegexPattern(*spec.format);
      if (regex.has_value()) {
        return RegexExpression(*regex, false, true);
      }
    }
    if (spec.pattern.has_value()) {
      return RegexExpression(*spec.pattern, false, /*force_cfg_expansion=*/true);
    }
    const bool bounded = spec.min_length != 0 || spec.max_length != -1;
    // Without exclusions, a length bound or an unrecognized format keeps the plain repetition.
    if (excludes_.empty() && (bounded || spec.format.has_value())) {
      return Repeat(
          rule_name + "_characters",
          builder_.AddCharacterClass({{0, 0x10ffff}}),
          spec.min_length,
          spec.max_length
      );
    }
    // With exclusions the raw string keeps only the exclusions: length constraints are dropped
    // (see JSONSchemaConverter::GenerateString) and an unrecognized format is unconstrained.
    if (bounded) {
      WarnDroppedLengthConstraints(spec, rule_name);
    }
    return RuleRef(kXMLString);
  }
  return JSONSchemaConverter::GenerateString(spec, rule_name);
}

int32_t XMLToolCallingConverter::GenerateAny(const AnySpec& spec, const std::string& rule_name) {
  if (nested_object_level_ == 0) {
    return RuleRef(kXMLObject);
  }
  if (nested_object_level_ == 1) {
    return Choice({RuleRef(kXMLString), RuleRef(kBasicArray), RuleRef(kBasicObject)});
  }
  return JSONSchemaConverter::GenerateAny(spec, rule_name);
}

int32_t XMLToolCallingConverter::GenerateArray(
    const ArraySpec& spec, const std::string& rule_name
) {
  nested_object_level_++;
  auto result = JSONSchemaConverter::GenerateArray(spec, rule_name);
  nested_object_level_--;
  return result;
}

int32_t XMLToolCallingConverter::GenerateConst(
    const ConstSpec& spec, const std::string& rule_name
) {
  if (nested_object_level_ == 0) {
    picojson::value value;
    XGRAMMAR_CHECK(ParseJSON(value, spec.json_value).empty());
    if (value.is<picojson::object>()) {
      // A root object is a parameter list, including when all its values are fixed.
      // Nested object constants still use the JSON representation below.
      ObjectSpec object;
      object.allow_unevaluated_properties = false;
      const auto& properties = value.get<picojson::object>();
      for (const auto& key : properties.ordered_keys()) {
        object.properties.push_back(
            {key, SchemaSpec::Make(ConstSpec{properties.at(key).serialize()})}
        );
        object.required.insert(key);
      }
      // As with JSON literals, keep a fixed order even when any_order is enabled.
      // The general any-order object rule permits repeated keys and is not exact for const.
      bool saved_any_order = any_order_;
      any_order_ = false;
      int32_t result = GenerateObject(object, rule_name);
      any_order_ = saved_any_order;
      return result;
    }
  }
  if (nested_object_level_ <= 1) {
    if (!IsAllowedJSONLiteral(spec.json_value, /*raw_string=*/true)) {
      return Unsatisfiable();
    }
    return ByteString(XMLValue(spec.json_value));
  }
  return JSONSchemaConverter::GenerateConst(spec, rule_name);
}

int32_t XMLToolCallingConverter::GenerateEnum(const EnumSpec& spec, const std::string& rule_name) {
  XGRAMMAR_DCHECK(!spec.json_values.empty())
      << "GenerateEnum called with empty enum spec for rule: " << rule_name;
  if (nested_object_level_ <= 1) {
    std::vector<int32_t> values;
    values.reserve(spec.json_values.size());
    for (const auto& value : spec.json_values) {
      values.push_back(GenerateConst(ConstSpec{value}, rule_name));
    }
    return Choice(values);
  }
  return JSONSchemaConverter::GenerateEnum(spec, rule_name);
}

int32_t XMLToolCallingConverter::FormatPropertyKey(
    const std::string& key, const SchemaSpecPtr& schema
) {
  if (nested_object_level_ <= 1) {
    // Only kimi_k3_xml encodes the value's type next to the key; the other formats would
    // discard the result, so don't walk the schema for them.
    std::optional<std::string> pinned_type;
    if (json_format_ == JSONFormat::kKimiK3XML) {
      pinned_type = GetRenderedJSONType(schema);
    }
    if (!IsAllowedString(EscapeAttrValue(key))) {
      return Unsatisfiable();
    }
    return Sequence(
        {ByteString(xml_wrapper_.key_wrapper_prefix + EscapeAttrValue(key)),
         XMLKeySuffix(pinned_type)}
    );
  }
  return JSONSchemaConverter::FormatPropertyKey(key, schema);
}

int32_t XMLToolCallingConverter::FormatProperty(
    const std::string& key,
    int32_t value_rule_id,
    const std::string& rule_name,
    int64_t idx,
    const SchemaSpecPtr& schema
) {
  if (nested_object_level_ <= 1) {
    if (json_format_ == JSONFormat::kDeepSeekXML || json_format_ == JSONFormat::kDeepSeekV41XML) {
      if (!IsAllowedString(key)) {
        return Unsatisfiable();
      }
      return Sequence(
          {ByteString(xml_wrapper_.key_wrapper_prefix + key),
           FormatDeepSeekParamSuffix(schema, value_rule_id)}
      );
    }
    std::vector<int32_t> elements = {FormatPropertyKey(key, schema)};
    if (!xml_wrapper_.value_wrapper_prefix.empty()) {
      elements.push_back(WhitespaceExpression());
      elements.push_back(ByteString(xml_wrapper_.value_wrapper_prefix));
    }
    // xml_string already accepts whitespace. Adding whitespace repetitions around it preserves the
    // language but creates one Earley state for every possible split with the string body.
    if (value_rule_id == builder_.GetRuleId(kXMLString)) {
      elements.push_back(RuleRef(value_rule_id));
    } else {
      elements.push_back(WhitespaceExpression());
      elements.push_back(RuleRef(value_rule_id));
      elements.push_back(WhitespaceExpression());
    }
    elements.push_back(ByteString(xml_wrapper_.parameter_suffix));
    return Sequence(elements);
  }
  return JSONSchemaConverter::FormatProperty(key, value_rule_id, rule_name, idx, schema);
}

int32_t XMLToolCallingConverter::FormatOtherProperty(
    int32_t key_pattern_expr,
    int32_t value_rule_id,
    const std::string& rule_name,
    const std::string& rule_name_suffix,
    const SchemaSpecPtr& schema
) {
  if (nested_object_level_ <= 1) {
    if (json_format_ == JSONFormat::kDeepSeekXML || json_format_ == JSONFormat::kDeepSeekV41XML) {
      return Sequence(
          {ByteString(xml_wrapper_.key_wrapper_prefix),
           key_pattern_expr,
           FormatDeepSeekParamSuffix(schema, value_rule_id)}
      );
    }
    std::vector<int32_t> elements = {
        ByteString(xml_wrapper_.key_wrapper_prefix),
        key_pattern_expr,
        XMLKeySuffix(
            json_format_ == JSONFormat::kKimiK3XML ? GetRenderedJSONType(schema) : std::nullopt
        )
    };
    if (!xml_wrapper_.value_wrapper_prefix.empty()) {
      elements.push_back(WhitespaceExpression());
      elements.push_back(ByteString(xml_wrapper_.value_wrapper_prefix));
    }
    if (value_rule_id == builder_.GetRuleId(kXMLString)) {
      elements.push_back(RuleRef(value_rule_id));
    } else {
      elements.push_back(WhitespaceExpression());
      elements.push_back(RuleRef(value_rule_id));
      elements.push_back(WhitespaceExpression());
    }
    elements.push_back(ByteString(xml_wrapper_.parameter_suffix));
    return Sequence(elements);
  }
  return JSONSchemaConverter::FormatOtherProperty(
      key_pattern_expr, value_rule_id, rule_name, rule_name_suffix, schema
  );
}

int32_t XMLToolCallingConverter::GenerateObject(
    const ObjectSpec& spec, const std::string& rule_name, bool dummy_need_braces
) {
  nested_object_level_++;
  bool need_brace = nested_object_level_ > 1;
  auto result = JSONSchemaConverter::GenerateObject(spec, rule_name, need_brace);
  nested_object_level_--;
  return result;
}

void XMLToolCallingConverter::AddCache(const std::string& key, int32_t rule_id) {
  if (key.empty()) {
    return;
  }
  rule_cache_manager_.AddCache(key, EncodingContext(), rule_id);
}

std::optional<int32_t> XMLToolCallingConverter::GetCache(const std::string& key) const {
  if (key.empty()) {
    return std::nullopt;
  }
  if ((json_format_ == JSONFormat::kQwenXML || json_format_ == JSONFormat::kDeepSeekXML ||
       json_format_ == JSONFormat::kDeepSeekV41XML) &&
      nested_object_level_ == 0 && key == "{}") {
    // Unconstrained tool arguments are an XML parameter list, not one parameter's raw value.
    return rule_cache_manager_.GetCache(kObjectCacheKey, 0);
  }
  // At level 0, {"type":"object"} is the root tool-arguments object and uses XML parameter
  // tags. At level 1 it is the value of one such parameter and must use the inner JSON object
  // rule, including braces. Without this distinction, the outer XML object cache is reused for
  // the value before GenerateObject() can advance nested_object_level_.
  if (nested_object_level_ == 1 && key == kObjectCacheKey) {
    return rule_cache_manager_.GetCache(key, 2);
  }
  return rule_cache_manager_.GetCache(key, EncodingContext());
}

std::string XMLToolCallingConverter::RefCacheKey(const std::string& uri) const {
  return std::to_string(EncodingContext()) + ":" + uri;
}

// ==================== Range Regex Generation ====================

// Stateless utility that turns a numeric range into an anchored regex matching
// exactly the JSON integers / numbers inside it. Every method is static; the
// class exists only to group the helpers and keep the internal ones private.
class NumberGenerator {
 public:
  // Anchored regex matching every integer x with start <= x <= end. Either bound
  // may be std::nullopt for an open side; an empty range yields "^()$". Bounds
  // span the whole int64 range (|INT64_MIN| is handled without negation overflow).
  static std::string IntegerRangeRegex(std::optional<int64_t> start, std::optional<int64_t> end);

  // Anchored regex matching every number in the range, written with up to
  // `precision` fraction digits. `exclusive_start` / `exclusive_end` exclude the
  // boundary value itself (turning >= / <= into > / <). Either bound may be
  // std::nullopt for an open side; an empty range yields "^()$".
  static std::string FloatRangeRegex(
      std::optional<double> start,
      std::optional<double> end,
      int precision,
      bool exclusive_start,
      bool exclusive_end
  );

 private:
  // Regex alternatives for the fraction digits following a decimal point.
  struct FracPatternSet {
    // Each pattern matches a non-empty fraction digit string.
    std::vector<std::string> parts;
    // Whether having no fraction digits at all also satisfies the bound.
    bool include_empty = false;
  };

  // --- Regex fragment primitives ---
  static std::string DigitClass(char lo, char hi);  // one digit in [lo, hi] (or \d)
  static std::string ExactDigits(int k);            // exactly k free digits: \d{k}
  static std::string FreeDigits(int max_count);     // 0..max_count free digits: \d{0,n}
  static std::string OptionalZeros(int max_count);  // 0..max_count zeros: 0{0,n}
  static std::string SomeZeros(int max_count);      // 1..max_count zeros: 0{1,n}
  static bool AllChar(const std::string& s, char c);

  // --- Integer range (operate on non-negative decimal magnitude strings) ---
  static std::string AbsDigits(int64_t v);
  static int CompareDigitStr(const std::string& a, const std::string& b);
  static std::vector<std::string> IntSameLen(const std::string& a, const std::string& b);
  static std::vector<std::string> NumberPatternsStr(const std::string& lo, const std::string& hi);
  static std::string SubRangeRegexStr(const std::string& lo, const std::string& hi);
  static std::vector<std::string> AtLeastPositivePatternsStr(const std::string& v_str);

  // --- Float range ---
  static std::string FormatFloat(double value, int precision);
  // Snaps a non-negative bound to the precision grid in the direction that keeps
  // the range sound: a lower bound rounds up, an upper bound rounds down, so no
  // out-of-range value is ever admitted. Returns the canonical grid string and,
  // via strict_out, whether the boundary value must still be excluded.
  static std::string RoundBoundToGrid(
      double value, int precision, bool is_lower, bool strict_in, bool* strict_out
  );
  // Adds (inc) or subtracts (!inc) one grid step (10^-precision) to a canonical
  // non-negative decimal string, returning the canonical result.
  static std::string AdjustGrid(const std::string& s, int precision, bool inc);
  static void SplitDecimal(const std::string& s, std::string* int_part, std::string* frac_part);
  static int CompareDecimal(
      const std::string& int_a,
      const std::string& frac_a,
      const std::string& int_b,
      const std::string& frac_b
  );
  static std::string StripAnchors(const std::string& regex);
  static int64_t ParseIntCapped(const std::string& digits);
  static FracPatternSet FracGreaterPatterns(const std::string& s, bool strict, int max_len);
  static FracPatternSet FracLessPatterns(const std::string& s, bool strict, int max_len);
  static FracPatternSet FracBetweenPatterns(
      const std::string& a, bool strict_a, const std::string& b, bool strict_b, int max_len
  );
  static std::vector<std::string> PositiveRangeParts(
      const std::string& low,
      bool strict_low,
      const std::optional<std::string>& high,
      bool strict_high,
      int precision
  );
};

// Helpers for integer range regex generation. They operate purely on
// fixed-length decimal digit strings (suffixes may carry leading zeros), so the
// patterns are correct by construction regardless of digit position.

// A regex fragment matching a single digit in [lo, hi].
std::string NumberGenerator::DigitClass(char lo, char hi) {
  if (lo == hi) {
    return std::string(1, lo);
  }
  if (lo == '0' && hi == '9') {
    return "\\d";
  }
  return "[" + std::string(1, lo) + "-" + std::string(1, hi) + "]";
}

// A regex fragment matching k free digits (each 0-9). Empty when k <= 0.
std::string NumberGenerator::ExactDigits(int k) {
  if (k <= 0) {
    return "";
  }
  if (k == 1) {
    return "\\d";
  }
  return "\\d{" + std::to_string(k) + "}";
}

bool NumberGenerator::AllChar(const std::string& s, char c) {
  return std::all_of(s.begin(), s.end(), [c](char ch) { return ch == c; });
}

// Patterns matching every equal-length digit string t with
// value(a) <= value(t) <= value(b). Requires a.size() == b.size() and
// value(a) <= value(b). Partitions t by its first digit:
//   * first digit == a[0]: the suffix must be >= a's suffix (<= 99..9);
//   * first digit strictly between a[0] and b[0]: the suffix is unconstrained;
//   * first digit == b[0]: the suffix must be <= b's suffix (>= 00..0).
// The partition is exact and non-overlapping, so the union is sound and
// complete for [a, b].
std::vector<std::string> NumberGenerator::IntSameLen(const std::string& a, const std::string& b) {
  int n = static_cast<int>(a.size());
  if (a == b) {
    return {a};
  }
  if (n == 1) {
    return {DigitClass(a[0], b[0])};
  }
  if (a[0] == b[0]) {
    std::vector<std::string> res;
    for (auto& p : IntSameLen(a.substr(1), b.substr(1))) {
      res.push_back(std::string(1, a[0]) + p);
    }
    return res;
  }
  // a[0] < b[0]
  std::string a_suf = a.substr(1);
  std::string b_suf = b.substr(1);
  if (AllChar(a_suf, '0') && AllChar(b_suf, '9')) {
    // The whole suffix space is free: collapse to one box pattern.
    if (a[0] == '0' && b[0] == '9') {
      return {ExactDigits(n)};
    }
    return {DigitClass(a[0], b[0]) + ExactDigits(n - 1)};
  }
  std::vector<std::string> res;
  std::string nines(n - 1, '9');
  std::string zeros(n - 1, '0');
  for (auto& p : IntSameLen(a_suf, nines)) {
    res.push_back(std::string(1, a[0]) + p);
  }
  if (b[0] - a[0] >= 2) {
    res.push_back(
        DigitClass(static_cast<char>(a[0] + 1), static_cast<char>(b[0] - 1)) + ExactDigits(n - 1)
    );
  }
  for (auto& p : IntSameLen(zeros, b_suf)) {
    res.push_back(std::string(1, b[0]) + p);
  }
  return res;
}

// Compares two non-negative decimal magnitude strings (no leading zeros except
// "0") by value.
int NumberGenerator::CompareDigitStr(const std::string& a, const std::string& b) {
  if (a.size() != b.size()) {
    return a.size() < b.size() ? -1 : 1;
  }
  if (a < b) {
    return -1;
  }
  return a > b ? 1 : 0;
}

// Patterns matching every integer whose magnitude has value in [lo, hi], where
// lo and hi are non-negative decimal magnitude strings (no leading zeros except
// "0"). An empty range (value(lo) > value(hi)) yields no patterns. Operating on
// strings keeps the whole int64 range representable, including
// |INT64_MIN| = 9223372036854775808, which does not fit in int64.
std::vector<std::string> NumberGenerator::NumberPatternsStr(
    const std::string& lo, const std::string& hi
) {
  std::vector<std::string> patterns;
  if (CompareDigitStr(lo, hi) > 0) {
    return patterns;
  }
  int lo_len = static_cast<int>(lo.size());
  int hi_len = static_cast<int>(hi.size());
  // Split [lo, hi] by digit length; each length yields a same-length segment
  // handled exactly by IntSameLen.
  for (int len = lo_len; len <= hi_len; ++len) {
    std::string a_str = (len == lo_len) ? lo : ("1" + std::string(len - 1, '0'));
    std::string b_str = (len == hi_len) ? hi : std::string(len, '9');
    for (auto& p : IntSameLen(a_str, b_str)) {
      patterns.push_back(p);
    }
  }
  return patterns;
}

// Joins NumberPatternsStr alternatives into a parenthesised regex group.
std::string NumberGenerator::SubRangeRegexStr(const std::string& lo, const std::string& hi) {
  std::vector<std::string> patterns = NumberPatternsStr(lo, hi);
  std::string joined;
  for (size_t i = 0; i < patterns.size(); ++i) {
    if (i > 0) {
      joined += "|";
    }
    joined += patterns[i];
  }
  return "(" + joined + ")";
}

// Patterns matching every integer in [value(v_str), +infinity) for v_str a
// positive magnitude string (no leading zeros). Same-length values come from
// IntSameLen(v_str, 99..9); strictly longer values are any non-zero-led number.
std::vector<std::string> NumberGenerator::AtLeastPositivePatternsStr(const std::string& v_str) {
  int len = static_cast<int>(v_str.size());
  std::vector<std::string> res = IntSameLen(v_str, std::string(len, '9'));
  res.push_back("[1-9]\\d{" + std::to_string(len) + ",}");
  return res;
}

// The magnitude (absolute value) of v as a decimal string. Derived from the
// signed text rather than by negating v, so INT64_MIN is handled correctly.
std::string NumberGenerator::AbsDigits(int64_t v) {
  std::string s = std::to_string(v);
  return (!s.empty() && s[0] == '-') ? s.substr(1) : s;
}

std::string NumberGenerator::IntegerRangeRegex(
    std::optional<int64_t> start, std::optional<int64_t> end
) {
  std::vector<std::string> parts;
  std::ostringstream result;

  if (!start && !end) {
    return "^-?\\d+$";
  }

  if (start && !end) {
    if (start.value() <= 0) {
      if (start.value() < 0) {
        // Negatives in [start, -1] are the magnitudes [1, |start|], negated.
        parts.push_back("-" + SubRangeRegexStr("1", AbsDigits(start.value())));
      }
      parts.push_back("0");
      parts.push_back("[1-9]\\d*");
    } else {
      // x >= start with start > 0: same-length values >= start, plus every
      // value with strictly more digits.
      for (auto& p : AtLeastPositivePatternsStr(std::to_string(start.value()))) {
        parts.push_back(p);
      }
    }
  }

  if (!start && end) {
    if (end.value() >= 0) {
      parts.push_back("-[1-9]\\d*");
      parts.push_back("0");
      if (end.value() > 0) {
        parts.push_back(SubRangeRegexStr("1", std::to_string(end.value())));
      }
    } else {
      // x <= end with end < 0: x = -a where a >= |end| > 0, so negate every
      // pattern for the range [|end|, +infinity).
      for (auto& p : AtLeastPositivePatternsStr(AbsDigits(end.value()))) {
        parts.push_back("-" + p);
      }
    }
  }

  if (start && end) {
    int64_t range_start = start.value();
    int64_t range_end = end.value();

    if (range_start > range_end) {
      return "^()$";
    }

    if (range_start < 0) {
      int64_t neg_start = range_start;
      int64_t neg_end = std::min(static_cast<int64_t>(-1), range_end);
      // Negatives in [neg_start, neg_end] are the magnitudes
      // [|neg_end|, |neg_start|], negated.
      parts.push_back("-" + SubRangeRegexStr(AbsDigits(neg_end), AbsDigits(neg_start)));
    }

    if (range_start <= 0 && range_end >= 0) {
      parts.push_back("0");
    }

    if (range_end > 0) {
      int64_t pos_start = std::max(static_cast<int64_t>(1), range_start);
      parts.push_back(SubRangeRegexStr(std::to_string(pos_start), std::to_string(range_end)));
    }
  }

  result << "^(";
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i > 0) {
      result << "|";
    }
    result << parts[i];
  }
  result << ")$";

  return result.str();
}

std::string NumberGenerator::FormatFloat(double value, int precision) {
  // Casting a double outside [INT64_MIN, INT64_MAX] (or NaN/Inf) to int64_t is
  // undefined behavior, so range-check before the integer fast path. 2^63 ==
  // 9223372036854775808.0 is exactly representable and one past INT64_MAX, so the
  // upper comparison must be strict.
  if (value >= -9223372036854775808.0 && value < 9223372036854775808.0 &&
      value == static_cast<int64_t>(value)) {
    return std::to_string(static_cast<int64_t>(value));
  }

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision) << value;
  std::string result = oss.str();

  size_t decimalPos = result.find('.');
  if (decimalPos != std::string::npos) {
    size_t lastNonZero = result.find_last_not_of('0');
    if (lastNonZero != std::string::npos && lastNonZero > decimalPos) {
      result.erase(lastNonZero + 1);
    } else if (lastNonZero == decimalPos) {
      result.erase(decimalPos);
    }
  }

  return result;
}

std::string NumberGenerator::AdjustGrid(const std::string& s, int precision, bool inc) {
  std::string int_part, frac_part;
  SplitDecimal(s, &int_part, &frac_part);
  // Build the scaled-integer numerator (value * 10^precision) as a digit string.
  // Callers only pass FormatFloat output (<= precision fraction digits); guard
  // the count so a longer string can never wrap the unsigned append count.
  frac_part.append(std::max(0, precision - static_cast<int>(frac_part.size())), '0');
  std::string num = int_part + frac_part;

  if (inc) {
    int i = static_cast<int>(num.size()) - 1;
    for (; i >= 0 && num[i] == '9'; --i) {
      num[i] = '0';
    }
    if (i < 0) {
      num.insert(num.begin(), '1');
    } else {
      num[i]++;
    }
  } else {
    int i = static_cast<int>(num.size()) - 1;
    for (; i >= 0 && num[i] == '0'; --i) {
      num[i] = '9';
    }
    if (i < 0) {
      // Underflow below zero; clamp to zero (does not occur for the bounds the
      // float pipeline feeds in, which are all >= one grid step when decremented).
      num.assign(num.size(), '0');
    } else {
      num[i]--;
    }
  }

  // Re-split into integer and `precision`-digit fraction, then canonicalize.
  while (static_cast<int>(num.size()) <= precision) {
    num.insert(num.begin(), '0');
  }
  std::string new_int = num.substr(0, num.size() - precision);
  std::string new_frac = num.substr(num.size() - precision);
  size_t nz = new_int.find_first_not_of('0');
  new_int = (nz == std::string::npos) ? "0" : new_int.substr(nz);
  size_t lnz = new_frac.find_last_not_of('0');
  new_frac = (lnz == std::string::npos) ? "" : new_frac.substr(0, lnz + 1);
  return new_frac.empty() ? new_int : new_int + "." + new_frac;
}

std::string NumberGenerator::RoundBoundToGrid(
    double value, int precision, bool is_lower, bool strict_in, bool* strict_out
) {
  // FormatFloat rounds to the nearest grid point; if that lands exactly on the
  // bound, keep the original strictness. Otherwise step to the grid point just
  // inside the range so no out-of-range value is admitted, and the boundary is
  // now strictly interior, so it becomes inclusive.
  std::string r = FormatFloat(value, precision);
  double rv = std::stod(r);
  if (rv == value) {
    *strict_out = strict_in;
    return r;
  }
  *strict_out = false;
  if (is_lower && rv < value) {
    // Rounded below a lower bound: move up to the smallest grid point >= value.
    r = AdjustGrid(r, precision, /*inc=*/true);
  } else if (!is_lower && rv > value) {
    // Rounded above an upper bound: move down to the largest grid point <= value.
    r = AdjustGrid(r, precision, /*inc=*/false);
  }
  return r;
}

// Helpers for GenerateFloatRangeRegex. Fraction patterns operate on the
// digit string after the decimal point, compared against a canonical bound
// fraction (canonical: produced by FormatFloat, so no trailing zeros).

// Matches 0 to max_count free digits.
std::string NumberGenerator::FreeDigits(int max_count) {
  if (max_count <= 0) {
    return "";
  }
  return "\\d{0," + std::to_string(max_count) + "}";
}

// Matches 0 to max_count zeros.
std::string NumberGenerator::OptionalZeros(int max_count) {
  if (max_count <= 0) {
    return "";
  }
  return "0{0," + std::to_string(max_count) + "}";
}

// Matches 1 to max_count zeros.
std::string NumberGenerator::SomeZeros(int max_count) {
  return "0{1," + std::to_string(max_count) + "}";
}

// Patterns for fraction strings t (1 <= |t| <= max_len) whose value 0.t is
// greater than 0.s (or equal when !strict). |s| <= max_len.
NumberGenerator::FracPatternSet NumberGenerator::FracGreaterPatterns(
    const std::string& s, bool strict, int max_len
) {
  FracPatternSet result;
  int n = static_cast<int>(s.size());
  // t agrees with s up to position i, then has a larger digit
  for (int i = 0; i < n; ++i) {
    if (s[i] < '9') {
      result.parts.push_back(
          s.substr(0, i) + DigitClass(s[i] + 1, '9') + FreeDigits(max_len - i - 1)
      );
    }
  }
  // t extends s with a nonzero digit (after optional zeros)
  for (int k = 0; n + k + 1 <= max_len; ++k) {
    result.parts.push_back(s + std::string(k, '0') + "[1-9]" + FreeDigits(max_len - n - k - 1));
  }
  if (!strict) {
    // t has the same value as s: s plus optional trailing zeros
    if (n > 0) {
      result.parts.push_back(s + OptionalZeros(max_len - n));
    } else {
      result.include_empty = true;
      if (max_len >= 1) {
        result.parts.push_back(SomeZeros(max_len));
      }
    }
  }
  return result;
}

// Patterns for fraction strings t (1 <= |t| <= max_len) whose value 0.t is
// less than 0.s (or equal when !strict). |s| <= max_len.
NumberGenerator::FracPatternSet NumberGenerator::FracLessPatterns(
    const std::string& s, bool strict, int max_len
) {
  FracPatternSet result;
  int n = static_cast<int>(s.size());
  // t agrees with s up to position i, then has a smaller digit
  for (int i = 0; i < n; ++i) {
    if (s[i] > '0') {
      result.parts.push_back(
          s.substr(0, i) + DigitClass('0', s[i] - 1) + FreeDigits(max_len - i - 1)
      );
    }
  }
  // t is a proper prefix of s plus optional trailing zeros: strictly smaller,
  // since the remaining digits of s contain a nonzero one
  for (int i = 0; i < n; ++i) {
    if (i == 0) {
      if (max_len >= 1) {
        result.parts.push_back(SomeZeros(max_len));
      }
    } else {
      result.parts.push_back(s.substr(0, i) + OptionalZeros(max_len - i));
    }
  }
  if (!strict) {
    // t has the same value as s
    if (n > 0) {
      result.parts.push_back(s + OptionalZeros(max_len - n));
    } else if (max_len >= 1) {
      result.parts.push_back(SomeZeros(max_len));
    }
  }
  result.include_empty = n > 0 || !strict;
  return result;
}

// Patterns for fraction strings t whose value 0.t lies between 0.a and 0.b.
// Requires value(0.a) < value(0.b) and b non-empty.
NumberGenerator::FracPatternSet NumberGenerator::FracBetweenPatterns(
    const std::string& a, bool strict_a, const std::string& b, bool strict_b, int max_len
) {
  FracPatternSet result;
  // Longest common prefix of b and zero-padded a. Always stops before |b|:
  // value(0.a) < value(0.b) implies b is not a prefix of padded a.
  int common_len = 0;
  while (common_len < static_cast<int>(b.size()) &&
         (common_len < static_cast<int>(a.size()) ? a[common_len] : '0') == b[common_len]) {
    ++common_len;
  }
  std::string common = b.substr(0, common_len);
  char digit_a = common_len < static_cast<int>(a.size()) ? a[common_len] : '0';
  char digit_b = b[common_len];

  // a digit strictly between the bounds' digits, then anything
  if (digit_b - digit_a >= 2) {
    result.parts.push_back(
        common + DigitClass(digit_a + 1, digit_b - 1) + FreeDigits(max_len - common_len - 1)
    );
  }
  // lower boundary: t continues with digit_a, the rest must exceed a's suffix
  if (common_len < static_cast<int>(a.size())) {
    FracPatternSet sub_lower =
        FracGreaterPatterns(a.substr(common_len + 1), strict_a, max_len - common_len - 1);
    for (auto& part : sub_lower.parts) {
      result.parts.push_back(common + digit_a + std::move(part));
    }
    if (sub_lower.include_empty) {
      result.parts.push_back(common + std::string(1, digit_a));
    }
  } else {
    // a's value equals value(0.common): only nonzero extensions of
    // common + digit_a ('0') are strictly greater
    FracPatternSet sub_lower = FracGreaterPatterns("", true, max_len - common_len - 1);
    for (auto& part : sub_lower.parts) {
      result.parts.push_back(common + digit_a + std::move(part));
    }
    if (!strict_a) {
      // t has the same value as a
      if (!a.empty()) {
        result.parts.push_back(a + OptionalZeros(max_len - static_cast<int>(a.size())));
      } else {
        result.include_empty = true;
        if (max_len >= 1) {
          result.parts.push_back(SomeZeros(max_len));
        }
      }
    }
  }
  // upper boundary: t continues with digit_b, the rest must stay below b's suffix
  FracPatternSet sub_upper =
      FracLessPatterns(b.substr(common_len + 1), strict_b, max_len - common_len - 1);
  for (auto& part : sub_upper.parts) {
    result.parts.push_back(common + digit_b + std::move(part));
  }
  if (sub_upper.include_empty) {
    result.parts.push_back(common + std::string(1, digit_b));
  }
  return result;
}

// Splits a canonical decimal string from FormatFloat ("12" or "12.34") into
// integer and fraction parts.
void NumberGenerator::SplitDecimal(
    const std::string& s, std::string* int_part, std::string* frac_part
) {
  size_t dot = s.find('.');
  if (dot == std::string::npos) {
    *int_part = s;
    frac_part->clear();
  } else {
    *int_part = s.substr(0, dot);
    *frac_part = s.substr(dot + 1);
  }
}

// Compares the values of two canonical non-negative decimals.
int NumberGenerator::CompareDecimal(
    const std::string& int_a,
    const std::string& frac_a,
    const std::string& int_b,
    const std::string& frac_b
) {
  if (int_a.size() != int_b.size()) {
    return int_a.size() < int_b.size() ? -1 : 1;
  }
  if (int_a != int_b) {
    return int_a < int_b ? -1 : 1;
  }
  size_t max_frac = std::max(frac_a.size(), frac_b.size());
  for (size_t i = 0; i < max_frac; ++i) {
    char da = i < frac_a.size() ? frac_a[i] : '0';
    char db = i < frac_b.size() ? frac_b[i] : '0';
    if (da != db) {
      return da < db ? -1 : 1;
    }
  }
  return 0;
}

// Strips the ^( )$ anchors added by IntegerRangeRegex, keeping the group.
std::string NumberGenerator::StripAnchors(const std::string& regex) {
  return regex.substr(1, regex.size() - 2);
}

int64_t NumberGenerator::ParseIntCapped(const std::string& digits) {
  // `digits` is a canonical non-negative integer string (no leading zeros).
  // Parse it exactly when it fits in int64; clamp to INT64_MAX otherwise (such
  // magnitudes are beyond practical float bounds and double integer precision).
  static const std::string kMaxInt64 = std::to_string(std::numeric_limits<int64_t>::max());
  if (digits.size() > kMaxInt64.size() ||
      (digits.size() == kMaxInt64.size() && digits > kMaxInt64)) {
    return std::numeric_limits<int64_t>::max();
  }
  return std::stoll(digits);
}

// Patterns for unsigned decimals (integer part plus optional fraction of up
// to `precision` digits) within the given bounds. `low` is required and
// non-negative; `high` is optional. Patterns for the value 0 are never
// produced: when low's value is 0 the bound is treated as strict, and the
// caller emits the zero pattern itself.
std::vector<std::string> NumberGenerator::PositiveRangeParts(
    const std::string& low,
    bool strict_low,
    const std::optional<std::string>& high,
    bool strict_high,
    int precision
) {
  std::vector<std::string> parts;
  std::string int_low, frac_low;
  SplitDecimal(low, &int_low, &frac_low);
  if (int_low == "0" && frac_low.empty()) {
    strict_low = true;
  }
  int64_t int_low_value = ParseIntCapped(int_low);
  std::string opt_any_frac = "(\\.\\d{1," + std::to_string(precision) + "})?";

  auto add_with_int_part = [&](const std::string& int_part, const FracPatternSet& set) {
    for (const auto& part : set.parts) {
      parts.push_back(int_part + "\\." + part);
    }
    if (set.include_empty) {
      parts.push_back(int_part);
    }
  };

  if (!high.has_value()) {
    add_with_int_part(int_low, FracGreaterPatterns(frac_low, strict_low, precision));
    // Guard the +1 against int64 overflow (int_low_value may be clamped to
    // INT64_MAX for very large bounds).
    if (int_low_value < std::numeric_limits<int64_t>::max()) {
      parts.push_back(
          StripAnchors(IntegerRangeRegex(int_low_value + 1, std::nullopt)) + opt_any_frac
      );
    }
    return parts;
  }

  std::string int_high, frac_high;
  SplitDecimal(*high, &int_high, &frac_high);
  int64_t int_high_value = ParseIntCapped(int_high);
  int cmp = CompareDecimal(int_low, frac_low, int_high, frac_high);
  if (cmp > 0 || (cmp == 0 && (strict_low || strict_high))) {
    return parts;
  }
  if (cmp == 0) {
    // single representable value, with optional redundant trailing zeros
    if (frac_low.empty()) {
      parts.push_back(int_low + "(\\." + SomeZeros(precision) + ")?");
    } else {
      parts.push_back(
          int_low + "\\." + frac_low + OptionalZeros(precision - static_cast<int>(frac_low.size()))
      );
    }
    return parts;
  }
  if (int_low == int_high) {
    add_with_int_part(
        int_low, FracBetweenPatterns(frac_low, strict_low, frac_high, strict_high, precision)
    );
  } else {
    add_with_int_part(int_low, FracGreaterPatterns(frac_low, strict_low, precision));
    if (int_high_value - int_low_value >= 2) {
      parts.push_back(
          StripAnchors(IntegerRangeRegex(int_low_value + 1, int_high_value - 1)) + opt_any_frac
      );
    }
    add_with_int_part(int_high, FracLessPatterns(frac_high, strict_high, precision));
  }
  return parts;
}

std::string NumberGenerator::FloatRangeRegex(
    std::optional<double> start,
    std::optional<double> end,
    int precision,
    bool exclusive_start,
    bool exclusive_end
) {
  if (start && end) {
    if (start.value() > end.value() ||
        (start.value() == end.value() && (exclusive_start || exclusive_end))) {
      return "^()$";
    }
  }

  if (!start && !end) {
    return "^-?\\d+(\\.\\d{1," + std::to_string(precision) + "})?$";
  }

  std::vector<std::string> parts;

  // Negative values: x is in [start, end] iff -x is in [-end, -start], so the
  // positive-range patterns are reused on the negated bounds and prefixed
  // with '-'.
  bool negatives_in_range = !start.has_value() || start.value() < 0;
  if (negatives_in_range) {
    std::string low = "0";
    bool strict_low = true;
    if (end.has_value() && end.value() < 0) {
      low =
          RoundBoundToGrid(-end.value(), precision, /*is_lower=*/true, exclusive_end, &strict_low);
    }
    std::optional<std::string> high;
    bool strict_high = false;
    if (start.has_value()) {
      high = RoundBoundToGrid(
          -start.value(), precision, /*is_lower=*/false, exclusive_start, &strict_high
      );
    }
    for (auto& part : PositiveRangeParts(low, strict_low, high, strict_high, precision)) {
      parts.push_back("-" + std::move(part));
    }
  }

  bool zero_allowed =
      (!start.has_value() || start.value() < 0 || (start.value() == 0 && !exclusive_start)) &&
      (!end.has_value() || end.value() > 0 || (end.value() == 0 && !exclusive_end));
  if (zero_allowed) {
    parts.push_back("0(\\." + SomeZeros(precision) + ")?");
    // Negative zero written with an all-zero fraction ("-0.0".."-0.000000") also
    // denotes 0. PositiveRangeParts never emits magnitude 0, so add these forms
    // explicitly when the range covers the negative side.
    if (negatives_in_range) {
      parts.push_back("-0(\\." + SomeZeros(precision) + ")");
    }
  }

  // Positive values
  if (!end.has_value() || end.value() > 0) {
    std::string low = "0";
    bool strict_low = true;
    if (start.has_value() && start.value() > 0) {
      low = RoundBoundToGrid(
          start.value(), precision, /*is_lower=*/true, exclusive_start, &strict_low
      );
    }
    std::optional<std::string> high;
    bool strict_high = false;
    if (end.has_value()) {
      high =
          RoundBoundToGrid(end.value(), precision, /*is_lower=*/false, exclusive_end, &strict_high);
    }
    for (auto& part : PositiveRangeParts(low, strict_low, high, strict_high, precision)) {
      parts.push_back(std::move(part));
    }
  }

  std::ostringstream result;
  result << "^(";
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i > 0) {
      result << "|";
    }
    result << parts[i];
  }
  result << ")$";

  return result.str();
}

std::string JSONSchemaConverter::GenerateRangeRegex(
    std::optional<int64_t> start, std::optional<int64_t> end
) {
  return NumberGenerator::IntegerRangeRegex(start, end);
}

std::string JSONSchemaConverter::GenerateFloatRangeRegex(
    std::optional<double> start,
    std::optional<double> end,
    int precision,
    bool exclusive_start,
    bool exclusive_end
) {
  return NumberGenerator::FloatRangeRegex(start, end, precision, exclusive_start, exclusive_end);
}

// ==================== Public API Functions ====================

std::optional<JSONFormat> JSONFormatFromString(const std::string& format) {
  static const std::unordered_map<std::string, JSONFormat> kNameToFormat = {
      {"json", JSONFormat::kJSON},
      {"qwen_xml", JSONFormat::kQwenXML},
      {"minimax_xml", JSONFormat::kMiniMaxXML},
      {"minimax_m3_xml", JSONFormat::kMiniMaxM3XML},
      {"deepseek_xml", JSONFormat::kDeepSeekXML},
      {"deepseek_v4_1_xml", JSONFormat::kDeepSeekV41XML},
      {"glm_xml", JSONFormat::kGlmXML},
      {"cohere_xml", JSONFormat::kCohereXML},
      {"kimi_k3_xml", JSONFormat::kKimiK3XML},
  };
  auto it = kNameToFormat.find(format);
  if (it == kNameToFormat.end()) {
    return std::nullopt;
  }
  return it->second;
}

Grammar JSONSchemaToGrammar(
    const std::string& schema,
    bool any_whitespace,
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool strict_mode,
    std::optional<int> max_whitespace_cnt,
    bool any_order,
    JSONFormat json_format,
    std::vector<std::string> excludes
) {
  picojson::value schema_value;
  std::string error = ParseJSON(schema_value, schema);
  XGRAMMAR_CHECK(error.empty()) << "Failed to parse JSON: " << error
                                << ". The JSON string is:" << schema;
  SchemaParser parser(schema_value, {strict_mode, json_format});
  auto spec_result = parser.Parse(schema_value, "root");
  if (spec_result.IsErr()) {
    XGRAMMAR_LOG(FATAL) << std::move(spec_result).UnwrapErr().what();
  }
  auto spec = std::move(spec_result).Unwrap();
  auto ref_resolver = [&parser](const std::string& uri, const std::string& rule_name_hint) {
    auto result = parser.ResolveRef(uri, rule_name_hint);
    if (result.IsErr()) {
      XGRAMMAR_LOG(FATAL) << std::move(result).UnwrapErr().what();
    }
    return std::move(result).Unwrap();
  };

  switch (json_format) {
    case JSONFormat::kJSON: {
      JSONSchemaConverter converter(
          indent,
          std::move(separators),
          any_whitespace,
          max_whitespace_cnt,
          std::move(ref_resolver),
          any_order,
          std::move(excludes)
      );
      return converter.Convert(spec);
    }
    case JSONFormat::kQwenXML:
    case JSONFormat::kMiniMaxXML:
    case JSONFormat::kDeepSeekXML:
    case JSONFormat::kDeepSeekV41XML:
    case JSONFormat::kGlmXML:
    case JSONFormat::kKimiK3XML: {
      XMLToolCallingConverter converter(
          indent,
          std::move(separators),
          any_whitespace,
          max_whitespace_cnt,
          std::move(ref_resolver),
          json_format,
          any_order,
          std::move(excludes)
      );
      return converter.Convert(spec);
    }
    case JSONFormat::kMiniMaxM3XML: {
      XGRAMMAR_CHECK(excludes.empty())
          << "JSONSchemaFormat.excludes is not supported for minimax_m3_xml";
      MiniMaxM3XMLToolCallingConverter converter(
          indent,
          std::move(separators),
          any_whitespace,
          max_whitespace_cnt,
          std::move(ref_resolver),
          any_order
      );
      return converter.Convert(spec);
    }
    case JSONFormat::kCohereXML: {
      XGRAMMAR_CHECK(excludes.empty())
          << "JSONSchemaFormat.excludes is not supported for cohere_xml";
      CohereXMLToolCallingConverter converter(
          indent,
          std::move(separators),
          any_whitespace,
          max_whitespace_cnt,
          std::move(ref_resolver),
          any_order
      );
      return converter.Convert(spec);
    }
    default:
      XGRAMMAR_LOG(FATAL) << "Invalid JSON format: " << static_cast<int>(json_format);
  }
  XGRAMMAR_UNREACHABLE();
}

std::string JSONSchemaToEBNF(
    const std::string& schema,
    bool any_whitespace,
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool strict_mode,
    std::optional<int> max_whitespace_cnt,
    JSONFormat json_format,
    bool any_order
) {
  picojson::value schema_value;
  std::string err = ParseJSON(schema_value, schema);
  XGRAMMAR_CHECK(err.empty()) << "Failed to parse JSON: " << err
                              << ". The JSON string is:" << schema;
  return JSONSchemaToEBNF(
      schema_value,
      any_whitespace,
      indent,
      separators,
      strict_mode,
      max_whitespace_cnt,
      json_format,
      any_order
  );
}

std::string JSONSchemaToEBNF(
    const picojson::value& schema,
    bool any_whitespace,
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool strict_mode,
    std::optional<int> max_whitespace_cnt,
    JSONFormat json_format,
    bool any_order
) {
  // Parse JSON Schema to SchemaSpec
  SchemaParser parser(schema, {strict_mode, json_format});
  auto spec_result = parser.Parse(schema, "root");
  if (spec_result.IsErr()) {
    XGRAMMAR_LOG(FATAL) << std::move(spec_result).UnwrapErr().what();
  }
  auto spec = std::move(spec_result).Unwrap();

  auto ref_resolver = [&parser](const std::string& uri, const std::string& rule_name_hint) {
    auto r = parser.ResolveRef(uri, rule_name_hint);
    if (r.IsErr()) {
      XGRAMMAR_LOG(FATAL) << std::move(r).UnwrapErr().what();
    }
    return std::move(r).Unwrap();
  };

  // Create converter based on format
  switch (json_format) {
    case JSONFormat::kJSON: {
      JSONSchemaConverter converter(
          indent, separators, any_whitespace, max_whitespace_cnt, ref_resolver, any_order
      );
      return GrammarNormalizer::Apply(converter.Convert(spec)).ToString();
    }
    case JSONFormat::kQwenXML:
    case JSONFormat::kMiniMaxXML:
    case JSONFormat::kDeepSeekXML:
    case JSONFormat::kDeepSeekV41XML:
    case JSONFormat::kGlmXML:
    case JSONFormat::kKimiK3XML: {
      XMLToolCallingConverter converter(
          indent,
          separators,
          any_whitespace,
          max_whitespace_cnt,
          ref_resolver,
          json_format,
          any_order
      );
      return GrammarNormalizer::Apply(converter.Convert(spec)).ToString();
    }
    case JSONFormat::kMiniMaxM3XML: {
      MiniMaxM3XMLToolCallingConverter converter(
          indent, separators, any_whitespace, max_whitespace_cnt, ref_resolver, any_order
      );
      return GrammarNormalizer::Apply(converter.Convert(spec)).ToString();
    }
    case JSONFormat::kCohereXML: {
      CohereXMLToolCallingConverter converter(
          indent, separators, any_whitespace, max_whitespace_cnt, ref_resolver, any_order
      );
      return GrammarNormalizer::Apply(converter.Convert(spec)).ToString();
    }
    default:
      XGRAMMAR_LOG(FATAL) << "Invalid JSON format: " << static_cast<int>(json_format);
  }
  XGRAMMAR_UNREACHABLE();
}

// Wrapper functions for testing
std::string GenerateRangeRegex(std::optional<int64_t> start, std::optional<int64_t> end) {
  return JSONSchemaConverter::GenerateRangeRegex(start, end);
}

std::string GenerateFloatRangeRegex(
    std::optional<double> start, std::optional<double> end, bool exclusive_start, bool exclusive_end
) {
  return JSONSchemaConverter::GenerateFloatRangeRegex(
      start, end, 6, exclusive_start, exclusive_end
  );
}

}  // namespace xgrammar
