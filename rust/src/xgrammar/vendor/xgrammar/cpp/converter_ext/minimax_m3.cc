/*!
 *  Copyright (c) 2026 by Contributors
 * \file xgrammar/converter_ext/minimax_m3.cc
 * \brief Implementation of the MiniMax M3 XML Tool Calling converter.
 */
#include <picojson.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <unordered_set>
#include <vector>

#include "../json_schema_converter_ext.h"
#include "../support/encoding.h"
#include "../support/logging.h"

namespace xgrammar {

namespace {

constexpr const char* kStringCacheKey = "{\"type\":\"string\"}";
constexpr const char* kMiniMaxM3Namespace = "]<]minimax[>[";
constexpr const char* kMiniMaxM3ArrayItemName = "item";

bool IsASCIIWhitespace(uint8_t byte) {
  return byte == ' ' || byte == '\t' || byte == '\n' || byte == '\r' || byte == '\f' ||
         byte == '\v';
}

bool IsCanonicalUTF8(const std::string& text) {
  for (size_t offset = 0; offset < text.size();) {
    auto [codepoint, num_bytes] = ParseNextUTF8(text.data() + offset);
    if (codepoint == CharHandlingError::kInvalidUTF8 || num_bytes <= 0 ||
        offset + num_bytes > text.size() || (codepoint >= 0xd800 && codepoint <= 0xdfff) ||
        codepoint > 0x10ffff || text.compare(offset, num_bytes, CharToUTF8(codepoint)) != 0) {
      return false;
    }
    offset += num_bytes;
  }
  return true;
}

}  // namespace

MiniMaxM3XMLToolCallingConverter::MiniMaxM3XMLToolCallingConverter(
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool any_whitespace,
    std::optional<int> max_whitespace_cnt,
    RefResolver ref_resolver,
    bool any_order
)
    : JSONSchemaConverter(
          indent, separators, any_whitespace, max_whitespace_cnt, ref_resolver, any_order
      ) {}

void MiniMaxM3XMLToolCallingConverter::AddBasicRules() {
  for (const auto& name : {kBasicInteger, kBasicNumber, kBasicString, kBasicBoolean, kBasicNull}) {
    builder_.AddEmptyRule(name);
  }

  builder_.UpdateRuleBody(
      kBasicInteger, JSONSchemaConverter::GenerateInteger(IntegerSpec{}, kBasicInteger)
  );
  AddCache("{\"type\":\"integer\"}", builder_.GetRuleId(kBasicInteger));

  builder_.UpdateRuleBody(
      kBasicNumber, JSONSchemaConverter::GenerateNumber(NumberSpec{}, kBasicNumber)
  );
  AddCache("{\"type\":\"number\"}", builder_.GetRuleId(kBasicNumber));

  builder_.UpdateRuleBody(kBasicString, TagDispatch(false, {kMiniMaxM3Namespace}));
  AddCache(kStringCacheKey, builder_.GetRuleId(kBasicString));

  builder_.UpdateRuleBody(
      kBasicBoolean, JSONSchemaConverter::GenerateBoolean(BooleanSpec{}, kBasicBoolean)
  );
  AddCache("{\"type\":\"boolean\"}", builder_.GetRuleId(kBasicBoolean));

  builder_.UpdateRuleBody(kBasicNull, JSONSchemaConverter::GenerateNull(NullSpec{}, kBasicNull));
  AddCache("{\"type\":\"null\"}", builder_.GetRuleId(kBasicNull));
}

int32_t MiniMaxM3XMLToolCallingConverter::GenerateString(
    const StringSpec& spec, const std::string& rule_name
) {
  const bool has_known_format =
      spec.format.has_value() && JSONFormatToRegexPattern(*spec.format).has_value();
  XGRAMMAR_CHECK(
      !spec.pattern.has_value() && !has_known_format && spec.min_length == 0 &&
      spec.max_length == -1
  ) << "String pattern, recognized format, and length constraints are not supported by "
       "minimax_m3_xml because they cannot be combined with the namespace-marker exclusion";
  return RuleRef(kBasicString);
}

int32_t MiniMaxM3XMLToolCallingConverter::GenerateArray(
    const ArraySpec& spec, const std::string& rule_name
) {
  constexpr int64_t kMaxRepeatCount = std::numeric_limits<int32_t>::max();
  XGRAMMAR_CHECK(
      spec.min_items <= kMaxRepeatCount &&
      (spec.max_items == -1 || spec.max_items <= kMaxRepeatCount) &&
      spec.prefix_items.size() <= static_cast<size_t>(kMaxRepeatCount)
  ) << "minimax_m3_xml array bounds exceed the supported range";
  XGRAMMAR_CHECK(!spec.allow_additional_items || spec.additional_items != nullptr)
      << "minimax_m3_xml requires a fixed schema for array items";

  std::vector<int32_t> prefix_items;
  prefix_items.reserve(spec.prefix_items.size());
  for (size_t index = 0; index < spec.prefix_items.size(); ++index) {
    int32_t item_rule_id =
        CreateRule(spec.prefix_items[index], rule_name + "_item_" + std::to_string(index));
    prefix_items.push_back(FormatElement(kMiniMaxM3ArrayItemName, item_rule_id));
  }

  std::optional<int32_t> additional_item;
  if (spec.allow_additional_items) {
    int32_t item_rule_id = CreateRule(spec.additional_items, rule_name + "_additional");
    additional_item = FormatElement(kMiniMaxM3ArrayItemName, item_rule_id);
  }

  int32_t empty = Empty();
  int32_t whitespace = WhitespaceExpression();
  if (prefix_items.empty()) {
    if (!additional_item.has_value() || spec.max_items == 0) {
      return empty;
    }
    int32_t min_items = static_cast<int32_t>(spec.min_items);
    int32_t max_items = spec.max_items == -1 ? -1 : static_cast<int32_t>(spec.max_items);
    int32_t nonempty = Sequence(
        {whitespace,
         *additional_item,
         Repeat(
             rule_name + "_items",
             Sequence({whitespace, *additional_item}),
             std::max(0, min_items - 1),
             max_items == -1 ? -1 : std::max(0, max_items - 1)
         ),
         whitespace}
    );
    return min_items == 0 ? Choice({nonempty, empty}) : nonempty;
  }

  int32_t prefix_count = static_cast<int32_t>(prefix_items.size());
  int32_t tail = empty;
  if (additional_item.has_value()) {
    int32_t min_additional = std::max(0, static_cast<int32_t>(spec.min_items) - prefix_count);
    int32_t max_additional = spec.max_items == -1
                                 ? -1
                                 : std::max(0, static_cast<int32_t>(spec.max_items) - prefix_count);
    tail = Repeat(
        rule_name + "_additional_items",
        Sequence({whitespace, *additional_item}),
        min_additional,
        max_additional
    );
  }

  // A prefixItems entry constrains its position but does not make that position mandatory. Build
  // a linear chain whose suffix can stop once minItems is satisfied.
  for (int32_t index = prefix_count - 2; index >= 0; --index) {
    int32_t emitted_count = index + 1;
    bool can_stop = emitted_count >= spec.min_items;
    bool can_continue = spec.max_items == -1 || emitted_count < spec.max_items;
    int32_t body = empty;
    if (can_continue) {
      int32_t continuation = Sequence({whitespace, prefix_items[index + 1], tail});
      body = can_stop ? Choice({continuation, empty}) : continuation;
    }
    int32_t tail_rule_id =
        builder_.AddRuleWithHint(rule_name + "_prefix_tail_" + std::to_string(index), body);
    tail = RuleRef(tail_rule_id);
  }

  if (spec.max_items == 0) {
    return empty;
  }
  int32_t nonempty = Sequence({whitespace, prefix_items[0], tail, whitespace});
  return spec.min_items == 0 ? Choice({nonempty, empty}) : nonempty;
}

void MiniMaxM3XMLToolCallingConverter::ValidateObject(const ObjectSpec& spec) const {
  XGRAMMAR_CHECK(
      !spec.allow_additional_properties && spec.additional_properties_schema == nullptr &&
      !spec.allow_unevaluated_properties && spec.unevaluated_properties_schema == nullptr &&
      spec.pattern_properties.empty() && spec.property_names == nullptr
  ) << "minimax_m3_xml requires fixed object property names; additionalProperties, "
       "unevaluatedProperties, patternProperties, and propertyNames are not supported";

  std::unordered_set<std::string> property_names;
  for (const auto& property : spec.properties) {
    XGRAMMAR_CHECK(property.schema != nullptr)
        << "minimax_m3_xml property must have a fixed schema: " << property.name;
    ValidateElementName(property.name);
    property_names.insert(property.name);
  }
  for (const auto& required : spec.required) {
    XGRAMMAR_CHECK(property_names.count(required) != 0)
        << "minimax_m3_xml required property has no fixed schema: " << required;
  }
}

int32_t MiniMaxM3XMLToolCallingConverter::GenerateObject(
    const ObjectSpec& spec, const std::string& rule_name, bool dummy_need_braces
) {
  ValidateObject(spec);
  bool saved_any_whitespace = any_whitespace_;
  any_whitespace_ = false;
  int32_t result = JSONSchemaConverter::GenerateObject(spec, rule_name, /*need_braces=*/false);
  any_whitespace_ = saved_any_whitespace;
  return result;
}

int32_t MiniMaxM3XMLToolCallingConverter::GenerateAny(
    const AnySpec& spec, const std::string& rule_name
) {
  XGRAMMAR_LOG(FATAL) << "minimax_m3_xml does not support unconstrained schemas";
  XGRAMMAR_UNREACHABLE();
}

int32_t MiniMaxM3XMLToolCallingConverter::GenerateLiteral(const picojson::value& value) {
  if (value.is<std::string>()) {
    const std::string& text = value.get<std::string>();
    XGRAMMAR_CHECK(text.find(kMiniMaxM3Namespace) == std::string::npos)
        << "A minimax_m3_xml string literal cannot contain the namespace marker";
    return ByteString(text);
  }
  if (value.is<picojson::object>()) {
    const auto& object = value.get<picojson::object>();
    std::vector<int32_t> properties;
    properties.reserve(object.size());
    for (const auto& key : object.ordered_keys()) {
      int32_t value_expr = GenerateLiteral(object.at(key));
      int32_t value_rule_id = builder_.AddRuleWithHint("literal_" + key, value_expr);
      properties.push_back(FormatElement(key, value_rule_id));
    }
    return Sequence(properties);
  }
  if (value.is<picojson::array>()) {
    const auto& array = value.get<picojson::array>();
    std::vector<int32_t> items;
    items.reserve(array.size());
    for (size_t index = 0; index < array.size(); ++index) {
      int32_t value_expr = GenerateLiteral(array[index]);
      int32_t value_rule_id =
          builder_.AddRuleWithHint("literal_item_" + std::to_string(index), value_expr);
      items.push_back(FormatElement(kMiniMaxM3ArrayItemName, value_rule_id));
    }
    return Sequence(items);
  }
  return ByteString(value.serialize());
}

int32_t MiniMaxM3XMLToolCallingConverter::GenerateConst(
    const ConstSpec& spec, const std::string& rule_name
) {
  picojson::value value;
  std::string error = picojson::parse(value, spec.json_value);
  XGRAMMAR_CHECK(error.empty()) << "Invalid const JSON value: " << error;
  return GenerateLiteral(value);
}

int32_t MiniMaxM3XMLToolCallingConverter::GenerateEnum(
    const EnumSpec& spec, const std::string& rule_name
) {
  XGRAMMAR_DCHECK(!spec.json_values.empty())
      << "GenerateEnum called with empty enum spec for rule: " << rule_name;
  std::vector<int32_t> values;
  values.reserve(spec.json_values.size());
  for (const auto& json_value : spec.json_values) {
    picojson::value value;
    std::string error = picojson::parse(value, json_value);
    XGRAMMAR_CHECK(error.empty()) << "Invalid enum JSON value: " << error;
    values.push_back(GenerateLiteral(value));
  }
  return Choice(values);
}

void MiniMaxM3XMLToolCallingConverter::ValidateElementName(const std::string& name) {
  XGRAMMAR_CHECK(!name.empty() && name.front() != '/' && name.find('>') == std::string::npos)
      << "Invalid minimax_m3_xml element name: " << name;
  XGRAMMAR_CHECK(IsCanonicalUTF8(name)) << "minimax_m3_xml element names must be valid UTF-8";
  XGRAMMAR_CHECK(std::any_of(name.begin(), name.end(), [](unsigned char byte) {
    return !IsASCIIWhitespace(byte);
  })) << "minimax_m3_xml element names cannot be blank";
}

int32_t MiniMaxM3XMLToolCallingConverter::FormatElement(
    const std::string& name, int32_t value_rule_id
) {
  ValidateElementName(name);
  return Sequence(
      {ByteString(std::string(kMiniMaxM3Namespace) + "<" + name + ">"),
       RuleRef(value_rule_id),
       ByteString(std::string(kMiniMaxM3Namespace) + "</" + name + ">")}
  );
}

int32_t MiniMaxM3XMLToolCallingConverter::FormatProperty(
    const std::string& key,
    int32_t value_rule_id,
    const std::string& rule_name,
    int64_t idx,
    const SchemaSpecPtr& schema
) {
  return FormatElement(key, value_rule_id);
}

std::string MiniMaxM3XMLToolCallingConverter::NextSeparator(bool is_end) {
  return GetWhitespacePattern();
}

}  // namespace xgrammar
