/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/structural_tag.cc
 */
#include "structural_tag.h"

#include <picojson.h>
#include <xgrammar/exception.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "grammar_builder.h"
#include "grammar_functor.h"
#include "grammar_impl.h"
#include "json_schema_converter.h"
#include "support/json_parse.h"
#include "support/logging.h"
#include "support/recursion_guard.h"
#include "support/utils.h"
#include "tokenizer_info_impl.h"
#include "xgrammar/grammar.h"

namespace xgrammar {

// Short alias for the error type.
using ISTError = InvalidStructuralTagError;

// Forward declaration for helpers that convert Format to picojson::value.
picojson::value FormatToJSONValue(const Format& format);

picojson::value StringVectorToJSONArray(const std::vector<std::string>& vector) {
  picojson::array array;
  array.reserve(vector.size());
  for (const auto& string : vector) {
    array.push_back(picojson::value(string));
  }
  return picojson::value(std::move(array));
}

picojson::value FormatVectorToJSONArray(const std::vector<xgrammar::Format>& vector) {
  picojson::array array;
  array.reserve(vector.size());
  for (const auto& format : vector) {
    array.push_back(xgrammar::FormatToJSONValue(format));
  }
  return picojson::value(std::move(array));
}

picojson::value TagVectorToJSONArray(const std::vector<xgrammar::TagFormat>& vector) {
  picojson::array array;
  array.reserve(vector.size());
  for (const auto& tag : vector) {
    array.push_back(tag.ToJSON());
  }
  return picojson::value(std::move(array));
}

picojson::value IntOrStringVectorToJSONArray(
    const std::vector<std::variant<int32_t, std::string>>& vec
) {
  picojson::array array;
  array.reserve(vec.size());
  for (const auto& item : vec) {
    if (std::holds_alternative<int32_t>(item)) {
      array.push_back(picojson::value(static_cast<double>(std::get<int32_t>(item))));
    } else {
      array.push_back(picojson::value(std::get<std::string>(item)));
    }
  }
  return picojson::value(std::move(array));
}

/******************** Format To JSON ********************/
std::string FormatToJSON(const Format& format) { return FormatToJSONValue(format).serialize(); }

picojson::value FormatToJSONValue(const Format& format) {
  return std::visit([&](auto&& arg) { return arg.ToJSON(); }, format);
}

picojson::value ConstStringFormat::ToJSON() const {
  picojson::object obj;
  obj["type"] = picojson::value(type);
  obj["value"] = picojson::value(value);
  return picojson::value(std::move(obj));
}

picojson::value JSONSchemaFormat::ToJSON() const {
  picojson::object obj;
  obj["type"] = picojson::value(type);
  picojson::value schema_val;
  if (ParseJSON(schema_val, json_schema).empty()) {
    obj["json_schema"] = schema_val;
  } else {
    obj["json_schema"] = picojson::value(json_schema);
  }
  obj["style"] = picojson::value(style);
  obj["any_order"] = picojson::value(any_order);
  if (!excludes.empty()) {
    obj["excludes"] = StringVectorToJSONArray(excludes);
  }
  if (max_whitespace_cnt.has_value()) {
    obj["max_whitespace_cnt"] = picojson::value(static_cast<int64_t>(*max_whitespace_cnt));
  } else {
    obj["max_whitespace_cnt"] = picojson::value();
  }
  return picojson::value(std::move(obj));
}

picojson::value GrammarFormat::ToJSON() const {
  picojson::object obj;
  obj["type"] = picojson::value(type);
  obj["grammar"] = picojson::value(grammar);
  return picojson::value(std::move(obj));
}

picojson::value RegexFormat::ToJSON() const {
  picojson::object obj;
  obj["type"] = picojson::value(type);
  obj["pattern"] = picojson::value(pattern);
  return picojson::value(std::move(obj));
}

picojson::value AnyTextFormat::ToJSON() const {
  picojson::object obj;
  obj["type"] = picojson::value(type);
  obj["excludes"] = StringVectorToJSONArray(excludes);
  obj["detected_end_strs"] = StringVectorToJSONArray(detected_end_strs_);
  if (max_tokens >= 0) {
    obj["max_tokens"] = picojson::value(static_cast<int64_t>(max_tokens));
  }
  if (max_chars >= 0) {
    obj["max_chars"] = picojson::value(static_cast<int64_t>(max_chars));
  }
  return picojson::value(std::move(obj));
}

// These two constructors are defined here rather than inline because instantiating
// vector<Format> against the still-incomplete Format variant is ill-formed under C++20.
SequenceFormat::SequenceFormat(std::vector<Format> elements) : elements(std::move(elements)) {}

OrFormat::OrFormat(std::vector<Format> elements) : elements(std::move(elements)) {}

picojson::value SequenceFormat::ToJSON() const {
  picojson::object obj;
  obj["type"] = picojson::value(type);
  obj["elements"] = FormatVectorToJSONArray(elements);
  return picojson::value(std::move(obj));
}

picojson::value OrFormat::ToJSON() const {
  picojson::object obj;
  obj["type"] = picojson::value(type);
  obj["elements"] = FormatVectorToJSONArray(elements);
  return picojson::value(std::move(obj));
}

picojson::value TagFormat::ToJSON() const {
  picojson::object obj;
  obj["type"] = picojson::value(type);
  if (std::holds_alternative<std::string>(begin)) {
    obj["begin"] = picojson::value(std::get<std::string>(begin));
  } else {
    obj["begin"] = std::get<TokenFormat>(begin).ToJSON();
  }
  if (content) {
    obj["content"] = FormatToJSONValue(*content);
  } else {
    obj["content"] = picojson::value();
  }
  if (std::holds_alternative<TokenFormat>(end)) {
    obj["end"] = std::get<TokenFormat>(end).ToJSON();
  } else {
    const auto& end_strs = std::get<std::vector<std::string>>(end);
    if (end_strs.size() == 1) {
      obj["end"] = picojson::value(end_strs[0]);
    } else {
      obj["end"] = StringVectorToJSONArray(end_strs);
    }
  }
  return picojson::value(std::move(obj));
}

picojson::value TokenFormat::ToJSON() const {
  picojson::object obj;
  obj["type"] = picojson::value(type);
  if (std::holds_alternative<int32_t>(token)) {
    obj["token"] = picojson::value(static_cast<double>(std::get<int32_t>(token)));
  } else {
    obj["token"] = picojson::value(std::get<std::string>(token));
  }
  return picojson::value(std::move(obj));
}

picojson::value ExcludeTokenFormat::ToJSON() const {
  picojson::object obj;
  obj["type"] = picojson::value(type);
  obj["exclude_tokens"] = IntOrStringVectorToJSONArray(exclude_tokens);
  return picojson::value(std::move(obj));
}

picojson::value AnyTokensFormat::ToJSON() const {
  picojson::object obj;
  obj["type"] = picojson::value(type);
  obj["exclude_tokens"] = IntOrStringVectorToJSONArray(exclude_tokens);
  if (max_tokens >= 0) {
    obj["max_tokens"] = picojson::value(static_cast<int64_t>(max_tokens));
  }
  return picojson::value(std::move(obj));
}

picojson::value TokenTriggeredTagsFormat::ToJSON() const {
  picojson::object obj;
  obj["type"] = picojson::value(type);
  obj["trigger_tokens"] = IntOrStringVectorToJSONArray(trigger_tokens);
  obj["tags"] = TagVectorToJSONArray(tags);
  obj["exclude_tokens"] = IntOrStringVectorToJSONArray(exclude_tokens);
  obj["at_least_one"] = picojson::value(at_least_one);
  obj["stop_after_first"] = picojson::value(stop_after_first);
  return picojson::value(std::move(obj));
}

picojson::value TriggeredTagsFormat::ToJSON() const {
  picojson::object obj;
  obj["type"] = picojson::value(type);
  obj["triggers"] = StringVectorToJSONArray(triggers);
  obj["tags"] = TagVectorToJSONArray(tags);
  obj["excludes"] = StringVectorToJSONArray(excludes);
  obj["at_least_one"] = picojson::value(at_least_one);
  obj["stop_after_first"] = picojson::value(stop_after_first);
  obj["detected_end_strs"] = StringVectorToJSONArray(detected_end_strs_);
  return picojson::value(std::move(obj));
}

picojson::value TagsWithSeparatorFormat::ToJSON() const {
  picojson::object obj;
  obj["type"] = picojson::value(type);
  obj["tags"] = TagVectorToJSONArray(tags);
  obj["separator"] = picojson::value(separator);
  obj["at_least_one"] = picojson::value(at_least_one);
  obj["stop_after_first"] = picojson::value(stop_after_first);
  return picojson::value(std::move(obj));
}

picojson::value OptionalFormat::ToJSON() const {
  picojson::object obj;
  obj["type"] = picojson::value(type);
  obj["content"] = FormatToJSONValue(*content);
  return picojson::value(std::move(obj));
}

picojson::value PlusFormat::ToJSON() const {
  picojson::object obj;
  obj["type"] = picojson::value(type);
  obj["content"] = FormatToJSONValue(*content);
  return picojson::value(std::move(obj));
}

picojson::value StarFormat::ToJSON() const {
  picojson::object obj;
  obj["type"] = picojson::value(type);
  obj["content"] = FormatToJSONValue(*content);
  return picojson::value(std::move(obj));
}

picojson::value RepeatFormat::ToJSON() const {
  picojson::object obj;
  obj["type"] = picojson::value(type);
  obj["min"] = picojson::value(static_cast<int64_t>(min));
  obj["max"] = picojson::value(static_cast<int64_t>(max));
  obj["content"] = FormatToJSONValue(*content);
  return picojson::value(std::move(obj));
}

picojson::value DispatchFormat::ToJSON() const {
  picojson::object obj;
  obj["type"] = picojson::value(type);
  picojson::array rules_arr;
  rules_arr.reserve(rules.size());
  for (const auto& pair : rules) {
    picojson::array pair_arr;
    pair_arr.push_back(picojson::value(pair.first));
    if (pair.second) {
      pair_arr.push_back(FormatToJSONValue(*pair.second));
    } else {
      pair_arr.push_back(picojson::value());
    }
    rules_arr.push_back(picojson::value(std::move(pair_arr)));
  }
  obj["rules"] = picojson::value(std::move(rules_arr));
  obj["loop"] = picojson::value(loop);
  obj["excludes"] = StringVectorToJSONArray(excludes);
  return picojson::value(std::move(obj));
}

picojson::value TokenDispatchFormat::ToJSON() const {
  picojson::object obj;
  obj["type"] = picojson::value(type);
  picojson::array rules_arr;
  rules_arr.reserve(rules.size());
  for (const auto& pair : rules) {
    picojson::array pair_arr;
    if (std::holds_alternative<int32_t>(pair.first)) {
      pair_arr.push_back(picojson::value(static_cast<double>(std::get<int32_t>(pair.first))));
    } else {
      pair_arr.push_back(picojson::value(std::get<std::string>(pair.first)));
    }
    if (pair.second) {
      pair_arr.push_back(FormatToJSONValue(*pair.second));
    } else {
      pair_arr.push_back(picojson::value());
    }
    rules_arr.push_back(picojson::value(std::move(pair_arr)));
  }
  obj["rules"] = picojson::value(std::move(rules_arr));
  obj["loop"] = picojson::value(loop);
  obj["exclude_tokens"] = IntOrStringVectorToJSONArray(exclude_tokens);
  return picojson::value(std::move(obj));
}

/************** StructuralTag Parser **************/

class StructuralTagParser {
 public:
  static Result<StructuralTag, StructuralTagError> FromJSON(const std::string& json);

 private:
  Result<StructuralTag, ISTError> ParseStructuralTag(const picojson::value& value);

  /*!
   * \brief Parse a Format object from a JSON value.
   * \param value The JSON value to parse.
   * \return A Format object if the JSON is valid, otherwise an error message in std::runtime_error.
   * \note The "type" field is checked in this function, and not checked in the Parse*Format
   * functions.
   */
  Result<Format, ISTError> ParseFormat(const picojson::value& value);
  Result<ConstStringFormat, ISTError> ParseConstStringFormat(const picojson::object& value);
  Result<JSONSchemaFormat, ISTError> ParseJSONSchemaFormat(
      const picojson::object& value, std::optional<std::string> style_override = std::nullopt
  );
  Result<AnyTextFormat, ISTError> ParseAnyTextFormat(const picojson::object& value);
  Result<GrammarFormat, ISTError> ParseGrammarFormat(const picojson::object& value);
  Result<RegexFormat, ISTError> ParseRegexFormat(const picojson::object& value);
  Result<SequenceFormat, ISTError> ParseSequenceFormat(const picojson::object& value);
  Result<OrFormat, ISTError> ParseOrFormat(const picojson::object& value);
  /*! \brief ParseTagFormat with extra check for object and the type field. */
  Result<TagFormat, ISTError> ParseTagFormat(const picojson::value& value);
  Result<TagFormat, ISTError> ParseTagFormat(const picojson::object& value);
  Result<TriggeredTagsFormat, ISTError> ParseTriggeredTagsFormat(const picojson::object& value);
  Result<TagsWithSeparatorFormat, ISTError> ParseTagsWithSeparatorFormat(
      const picojson::object& value
  );
  Result<OptionalFormat, ISTError> ParseOptionalFormat(const picojson::object& value);
  Result<PlusFormat, ISTError> ParsePlusFormat(const picojson::object& value);
  Result<StarFormat, ISTError> ParseStarFormat(const picojson::object& value);
  Result<RepeatFormat, ISTError> ParseRepeatFormat(const picojson::object& value);
  Result<TokenFormat, ISTError> ParseTokenFormat(const picojson::object& value);
  Result<ExcludeTokenFormat, ISTError> ParseExcludeTokenFormat(const picojson::object& value);
  Result<AnyTokensFormat, ISTError> ParseAnyTokensFormat(const picojson::object& value);
  Result<TokenTriggeredTagsFormat, ISTError> ParseTokenTriggeredTagsFormat(
      const picojson::object& value
  );
  Result<DispatchFormat, ISTError> ParseDispatchFormat(const picojson::object& value);
  Result<TokenDispatchFormat, ISTError> ParseTokenDispatchFormat(const picojson::object& value);

  int parse_format_recursion_depth_ = 0;
};

Result<StructuralTag, StructuralTagError> StructuralTagParser::FromJSON(const std::string& json) {
  picojson::value value;
  std::string err = ParseJSON(value, json);
  if (!err.empty()) {
    return ResultErr<InvalidJSONError>("Failed to parse JSON: " + err);
  }
  return Result<StructuralTag, StructuralTagError>::Convert(
      StructuralTagParser().ParseStructuralTag(value)
  );
}

Result<StructuralTag, ISTError> StructuralTagParser::ParseStructuralTag(const picojson::value& value
) {
  if (!value.is<picojson::object>()) {
    return ResultErr<ISTError>("Structural tag must be an object");
  }
  const auto& obj = value.get<picojson::object>();
  // The type field is optional but must be "structural_tag" if present.
  if (obj.find("type") != obj.end()) {
    if (!obj["type"].is<std::string>() || obj["type"].get<std::string>() != "structural_tag") {
      return ResultErr<ISTError>("Structural tag's type must be a string \"structural_tag\"");
    }
  }
  // The format field is required.
  if (obj.find("format") == obj.end()) {
    return ResultErr<ISTError>("Structural tag must have a format field");
  }
  auto format = ParseFormat(obj["format"]);
  if (format.IsErr()) {
    return ResultErr<ISTError>(std::move(format).UnwrapErr());
  }
  return ResultOk<StructuralTag>(std::move(format).Unwrap());
}

Result<Format, ISTError> StructuralTagParser::ParseFormat(const picojson::value& value) {
  RecursionGuard guard(&parse_format_recursion_depth_);
  // The global recursion limit is far deeper than the native stack allows for the recursive passes
  // over the format tree (a few KB per level, and Windows has a 1 MB main-thread stack), so cap
  // the nesting of formats explicitly.
  static constexpr int kMaxFormatDepth = 100;
  if (parse_format_recursion_depth_ > kMaxFormatDepth) {
    return ResultErr<ISTError>(
        "Formats are nested deeper than " + std::to_string(kMaxFormatDepth) + " levels"
    );
  }
  if (!value.is<picojson::object>()) {
    return ResultErr<ISTError>("Format must be an object");
  }
  const auto& obj = value.get<picojson::object>();
  // If type is present, use it to determine the format.
  if (obj.find("type") != obj.end()) {
    if (!obj["type"].is<std::string>()) {
      return ResultErr<ISTError>("Format's type must be a string");
    }
    auto type = obj["type"].get<std::string>();
    if (type == "const_string") {
      return Result<Format, ISTError>::Convert(ParseConstStringFormat(obj));
    } else if (type == "json_schema") {
      return Result<Format, ISTError>::Convert(ParseJSONSchemaFormat(obj));
    } else if (type == "any_text") {
      return Result<Format, ISTError>::Convert(ParseAnyTextFormat(obj));
    } else if (type == "sequence") {
      return Result<Format, ISTError>::Convert(ParseSequenceFormat(obj));
    } else if (type == "or") {
      return Result<Format, ISTError>::Convert(ParseOrFormat(obj));
    } else if (type == "tag") {
      return Result<Format, ISTError>::Convert(ParseTagFormat(obj));
    } else if (type == "triggered_tags") {
      return Result<Format, ISTError>::Convert(ParseTriggeredTagsFormat(obj));
    } else if (type == "tags_with_separator") {
      return Result<Format, ISTError>::Convert(ParseTagsWithSeparatorFormat(obj));
    } else if (type == "optional") {
      return Result<Format, ISTError>::Convert(ParseOptionalFormat(obj));
    } else if (type == "plus") {
      return Result<Format, ISTError>::Convert(ParsePlusFormat(obj));
    } else if (type == "star") {
      return Result<Format, ISTError>::Convert(ParseStarFormat(obj));
    } else if (type == "repeat") {
      return Result<Format, ISTError>::Convert(ParseRepeatFormat(obj));
    } else if (type == "qwen_xml_parameter") {
      return Result<Format, ISTError>::Convert(ParseJSONSchemaFormat(obj, "qwen_xml"));
    } else if (type == "grammar") {
      return Result<Format, ISTError>::Convert(ParseGrammarFormat(obj));
    } else if (type == "regex") {
      return Result<Format, ISTError>::Convert(ParseRegexFormat(obj));
    } else if (type == "token") {
      return Result<Format, ISTError>::Convert(ParseTokenFormat(obj));
    } else if (type == "exclude_token") {
      return Result<Format, ISTError>::Convert(ParseExcludeTokenFormat(obj));
    } else if (type == "any_tokens") {
      return Result<Format, ISTError>::Convert(ParseAnyTokensFormat(obj));
    } else if (type == "token_triggered_tags") {
      return Result<Format, ISTError>::Convert(ParseTokenTriggeredTagsFormat(obj));
    } else if (type == "dispatch") {
      return Result<Format, ISTError>::Convert(ParseDispatchFormat(obj));
    } else if (type == "token_dispatch") {
      return Result<Format, ISTError>::Convert(ParseTokenDispatchFormat(obj));
    } else {
      return ResultErr<ISTError>("Format type not recognized: " + type);
    }
  }

  // If type is not present, try every format type one by one. Tag is prioritized.
  auto tag_format = ParseTagFormat(obj);
  if (!tag_format.IsErr()) {
    return ResultOk<Format>(std::move(tag_format).Unwrap());
  }
  auto const_string_format = ParseConstStringFormat(obj);
  if (!const_string_format.IsErr()) {
    return ResultOk<Format>(std::move(const_string_format).Unwrap());
  }
  auto json_schema_format = ParseJSONSchemaFormat(obj);
  if (!json_schema_format.IsErr()) {
    return ResultOk<Format>(std::move(json_schema_format).Unwrap());
  }
  auto any_text_format = ParseAnyTextFormat(obj);
  if (!any_text_format.IsErr()) {
    return ResultOk<Format>(std::move(any_text_format).Unwrap());
  }
  auto sequence_format = ParseSequenceFormat(obj);
  if (!sequence_format.IsErr()) {
    return ResultOk<Format>(std::move(sequence_format).Unwrap());
  }
  auto or_format = ParseOrFormat(obj);
  if (!or_format.IsErr()) {
    return ResultOk<Format>(std::move(or_format).Unwrap());
  }
  auto triggered_tags_format = ParseTriggeredTagsFormat(obj);
  if (!triggered_tags_format.IsErr()) {
    return ResultOk<Format>(std::move(triggered_tags_format).Unwrap());
  }
  auto tags_with_separator_format = ParseTagsWithSeparatorFormat(obj);
  if (!tags_with_separator_format.IsErr()) {
    return ResultOk<Format>(std::move(tags_with_separator_format).Unwrap());
  }
  auto optional_format = ParseOptionalFormat(obj);
  if (!optional_format.IsErr()) {
    return ResultOk<Format>(std::move(optional_format).Unwrap());
  }
  auto plus_format = ParsePlusFormat(obj);
  if (!plus_format.IsErr()) {
    return ResultOk<Format>(std::move(plus_format).Unwrap());
  }
  auto star_format = ParseStarFormat(obj);
  if (!star_format.IsErr()) {
    return ResultOk<Format>(std::move(star_format).Unwrap());
  }
  auto repeat_format = ParseRepeatFormat(obj);
  if (!repeat_format.IsErr()) {
    return ResultOk<Format>(std::move(repeat_format).Unwrap());
  }
  auto tag_dispatch_format = ParseDispatchFormat(obj);
  if (!tag_dispatch_format.IsErr()) {
    return ResultOk<Format>(std::move(tag_dispatch_format).Unwrap());
  }
  auto token_tag_dispatch_format = ParseTokenDispatchFormat(obj);
  if (!token_tag_dispatch_format.IsErr()) {
    return ResultOk<Format>(std::move(token_tag_dispatch_format).Unwrap());
  }
  return ResultErr<ISTError>("Invalid format: " + value.serialize(false));
}

Result<ConstStringFormat, ISTError> StructuralTagParser::ParseConstStringFormat(
    const picojson::object& obj
) {
  // value is required.
  auto value_it = obj.find("value");
  if (value_it == obj.end() || !value_it->second.is<std::string>()) {
    return ResultErr<ISTError>("ConstString format must have a value field with a string");
  }
  return ResultOk<ConstStringFormat>(value_it->second.get<std::string>());
}

Result<JSONSchemaFormat, ISTError> StructuralTagParser::ParseJSONSchemaFormat(
    const picojson::object& obj, std::optional<std::string> style_override
) {
  // json_schema is required.
  auto json_schema_it = obj.find("json_schema");
  if (json_schema_it == obj.end() ||
      !(json_schema_it->second.is<picojson::object>() || json_schema_it->second.is<bool>())) {
    return ResultErr<ISTError>(
        "JSON schema format must have a json_schema field with a object or boolean value"
    );
  }
  std::string style = "json";
  if (style_override.has_value()) {
    style = *style_override;
  } else {
    auto it = obj.find("style");
    if (it != obj.end() && it->second.is<std::string>()) {
      style = it->second.get<std::string>();
      if (style != "json" && style != "qwen_xml" && style != "minimax_xml" &&
          style != "minimax_m3_xml" && style != "deepseek_xml" && style != "glm_xml" &&
          style != "cohere_xml" && style != "kimi_k3_xml" && style != "deepseek_v4_1_xml") {
        return ResultErr<ISTError>(
            "style must be \"json\", \"qwen_xml\", \"minimax_xml\", \"minimax_m3_xml\", "
            "\"deepseek_xml\", \"glm_xml\", \"cohere_xml\", \"kimi_k3_xml\", or "
            "\"deepseek_v4_1_xml\""
        );
      }
    }
  }
  bool any_order = false;
  auto any_order_it = obj.find("any_order");
  if (any_order_it != obj.end()) {
    if (!any_order_it->second.is<bool>()) {
      return ResultErr<ISTError>("any_order must be a boolean");
    }
    any_order = any_order_it->second.get<bool>();
  }
  std::optional<int> max_whitespace_cnt = std::nullopt;
  auto max_whitespace_cnt_it = obj.find("max_whitespace_cnt");
  if (max_whitespace_cnt_it != obj.end() && !max_whitespace_cnt_it->second.is<picojson::null>()) {
    if (!max_whitespace_cnt_it->second.is<double>()) {
      return ResultErr<ISTError>("max_whitespace_cnt must be an integer or null");
    }
    max_whitespace_cnt = static_cast<int>(max_whitespace_cnt_it->second.get<int64_t>());
  }
  std::vector<std::string> excludes;
  if (auto it = obj.find("excludes"); it != obj.end()) {
    if (!it->second.is<picojson::array>()) {
      return ResultErr<ISTError>(
          "JSONSchema format's excludes must be an array of non-empty strings"
      );
    }
    for (const auto& value : it->second.get<picojson::array>()) {
      if (!value.is<std::string>() || value.get<std::string>().empty()) {
        return ResultErr<ISTError>(
            "JSONSchema format's excludes must be an array of non-empty strings"
        );
      }
      excludes.push_back(value.get<std::string>());
    }
    std::sort(excludes.begin(), excludes.end());
    excludes.erase(std::unique(excludes.begin(), excludes.end()), excludes.end());
  }
  // here introduces a serialization/deserialization overhead; try to avoid it in the future.
  return ResultOk<JSONSchemaFormat>(
      json_schema_it->second.serialize(false),
      style,
      any_order,
      max_whitespace_cnt,
      std::move(excludes)
  );
}

Result<int32_t, ISTError> ParseOptionalBudget(
    const picojson::object& obj, const std::string& field_name, const std::string& format_name
) {
  auto it = obj.find(field_name);
  if (it == obj.end() || it->second.is<picojson::null>()) {
    return ResultOk<int32_t>(-1);
  }
  if (!it->second.is<double>()) {
    return ResultErr<ISTError>(
        field_name + " in " + format_name + " must be a non-negative 32-bit integer or null"
    );
  }
  double value = it->second.get<double>();
  if (value < 0 || value > std::numeric_limits<int32_t>::max() || value != std::floor(value)) {
    return ResultErr<ISTError>(
        field_name + " in " + format_name + " must be a non-negative 32-bit integer or null"
    );
  }
  return ResultOk<int32_t>(static_cast<int32_t>(value));
}

Result<AnyTextFormat, ISTError> StructuralTagParser::ParseAnyTextFormat(const picojson::object& obj
) {
  auto max_tokens_result = ParseOptionalBudget(obj, "max_tokens", "any_text");
  if (max_tokens_result.IsErr()) {
    return ResultErr<ISTError>(std::move(max_tokens_result).UnwrapErr());
  }
  auto max_chars_result = ParseOptionalBudget(obj, "max_chars", "any_text");
  if (max_chars_result.IsErr()) {
    return ResultErr<ISTError>(std::move(max_chars_result).UnwrapErr());
  }
  int32_t max_tokens = std::move(max_tokens_result).Unwrap();
  int32_t max_chars = std::move(max_chars_result).Unwrap();

  auto excluded_strs_it = obj.find("excludes");
  if (excluded_strs_it == obj.end()) {
    if ((obj.find("type") == obj.end())) {
      return ResultErr<ISTError>("Any text format should not have any fields other than type");
    }
    return ResultOk<AnyTextFormat>(std::vector<std::string>{}, max_tokens, max_chars);
  }
  if (!excluded_strs_it->second.is<picojson::array>()) {
    return ResultErr<ISTError>("AnyText format's excluded_strs field must be an array");
  }
  const auto& excluded_strs_array = excluded_strs_it->second.get<picojson::array>();
  std::vector<std::string> excluded_strs;
  excluded_strs.reserve(excluded_strs_array.size());
  for (const auto& excluded_str : excluded_strs_array) {
    if (!excluded_str.is<std::string>()) {
      return ResultErr<ISTError>("AnyText format's excluded_strs array must contain strings");
    }
    excluded_strs.push_back(excluded_str.get<std::string>());
  }
  return ResultOk<AnyTextFormat>(std::move(excluded_strs), max_tokens, max_chars);
}

Result<GrammarFormat, ISTError> StructuralTagParser::ParseGrammarFormat(const picojson::object& obj
) {
  // grammar is required.
  auto grammar_it = obj.find("grammar");
  if (grammar_it == obj.end() || !grammar_it->second.is<std::string>() ||
      grammar_it->second.get<std::string>().empty()) {
    return ResultErr<ISTError>("Grammar format must have a grammar field with a non-empty string");
  }
  return ResultOk<GrammarFormat>(grammar_it->second.get<std::string>());
}

Result<RegexFormat, ISTError> StructuralTagParser::ParseRegexFormat(const picojson::object& obj) {
  // pattern is required.
  auto pattern_it = obj.find("pattern");
  if (pattern_it == obj.end() || !pattern_it->second.is<std::string>() ||
      pattern_it->second.get<std::string>().empty()) {
    return ResultErr<ISTError>("Regex format must have a pattern field with a non-empty string");
  }
  return ResultOk<RegexFormat>(pattern_it->second.get<std::string>());
}

Result<SequenceFormat, ISTError> StructuralTagParser::ParseSequenceFormat(
    const picojson::object& obj
) {
  // elements is required.
  auto elements_it = obj.find("elements");
  if (elements_it == obj.end() || !elements_it->second.is<picojson::array>()) {
    return ResultErr<ISTError>("Sequence format must have an elements field with an array");
  }
  const auto& elements_array = elements_it->second.get<picojson::array>();
  std::vector<Format> elements;
  elements.reserve(elements_array.size());
  for (const auto& element : elements_array) {
    auto format = ParseFormat(element);
    if (format.IsErr()) {
      return ResultErr<ISTError>(std::move(format).UnwrapErr());
    }
    elements.push_back(std::move(format).Unwrap());
  }
  if (elements.size() == 0) {
    return ResultErr<ISTError>("Sequence format must have at least one element");
  }
  return ResultOk<SequenceFormat>(std::move(elements));
}

Result<OrFormat, ISTError> StructuralTagParser::ParseOrFormat(const picojson::object& obj) {
  // elements is required.
  auto elements_it = obj.find("elements");
  if (elements_it == obj.end() || !elements_it->second.is<picojson::array>()) {
    return ResultErr<ISTError>("Or format must have an elements field with an array");
  }
  const auto& elements_array = elements_it->second.get<picojson::array>();
  std::vector<Format> elements;
  elements.reserve(elements_array.size());
  for (const auto& element : elements_array) {
    auto format = ParseFormat(element);
    if (format.IsErr()) {
      return ResultErr<ISTError>(std::move(format).UnwrapErr());
    }
    elements.push_back(std::move(format).Unwrap());
  }
  if (elements.size() == 0) {
    return ResultErr<ISTError>("Or format must have at least one element");
  }
  return ResultOk<OrFormat>(std::move(elements));
}

Result<TagFormat, ISTError> StructuralTagParser::ParseTagFormat(const picojson::value& value) {
  if (!value.is<picojson::object>()) {
    return ResultErr<ISTError>("Tag format must be an object");
  }
  const auto& obj = value.get<picojson::object>();
  if (obj.find("type") != obj.end() &&
      (!obj["type"].is<std::string>() || obj["type"].get<std::string>() != "tag")) {
    return ResultErr<ISTError>("Tag format's type must be a string \"tag\"");
  }
  return ParseTagFormat(obj);
}

Result<TagFormat, ISTError> StructuralTagParser::ParseTagFormat(const picojson::object& obj) {
  // begin is required: string or TokenFormat object
  auto begin_it = obj.find("begin");
  if (begin_it == obj.end()) {
    return ResultErr<ISTError>("Tag format's begin field must be a string");
  }
  std::variant<std::string, TokenFormat> begin;
  if (begin_it->second.is<std::string>()) {
    begin = begin_it->second.get<std::string>();
  } else if (begin_it->second.is<picojson::object>()) {
    auto tf = ParseTokenFormat(begin_it->second.get<picojson::object>());
    if (tf.IsErr()) {
      return ResultErr<ISTError>(std::move(tf).UnwrapErr());
    }
    begin = std::move(tf).Unwrap();
  } else {
    return ResultErr<ISTError>("Tag format's begin field must be a string");
  }

  // content is required.
  auto content_it = obj.find("content");
  if (content_it == obj.end()) {
    return ResultErr<ISTError>("Tag format must have a content field");
  }
  auto content = ParseFormat(content_it->second);
  if (content.IsErr()) {
    return ResultErr<ISTError>(std::move(content).UnwrapErr());
  }

  // end is required: string, array of strings, or TokenFormat object
  auto end_it = obj.find("end");
  if (end_it == obj.end()) {
    return ResultErr<ISTError>("Tag format must have an end field");
  }

  std::variant<std::vector<std::string>, TokenFormat> end;
  if (end_it->second.is<std::string>()) {
    end = std::vector<std::string>{end_it->second.get<std::string>()};
  } else if (end_it->second.is<picojson::array>()) {
    const auto& end_array = end_it->second.get<picojson::array>();
    if (end_array.empty()) {
      return ResultErr<ISTError>("Tag format's end array cannot be empty");
    }
    std::vector<std::string> end_strings;
    for (const auto& item : end_array) {
      if (!item.is<std::string>()) {
        return ResultErr<ISTError>("Tag format's end array must contain only strings");
      }
      end_strings.push_back(item.get<std::string>());
    }
    end = std::move(end_strings);
  } else if (end_it->second.is<picojson::object>()) {
    auto tf = ParseTokenFormat(end_it->second.get<picojson::object>());
    if (tf.IsErr()) {
      return ResultErr<ISTError>(std::move(tf).UnwrapErr());
    }
    end = std::move(tf).Unwrap();
  } else {
    return ResultErr<ISTError>("Tag format's end field must be a string or array of strings");
  }

  return ResultOk<TagFormat>(
      std::move(begin), std::make_shared<Format>(std::move(content).Unwrap()), std::move(end)
  );
}

Result<TriggeredTagsFormat, ISTError> StructuralTagParser::ParseTriggeredTagsFormat(
    const picojson::object& obj
) {
  // triggers is required.
  auto triggers_it = obj.find("triggers");
  if (triggers_it == obj.end() || !triggers_it->second.is<picojson::array>()) {
    return ResultErr<ISTError>("Triggered tags format must have a triggers field with an array");
  }
  const auto& triggers_array = triggers_it->second.get<picojson::array>();
  std::vector<std::string> excluded_strs;
  std::vector<std::string> triggers;
  triggers.reserve(triggers_array.size());
  for (const auto& trigger : triggers_array) {
    if (!trigger.is<std::string>() || trigger.get<std::string>().empty()) {
      return ResultErr<ISTError>("Triggered tags format's triggers must be non-empty strings");
    }
    triggers.push_back(trigger.get<std::string>());
  }
  if (triggers.size() == 0) {
    return ResultErr<ISTError>("Triggered tags format's triggers must be non-empty");
  }
  // tags is required.
  auto tags_it = obj.find("tags");
  if (tags_it == obj.end() || !tags_it->second.is<picojson::array>()) {
    return ResultErr<ISTError>("Triggered tags format must have a tags field with an array");
  }
  const auto& tags_array = tags_it->second.get<picojson::array>();
  std::vector<TagFormat> tags;
  tags.reserve(tags_array.size());
  for (const auto& tag : tags_array) {
    auto tag_format = ParseTagFormat(tag);
    if (tag_format.IsErr()) {
      return ResultErr<ISTError>(std::move(tag_format).UnwrapErr());
    }
    tags.push_back(std::move(tag_format).Unwrap());
  }
  if (tags.size() == 0) {
    return ResultErr<ISTError>("Triggered tags format's tags must be non-empty");
  }
  // excludes is optional.
  auto excludes_it = obj.find("excludes");
  if (excludes_it != obj.end()) {
    if (!excludes_it->second.is<picojson::array>()) {
      return ResultErr<ISTError>("Triggered tags format should have a excludes field with an array"
      );
    }
    const auto& excludes_array = excludes_it->second.get<picojson::array>();
    excluded_strs.reserve(excludes_array.size());
    for (const auto& excluded_str : excludes_array) {
      if (!excluded_str.is<std::string>() || excluded_str.get<std::string>().empty()) {
        return ResultErr<ISTError>("Triggered tags format's excluded_strs must be non-empty strings"
        );
      }
      excluded_strs.push_back(excluded_str.get<std::string>());
    }
  }

  // at_least_one is optional.
  bool at_least_one = false;
  auto at_least_one_it = obj.find("at_least_one");
  if (at_least_one_it != obj.end()) {
    if (!at_least_one_it->second.is<bool>()) {
      return ResultErr<ISTError>("at_least_one must be a boolean");
    }
    at_least_one = at_least_one_it->second.get<bool>();
  }
  // stop_after_first is optional.
  bool stop_after_first = false;
  auto stop_after_first_it = obj.find("stop_after_first");
  if (stop_after_first_it != obj.end()) {
    if (!stop_after_first_it->second.is<bool>()) {
      return ResultErr<ISTError>("stop_after_first must be a boolean");
    }
    stop_after_first = stop_after_first_it->second.get<bool>();
  }
  return ResultOk<TriggeredTagsFormat>(
      std::move(triggers), std::move(tags), std::move(excluded_strs), at_least_one, stop_after_first
  );
}

Result<TagsWithSeparatorFormat, ISTError> StructuralTagParser::ParseTagsWithSeparatorFormat(
    const picojson::object& obj
) {
  // tags is required.
  auto tags_it = obj.find("tags");
  if (tags_it == obj.end() || !tags_it->second.is<picojson::array>()) {
    return ResultErr<ISTError>("Tags with separator format must have a tags field with an array");
  }
  const auto& tags_array = tags_it->second.get<picojson::array>();
  std::vector<TagFormat> tags;
  tags.reserve(tags_array.size());
  for (const auto& tag : tags_array) {
    auto tag_format = ParseTagFormat(tag);
    if (tag_format.IsErr()) {
      return ResultErr<ISTError>(std::move(tag_format).UnwrapErr());
    }
    tags.push_back(std::move(tag_format).Unwrap());
  }
  if (tags.size() == 0) {
    return ResultErr<ISTError>("Tags with separator format's tags must be non-empty");
  }
  // separator is required (can be empty string).
  auto separator_it = obj.find("separator");
  if (separator_it == obj.end() || !separator_it->second.is<std::string>()) {
    return ResultErr<ISTError>("Tags with separator format's separator field must be a string");
  }
  // at_least_one is optional.
  bool at_least_one = false;
  auto at_least_one_it = obj.find("at_least_one");
  if (at_least_one_it != obj.end()) {
    if (!at_least_one_it->second.is<bool>()) {
      return ResultErr<ISTError>("at_least_one must be a boolean");
    }
    at_least_one = at_least_one_it->second.get<bool>();
  }
  // stop_after_first is optional.
  bool stop_after_first = false;
  auto stop_after_first_it = obj.find("stop_after_first");
  if (stop_after_first_it != obj.end()) {
    if (!stop_after_first_it->second.is<bool>()) {
      return ResultErr<ISTError>("stop_after_first must be a boolean");
    }
    stop_after_first = stop_after_first_it->second.get<bool>();
  }
  return ResultOk<TagsWithSeparatorFormat>(
      std::move(tags), separator_it->second.get<std::string>(), at_least_one, stop_after_first
  );
}

Result<OptionalFormat, ISTError> StructuralTagParser::ParseOptionalFormat(
    const picojson::object& obj
) {
  auto content_it = obj.find("content");
  if (content_it == obj.end()) {
    return ResultErr<ISTError>("Optional format must have a content field");
  }
  auto content = ParseFormat(content_it->second);
  if (content.IsErr()) {
    return ResultErr<ISTError>(std::move(content).UnwrapErr());
  }
  return ResultOk<OptionalFormat>(std::make_shared<Format>(std::move(content).Unwrap()));
}

Result<PlusFormat, ISTError> StructuralTagParser::ParsePlusFormat(const picojson::object& obj) {
  auto content_it = obj.find("content");
  if (content_it == obj.end()) {
    return ResultErr<ISTError>("Plus format must have a content field");
  }
  auto content = ParseFormat(content_it->second);
  if (content.IsErr()) {
    return ResultErr<ISTError>(std::move(content).UnwrapErr());
  }
  return ResultOk<PlusFormat>(std::make_shared<Format>(std::move(content).Unwrap()));
}

Result<StarFormat, ISTError> StructuralTagParser::ParseStarFormat(const picojson::object& obj) {
  auto content_it = obj.find("content");
  if (content_it == obj.end()) {
    return ResultErr<ISTError>("Star format must have a content field");
  }
  auto content = ParseFormat(content_it->second);
  if (content.IsErr()) {
    return ResultErr<ISTError>(std::move(content).UnwrapErr());
  }
  return ResultOk<StarFormat>(std::make_shared<Format>(std::move(content).Unwrap()));
}

Result<TokenFormat, ISTError> StructuralTagParser::ParseTokenFormat(const picojson::object& obj) {
  auto token_it = obj.find("token");
  if (token_it == obj.end()) {
    return ResultErr<ISTError>("TokenFormat must have a token field");
  }
  if (token_it->second.is<double>()) {
    double d = token_it->second.get<double>();
    if (d != static_cast<double>(static_cast<int32_t>(d))) {
      return ResultErr<ISTError>("Token ID must be an integer");
    }
    int32_t id = static_cast<int32_t>(d);
    if (id < 0) {
      return ResultErr<ISTError>("Token ID must be non-negative");
    }
    return ResultOk<TokenFormat>(std::variant<int32_t, std::string>(id));
  } else if (token_it->second.is<std::string>()) {
    auto s = token_it->second.get<std::string>();
    if (s.empty()) {
      return ResultErr<ISTError>("Token string must be non-empty");
    }
    return ResultOk<TokenFormat>(std::variant<int32_t, std::string>(std::move(s)));
  }
  return ResultErr<ISTError>("TokenFormat's token must be an integer or string");
}

Result<std::vector<std::variant<int32_t, std::string>>, ISTError> ParseIntOrStringArray(
    const picojson::value& val, const std::string& field_name
) {
  std::vector<std::variant<int32_t, std::string>> result;
  if (!val.is<picojson::array>()) {
    return ResultErr<ISTError>(field_name + " must be an array");
  }
  for (const auto& v : val.get<picojson::array>()) {
    if (v.is<double>()) {
      double d = v.get<double>();
      if (d != static_cast<double>(static_cast<int32_t>(d))) {
        return ResultErr<ISTError>(field_name + " elements must be integers, not floats");
      }
      int32_t id = static_cast<int32_t>(d);
      if (id < 0) {
        return ResultErr<ISTError>(
            field_name + " elements must be non-negative integers or strings"
        );
      }
      result.emplace_back(std::in_place_type<int32_t>, id);
    } else if (v.is<std::string>()) {
      auto s = v.get<std::string>();
      if (s.empty()) {
        return ResultErr<ISTError>(field_name + " string elements must be non-empty");
      }
      result.push_back(std::move(s));
    } else {
      return ResultErr<ISTError>(field_name + " elements must be integers or strings");
    }
  }
  return ResultOk(std::move(result));
}

Result<ExcludeTokenFormat, ISTError> StructuralTagParser::ParseExcludeTokenFormat(
    const picojson::object& obj
) {
  std::vector<std::variant<int32_t, std::string>> exclude_tokens;
  auto it = obj.find("exclude_tokens");
  if (it != obj.end()) {
    auto parsed = ParseIntOrStringArray(it->second, "exclude_tokens");
    if (parsed.IsErr()) {
      return ResultErr<ISTError>(std::move(parsed).UnwrapErr());
    }
    exclude_tokens = std::move(parsed).Unwrap();
  }
  return ResultOk<ExcludeTokenFormat>(std::move(exclude_tokens));
}

Result<AnyTokensFormat, ISTError> StructuralTagParser::ParseAnyTokensFormat(
    const picojson::object& obj
) {
  std::vector<std::variant<int32_t, std::string>> exclude_tokens;
  auto it = obj.find("exclude_tokens");
  if (it != obj.end()) {
    auto parsed = ParseIntOrStringArray(it->second, "exclude_tokens");
    if (parsed.IsErr()) {
      return ResultErr<ISTError>(std::move(parsed).UnwrapErr());
    }
    exclude_tokens = std::move(parsed).Unwrap();
  }
  auto max_tokens_result = ParseOptionalBudget(obj, "max_tokens", "any_tokens");
  if (max_tokens_result.IsErr()) {
    return ResultErr<ISTError>(std::move(max_tokens_result).UnwrapErr());
  }
  return ResultOk<AnyTokensFormat>(
      std::move(exclude_tokens), std::move(max_tokens_result).Unwrap()
  );
}

Result<TokenTriggeredTagsFormat, ISTError> StructuralTagParser::ParseTokenTriggeredTagsFormat(
    const picojson::object& obj
) {
  // trigger_tokens is required
  auto triggers_it = obj.find("trigger_tokens");
  if (triggers_it == obj.end()) {
    return ResultErr<ISTError>("TokenTriggeredTagsFormat must have a trigger_tokens field");
  }
  auto triggers = ParseIntOrStringArray(triggers_it->second, "trigger_tokens");
  if (triggers.IsErr()) {
    return ResultErr<ISTError>(std::move(triggers).UnwrapErr());
  }
  auto trigger_tokens = std::move(triggers).Unwrap();
  if (trigger_tokens.empty()) {
    return ResultErr<ISTError>("trigger_tokens must be non-empty");
  }

  // tags is required
  auto tags_it = obj.find("tags");
  if (tags_it == obj.end() || !tags_it->second.is<picojson::array>()) {
    return ResultErr<ISTError>("TokenTriggeredTagsFormat must have a tags field with an array");
  }
  std::vector<TagFormat> tags;
  for (const auto& tag : tags_it->second.get<picojson::array>()) {
    auto tag_format = ParseTagFormat(tag);
    if (tag_format.IsErr()) {
      return ResultErr<ISTError>(std::move(tag_format).UnwrapErr());
    }
    tags.push_back(std::move(tag_format).Unwrap());
  }
  if (tags.empty()) {
    return ResultErr<ISTError>("TokenTriggeredTagsFormat tags must be non-empty");
  }

  // exclude_tokens is optional
  std::vector<std::variant<int32_t, std::string>> exclude_tokens;
  auto excludes_it = obj.find("exclude_tokens");
  if (excludes_it != obj.end()) {
    auto parsed = ParseIntOrStringArray(excludes_it->second, "exclude_tokens");
    if (parsed.IsErr()) {
      return ResultErr<ISTError>(std::move(parsed).UnwrapErr());
    }
    exclude_tokens = std::move(parsed).Unwrap();
  }

  bool at_least_one = false;
  auto alo_it = obj.find("at_least_one");
  if (alo_it != obj.end()) {
    if (!alo_it->second.is<bool>()) {
      return ResultErr<ISTError>("at_least_one must be a boolean");
    }
    at_least_one = alo_it->second.get<bool>();
  }

  bool stop_after_first = false;
  auto saf_it = obj.find("stop_after_first");
  if (saf_it != obj.end()) {
    if (!saf_it->second.is<bool>()) {
      return ResultErr<ISTError>("stop_after_first must be a boolean");
    }
    stop_after_first = saf_it->second.get<bool>();
  }

  return ResultOk<TokenTriggeredTagsFormat>(
      std::move(trigger_tokens),
      std::move(tags),
      std::move(exclude_tokens),
      at_least_one,
      stop_after_first
  );
}

Result<DispatchFormat, ISTError> StructuralTagParser::ParseDispatchFormat(
    const picojson::object& obj
) {
  auto rules_it = obj.find("rules");
  if (rules_it == obj.end() || !rules_it->second.is<picojson::array>()) {
    return ResultErr<ISTError>("TagDispatch format must have a rules field with an array");
  }
  const auto& rules_array = rules_it->second.get<picojson::array>();
  if (rules_array.empty()) {
    return ResultErr<ISTError>("TagDispatch format rules must be non-empty");
  }
  std::vector<std::pair<std::string, std::shared_ptr<Format>>> rules;
  rules.reserve(rules_array.size());
  for (const auto& item : rules_array) {
    if (!item.is<picojson::array>()) {
      return ResultErr<ISTError>("TagDispatch pair must be a 2-element array");
    }
    const auto& pair_arr = item.get<picojson::array>();
    if (pair_arr.size() != 2) {
      return ResultErr<ISTError>("TagDispatch pair must be a 2-element array");
    }
    if (!pair_arr[0].is<std::string>()) {
      return ResultErr<ISTError>("TagDispatch pair first element must be a string");
    }
    std::string trigger = pair_arr[0].get<std::string>();
    auto content = ParseFormat(pair_arr[1]);
    if (content.IsErr()) {
      return ResultErr<ISTError>(std::move(content).UnwrapErr());
    }
    rules.push_back({std::move(trigger), std::make_shared<Format>(std::move(content).Unwrap())});
  }

  bool loop = true;
  auto loop_it = obj.find("loop");
  if (loop_it != obj.end()) {
    if (!loop_it->second.is<bool>()) {
      return ResultErr<ISTError>("loop must be a boolean");
    }
    loop = loop_it->second.get<bool>();
  }

  std::vector<std::string> excludes;
  auto excludes_it = obj.find("excludes");
  if (excludes_it != obj.end()) {
    if (!excludes_it->second.is<picojson::array>()) {
      return ResultErr<ISTError>("excludes must be an array");
    }
    for (const auto& e : excludes_it->second.get<picojson::array>()) {
      if (!e.is<std::string>() || e.get<std::string>().empty()) {
        return ResultErr<ISTError>("excludes must contain non-empty strings");
      }
      excludes.push_back(e.get<std::string>());
    }
  }

  return ResultOk<DispatchFormat>(std::move(rules), loop, std::move(excludes));
}

Result<TokenDispatchFormat, ISTError> StructuralTagParser::ParseTokenDispatchFormat(
    const picojson::object& obj
) {
  auto rules_it = obj.find("rules");
  if (rules_it == obj.end() || !rules_it->second.is<picojson::array>()) {
    return ResultErr<ISTError>("TokenTagDispatch format must have a rules field with an array");
  }
  const auto& rules_array = rules_it->second.get<picojson::array>();
  if (rules_array.empty()) {
    return ResultErr<ISTError>("TokenTagDispatch format rules must be non-empty");
  }
  std::vector<std::pair<std::variant<int32_t, std::string>, std::shared_ptr<Format>>> rules;
  rules.reserve(rules_array.size());
  for (const auto& item : rules_array) {
    if (!item.is<picojson::array>()) {
      return ResultErr<ISTError>("TokenTagDispatch pair must be a 2-element array");
    }
    const auto& pair_arr = item.get<picojson::array>();
    if (pair_arr.size() != 2) {
      return ResultErr<ISTError>("TokenTagDispatch pair must be a 2-element array");
    }
    std::variant<int32_t, std::string> trigger;
    if (pair_arr[0].is<double>()) {
      double d = pair_arr[0].get<double>();
      if (d != static_cast<double>(static_cast<int32_t>(d))) {
        return ResultErr<ISTError>("Token ID must be an integer");
      }
      trigger = static_cast<int32_t>(d);
    } else if (pair_arr[0].is<std::string>()) {
      trigger = pair_arr[0].get<std::string>();
    } else {
      return ResultErr<ISTError>("TokenTagDispatch pair first element must be an integer or string"
      );
    }
    auto content = ParseFormat(pair_arr[1]);
    if (content.IsErr()) {
      return ResultErr<ISTError>(std::move(content).UnwrapErr());
    }
    rules.push_back({std::move(trigger), std::make_shared<Format>(std::move(content).Unwrap())});
  }

  bool loop = true;
  auto loop_it = obj.find("loop");
  if (loop_it != obj.end()) {
    if (!loop_it->second.is<bool>()) {
      return ResultErr<ISTError>("loop must be a boolean");
    }
    loop = loop_it->second.get<bool>();
  }

  std::vector<std::variant<int32_t, std::string>> exclude_tokens;
  auto excludes_it = obj.find("exclude_tokens");
  if (excludes_it != obj.end()) {
    auto parsed = ParseIntOrStringArray(excludes_it->second, "exclude_tokens");
    if (parsed.IsErr()) {
      return ResultErr<ISTError>(std::move(parsed).UnwrapErr());
    }
    exclude_tokens = std::move(parsed).Unwrap();
  }

  return ResultOk<TokenDispatchFormat>(std::move(rules), loop, std::move(exclude_tokens));
}

/************** StructuralTagTokenResolver **************/

class StructuralTagTokenResolver {
 public:
  static std::optional<ISTError> Resolve(
      StructuralTag* structural_tag, const std::optional<TokenizerInfo>& tokenizer_info
  );

 private:
  explicit StructuralTagTokenResolver(const std::optional<TokenizerInfo>& tokenizer_info)
      : tokenizer_info_(tokenizer_info) {}

  std::optional<ISTError> ResolveFormat(Format* format);
  std::optional<ISTError> ResolveTagFormat(TagFormat* tag);
  std::optional<ISTError> ResolveTokenFormat(TokenFormat* tf);
  std::optional<ISTError> ResolveIntOrStringVec(
      const std::vector<std::variant<int32_t, std::string>>& input, std::vector<int32_t>* output
  );

  const std::optional<TokenizerInfo>& tokenizer_info_;
};

std::optional<ISTError> StructuralTagTokenResolver::Resolve(
    StructuralTag* structural_tag, const std::optional<TokenizerInfo>& tokenizer_info
) {
  return StructuralTagTokenResolver(tokenizer_info).ResolveFormat(&structural_tag->format);
}

std::optional<ISTError> StructuralTagTokenResolver::ResolveTokenFormat(TokenFormat* tf) {
  if (tf->resolved_token_id_ >= 0) return std::nullopt;
  if (!std::holds_alternative<std::string>(tf->token)) return std::nullopt;
  if (!tokenizer_info_) {
    return ISTError("Token string resolution requires tokenizer_info");
  }
  const auto& token_str = std::get<std::string>(tf->token);
  const auto& vocab = tokenizer_info_->GetDecodedVocab();
  for (int32_t i = 0; i < static_cast<int32_t>(vocab.size()); ++i) {
    if (vocab[i] == token_str) {
      tf->resolved_token_id_ = i;
      return std::nullopt;
    }
  }
  return ISTError("Token string \"" + token_str + "\" not found in vocabulary");
}

std::optional<ISTError> StructuralTagTokenResolver::ResolveIntOrStringVec(
    const std::vector<std::variant<int32_t, std::string>>& input, std::vector<int32_t>* output
) {
  output->clear();
  output->reserve(input.size());
  for (const auto& item : input) {
    if (std::holds_alternative<int32_t>(item)) {
      output->push_back(std::get<int32_t>(item));
    } else {
      if (!tokenizer_info_) {
        return ISTError("Token string resolution requires tokenizer_info");
      }
      const auto& s = std::get<std::string>(item);
      const auto& vocab = tokenizer_info_->GetDecodedVocab();
      bool found = false;
      for (int32_t i = 0; i < static_cast<int32_t>(vocab.size()); ++i) {
        if (vocab[i] == s) {
          output->push_back(i);
          found = true;
          break;
        }
      }
      if (!found) {
        return ISTError("Token string \"" + s + "\" not found in vocabulary");
      }
    }
  }
  return std::nullopt;
}

std::optional<ISTError> StructuralTagTokenResolver::ResolveTagFormat(TagFormat* tag) {
  if (std::holds_alternative<TokenFormat>(tag->begin)) {
    auto err = ResolveTokenFormat(&std::get<TokenFormat>(tag->begin));
    if (err) return err;
  }
  if (std::holds_alternative<TokenFormat>(tag->end)) {
    auto err = ResolveTokenFormat(&std::get<TokenFormat>(tag->end));
    if (err) return err;
  }
  return ResolveFormat(tag->content.get());
}

std::optional<ISTError> StructuralTagTokenResolver::ResolveFormat(Format* format) {
  return std::visit(
      [&](auto&& arg) -> std::optional<ISTError> {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, TokenFormat>) {
          return ResolveTokenFormat(&arg);
        } else if constexpr (std::is_same_v<T, ExcludeTokenFormat>) {
          return ResolveIntOrStringVec(arg.exclude_tokens, &arg.resolved_token_ids_);
        } else if constexpr (std::is_same_v<T, AnyTokensFormat>) {
          return ResolveIntOrStringVec(arg.exclude_tokens, &arg.resolved_exclude_token_ids_);
        } else if constexpr (std::is_same_v<T, TokenTriggeredTagsFormat>) {
          auto err = ResolveIntOrStringVec(arg.trigger_tokens, &arg.resolved_trigger_token_ids_);
          if (err) return err;
          err = ResolveIntOrStringVec(arg.exclude_tokens, &arg.resolved_exclude_token_ids_);
          if (err) return err;
          for (auto& tag : arg.tags) {
            err = ResolveTagFormat(&tag);
            if (err) return err;
          }
          return std::nullopt;
        } else if constexpr (std::is_same_v<T, TokenDispatchFormat>) {
          std::vector<std::variant<int32_t, std::string>> trigger_tokens;
          trigger_tokens.reserve(arg.rules.size());
          for (const auto& p : arg.rules) {
            trigger_tokens.push_back(p.first);
          }
          auto err = ResolveIntOrStringVec(trigger_tokens, &arg.resolved_trigger_token_ids_);
          if (err) return err;
          err = ResolveIntOrStringVec(arg.exclude_tokens, &arg.resolved_exclude_token_ids_);
          if (err) return err;
          for (auto& p : arg.rules) {
            if (p.second) {
              auto e = ResolveFormat(p.second.get());
              if (e) return e;
            }
          }
          return std::nullopt;
        } else if constexpr (std::is_same_v<T, DispatchFormat>) {
          for (auto& p : arg.rules) {
            if (p.second) {
              auto err = ResolveFormat(p.second.get());
              if (err) return err;
            }
          }
          return std::nullopt;
        } else if constexpr (std::is_same_v<T, TagFormat>) {
          return ResolveTagFormat(&arg);
        } else if constexpr (std::is_same_v<T, SequenceFormat>) {
          for (auto& elem : arg.elements) {
            auto err = ResolveFormat(&elem);
            if (err) return err;
          }
          return std::nullopt;
        } else if constexpr (std::is_same_v<T, OrFormat>) {
          for (auto& elem : arg.elements) {
            auto err = ResolveFormat(&elem);
            if (err) return err;
          }
          return std::nullopt;
        } else if constexpr (std::is_same_v<T, TriggeredTagsFormat>) {
          for (auto& tag : arg.tags) {
            auto err = ResolveTagFormat(&tag);
            if (err) return err;
          }
          return std::nullopt;
        } else if constexpr (std::is_same_v<T, TagsWithSeparatorFormat>) {
          for (auto& tag : arg.tags) {
            auto err = ResolveTagFormat(&tag);
            if (err) return err;
          }
          return std::nullopt;
        } else if constexpr (std::is_same_v<T, OptionalFormat> || std::is_same_v<T, PlusFormat> ||
                             std::is_same_v<T, StarFormat>) {
          return ResolveFormat(arg.content.get());
        } else {
          return std::nullopt;
        }
      },
      *format
  );
}

Result<RepeatFormat, ISTError> StructuralTagParser::ParseRepeatFormat(const picojson::object& obj) {
  auto min_it = obj.find("min");
  if (min_it == obj.end() || !min_it->second.is<double>()) {
    return ResultErr<ISTError>("Repeat format must have a min field (number)");
  }
  auto max_it = obj.find("max");
  if (max_it == obj.end() || !max_it->second.is<double>()) {
    return ResultErr<ISTError>("Repeat format must have a max field (number)");
  }
  int64_t min = min_it->second.get<int64_t>();
  int64_t max = max_it->second.get<int64_t>();
  int32_t max_value_int32 = std::numeric_limits<int32_t>::max();
  if (max >= 0 && min > max) {
    return ResultErr<ISTError>("Repeat min must be <= max");
  }
  if (min < 0) {
    return ResultErr<ISTError>("Repeat min must be >= 0");
  }
  if (max < -1) {
    return ResultErr<ISTError>("Repeat max must be -1 (unbounded) or >= 0");
  }
  if (max > static_cast<int64_t>(max_value_int32)) {
    XGRAMMAR_LOG(WARNING) << "Repeat max is too large, will be set as not limited";
    max = -1;  // -1 means unlimited
  }
  if (min > static_cast<int64_t>(max_value_int32)) {
    return ResultErr<ISTError>(
        "Repeat min is too large, must be <= " + std::to_string(max_value_int32)
    );
  }
  auto content_it = obj.find("content");
  if (content_it == obj.end()) {
    return ResultErr<ISTError>("Repeat format must have a content field");
  }
  auto content = ParseFormat(content_it->second);
  if (content.IsErr()) {
    return ResultErr<ISTError>(std::move(content).UnwrapErr());
  }
  return ResultOk<RepeatFormat>(
      static_cast<int32_t>(min),
      static_cast<int32_t>(max),
      std::make_shared<Format>(std::move(content).Unwrap())
  );
}

/************** StructuralTag Analyzer **************/

/*!
 * \brief Analyze a StructuralTag and extract useful information for conversion to Grammar.
 */
class StructuralTagAnalyzer {
 public:
  static std::optional<ISTError> Analyze(StructuralTag* structural_tag);

 private:
  /*! \brief A variant that can hold the pointer of any Format types. */
  using FormatPtrVariant = std::variant<
      ConstStringFormat*,
      JSONSchemaFormat*,
      AnyTextFormat*,
      GrammarFormat*,
      RegexFormat*,
      SequenceFormat*,
      OrFormat*,
      TagFormat*,
      TriggeredTagsFormat*,
      TagsWithSeparatorFormat*,
      OptionalFormat*,
      PlusFormat*,
      StarFormat*,
      RepeatFormat*,
      TokenFormat*,
      ExcludeTokenFormat*,
      AnyTokensFormat*,
      TokenTriggeredTagsFormat*,
      DispatchFormat*,
      TokenDispatchFormat*>;

  // Call this if we have a pointer to a Format.
  std::optional<ISTError> Visit(Format* format);
  // Call this if we have a pointer to a variant of Format.
  std::optional<ISTError> Visit(FormatPtrVariant format);

  // The following is dispatched from Visit. Don't call them directly because they don't handle
  // stack logics.
  std::optional<ISTError> VisitSub(ConstStringFormat* format);
  std::optional<ISTError> VisitSub(JSONSchemaFormat* format);
  std::optional<ISTError> VisitSub(AnyTextFormat* format);
  std::optional<ISTError> VisitSub(GrammarFormat* format);
  std::optional<ISTError> VisitSub(RegexFormat* format);
  std::optional<ISTError> VisitSub(SequenceFormat* format);
  std::optional<ISTError> VisitSub(OrFormat* format);
  std::optional<ISTError> VisitSub(TagFormat* format);
  std::optional<ISTError> VisitSub(TriggeredTagsFormat* format);
  std::optional<ISTError> VisitSub(TagsWithSeparatorFormat* format);
  std::optional<ISTError> VisitSub(OptionalFormat* format);
  std::optional<ISTError> VisitSub(PlusFormat* format);
  std::optional<ISTError> VisitSub(StarFormat* format);
  std::optional<ISTError> VisitSub(TokenFormat* format);
  std::optional<ISTError> VisitSub(ExcludeTokenFormat* format);
  std::optional<ISTError> VisitSub(AnyTokensFormat* format);
  std::optional<ISTError> VisitSub(TokenTriggeredTagsFormat* format);
  std::optional<ISTError> VisitSub(RepeatFormat* format);
  std::optional<ISTError> VisitSub(DispatchFormat* format);
  std::optional<ISTError> VisitSub(TokenDispatchFormat* format);

  std::vector<std::string> DetectEndStrings();
  std::vector<int32_t> DetectEndTokenIds();
  bool IsUnlimited(const Format& format);
  bool IsExcluded(const Format& format);

  int visit_format_recursion_depth_ = 0;
  std::vector<FormatPtrVariant> stack_;
};

std::optional<ISTError> StructuralTagAnalyzer::Analyze(StructuralTag* structural_tag) {
  return StructuralTagAnalyzer().Visit(&structural_tag->format);
}

std::vector<std::string> StructuralTagAnalyzer::DetectEndStrings() {
  for (int i = static_cast<int>(stack_.size()) - 1; i >= 0; --i) {
    auto& format = stack_[i];
    if (std::holds_alternative<TagFormat*>(format)) {
      auto* tag = std::get<TagFormat*>(format);
      if (std::holds_alternative<std::vector<std::string>>(tag->end)) {
        return std::get<std::vector<std::string>>(tag->end);
      }
      return {};  // TokenFormat end — propagated via DetectEndTokenIds
    }
  }
  return {};
}

std::vector<int32_t> StructuralTagAnalyzer::DetectEndTokenIds() {
  for (int i = static_cast<int>(stack_.size()) - 1; i >= 0; --i) {
    auto& format = stack_[i];
    if (std::holds_alternative<TagFormat*>(format)) {
      auto* tag = std::get<TagFormat*>(format);
      if (std::holds_alternative<TokenFormat>(tag->end)) {
        auto& tf = std::get<TokenFormat>(tag->end);
        return {tf.resolved_token_id_};
      }
      return {};
    }
  }
  return {};
}

bool StructuralTagAnalyzer::IsUnlimited(const Format& format) {
  return std::visit(
      [&](auto&& arg) -> bool {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, AnyTextFormat>) {
          return true;
        } else if constexpr (std::is_same_v<T, TriggeredTagsFormat>) {
          return true;
        } else if constexpr (std::is_same_v<T, TokenTriggeredTagsFormat>) {
          return true;
        } else if constexpr (std::is_same_v<T, DispatchFormat>) {
          return true;
        } else if constexpr (std::is_same_v<T, TokenDispatchFormat>) {
          return true;
        } else if constexpr (std::is_same_v<T, AnyTokensFormat>) {
          return true;
        } else if constexpr (std::is_same_v<T, TagsWithSeparatorFormat>) {
          return true;
        } else if constexpr (std::is_same_v<T, SequenceFormat>) {
          return arg.is_unlimited_;
        } else if constexpr (std::is_same_v<T, OrFormat>) {
          return arg.is_unlimited_;
        } else if constexpr (std::is_same_v<T, OptionalFormat>) {
          return IsUnlimited(*arg.content);
        } else if constexpr (std::is_same_v<T, StarFormat> || std::is_same_v<T, PlusFormat>) {
          return true;
        } else if constexpr (std::is_same_v<T, RepeatFormat>) {
          return arg.max == -1 || (arg.max != 0 && IsUnlimited(*arg.content));
        } else {
          return false;
        }
      },
      format
  );
}

bool StructuralTagAnalyzer::IsExcluded(const Format& format) {
  return std::visit(
      [&](auto&& arg) -> bool {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, AnyTextFormat>) {
          return !arg.excludes.empty();
        } else if constexpr (std::is_same_v<T, TriggeredTagsFormat>) {
          return !arg.excludes.empty();
        } else if constexpr (std::is_same_v<T, TokenTriggeredTagsFormat>) {
          return !arg.exclude_tokens.empty();
        } else if constexpr (std::is_same_v<T, DispatchFormat>) {
          return !arg.excludes.empty();
        } else if constexpr (std::is_same_v<T, TokenDispatchFormat>) {
          return !arg.exclude_tokens.empty();
        } else if constexpr (std::is_same_v<T, AnyTokensFormat>) {
          return !arg.exclude_tokens.empty();
        } else {
          return false;
        }
      },
      format
  );
}

std::optional<ISTError> StructuralTagAnalyzer::Visit(Format* format) {
  FormatPtrVariant format_ptr_variant =
      std::visit([&](auto&& arg) -> FormatPtrVariant { return &arg; }, *format);
  return Visit(format_ptr_variant);
}

std::optional<ISTError> StructuralTagAnalyzer::Visit(FormatPtrVariant format) {
  RecursionGuard guard(&visit_format_recursion_depth_);

  // Push format to stack
  stack_.push_back(format);

  // Dispatch to the corresponding visit function
  auto result =
      std::visit([&](auto&& arg) -> std::optional<ISTError> { return VisitSub(arg); }, format);

  // Pop format from stack
  stack_.pop_back();

  return result;
}

std::optional<ISTError> StructuralTagAnalyzer::VisitSub(ConstStringFormat* format) {
  return std::nullopt;
}

std::optional<ISTError> StructuralTagAnalyzer::VisitSub(JSONSchemaFormat* format) {
  return std::nullopt;
}

std::optional<ISTError> StructuralTagAnalyzer::VisitSub(AnyTextFormat* format) {
  format->detected_end_strs_ = DetectEndStrings();
  return std::nullopt;
}

std::optional<ISTError> StructuralTagAnalyzer::VisitSub(GrammarFormat* format) {
  return std::nullopt;
}

std::optional<ISTError> StructuralTagAnalyzer::VisitSub(RegexFormat* format) {
  return std::nullopt;
}

std::optional<ISTError> StructuralTagAnalyzer::VisitSub(SequenceFormat* format) {
  bool is_any_unlimited = false;
  for (auto& element : format->elements) {
    auto err = Visit(&element);
    if (err.has_value()) {
      return err;
    }
    is_any_unlimited |= IsUnlimited(element) && !IsExcluded(element);
  }
  format->is_unlimited_ = is_any_unlimited;
  return std::nullopt;
}

std::optional<ISTError> StructuralTagAnalyzer::VisitSub(OrFormat* format) {
  bool is_any_unlimited = false;
  for (auto& element : format->elements) {
    auto err = Visit(&element);
    if (err.has_value()) {
      return err;
    }
    is_any_unlimited |= IsUnlimited(element) && !IsExcluded(element);
  }
  format->is_unlimited_ = is_any_unlimited;
  return std::nullopt;
}

std::optional<ISTError> StructuralTagAnalyzer::VisitSub(TagFormat* format) {
  auto err = Visit(format->content.get());
  if (err.has_value()) {
    return err;
  }
  auto is_content_unlimited = IsUnlimited(*(format->content));
  if (is_content_unlimited) {
    if (std::holds_alternative<std::vector<std::string>>(format->end)) {
      const auto& ends = std::get<std::vector<std::string>>(format->end);
      bool has_non_empty_end = false;
      for (const auto& end_str : ends) {
        if (!end_str.empty()) {
          has_non_empty_end = true;
          break;
        }
      }
      if (!has_non_empty_end && !IsExcluded(*format->content)) {
        return ISTError("When the content is unlimited, at least one end string must be non-empty");
      }
    }
    // TokenFormat end is always non-empty → no error needed
  }
  return std::nullopt;
}

std::optional<ISTError> StructuralTagAnalyzer::VisitSub(TriggeredTagsFormat* format) {
  for (auto& tag : format->tags) {
    auto err = Visit(&tag);
    if (err.has_value()) {
      return err;
    }
  }
  format->detected_end_strs_ = DetectEndStrings();
  return std::nullopt;
}

std::optional<ISTError> StructuralTagAnalyzer::VisitSub(TagsWithSeparatorFormat* format) {
  for (auto& tag : format->tags) {
    auto err = Visit(&tag);
    if (err.has_value()) {
      return err;
    }
  }
  return std::nullopt;
}

std::optional<ISTError> StructuralTagAnalyzer::VisitSub(OptionalFormat* format) {
  return Visit(format->content.get());
}

std::optional<ISTError> StructuralTagAnalyzer::VisitSub(PlusFormat* format) {
  return Visit(format->content.get());
}

std::optional<ISTError> StructuralTagAnalyzer::VisitSub(StarFormat* format) {
  return Visit(format->content.get());
}

std::optional<ISTError> StructuralTagAnalyzer::VisitSub(TokenFormat* format) {
  return std::nullopt;
}

std::optional<ISTError> StructuralTagAnalyzer::VisitSub(ExcludeTokenFormat* format) {
  format->detected_end_token_ids_ = DetectEndTokenIds();
  return std::nullopt;
}

std::optional<ISTError> StructuralTagAnalyzer::VisitSub(AnyTokensFormat* format) {
  format->detected_end_token_ids_ = DetectEndTokenIds();
  return std::nullopt;
}

std::optional<ISTError> StructuralTagAnalyzer::VisitSub(TokenTriggeredTagsFormat* format) {
  for (auto& tag : format->tags) {
    auto err = Visit(&tag);
    if (err.has_value()) {
      return err;
    }
  }
  format->detected_end_token_ids_ = DetectEndTokenIds();
  return std::nullopt;
}

std::optional<ISTError> StructuralTagAnalyzer::VisitSub(DispatchFormat* format) {
  for (auto& pair : format->rules) {
    if (pair.second) {
      auto err = Visit(pair.second.get());
      if (err.has_value()) return err;
    }
  }
  return std::nullopt;
}

std::optional<ISTError> StructuralTagAnalyzer::VisitSub(TokenDispatchFormat* format) {
  for (auto& pair : format->rules) {
    if (pair.second) {
      auto err = Visit(pair.second.get());
      if (err.has_value()) return err;
    }
  }
  return std::nullopt;
}

std::optional<ISTError> StructuralTagAnalyzer::VisitSub(RepeatFormat* format) {
  return Visit(format->content.get());
}

/************** StructuralTag to Grammar Converter **************/

class StructuralTagGrammarConverter {
 public:
  static Result<Grammar, ISTError> Convert(const StructuralTag& structural_tag);

 private:
  StructuralTagGrammarConverter() = default;

  /*!
   * \brief Visit a Format and return the rule id of the added rule.
   * \param format The Format to visit.
   * \return The rule id of the added rule. If the visit fails, the error is returned.
   * \note This method uses serialization to deduplicate identical formats.
   */
  Result<int, ISTError> Visit(const Format& format);
  Result<int, ISTError> VisitSub(const ConstStringFormat& format);
  Result<int, ISTError> VisitSub(const JSONSchemaFormat& format);
  Result<int, ISTError> VisitSub(const AnyTextFormat& format);
  Result<int, ISTError> VisitSub(const GrammarFormat& format);
  Result<int, ISTError> VisitSub(const RegexFormat& format);
  Result<int, ISTError> VisitSub(const SequenceFormat& format);
  Result<int, ISTError> VisitSub(const OrFormat& format);
  Result<int, ISTError> VisitSub(const TagFormat& format);
  Result<int, ISTError> VisitSub(const TriggeredTagsFormat& format);
  Result<int, ISTError> VisitSub(const TagsWithSeparatorFormat& format);
  Result<int, ISTError> VisitSub(const OptionalFormat& format);
  Result<int, ISTError> VisitSub(const PlusFormat& format);
  Result<int, ISTError> VisitSub(const StarFormat& format);
  Result<int, ISTError> VisitSub(const TokenFormat& format);
  Result<int, ISTError> VisitSub(const ExcludeTokenFormat& format);
  Result<int, ISTError> VisitSub(const AnyTokensFormat& format);
  Result<int, ISTError> VisitSub(const TokenTriggeredTagsFormat& format);
  Result<int, ISTError> VisitSub(const RepeatFormat& format);
  Result<int, ISTError> VisitSub(const DispatchFormat& format);
  Result<int, ISTError> VisitSub(const TokenDispatchFormat& format);
  Grammar AddRootRuleAndGetGrammar(int ref_rule_id);
  void SetRuleBodyAndBudgets(
      int32_t rule_id, int body_expr_id, int32_t max_tokens, int32_t max_chars = -1
  );

  bool IsPrefix(const std::string& prefix, const std::string& full_str);
  int BuildBeginExpr(const TagFormat& tag);
  int BuildEndExpr(const TagFormat& tag);

  GrammarBuilder grammar_builder_;

  /*!
   * \brief Cache from format serialization to rule id.
   * This enables deduplication of identical formats to reduce grammar size.
   */
  std::unordered_map<std::string, int> serialization_to_rule_id_;
};

bool StructuralTagGrammarConverter::IsPrefix(
    const std::string& prefix, const std::string& full_str
) {
  return prefix.size() <= full_str.size() &&
         std::string_view(full_str).substr(0, prefix.size()) == prefix;
}

Result<Grammar, ISTError> StructuralTagGrammarConverter::Convert(const StructuralTag& structural_tag
) {
  StructuralTagGrammarConverter converter;
  auto result = converter.Visit(structural_tag.format);
  if (result.IsErr()) {
    return ResultErr(std::move(result).UnwrapErr());
  }
  // Add a root rule
  auto root_rule_id = std::move(result).Unwrap();
  return ResultOk(converter.AddRootRuleAndGetGrammar(root_rule_id));
}

Grammar StructuralTagGrammarConverter::AddRootRuleAndGetGrammar(int ref_rule_id) {
  auto expr = grammar_builder_.AddRuleRef(ref_rule_id);
  auto sequence_expr = grammar_builder_.AddSequence({expr});
  auto choices_expr = grammar_builder_.AddChoices({sequence_expr});
  auto root_rule_id = grammar_builder_.AddRuleWithHint("root", choices_expr);
  return grammar_builder_.Get(root_rule_id);
}

void StructuralTagGrammarConverter::SetRuleBodyAndBudgets(
    int32_t rule_id, int body_expr_id, int32_t max_tokens, int32_t max_chars
) {
  // A zero-token region is empty by definition. Materialize that in the grammar because -1 is
  // also the parser's pre-first-token deadline sentinel.
  if (max_tokens == 0) {
    body_expr_id = grammar_builder_.AddChoices({grammar_builder_.AddEmptyStr()});
  }
  grammar_builder_.UpdateRuleBody(rule_id, body_expr_id);
  if (max_tokens >= 0) {
    grammar_builder_.UpdateMaxTokens(rule_id, max_tokens);
  }
  if (max_chars >= 0) {
    grammar_builder_.UpdateMaxChars(rule_id, max_chars);
  }
}

Result<int, ISTError> StructuralTagGrammarConverter::Visit(const Format& format) {
  std::string fingerprint = FormatToJSONValue(format).serialize();

  // Check if we've already processed an identical format
  auto it = serialization_to_rule_id_.find(fingerprint);
  if (it != serialization_to_rule_id_.end()) {
    return ResultOk(it->second);
  }

  // Process the format and cache the result
  auto result =
      std::visit([&](auto&& arg) -> Result<int, ISTError> { return VisitSub(arg); }, format);
  if (result.IsOk()) {
    int rule_id = std::move(result).Unwrap();
    serialization_to_rule_id_[fingerprint] = rule_id;
    return ResultOk(rule_id);
  }
  return result;
}

Result<int, ISTError> StructuralTagGrammarConverter::VisitSub(const ConstStringFormat& format) {
  auto expr = format.value.empty() ? grammar_builder_.AddEmptyStr()
                                   : grammar_builder_.AddByteString(format.value);
  auto sequence_expr = grammar_builder_.AddSequence({expr});
  auto choices_expr = grammar_builder_.AddChoices({sequence_expr});
  return ResultOk(grammar_builder_.AddRuleWithHint("const_string", choices_expr));
}

Result<int, ISTError> StructuralTagGrammarConverter::VisitSub(const JSONSchemaFormat& format) {
  auto json_format = JSONFormatFromString(format.style);
  if (!json_format.has_value()) {
    return ResultErr<ISTError>("Unsupported parsing type: " + format.style);
  }
  // The whitespace cap comes from the JSONSchemaFormat node (per-tag).
  auto sub_grammar = GrammarNormalizer::Apply(JSONSchemaToGrammar(
      format.json_schema,
      /*any_whitespace=*/true,
      /*indent=*/std::nullopt,
      /*separators=*/std::nullopt,
      /*strict_mode=*/true,
      /*max_whitespace_cnt=*/format.max_whitespace_cnt,
      /*any_order=*/format.any_order,
      /*json_format=*/*json_format,
      /*excludes=*/format.excludes
  ));
  auto added_root_rule_id = SubGrammarAdder().Apply(&grammar_builder_, sub_grammar);
  return ResultOk(added_root_rule_id);
}

Result<int, ISTError> StructuralTagGrammarConverter::VisitSub(const GrammarFormat& format) {
  auto sub_grammar = Grammar::FromEBNF(format.grammar);
  auto added_root_rule_id = SubGrammarAdder().Apply(&grammar_builder_, sub_grammar);
  return ResultOk(added_root_rule_id);
}

Result<int, ISTError> StructuralTagGrammarConverter::VisitSub(const RegexFormat& format) {
  auto sub_grammar = Grammar::FromRegex(format.pattern);
  auto added_root_rule_id = SubGrammarAdder().Apply(&grammar_builder_, sub_grammar);
  return ResultOk(added_root_rule_id);
}

Result<int, ISTError> StructuralTagGrammarConverter::VisitSub(const AnyTextFormat& format) {
  std::vector<std::string> all_excludes = format.excludes;
  for (const auto& s : format.detected_end_strs_) {
    if (!s.empty()) {
      all_excludes.push_back(s);
    }
  }
  int body_expr_id;
  if (!all_excludes.empty()) {
    body_expr_id =
        grammar_builder_.AddTagDispatch(Grammar::Impl::TagDispatch{{}, false, all_excludes});
  } else {
    auto any_text_expr = grammar_builder_.AddCharacterClassStar({{0, 0x10FFFF}}, false);
    auto sequence_expr = grammar_builder_.AddSequence({any_text_expr});
    body_expr_id = grammar_builder_.AddChoices({sequence_expr});
  }
  int rule_id = grammar_builder_.AddEmptyRuleWithHint("any_text");
  SetRuleBodyAndBudgets(rule_id, body_expr_id, format.max_tokens, format.max_chars);
  return ResultOk(rule_id);
}

Result<int, ISTError> StructuralTagGrammarConverter::VisitSub(const SequenceFormat& format) {
  std::vector<int> rule_ref_ids;
  rule_ref_ids.reserve(format.elements.size());
  for (const auto& element : format.elements) {
    auto result = Visit(element);
    if (result.IsErr()) {
      return result;
    }
    int sub_rule_id = std::move(result).Unwrap();
    rule_ref_ids.push_back(grammar_builder_.AddRuleRef(sub_rule_id));
  }
  auto expr = grammar_builder_.AddChoices({grammar_builder_.AddSequence(rule_ref_ids)});
  return ResultOk(grammar_builder_.AddRuleWithHint("sequence", expr));
}

Result<int, ISTError> StructuralTagGrammarConverter::VisitSub(const OrFormat& format) {
  std::vector<int> sequence_ids;
  sequence_ids.reserve(format.elements.size());
  for (const auto& element : format.elements) {
    auto result = Visit(element);
    if (result.IsErr()) {
      return result;
    }
    int sub_rule_id = std::move(result).Unwrap();
    auto rule_ref_expr = grammar_builder_.AddRuleRef(sub_rule_id);
    sequence_ids.push_back(grammar_builder_.AddSequence({rule_ref_expr}));
  }
  auto expr = grammar_builder_.AddChoices(sequence_ids);
  return ResultOk(grammar_builder_.AddRuleWithHint("or", expr));
}

int StructuralTagGrammarConverter::BuildBeginExpr(const TagFormat& tag) {
  if (std::holds_alternative<std::string>(tag.begin)) {
    return grammar_builder_.AddByteString(std::get<std::string>(tag.begin));
  }
  return grammar_builder_.AddTokenSet({std::get<TokenFormat>(tag.begin).resolved_token_id_});
}

int StructuralTagGrammarConverter::BuildEndExpr(const TagFormat& tag) {
  if (std::holds_alternative<TokenFormat>(tag.end)) {
    return grammar_builder_.AddTokenSet({std::get<TokenFormat>(tag.end).resolved_token_id_});
  }
  const auto& ends = std::get<std::vector<std::string>>(tag.end);
  if (ends.size() == 1) {
    return ends[0].empty() ? grammar_builder_.AddEmptyStr()
                           : grammar_builder_.AddByteString(ends[0]);
  }
  std::vector<int> end_seq_ids;
  for (const auto& s : ends) {
    auto e = s.empty() ? grammar_builder_.AddEmptyStr() : grammar_builder_.AddByteString(s);
    end_seq_ids.push_back(grammar_builder_.AddSequence({e}));
  }
  auto choice = grammar_builder_.AddChoices(end_seq_ids);
  auto rule = grammar_builder_.AddRuleWithHint("tag_end", choice);
  return grammar_builder_.AddRuleRef(rule);
}

Result<int, ISTError> StructuralTagGrammarConverter::VisitSub(const TagFormat& format) {
  auto result = Visit(*format.content);
  if (result.IsErr()) {
    return result;
  }
  auto sub_rule_id = std::move(result).Unwrap();
  auto begin_expr = BuildBeginExpr(format);
  auto rule_ref_expr = grammar_builder_.AddRuleRef(sub_rule_id);
  auto end_expr = BuildEndExpr(format);

  auto sequence_expr_id = grammar_builder_.AddSequence({begin_expr, rule_ref_expr, end_expr});
  auto choices_expr = grammar_builder_.AddChoices({sequence_expr_id});
  return ResultOk(grammar_builder_.AddRuleWithHint("tag", choices_expr));
}

Result<int, ISTError> StructuralTagGrammarConverter::VisitSub(const TriggeredTagsFormat& format) {
  // Step 1. Visit all tags and add to grammar
  std::vector<std::vector<int>> trigger_to_tag_ids(format.triggers.size());
  std::vector<int> tag_content_rule_ids;
  tag_content_rule_ids.reserve(format.tags.size());

  for (int it_tag = 0; it_tag < static_cast<int>(format.tags.size()); ++it_tag) {
    const auto& tag = format.tags[it_tag];
    if (!std::holds_alternative<std::string>(tag.begin)) {
      return ResultErr<ISTError>(
          "Tags in triggered_tags must have a string begin, not a token format"
      );
    }
    const auto& tag_begin = std::get<std::string>(tag.begin);
    int matched_trigger_id = -1;
    for (int it_trigger = 0; it_trigger < static_cast<int>(format.triggers.size()); ++it_trigger) {
      const auto& trigger = format.triggers[it_trigger];
      if (IsPrefix(trigger, tag_begin)) {
        if (matched_trigger_id != -1) {
          return ResultErr<ISTError>("One tag matches multiple triggers in a triggered tags format"
          );
        }
        matched_trigger_id = it_trigger;
      }
    }
    if (matched_trigger_id == -1) {
      return ResultErr<ISTError>("One tag does not match any trigger in a triggered tags format");
    }
    trigger_to_tag_ids[matched_trigger_id].push_back(it_tag);

    auto result = Visit(*tag.content);
    if (result.IsErr()) {
      return result;
    }
    tag_content_rule_ids.push_back(std::move(result).Unwrap());
  }

  // Step 2. Special Case: at_least_one && stop_after_first.
  if (format.at_least_one && format.stop_after_first) {
    std::vector<int> choice_elements;
    for (int it_tag = 0; it_tag < static_cast<int>(format.tags.size()); ++it_tag) {
      const auto& tag = format.tags[it_tag];
      auto begin_expr_id = BuildBeginExpr(tag);
      auto rule_ref_expr_id = grammar_builder_.AddRuleRef(tag_content_rule_ids[it_tag]);
      auto end_expr_id = BuildEndExpr(tag);
      choice_elements.push_back(
          grammar_builder_.AddSequence({begin_expr_id, rule_ref_expr_id, end_expr_id})
      );
    }
    auto choice_expr_id = grammar_builder_.AddChoices(choice_elements);
    return ResultOk(grammar_builder_.AddRuleWithHint("triggered_tags", choice_expr_id));
  }

  // Step 3. Normal Case.
  // Step 3.1 Get tag_rule_pairs.
  std::vector<std::pair<std::string, int32_t>> tag_rule_pairs;
  for (int it_trigger = 0; it_trigger < static_cast<int>(format.triggers.size()); ++it_trigger) {
    const auto& trigger = format.triggers[it_trigger];
    std::vector<int> choice_elements;
    for (const auto& tag_id : trigger_to_tag_ids[it_trigger]) {
      const auto& tag = format.tags[tag_id];
      const auto& tag_begin = std::get<std::string>(tag.begin);
      int begin_expr_id = grammar_builder_.AddByteString(tag_begin.substr(trigger.size()));
      int rule_ref_expr_id = grammar_builder_.AddRuleRef(tag_content_rule_ids[tag_id]);
      int end_expr_id = BuildEndExpr(tag);
      choice_elements.push_back(
          grammar_builder_.AddSequence({begin_expr_id, rule_ref_expr_id, end_expr_id})
      );
    }
    auto choice_expr_id = grammar_builder_.AddChoices(choice_elements);
    auto sub_rule_id = grammar_builder_.AddRuleWithHint("triggered_tags_group", choice_expr_id);
    tag_rule_pairs.push_back(std::make_pair(trigger, sub_rule_id));
  }

  // Step 3.2 Add TagDispatch.
  int32_t rule_expr_id;
  bool loop_after_dispatch = !format.stop_after_first;
  std::vector<std::string> all_excludes = format.excludes;
  for (const auto& s : format.detected_end_strs_) {
    if (!s.empty()) {
      all_excludes.push_back(s);
    }
  }
  rule_expr_id = grammar_builder_.AddTagDispatch(
      Grammar::Impl::TagDispatch{tag_rule_pairs, loop_after_dispatch, all_excludes}
  );

  // Step 3.3 Consider at_least_one
  if (format.at_least_one) {
    std::vector<int> first_choice_elements;
    for (int it_tag = 0; it_tag < static_cast<int>(format.tags.size()); ++it_tag) {
      const auto& tag = format.tags[it_tag];
      auto begin_expr_id = BuildBeginExpr(tag);
      auto rule_ref_expr_id = grammar_builder_.AddRuleRef(tag_content_rule_ids[it_tag]);
      auto end_expr_id = BuildEndExpr(tag);
      first_choice_elements.push_back(
          grammar_builder_.AddSequence({begin_expr_id, rule_ref_expr_id, end_expr_id})
      );
    }
    auto first_choice_expr_id = grammar_builder_.AddChoices(first_choice_elements);
    auto first_rule_id =
        grammar_builder_.AddRuleWithHint("triggered_tags_first", first_choice_expr_id);

    auto tag_dispatch_rule_id =
        grammar_builder_.AddRuleWithHint("triggered_tags_sub", rule_expr_id);
    auto ref_first_rule_expr_id = grammar_builder_.AddRuleRef(first_rule_id);
    auto ref_tag_dispatch_rule_expr_id = grammar_builder_.AddRuleRef(tag_dispatch_rule_id);
    auto sequence_expr_id =
        grammar_builder_.AddSequence({ref_first_rule_expr_id, ref_tag_dispatch_rule_expr_id});
    rule_expr_id = grammar_builder_.AddChoices({sequence_expr_id});
  }

  auto rule_id = grammar_builder_.AddRuleWithHint("triggered_tags", rule_expr_id);
  return ResultOk(rule_id);
}

Result<int, ISTError> StructuralTagGrammarConverter::VisitSub(const TagsWithSeparatorFormat& format
) {
  // The grammar:
  // Step 1. tags_rule: call tags
  //   tags_rule ::= tag1 | tag2 | ... | tagN
  // Step 2. Special handling (stop_after_first is true):
  //   if at_least_one is false:
  //     root ::= tags_rule | ""
  //   if at_least_one is true:
  //     root ::= tags_rule
  // Step 3. Normal handling (stop_after_first is false):
  //   if at_least_one is false:
  //     root ::= tags_rule tags_rule_sub | ""
  //   if at_least_one is true:
  //     root ::= tags_rule tags_rule_sub
  //   tags_rule_sub ::= sep tags_rule tags_rule_sub | ""

  // Step 1. Construct a rule representing any tag
  std::vector<int> choice_ids;
  for (int it_tag = 0; it_tag < static_cast<int>(format.tags.size()); ++it_tag) {
    auto tag_rule_id = Visit(format.tags[it_tag]);
    if (tag_rule_id.IsErr()) {
      return tag_rule_id;
    }
    auto tag_rule_ref_id = grammar_builder_.AddRuleRef(std::move(tag_rule_id).Unwrap());
    auto sequence_expr_id = grammar_builder_.AddSequence({tag_rule_ref_id});
    choice_ids.push_back(sequence_expr_id);
  }
  auto choice_expr_id = grammar_builder_.AddChoices(choice_ids);
  auto all_tags_rule_id =
      grammar_builder_.AddRuleWithHint("tags_with_separator_tags", choice_expr_id);

  auto all_tags_rule_ref_id = grammar_builder_.AddRuleRef(all_tags_rule_id);

  // Step 2. Special case (stop_after_first is true):
  if (format.stop_after_first) {
    int32_t rule_body_expr_id;
    if (format.at_least_one) {
      // root ::= tags_rule
      rule_body_expr_id =
          grammar_builder_.AddChoices({grammar_builder_.AddSequence({all_tags_rule_ref_id})});
    } else {
      // root ::= tags_rule | ""
      rule_body_expr_id = grammar_builder_.AddChoices(
          {grammar_builder_.AddSequence({all_tags_rule_ref_id}), grammar_builder_.AddEmptyStr()}
      );
    }

    auto rule_id = grammar_builder_.AddRuleWithHint("tags_with_separator", rule_body_expr_id);
    return ResultOk(rule_id);
  }

  // Step 3. Normal handling (stop_after_first is false):
  // Step 3.1 Construct sub rule: sub ::= sep tags sub | ""
  auto sub_rule_id = grammar_builder_.AddEmptyRuleWithHint("tags_with_separator_sub");

  auto end_str_sequence_id = grammar_builder_.AddEmptyStr();

  std::vector<int> sub_sequence_elements;
  if (!format.separator.empty()) {
    sub_sequence_elements.push_back(grammar_builder_.AddByteString(format.separator));
  }
  sub_sequence_elements.push_back(all_tags_rule_ref_id);
  sub_sequence_elements.push_back(grammar_builder_.AddRuleRef(sub_rule_id));

  auto sub_rule_body_id = grammar_builder_.AddChoices(
      {grammar_builder_.AddSequence(sub_sequence_elements), end_str_sequence_id}
  );
  grammar_builder_.UpdateRuleBody(sub_rule_id, sub_rule_body_id);

  // Step 3.2 Construct root rule
  std::vector<int> choices = {
      grammar_builder_.AddSequence({all_tags_rule_ref_id, grammar_builder_.AddRuleRef(sub_rule_id)}
      ),
  };
  if (!format.at_least_one) {
    choices.push_back(end_str_sequence_id);
  }
  auto rule_body_expr_id = grammar_builder_.AddChoices(choices);
  auto rule_id = grammar_builder_.AddRuleWithHint("tags_with_separator", rule_body_expr_id);
  return ResultOk(rule_id);
}

Result<int, ISTError> StructuralTagGrammarConverter::VisitSub(const OptionalFormat& format) {
  // optional: 0 or 1 occurrence -> Choice(content, "")
  auto result = Visit(*format.content);
  if (result.IsErr()) {
    return result;
  }
  int content_rule_id = std::move(result).Unwrap();
  auto content_ref = grammar_builder_.AddRuleRef(content_rule_id);
  auto expr = grammar_builder_.AddChoices(
      {grammar_builder_.AddEmptyStr(), grammar_builder_.AddSequence({content_ref})}
  );
  return ResultOk(grammar_builder_.AddRuleWithHint("optional", expr));
}

Result<int, ISTError> StructuralTagGrammarConverter::VisitSub(const PlusFormat& format) {
  // plus: 1 or more occurrences -> content content_star, where content_star = content content_star
  // | ""
  auto result = Visit(*format.content);
  if (result.IsErr()) {
    return result;
  }
  int content_rule_id = std::move(result).Unwrap();
  auto content_ref = grammar_builder_.AddRuleRef(content_rule_id);
  auto star_rule_id = grammar_builder_.AddEmptyRuleWithHint("plus_star");
  auto star_ref = grammar_builder_.AddRuleRef(star_rule_id);
  auto star_body = grammar_builder_.AddChoices(
      {grammar_builder_.AddEmptyStr(), grammar_builder_.AddSequence({content_ref, star_ref})}
  );
  grammar_builder_.UpdateRuleBody(star_rule_id, star_body);
  auto plus_expr = grammar_builder_.AddSequence({content_ref, star_ref});
  return ResultOk(grammar_builder_.AddRuleWithHint("plus", plus_expr));
}

Result<int, ISTError> StructuralTagGrammarConverter::VisitSub(const StarFormat& format) {
  // star: 0 or more occurrences -> content_star, where content_star = content content_star | ""
  auto result = Visit(*format.content);
  if (result.IsErr()) {
    return result;
  }
  int content_rule_id = std::move(result).Unwrap();
  auto content_ref = grammar_builder_.AddRuleRef(content_rule_id);
  auto star_rule_id = grammar_builder_.AddEmptyRuleWithHint("star");
  auto star_ref = grammar_builder_.AddRuleRef(star_rule_id);
  auto star_body = grammar_builder_.AddChoices(
      {grammar_builder_.AddEmptyStr(), grammar_builder_.AddSequence({content_ref, star_ref})}
  );
  grammar_builder_.UpdateRuleBody(star_rule_id, star_body);
  return ResultOk(grammar_builder_.AddRuleWithHint("star", star_ref));
}

Result<int, ISTError> StructuralTagGrammarConverter::VisitSub(const TokenFormat& format) {
  XGRAMMAR_DCHECK(format.resolved_token_id_ >= 0)
      << "TokenFormat must be resolved before conversion";
  auto token_set_expr = grammar_builder_.AddTokenSet({format.resolved_token_id_});
  auto seq = grammar_builder_.AddSequence({token_set_expr});
  auto choices = grammar_builder_.AddChoices({seq});
  return ResultOk(grammar_builder_.AddRuleWithHint("token", choices));
}

Result<int, ISTError> StructuralTagGrammarConverter::VisitSub(const ExcludeTokenFormat& format) {
  std::vector<int32_t> all_excludes = format.resolved_token_ids_;
  for (auto tid : format.detected_end_token_ids_) {
    all_excludes.push_back(tid);
  }
  int expr = grammar_builder_.AddExcludeTokenSet(all_excludes);
  auto seq = grammar_builder_.AddSequence({expr});
  auto choices = grammar_builder_.AddChoices({seq});
  return ResultOk(grammar_builder_.AddRuleWithHint("exclude_token", choices));
}

Result<int, ISTError> StructuralTagGrammarConverter::VisitSub(const AnyTokensFormat& format) {
  std::vector<int32_t> all_excludes = format.resolved_exclude_token_ids_;
  for (auto tid : format.detected_end_token_ids_) {
    all_excludes.push_back(tid);
  }
  int exclude_expr = grammar_builder_.AddExcludeTokenSet(all_excludes);
  int exclude_seq = grammar_builder_.AddSequence({exclude_expr});
  int exclude_choices = grammar_builder_.AddChoices({exclude_seq});
  int inner_rule = grammar_builder_.AddRuleWithHint("any_tokens_inner", exclude_choices);
  auto inner_ref = grammar_builder_.AddRuleRef(inner_rule);
  auto star_rule_id = grammar_builder_.AddEmptyRuleWithHint("any_tokens");
  auto star_ref = grammar_builder_.AddRuleRef(star_rule_id);
  auto star_body = grammar_builder_.AddChoices(
      {grammar_builder_.AddEmptyStr(), grammar_builder_.AddSequence({inner_ref, star_ref})}
  );
  SetRuleBodyAndBudgets(star_rule_id, star_body, format.max_tokens);
  return ResultOk(star_rule_id);
}

Result<int, ISTError> StructuralTagGrammarConverter::VisitSub(const TokenTriggeredTagsFormat& format
) {
  // Step 1. Visit all tags, map trigger → tag IDs
  std::vector<std::vector<int>> trigger_to_tag_ids(format.trigger_tokens.size());
  std::vector<int> tag_content_rule_ids;
  tag_content_rule_ids.reserve(format.tags.size());

  for (int it_tag = 0; it_tag < static_cast<int>(format.tags.size()); ++it_tag) {
    const auto& tag = format.tags[it_tag];
    if (!std::holds_alternative<TokenFormat>(tag.begin)) {
      return ResultErr<ISTError>(
          "Tags in token_triggered_tags must have a token format begin, not a string"
      );
    }
    auto begin_token_id = std::get<TokenFormat>(tag.begin).resolved_token_id_;

    int matched = -1;
    for (int it_t = 0; it_t < static_cast<int>(format.resolved_trigger_token_ids_.size()); ++it_t) {
      if (format.resolved_trigger_token_ids_[it_t] == begin_token_id) {
        if (matched != -1) {
          return ResultErr<ISTError>("Tag matches multiple triggers");
        }
        matched = it_t;
      }
    }
    if (matched == -1) {
      return ResultErr<ISTError>("Tag does not match any trigger");
    }
    trigger_to_tag_ids[matched].push_back(it_tag);

    auto result = Visit(*tag.content);
    if (result.IsErr()) return result;
    tag_content_rule_ids.push_back(std::move(result).Unwrap());
  }

  // Step 2. Special case: at_least_one && stop_after_first
  if (format.at_least_one && format.stop_after_first) {
    std::vector<int> choice_elements;
    for (int it_tag = 0; it_tag < static_cast<int>(format.tags.size()); ++it_tag) {
      const auto& tag = format.tags[it_tag];
      auto begin_expr = BuildBeginExpr(tag);
      auto ref = grammar_builder_.AddRuleRef(tag_content_rule_ids[it_tag]);
      auto end_expr = BuildEndExpr(tag);
      choice_elements.push_back(grammar_builder_.AddSequence({begin_expr, ref, end_expr}));
    }
    auto choice = grammar_builder_.AddChoices(choice_elements);
    return ResultOk(grammar_builder_.AddRuleWithHint("token_triggered_tags", choice));
  }

  // Step 3. Normal case — TokenTagDispatch
  std::vector<std::pair<int32_t, int32_t>> trigger_rule_pairs;
  for (int it_t = 0; it_t < static_cast<int>(format.trigger_tokens.size()); ++it_t) {
    std::vector<int> choice_elements;
    for (auto tag_id : trigger_to_tag_ids[it_t]) {
      const auto& tag = format.tags[tag_id];
      auto ref = grammar_builder_.AddRuleRef(tag_content_rule_ids[tag_id]);
      auto end_expr = BuildEndExpr(tag);
      choice_elements.push_back(grammar_builder_.AddSequence({ref, end_expr}));
    }
    auto choice = grammar_builder_.AddChoices(choice_elements);
    auto sub_rule = grammar_builder_.AddRuleWithHint("token_triggered_tags_group", choice);
    trigger_rule_pairs.push_back({format.resolved_trigger_token_ids_[it_t], sub_rule});
  }

  bool loop = !format.stop_after_first;
  std::vector<int32_t> all_excludes = format.resolved_exclude_token_ids_;
  for (auto tid : format.detected_end_token_ids_) {
    all_excludes.push_back(tid);
  }
  auto ttd_expr = grammar_builder_.AddTokenTagDispatch(
      Grammar::Impl::TokenTagDispatch{trigger_rule_pairs, loop, all_excludes}
  );
  int32_t rule_expr_id = ttd_expr;

  if (format.at_least_one) {
    std::vector<int> first_choices;
    for (int it_tag = 0; it_tag < static_cast<int>(format.tags.size()); ++it_tag) {
      const auto& tag = format.tags[it_tag];
      auto begin_expr = BuildBeginExpr(tag);
      auto ref = grammar_builder_.AddRuleRef(tag_content_rule_ids[it_tag]);
      auto end_expr = BuildEndExpr(tag);
      first_choices.push_back(grammar_builder_.AddSequence({begin_expr, ref, end_expr}));
    }
    auto first_choice = grammar_builder_.AddChoices(first_choices);
    auto first_rule = grammar_builder_.AddRuleWithHint("token_triggered_tags_first", first_choice);
    auto dispatch_rule = grammar_builder_.AddRuleWithHint("token_triggered_tags_sub", rule_expr_id);
    auto seq = grammar_builder_.AddSequence(
        {grammar_builder_.AddRuleRef(first_rule), grammar_builder_.AddRuleRef(dispatch_rule)}
    );
    rule_expr_id = grammar_builder_.AddChoices({seq});
  }

  return ResultOk(grammar_builder_.AddRuleWithHint("token_triggered_tags", rule_expr_id));
}

Result<int, ISTError> StructuralTagGrammarConverter::VisitSub(const RepeatFormat& format) {
  auto result = Visit(*format.content);
  if (result.IsErr()) {
    return result;
  }
  int content_rule_id = std::move(result).Unwrap();
  int repeat_expr_id = grammar_builder_.AddRepeat(content_rule_id, format.min, format.max);
  return ResultOk(grammar_builder_.AddRuleWithHint("repeat", repeat_expr_id));
}

Result<int, ISTError> StructuralTagGrammarConverter::VisitSub(const DispatchFormat& format) {
  std::vector<std::pair<std::string, int32_t>> tag_rule_pairs;
  tag_rule_pairs.reserve(format.rules.size());
  for (const auto& pair : format.rules) {
    if (!pair.second) {
      return ResultErr<ISTError>("TagDispatch pair must have content");
    }
    auto result = Visit(*pair.second);
    if (result.IsErr()) {
      return result;
    }
    tag_rule_pairs.push_back({pair.first, std::move(result).Unwrap()});
  }
  auto rule_expr_id = grammar_builder_.AddTagDispatch(
      Grammar::Impl::TagDispatch{std::move(tag_rule_pairs), format.loop, format.excludes}
  );
  return ResultOk(grammar_builder_.AddRuleWithHint("tag_dispatch", rule_expr_id));
}

Result<int, ISTError> StructuralTagGrammarConverter::VisitSub(const TokenDispatchFormat& format) {
  XGRAMMAR_DCHECK(format.resolved_trigger_token_ids_.size() == format.rules.size())
      << "TokenDispatchFormat must be resolved before conversion";
  std::vector<std::pair<int32_t, int32_t>> trigger_rule_pairs;
  trigger_rule_pairs.reserve(format.rules.size());
  for (size_t i = 0; i < format.rules.size(); ++i) {
    const auto& pair = format.rules[i];
    if (!pair.second) {
      return ResultErr<ISTError>("TokenTagDispatch pair must have content");
    }
    auto result = Visit(*pair.second);
    if (result.IsErr()) {
      return result;
    }
    trigger_rule_pairs.push_back({format.resolved_trigger_token_ids_[i], std::move(result).Unwrap()}
    );
  }
  std::vector<int32_t> all_excludes = format.resolved_exclude_token_ids_;
  auto rule_expr_id = grammar_builder_.AddTokenTagDispatch(
      Grammar::Impl::TokenTagDispatch{trigger_rule_pairs, format.loop, all_excludes}
  );
  return ResultOk(grammar_builder_.AddRuleWithHint("token_tag_dispatch", rule_expr_id));
}

/************** StructuralTag Conversion Public API **************/

Result<Grammar, StructuralTagError> StructuralTagToGrammar(
    const std::string& structural_tag_json, const std::optional<TokenizerInfo>& tokenizer_info
) {
  auto structural_tag_result = StructuralTagParser::FromJSON(structural_tag_json);
  if (structural_tag_result.IsErr()) {
    return ResultErr(std::move(structural_tag_result).UnwrapErr());
  }
  auto structural_tag = std::move(structural_tag_result).Unwrap();

  auto resolve_err = StructuralTagTokenResolver::Resolve(&structural_tag, tokenizer_info);
  if (resolve_err.has_value()) {
    return ResultErr(std::move(resolve_err).value());
  }

  auto err = StructuralTagAnalyzer().Analyze(&structural_tag);
  if (err.has_value()) {
    return ResultErr(std::move(err).value());
  }

  auto result = StructuralTagGrammarConverter::Convert(structural_tag);
  if (result.IsErr()) {
    return ResultErr(std::move(result).UnwrapErr());
  }
  return ResultOk(GrammarNormalizer::Apply(std::move(result).Unwrap()));
}

}  // namespace xgrammar
