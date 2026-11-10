/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/converter_ext/cohere.cc
 * \brief Implementation of the Cohere XML Tool Calling converter.
 */
#include <picojson.h>

#include <algorithm>
#include <array>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../json_schema_converter_ext.h"
#include "../support/encoding.h"
#include "../support/json_parse.h"
#include "../support/logging.h"

namespace xgrammar {

namespace converter_ext {

XMLWrapper GetCohereXMLWrapper() { return {"<cofl:value", ">", "", "</cofl:value>"}; }

}  // namespace converter_ext

namespace {

constexpr std::array<std::pair<TCodepoint, const char*>, 4> kCohereKeyEntities = {
    std::pair<TCodepoint, const char*>{'&', "&amp;"},
    std::pair<TCodepoint, const char*>{'<', "&lt;"},
    std::pair<TCodepoint, const char*>{'>', "&gt;"},
    std::pair<TCodepoint, const char*>{'"', "&quot;"},
};

std::string SerializeCohereKeyCodepoint(TCodepoint codepoint) {
  for (const auto& [entity_codepoint, entity] : kCohereKeyEntities) {
    if (codepoint == entity_codepoint) {
      return entity;
    }
  }
  return CharToUTF8(codepoint);
}

std::vector<TCodepoint> ParseCohereKeyCodepoints(const std::string& key) {
  XGRAMMAR_CHECK(key.find('\0') == std::string::npos) << "Cohere property names cannot contain NUL";
  auto codepoints = ParseUTF8(key.c_str());
  XGRAMMAR_CHECK(codepoints.size() != 1 || codepoints[0] != CharHandlingError::kInvalidUTF8)
      << "Cohere property names must be valid UTF-8";
  return codepoints;
}

std::string SerializeCohereKey(const std::string& key) {
  std::string serialized;
  for (TCodepoint codepoint : ParseCohereKeyCodepoints(key)) {
    serialized += SerializeCohereKeyCodepoint(codepoint);
  }
  return serialized;
}

template <typename Children>
std::vector<GrammarBuilder::CharacterClassElement> CohereOrdinaryKeyRangesExcluding(
    const Children& children
) {
  constexpr TCodepoint kMaxUnicodeCodepoint = 0x10FFFF;

  // Ordinary key characters must not consume NUL, XML-sensitive characters (which are
  // represented by entity alternatives), or a codepoint handled by a child trie branch.
  std::vector<TCodepoint> excluded;
  excluded.reserve(1 + kCohereKeyEntities.size() + children.size());
  excluded.push_back('\0');
  for (const auto& entry : kCohereKeyEntities) {
    excluded.push_back(entry.first);
  }
  for (const auto& entry : children) {
    excluded.push_back(entry.first);
  }

  // Sorting makes duplicate exclusions adjacent so that unique + erase can remove them.
  std::sort(excluded.begin(), excluded.end());
  excluded.erase(std::unique(excluded.begin(), excluded.end()), excluded.end());

  // Build the positive character class as the gaps between excluded codepoints. Positive
  // ranges preserve full Unicode support when the grammar is lowered to an FSM.
  std::vector<GrammarBuilder::CharacterClassElement> ranges;
  TCodepoint range_start = 0;
  for (TCodepoint codepoint : excluded) {
    if (codepoint < range_start) {
      continue;
    }
    if (range_start < codepoint) {
      ranges.push_back({range_start, codepoint - 1});
    }
    range_start = codepoint + 1;
  }
  if (range_start <= kMaxUnicodeCodepoint) {
    ranges.push_back({range_start, kMaxUnicodeCodepoint});
  }
  return ranges;
}

}  // namespace

const std::string CohereXMLToolCallingConverter::kCohereKey = "cohere_key";
const std::string CohereXMLToolCallingConverter::kCohereAnyScalar = "cohere_any_scalar";
const std::string CohereXMLToolCallingConverter::kCohereAnyList = "cohere_any_list";

CohereXMLToolCallingConverter::CohereXMLToolCallingConverter(
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool any_whitespace,
    std::optional<int> max_whitespace_cnt,
    RefResolver ref_resolver,
    bool any_order
)
    : XMLToolCallingConverter(
          indent,
          separators,
          any_whitespace,
          max_whitespace_cnt,
          ref_resolver,
          JSONFormat::kCohereXML,
          any_order
      ) {}

void CohereXMLToolCallingConverter::AddBasicRules() {
  // Cohere's dynamic key and recursive Any rules must have stable targets before kXMLObject is
  // built, because its additional-property formatting reaches them through virtual dispatch.
  builder_.AddEmptyRule(kCohereKey);
  builder_.AddEmptyRule(kCohereAnyScalar);
  builder_.AddEmptyRule(kCohereAnyList);

  XMLToolCallingConverter::AddBasicRules();

  builder_.UpdateRuleBody(kCohereKey, RegexExpression(R"(([^\x00"&<>]|&amp;|&lt;|&gt;|&quot;)+)"));

  builder_.UpdateRuleBody(
      kCohereAnyScalar, Choice({RuleRef(kBasicNumber), RuleRef(kBasicBoolean), RuleRef(kBasicNull)})
  );
  // kXMLObject already provides named_any_value* through its dynamic-property formatting.
  // Lists need the corresponding recursive sequence of unnamed Any wrappers explicitly.
  int32_t unnamed_any_value = FormatAnyCohereParam(std::nullopt, std::nullopt);
  builder_.UpdateRuleBody(
      kCohereAnyList, Repeat("cohere_any_list_items", unnamed_any_value, 0, -1)
  );
}

bool CohereXMLToolCallingConverter::AtCohereRoot() const {
  return nested_object_level_ == 0 && object_stack_.empty() && cohere_array_level_ == 0;
}

bool CohereXMLToolCallingConverter::InCohereValueContext() const {
  return nested_object_level_ <= 1 || !object_stack_.empty() || cohere_array_level_ > 0;
}

std::string CohereXMLToolCallingConverter::RefCacheKey(const std::string& uri) const {
  // Recursive Cohere dict/list contents remain tagged values, regardless of nesting depth.
  int context = AtCohereRoot() ? 0 : (InCohereValueContext() ? 1 : 2);
  return std::to_string(context) + ":" + uri;
}

int32_t CohereXMLToolCallingConverter::FormatCohereValue(int32_t value_rule_id) {
  if (value_rule_id == builder_.GetRuleId(kXMLString)) {
    return RuleRef(value_rule_id);
  }
  return Sequence({WhitespaceExpression(), RuleRef(value_rule_id), WhitespaceExpression()});
}

std::string CohereXMLToolCallingConverter::CohereTypeForJSONLiteral(const std::string& json_value) {
  picojson::value value;
  std::string error = ParseJSON(value, json_value);
  // Const/enum object and array literals are emitted as JSON text today, not recursive Cohere
  // dict/list bodies, so only JSON strings get the raw Cohere type.
  return error.empty() && value.is<std::string>() ? "raw" : "json";
}

std::optional<std::string> CohereXMLToolCallingConverter::CommonCohereTypeForJSONLiterals(
    const std::vector<std::string>& json_values
) {
  std::optional<std::string> common_type;
  for (const auto& json_value : json_values) {
    auto type = CohereTypeForJSONLiteral(json_value);
    if (!common_type.has_value()) {
      common_type = type;
    } else if (*common_type != type) {
      return std::nullopt;
    }
  }
  return common_type;
}

int32_t CohereXMLToolCallingConverter::GetCohereTypePattern(const SchemaSpecPtr& schema) {
  return std::visit(
      [this](const auto& spec) -> int32_t {
        using T = std::decay_t<decltype(spec)>;
        if constexpr (std::is_same_v<T, StringSpec>) {
          return ByteString("raw");
        } else if constexpr (std::is_same_v<T, ObjectSpec>) {
          return ByteString("dict");
        } else if constexpr (std::is_same_v<T, ArraySpec>) {
          return ByteString("list");
        } else if constexpr (std::is_same_v<T, ConstSpec>) {
          return ByteString(CohereTypeForJSONLiteral(spec.json_value));
        } else if constexpr (std::is_same_v<T, EnumSpec>) {
          auto common_type = CommonCohereTypeForJSONLiterals(spec.json_values);
          // Mixed enums are branch-correlated by FormatCohereParam. This fallback is only used if
          // a mixed enum somehow reaches the single-wrapper path, where there is no one true type.
          return ByteString(common_type.has_value() ? *common_type : "json");
        } else {
          return ByteString("json");
        }
      },
      schema->spec
  );
}

std::optional<std::vector<SchemaSpecPtr>> CohereXMLToolCallingConverter::GetCohereCompositeOptions(
    const SchemaSpecPtr& schema
) const {
  if (schema == nullptr) {
    return std::nullopt;
  }
  return std::visit(
      [](const auto& spec) -> std::optional<std::vector<SchemaSpecPtr>> {
        using T = std::decay_t<decltype(spec)>;
        if constexpr (std::is_same_v<T, AnyOfSpec>) {
          return spec.options;
        } else if constexpr (std::is_same_v<T, OneOfSpec>) {
          return spec.options;
        } else if constexpr (std::is_same_v<T, AllOfSpec>) {
          if (spec.schemas.size() == 1) {
            return spec.schemas;
          }
          return std::nullopt;
        } else if constexpr (std::is_same_v<T, TypeArraySpec>) {
          return spec.type_schemas;
        } else if constexpr (std::is_same_v<T, EnumSpec>) {
          if (spec.json_values.empty() ||
              CommonCohereTypeForJSONLiterals(spec.json_values).has_value()) {
            return std::nullopt;
          }
          std::vector<SchemaSpecPtr> options;
          options.reserve(spec.json_values.size());
          for (size_t index = 0; index < spec.json_values.size(); ++index) {
            const auto& json_value = spec.json_values[index];
            ConstSpec const_spec;
            const_spec.json_value = json_value;
            options.push_back(
                SchemaSpec::Make(std::move(const_spec), "", "enum_case_" + std::to_string(index))
            );
          }
          return options;
        } else {
          return std::nullopt;
        }
      },
      schema->spec
  );
}

int32_t CohereXMLToolCallingConverter::CohereParamPrefix(
    const std::optional<std::string>& name, const std::optional<int32_t>& key_pattern_expr
) {
  std::vector<int32_t> elements = {ByteString(xml_wrapper_.key_wrapper_prefix)};
  if (name.has_value()) {
    elements.push_back(ByteString(" name=\"" + SerializeCohereKey(*name) + "\""));
  } else if (key_pattern_expr.has_value()) {
    elements.push_back(ByteString(" name=\""));
    elements.push_back(*key_pattern_expr);
    elements.push_back(ByteString("\""));
  }
  return Sequence(elements);
}

int32_t CohereXMLToolCallingConverter::FormatCohereSuffixWithType(
    int32_t type_expression, int32_t value_rule_id
) {
  std::vector<int32_t> elements = {
      ByteString(" type=\""), type_expression, ByteString("\"" + xml_wrapper_.key_wrapper_suffix)
  };
  if (!xml_wrapper_.value_wrapper_prefix.empty()) {
    elements.push_back(ByteString(xml_wrapper_.value_wrapper_prefix));
  }
  elements.push_back(FormatCohereValue(value_rule_id));
  elements.push_back(ByteString(xml_wrapper_.parameter_suffix));
  return Sequence(elements);
}

int32_t CohereXMLToolCallingConverter::FormatAnyCohereSuffix() {
  // kXMLAny is the aggregate body-only union. Wrapping it under every type would create a
  // type/body cross product, so each wrapper deliberately references its matching component.
  return Choice(
      {FormatCohereSuffixWithType(ByteString("raw"), builder_.GetRuleId(kXMLString)),
       FormatCohereSuffixWithType(ByteString("json"), builder_.GetRuleId(kCohereAnyScalar)),
       FormatCohereSuffixWithType(ByteString("dict"), builder_.GetRuleId(kXMLObject)),
       FormatCohereSuffixWithType(ByteString("list"), builder_.GetRuleId(kCohereAnyList))}
  );
}

int32_t CohereXMLToolCallingConverter::FormatAnyCohereParam(
    const std::optional<std::string>& name, const std::optional<int32_t>& key_pattern_expr
) {
  return Sequence({CohereParamPrefix(name, key_pattern_expr), FormatAnyCohereSuffix()});
}

int32_t CohereXMLToolCallingConverter::FormatCohereParam(
    const std::optional<std::string>& name,
    const std::optional<int32_t>& key_pattern_expr,
    const SchemaSpecPtr& schema,
    int32_t value_rule_id
) {
  return Sequence(
      {CohereParamPrefix(name, key_pattern_expr), FormatCohereParamSuffix(schema, value_rule_id)}
  );
}

int32_t CohereXMLToolCallingConverter::FormatCohereParamSuffix(
    const SchemaSpecPtr& schema, int32_t value_rule_id
) {
  // Copy the name before generating: GenerateFromSpec may add rules and reallocate the
  // builder's rule storage, invalidating references into it.
  std::string value_rule_name = builder_.GetRule(value_rule_id).name;
  if (const auto* ref = std::get_if<RefSpec>(&schema->spec); ref != nullptr) {
    auto cached = cohere_param_ref_rules_.find(ref->uri);
    if (cached != cohere_param_ref_rules_.end()) {
      return RuleRef(cached->second);
    }
    // Register the rule before resolving the reference. A schema that leads back to this URI,
    // directly or through nested dict/list items, then reuses the rule instead of expanding
    // again without bound, and shared acyclic references are built only once.
    int32_t param_rule_id = builder_.AddEmptyRuleWithHint(value_rule_name + "_cohere_param");
    cohere_param_ref_rules_.emplace(ref->uri, param_rule_id);
    SchemaSpecPtr resolved = ResolveRefSchema(*ref, value_rule_name);
    builder_.UpdateRuleBody(param_rule_id, FormatCohereParamSuffix(resolved, value_rule_id));
    return RuleRef(param_rule_id);
  }

  // CreateRule may return any aggregate rule (cached or freshly generated), but the schema is
  // retained along this call path so Any can select correlated wrappers here.
  if (std::holds_alternative<AnySpec>(schema->spec)) {
    return FormatAnyCohereSuffix();
  }
  if (const auto* all_of = std::get_if<AllOfSpec>(&schema->spec);
      all_of != nullptr && all_of->schemas.size() != 1) {
    // The base converter intentionally falls back to Any while multi-branch allOf support is
    // incomplete. Keep that fallback canonical instead of wrapping its aggregate body as json.
    return FormatAnyCohereSuffix();
  }

  auto options = GetCohereCompositeOptions(schema);
  if (!options.has_value()) {
    return FormatCohereSuffixWithType(GetCohereTypePattern(schema), value_rule_id);
  }

  std::vector<int32_t> choices;
  choices.reserve(options->size());
  for (size_t index = 0; index < options->size(); ++index) {
    const SchemaSpecPtr& option = (*options)[index];
    int32_t option_rule_id =
        CreateRule(option, value_rule_name + "_cohere_case_" + std::to_string(index));
    choices.push_back(FormatCohereParamSuffix(option, option_rule_id));
  }
  return choices.size() == 1 ? choices[0] : Choice(choices);
}

int32_t CohereXMLToolCallingConverter::GenerateString(
    const StringSpec& spec, const std::string& rule_name
) {
  if (!InCohereValueContext()) {
    return JSONSchemaConverter::GenerateString(spec, rule_name);
  }
  if (!spec.pattern.has_value() && !spec.format.has_value() && spec.min_length == 0 &&
      spec.max_length == -1) {
    return RuleRef(kXMLString);
  }
  if (spec.format.has_value()) {
    const std::string& format = *spec.format;
    auto regex_pattern = JSONFormatToRegexPattern(format);
    if (regex_pattern.has_value()) {
      return RegexExpression(regex_pattern.value(), false, true);
    }
  }
  if (spec.pattern.has_value()) {
    return RegexExpression(*spec.pattern, false, /*force_cfg_expansion=*/true);
  }
  if (spec.min_length != 0 || spec.max_length != -1) {
    return Repeat(
        rule_name + "_characters",
        builder_.AddCharacterClass({{0, 0x10ffff}}),
        spec.min_length,
        spec.max_length
    );
  }
  return JSONSchemaConverter::GenerateString(spec, rule_name);
}

int32_t CohereXMLToolCallingConverter::GenerateAny(
    const AnySpec& spec, const std::string& rule_name
) {
  if (!InCohereValueContext()) {
    return JSONSchemaConverter::GenerateAny(spec, rule_name);
  }
  if (AtCohereRoot()) {
    return RuleRef(kXMLObject);
  }
  return Choice(
      {RuleRef(kXMLString), RuleRef(kCohereAnyScalar), RuleRef(kXMLObject), RuleRef(kCohereAnyList)}
  );
}

int32_t CohereXMLToolCallingConverter::GenerateConst(
    const ConstSpec& spec, const std::string& rule_name
) {
  if (!InCohereValueContext()) {
    return JSONSchemaConverter::GenerateConst(spec, rule_name);
  }
  return ByteString(XMLValue(spec.json_value));
}

int32_t CohereXMLToolCallingConverter::GenerateEnum(
    const EnumSpec& spec, const std::string& rule_name
) {
  XGRAMMAR_DCHECK(!spec.json_values.empty())
      << "GenerateEnum called with empty enum spec for rule: " << rule_name;
  if (!InCohereValueContext()) {
    return JSONSchemaConverter::GenerateEnum(spec, rule_name);
  }
  std::vector<int32_t> values;
  values.reserve(spec.json_values.size());
  for (const auto& value : spec.json_values) {
    values.push_back(ByteString(XMLValue(value)));
  }
  return Choice(values);
}

int32_t CohereXMLToolCallingConverter::GenerateObject(
    const ObjectSpec& spec, const std::string& rule_name, bool dummy_need_braces
) {
  nested_object_level_++;
  bool use_cohere_object = InCohereValueContext();

  int32_t result;
  if (use_cohere_object) {
    SchemaSpecPtr additional_property;
    if (spec.allow_additional_properties && spec.additional_properties_schema) {
      additional_property = spec.additional_properties_schema;
    } else if (spec.allow_unevaluated_properties && spec.unevaluated_properties_schema) {
      additional_property = spec.unevaluated_properties_schema;
    } else if (spec.allow_additional_properties || spec.allow_unevaluated_properties) {
      additional_property = SchemaSpec::Make(AnySpec{}, "", "any");
    }

    object_stack_.push_back(&spec);
    additional_property_stack_.push_back(additional_property);
    result = JSONSchemaConverter::GenerateObject(spec, rule_name, false);
    additional_property_stack_.pop_back();
    object_stack_.pop_back();
  } else {
    result = JSONSchemaConverter::GenerateObject(spec, rule_name, nested_object_level_ > 1);
  }

  nested_object_level_--;
  return result;
}

int32_t CohereXMLToolCallingConverter::GenerateArray(
    const ArraySpec& spec, const std::string& rule_name
) {
  if (!InCohereValueContext()) {
    nested_object_level_++;
    auto result = JSONSchemaConverter::GenerateArray(spec, rule_name);
    nested_object_level_--;
    return result;
  }

  cohere_array_level_++;
  std::vector<int32_t> item_patterns;
  for (size_t i = 0; i < spec.prefix_items.size(); ++i) {
    int32_t item_rule_id =
        CreateRule(spec.prefix_items[i], rule_name + "_item_" + std::to_string(i));
    item_patterns.push_back(
        FormatCohereParam(std::nullopt, std::nullopt, spec.prefix_items[i], item_rule_id)
    );
  }

  std::optional<int32_t> additional_item_pattern;
  if (spec.allow_additional_items && spec.additional_items) {
    int32_t additional_rule_id = CreateRule(spec.additional_items, rule_name + "_additional");
    additional_item_pattern =
        FormatCohereParam(std::nullopt, std::nullopt, spec.additional_items, additional_rule_id);
  }
  cohere_array_level_--;

  if (item_patterns.empty()) {
    if (!additional_item_pattern.has_value() || spec.max_items == 0) {
      return Empty();
    }
    return Repeat(
        rule_name + "_items",
        *additional_item_pattern,
        static_cast<int>(spec.min_items),
        spec.max_items == -1 ? -1 : static_cast<int>(spec.max_items)
    );
  }

  int32_t prefix_part = Sequence(item_patterns);
  if (!additional_item_pattern.has_value()) {
    return prefix_part;
  }

  int64_t min_additional = std::max(
      static_cast<int64_t>(0), spec.min_items - static_cast<int64_t>(item_patterns.size())
  );
  int64_t max_additional =
      spec.max_items == -1 ? -1 : spec.max_items - static_cast<int64_t>(item_patterns.size());
  return Sequence(
      {prefix_part,
       Repeat(
           rule_name + "_additional_items",
           *additional_item_pattern,
           static_cast<int>(min_additional),
           max_additional == -1 ? -1 : static_cast<int>(max_additional)
       )}
  );
}

int32_t CohereXMLToolCallingConverter::FormatProperty(
    const std::string& key,
    int32_t value_rule_id,
    const std::string& rule_name,
    int64_t idx,
    const SchemaSpecPtr& schema
) {
  if (!object_stack_.empty() && idx >= 0 &&
      idx < static_cast<int64_t>(object_stack_.back()->properties.size())) {
    const auto& prop = object_stack_.back()->properties[idx];
    return FormatCohereParam(prop.name, std::nullopt, prop.schema, value_rule_id);
  }
  return XMLToolCallingConverter::FormatProperty(key, value_rule_id, rule_name, idx, schema);
}

int32_t CohereXMLToolCallingConverter::FormatOtherProperty(
    int32_t key_pattern_expr,
    int32_t value_rule_id,
    const std::string& rule_name,
    const std::string& rule_name_suffix,
    const SchemaSpecPtr& schema
) {
  SchemaSpecPtr value_schema = schema;
  if (!value_schema && !additional_property_stack_.empty()) {
    value_schema = additional_property_stack_.back();
  }
  if (!value_schema && InCohereValueContext()) {
    value_schema = SchemaSpec::Make(AnySpec{}, "", "any");
    value_rule_id = CreateRule(value_schema, rule_name + "_" + rule_name_suffix + "_cohere_any");
  }
  if (value_schema) {
    return FormatCohereParam(std::nullopt, key_pattern_expr, value_schema, value_rule_id);
  }
  return XMLToolCallingConverter::FormatOtherProperty(
      key_pattern_expr, value_rule_id, rule_name, rule_name_suffix, schema
  );
}

std::string CohereXMLToolCallingConverter::GetKeyPattern() const {
  if (InCohereValueContext()) {
    return kCohereKey;
  }
  return JSONSchemaConverter::GetKeyPattern();
}

int32_t CohereXMLToolCallingConverter::BuildCohereKeyExcludingBody(
    const CohereKeyTrieNode& node, int depth
) {
  std::vector<int32_t> choices;
  if (depth > 0 && !node.is_terminal) {
    choices.push_back(Empty());
  }

  int32_t optional_key_suffix = Choice({Empty(), RuleRef(kCohereKey)});
  int32_t ordinary_key_unit =
      builder_.AddCharacterClass(CohereOrdinaryKeyRangesExcluding(node.children));
  choices.push_back(Sequence({ordinary_key_unit, optional_key_suffix}));
  for (const auto& [codepoint, entity] : kCohereKeyEntities) {
    if (!node.children.count(codepoint)) {
      int32_t entity_key_unit = ByteString(entity);
      choices.push_back(Sequence({entity_key_unit, optional_key_suffix}));
    }
  }

  for (const auto& [codepoint, child] : node.children) {
    choices.push_back(Sequence(
        {ByteString(SerializeCohereKeyCodepoint(codepoint)),
         BuildCohereKeyExcludingBody(child, depth + 1)}
    ));
  }

  return Choice(choices);
}

int32_t CohereXMLToolCallingConverter::GetKeyPatternExcluding(
    const std::vector<ObjectSpec::Property>& properties, const std::string& rule_name
) {
  if (InCohereValueContext()) {
    if (properties.empty()) {
      return RuleRef(GetKeyPattern());
    }
    CohereKeyTrieNode root;
    for (const auto& prop : properties) {
      CohereKeyTrieNode* cur = &root;
      auto codepoints = ParseCohereKeyCodepoints(prop.name);
      for (TCodepoint codepoint : codepoints) {
        cur = &cur->children[codepoint];
      }
      if (!codepoints.empty()) {
        cur->is_terminal = true;
      }
    }
    int32_t key_rule_id = builder_.AddEmptyRuleWithHint(rule_name + "_cohere_addl_key");
    builder_.UpdateRuleBody(key_rule_id, BuildCohereKeyExcludingBody(root, 0));
    return RuleRef(key_rule_id);
  }
  return JSONSchemaConverter::GetKeyPatternExcluding(properties, rule_name);
}

std::string CohereXMLToolCallingConverter::NextSeparator(bool is_end) {
  if (InCohereValueContext()) {
    return GetWhitespacePattern();
  }
  return JSONSchemaConverter::NextSeparator(is_end);
}

void CohereXMLToolCallingConverter::AddCache(const std::string& key, int32_t rule_id) {
  if (key.empty()) {
    return;
  }
  rule_cache_manager_.AddCache(key, nested_object_level_ > 1 && !InCohereValueContext(), rule_id);
}

std::optional<int32_t> CohereXMLToolCallingConverter::GetCache(const std::string& key) const {
  if (key.empty()) {
    return std::nullopt;
  }
  // "true" and {} are equivalent schemas. At the tool-arguments root both are unrestricted
  // dictionaries, while nested {} keeps using the aggregate Any body rule.
  if (AtCohereRoot() && (key == "{}" || key == "true")) {
    return builder_.GetRuleId(kXMLObject);
  }
  return rule_cache_manager_.GetCache(key, nested_object_level_ > 1 && !InCohereValueContext());
}

}  // namespace xgrammar
