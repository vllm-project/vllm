/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/json_schema_converter_ext.h
 * \brief Extended format converters for JSON Schema, including XML Tool Calling format.
 */

#ifndef XGRAMMAR_JSON_SCHEMA_CONVERTER_EXT_H_
#define XGRAMMAR_JSON_SCHEMA_CONVERTER_EXT_H_

#include <algorithm>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "json_schema_converter.h"

namespace xgrammar {

namespace converter_ext {

// Wrapper strings for XML parameter tags.
struct XMLWrapper {
  std::string key_wrapper_prefix;
  std::string key_wrapper_suffix;
  std::string value_wrapper_prefix;
  std::string parameter_suffix;
};

}  // namespace converter_ext

/*!
 * \brief Converter for MiniMax M3's recursive namespace-prefixed XML format.
 *
 * This initial implementation supports schemas whose object property names are
 * known when the grammar is built. Schemas requiring runtime element names are
 * rejected explicitly.
 */
class MiniMaxM3XMLToolCallingConverter : public JSONSchemaConverter {
 public:
  MiniMaxM3XMLToolCallingConverter(
      std::optional<int> indent,
      std::optional<std::pair<std::string, std::string>> separators,
      bool any_whitespace,
      std::optional<int> max_whitespace_cnt,
      RefResolver ref_resolver = nullptr,
      bool any_order = false
  );

 protected:
  int32_t GenerateString(const StringSpec& spec, const std::string& rule_name) override;
  int32_t GenerateArray(const ArraySpec& spec, const std::string& rule_name) override;
  int32_t GenerateObject(
      const ObjectSpec& spec, const std::string& rule_name, bool dummy_need_braces = false
  ) override;
  int32_t GenerateAny(const AnySpec& spec, const std::string& rule_name) override;
  int32_t GenerateConst(const ConstSpec& spec, const std::string& rule_name) override;
  int32_t GenerateEnum(const EnumSpec& spec, const std::string& rule_name) override;

  int32_t FormatProperty(
      const std::string& key,
      int32_t value_rule_id,
      const std::string& rule_name,
      int64_t idx,
      const SchemaSpecPtr& schema
  ) override;
  std::string NextSeparator(bool is_end = false) override;
  void AddBasicRules() override;

 private:
  int32_t FormatElement(const std::string& name, int32_t value_rule_id);
  int32_t GenerateLiteral(const picojson::value& value);
  void ValidateObject(const ObjectSpec& spec) const;
  static void ValidateElementName(const std::string& name);
};

/*!
 * \brief Converter for XML Tool Calling format (e.g., Qwen style).
 *
 * This converter generates a grammar where:
 * - The outermost object uses XML format: <parameter=name>value</parameter>
 * - Inner values use standard JSON format
 */
class XMLToolCallingConverter : public JSONSchemaConverter {
 public:
  XMLToolCallingConverter(
      std::optional<int> indent,
      std::optional<std::pair<std::string, std::string>> separators,
      bool any_whitespace,
      std::optional<int> max_whitespace_cnt,
      RefResolver ref_resolver = nullptr,
      JSONFormat json_format = JSONFormat::kQwenXML,
      bool any_order = false,
      std::vector<std::string> excludes = {}
  );

  /*! \brief Convert SchemaSpec to grammar with XML format for root object. Note that this function
   * is not thread-safe.*/
  Grammar Convert(const SchemaSpecPtr& spec);

 protected:
  // Override methods for XML format
  int32_t GenerateString(const StringSpec& spec, const std::string& rule_name) override;
  int32_t GenerateObject(
      const ObjectSpec& spec, const std::string& rule_name, bool dummy_need_braces = false
  ) override;
  int32_t GenerateAny(const AnySpec& spec, const std::string& rule_name) override;
  int32_t GenerateArray(const ArraySpec& spec, const std::string& rule_name) override;
  int32_t GenerateConst(const ConstSpec& spec, const std::string& rule_name) override;
  int32_t GenerateEnum(const EnumSpec& spec, const std::string& rule_name) override;

  // Override format hooks
  int32_t FormatPropertyKey(const std::string& key, const SchemaSpecPtr& schema) override;
  int32_t FormatProperty(
      const std::string& key,
      int32_t value_rule_id,
      const std::string& rule_name,
      int64_t idx,
      const SchemaSpecPtr& schema
  ) override;
  int32_t FormatOtherProperty(
      int32_t key_pattern_expr,
      int32_t value_rule_id,
      const std::string& rule_name,
      const std::string& rule_name_suffix,
      const SchemaSpecPtr& schema
  ) override;

  std::string GetKeyPattern() const override;
  std::string GetBasicAnyRuleName() const override;
  int32_t GetKeyPatternExcluding(
      const std::vector<ObjectSpec::Property>& properties, const std::string& rule_name
  ) override;

  std::string NextSeparator(bool is_end = false) override;

  void AddBasicRules() override;

  void AddCache(const std::string& key, int32_t rule_id) override;
  std::optional<int32_t> GetCache(const std::string& key) const override;
  std::string RefCacheKey(const std::string& uri) const override;

 protected:
  using XMLWrapper = converter_ext::XMLWrapper;

  static const std::unordered_map<JSONFormat, XMLWrapper> kKeyWrapperMap;
  static const std::string kXMLString;
  static const std::string kXMLAny;
  static const std::string kXMLObject;
  static const std::string kXMLVariableName;

  std::string XMLValue(const std::string& json_value) const;
  std::string EscapeAttrValue(const std::string& value) const;

  /*!
   * \brief Return a single rendered JSON type when it can be determined without resolving
   * references or combinators, or std::nullopt otherwise (\p spec may be nullptr).
   *
   * Shared by Kimi-K3 type attributes and DeepSeek string attributes. Both integer and
   * number schemas render as "number"; string values are rendered as raw text.
   */
  static std::optional<std::string> GetRenderedJSONType(const SchemaSpecPtr& spec);

  /*!
   * \brief Build the expression between the property key and its value.
   * \param pinned_type For kimi_k3_xml, the single type attribute this property must carry.
   * std::nullopt keeps every type allowed, which is what free-form keys
   * (additionalProperties / patternProperties) need.
   */
  int32_t XMLKeySuffix(const std::optional<std::string>& pinned_type = std::nullopt);

  /*!
   * \brief Build a DeepSeek XML parameter's string attribute, value and closing tag.
   * string="true" wraps raw strings, string="false" wraps JSON values. Unions and mixed enums
   * produce one alternative per option; GetRenderedJSONType supplies the type classification.
   * A negative value_rule_id defers value-rule creation until a typed leaf is reached.
   */
  int32_t FormatDeepSeekParamSuffix(
      const SchemaSpecPtr& schema, int32_t value_rule_id, const std::string& rule_name_hint = ""
  );

  // Parameter suffix rules are independent of the key, so references can be shared across
  // named and dynamic parameters. Allocate them before resolving refs to handle cycles.
  std::unordered_map<std::string, int32_t> deepseek_param_ref_rules_;

  JSONFormat json_format_;
  // Root parameter lists, raw parameter values, and nested JSON have distinct grammars.
  int EncodingContext() const { return std::min(nested_object_level_, 2); }
  // Track if we're at the root object level
  int nested_object_level_ = 0;
  const XMLWrapper xml_wrapper_;
};

/*!
 * \brief Converter for Cohere XML Tool Calling format.
 *
 * This converter generates recursive Cohere value tags:
 * <cofl:value name="key" type="raw|json|dict|list">value</cofl:value>.
 * Object properties use named value tags. Array items use unnamed value tags.
 */
class CohereXMLToolCallingConverter : public XMLToolCallingConverter {
 public:
  CohereXMLToolCallingConverter(
      std::optional<int> indent,
      std::optional<std::pair<std::string, std::string>> separators,
      bool any_whitespace,
      std::optional<int> max_whitespace_cnt,
      RefResolver ref_resolver = nullptr,
      bool any_order = false
  );

 protected:
  int32_t GenerateString(const StringSpec& spec, const std::string& rule_name) override;
  int32_t GenerateObject(
      const ObjectSpec& spec, const std::string& rule_name, bool dummy_need_braces = false
  ) override;
  int32_t GenerateAny(const AnySpec& spec, const std::string& rule_name) override;
  int32_t GenerateArray(const ArraySpec& spec, const std::string& rule_name) override;
  int32_t GenerateConst(const ConstSpec& spec, const std::string& rule_name) override;
  int32_t GenerateEnum(const EnumSpec& spec, const std::string& rule_name) override;

  int32_t FormatProperty(
      const std::string& key,
      int32_t value_rule_id,
      const std::string& rule_name,
      int64_t idx,
      const SchemaSpecPtr& schema
  ) override;
  int32_t FormatOtherProperty(
      int32_t key_pattern_expr,
      int32_t value_rule_id,
      const std::string& rule_name,
      const std::string& rule_name_suffix,
      const SchemaSpecPtr& schema
  ) override;

  std::string GetKeyPattern() const override;
  int32_t GetKeyPatternExcluding(
      const std::vector<ObjectSpec::Property>& properties, const std::string& rule_name
  ) override;
  std::string NextSeparator(bool is_end = false) override;

  void AddBasicRules() override;
  void AddCache(const std::string& key, int32_t rule_id) override;
  std::optional<int32_t> GetCache(const std::string& key) const override;
  std::string RefCacheKey(const std::string& uri) const override;

 private:
  struct CohereKeyTrieNode {
    bool is_terminal = false;
    std::map<int32_t, CohereKeyTrieNode> children;
  };

  static const std::string kCohereKey;
  static const std::string kCohereAnyScalar;
  static const std::string kCohereAnyList;

  /*! \brief `<cofl:value` plus the optional ` name="..."` attribute of one parameter. */
  int32_t CohereParamPrefix(
      const std::optional<std::string>& name, const std::optional<int32_t>& key_pattern_expr
  );
  /*!
   * \brief The type attribute, value and closing tag of one parameter, correlated with
   * \p schema. The suffix does not depend on the parameter name, so `$ref` schemas get one
   * memoized rule per URI that is registered before the reference is resolved; recursive
   * references (directly, or through nested dict/list items) then point back at that rule.
   */
  int32_t FormatCohereParamSuffix(const SchemaSpecPtr& schema, int32_t value_rule_id);
  int32_t FormatCohereSuffixWithType(int32_t type_expression, int32_t value_rule_id);
  int32_t FormatAnyCohereSuffix();
  int32_t FormatCohereParam(
      const std::optional<std::string>& name,
      const std::optional<int32_t>& key_pattern_expr,
      const SchemaSpecPtr& schema,
      int32_t value_rule_id
  );
  int32_t FormatAnyCohereParam(
      const std::optional<std::string>& name, const std::optional<int32_t>& key_pattern_expr
  );
  int32_t FormatCohereValue(int32_t value_rule_id);
  int32_t GetCohereTypePattern(const SchemaSpecPtr& schema);
  static std::string CohereTypeForJSONLiteral(const std::string& json_value);
  static std::optional<std::string> CommonCohereTypeForJSONLiterals(
      const std::vector<std::string>& json_values
  );
  std::optional<std::vector<SchemaSpecPtr>> GetCohereCompositeOptions(const SchemaSpecPtr& schema
  ) const;
  int32_t BuildCohereKeyExcludingBody(const CohereKeyTrieNode& node, int depth);
  bool AtCohereRoot() const;
  bool InCohereValueContext() const;

  std::vector<const ObjectSpec*> object_stack_;
  std::vector<SchemaSpecPtr> additional_property_stack_;
  // Parameter suffix rules keyed by `$ref` URI; see FormatCohereParamSuffix.
  std::unordered_map<std::string, int32_t> cohere_param_ref_rules_;
  int cohere_array_level_ = 0;
};

namespace converter_ext {

XMLWrapper GetQwenXMLWrapper();
XMLWrapper GetMiniMaxXMLWrapper();
XMLWrapper GetDeepSeekXMLWrapper();
XMLWrapper GetDeepSeekV41XMLWrapper();
XMLWrapper GetGLMXMLWrapper();
XMLWrapper GetCohereXMLWrapper();
XMLWrapper GetKimiK3XMLWrapper();

struct XMLKeySuffix {
  const char* prefix;
  std::vector<const char*> values;
  const char* suffix;
};

const XMLKeySuffix& GetKimiK3XMLKeySuffix();

}  // namespace converter_ext
}  // namespace xgrammar

#endif  // XGRAMMAR_JSON_SCHEMA_CONVERTER_EXT_H_
