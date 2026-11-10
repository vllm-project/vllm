/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/json_schema_converter.h
 * \brief Convert a JSON Schema directly to a grammar AST.
 */

#ifndef XGRAMMAR_JSON_SCHEMA_CONVERTER_H_
#define XGRAMMAR_JSON_SCHEMA_CONVERTER_H_

#include <picojson.h>
#include <xgrammar/grammar.h>

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "grammar_builder.h"
#include "support/utils.h"

namespace xgrammar {

// ==================== SchemaSpec: Intermediate Representation for JSON Schema ====================

// Forward declaration
struct SchemaSpec;
using SchemaSpecPtr = std::shared_ptr<SchemaSpec>;

// Basic Type Specs
struct IntegerSpec {
  std::optional<int64_t> minimum;
  std::optional<int64_t> maximum;
  std::optional<int64_t> exclusive_minimum;
  std::optional<int64_t> exclusive_maximum;
  std::optional<int64_t> multiple_of;

  std::string ToString() const;
};

struct NumberSpec {
  std::optional<double> minimum;
  std::optional<double> maximum;
  std::optional<double> exclusive_minimum;
  std::optional<double> exclusive_maximum;

  std::string ToString() const;
};

struct StringSpec {
  std::optional<std::string> pattern;
  std::optional<std::string> format;
  int min_length = 0;
  int max_length = -1;  // -1 means no limit

  std::string ToString() const;
};

struct BooleanSpec {
  std::string ToString() const;
};

struct NullSpec {
  std::string ToString() const;
};

struct AnySpec {
  std::string ToString() const;
};

// Complex Type Specs
struct ArraySpec {
  std::vector<SchemaSpecPtr> prefix_items;
  bool allow_additional_items = true;
  SchemaSpecPtr additional_items;  // nullptr means not allowed
  int64_t min_items = 0;
  int64_t max_items = -1;  // -1 means no limit

  std::string ToString() const;
};

struct ObjectSpec {
  struct Property {
    std::string name;
    SchemaSpecPtr schema;
  };

  struct PatternProperty {
    std::string pattern;  // regex pattern for key
    SchemaSpecPtr schema;
  };

  std::vector<Property> properties;
  std::vector<PatternProperty> pattern_properties;
  std::unordered_set<std::string> required;

  bool allow_additional_properties = false;
  SchemaSpecPtr additional_properties_schema;
  bool allow_unevaluated_properties = true;
  SchemaSpecPtr unevaluated_properties_schema;
  SchemaSpecPtr property_names;

  int min_properties = 0;
  int max_properties = -1;  // -1 means no limit

  std::string ToString() const;
};

// Composite Type Specs
struct ConstSpec {
  std::string json_value;  // JSON serialized value

  std::string ToString() const;
};

struct EnumSpec {
  std::vector<std::string> json_values;  // JSON serialized values

  std::string ToString() const;
};

struct RefSpec {
  std::string uri;

  std::string ToString() const;
};

struct AnyOfSpec {
  std::vector<SchemaSpecPtr> options;

  std::string ToString() const;
};

struct OneOfSpec {
  std::vector<SchemaSpecPtr> options;

  std::string ToString() const;
};

struct AllOfSpec {
  std::vector<SchemaSpecPtr> schemas;

  std::string ToString() const;
};

struct TypeArraySpec {
  // Handle "type": ["string", "integer"] cases
  std::vector<SchemaSpecPtr> type_schemas;

  std::string ToString() const;
};

// Unified SchemaSpec
using SchemaSpecVariant = std::variant<
    IntegerSpec,
    NumberSpec,
    StringSpec,
    BooleanSpec,
    NullSpec,
    ArraySpec,
    ObjectSpec,
    AnySpec,
    ConstSpec,
    EnumSpec,
    RefSpec,
    AnyOfSpec,
    OneOfSpec,
    AllOfSpec,
    TypeArraySpec>;

struct SchemaSpec {
  SchemaSpecVariant spec;
  std::string cache_key;       // for deduplication
  std::string rule_name_hint;  // suggested rule name

  std::string ToString() const;

  // Helper method to create SchemaSpec
  template <typename T>
  static SchemaSpecPtr Make(T&& spec_value, std::string cache_key = "", std::string hint = "") {
    auto ptr = std::make_shared<SchemaSpec>();
    ptr->spec = std::forward<T>(spec_value);
    ptr->cache_key = std::move(cache_key);
    ptr->rule_name_hint = std::move(hint);
    return ptr;
  }
};

// ==================== JSONFormat Enum ====================

enum class JSONFormat : int {
  kJSON = 0,
  kQwenXML = 1,
  kMiniMaxXML = 2,
  kDeepSeekXML = 3,
  kGlmXML = 4,
  kCohereXML = 5,
  kKimiK3XML = 6,
  kMiniMaxM3XML = 7,
  kDeepSeekV41XML = 8,
};

/*!
 * \brief Convert a format name to JSONFormat.
 * \param format One of "json", "qwen_xml", "minimax_xml", "minimax_m3_xml", "deepseek_xml",
 * "glm_xml", "cohere_xml", "kimi_k3_xml", or "deepseek_v4_1_xml".
 * \return The corresponding JSONFormat, or std::nullopt if the name is not recognized.
 */
std::optional<JSONFormat> JSONFormatFromString(const std::string& format);

/*!
 * \brief Manage the rule generation cache. Wraps key-value cache for schema deduplication.
 * The cached value is the rule id in the grammar builder.
 */
class GenerateCacheManager {
 public:
  /*! \brief Add a key-value pair to the cache. */
  void AddCache(const std::string& key, int context, int32_t rule_id) {
    cache_[{key, context}] = rule_id;
  }

  /*! \brief Get cached rule id by key. Returns std::nullopt if not found. */
  std::optional<int32_t> GetCache(const std::string& key, int context) const {
    auto it = cache_.find({key, context});
    if (it != cache_.end()) {
      return it->second;
    }
    return std::nullopt;
  }

 private:
  std::unordered_map<std::pair<std::string, int>, int32_t> cache_;
};

/*!
 * \brief Manage the indent and separator for the generation of EBNF grammar.
 */
class IndentManager {
 public:
  IndentManager(
      std::optional<int> indent,
      const std::string& separator,
      bool any_whitespace,
      std::optional<int> max_whitespace_cnt
  );

  void StartIndent();
  void EndIndent();
  std::string StartSeparator();
  std::string MiddleSeparator();
  std::string EndSeparator();
  std::string EmptySeparator();
  std::string NextSeparator(bool is_end = false);

 private:
  bool any_whitespace_;
  bool enable_newline_;
  int64_t indent_;
  std::string separator_;
  int64_t total_indent_;
  std::vector<bool> is_first_;
  std::optional<int> max_whitespace_cnt_;

  friend class JSONSchemaConverter;
};

/*!
 * \brief Convert SchemaSpec directly to a grammar AST.
 *
 * This is the base class for grammar generation. It generates JSON-format grammar by default.
 * Subclasses can override virtual methods to generate different formats (e.g., XML).
 */
class JSONSchemaConverter {
 public:
  using RefResolver =
      std::function<SchemaSpecPtr(const std::string& uri, const std::string& rule_name_hint)>;

  JSONSchemaConverter(
      std::optional<int> indent,
      std::optional<std::pair<std::string, std::string>> separators,
      bool any_whitespace,
      std::optional<int> max_whitespace_cnt,
      RefResolver ref_resolver = nullptr,
      bool any_order = false,
      std::vector<std::string> excludes = {}
  );

  virtual ~JSONSchemaConverter() = default;

  /*!
   * \brief Convert SchemaSpec directly to a grammar AST.
   * \param spec The SchemaSpec to convert.
   * \return The grammar AST.
   */
  Grammar Convert(const SchemaSpecPtr& spec);

  /*! \brief Whether \p format is compiled to a regex, which shadows minLength/maxLength. */
  static bool IsBuiltinFormat(const std::string& format) {
    return JSONFormatToRegexPattern(format).has_value();
  }

 protected:
  using CharacterClassElement = GrammarBuilder::CharacterClassElement;

  // ==================== Virtual methods for generation ====================
  // Subclasses can override these to customize output format

  virtual int32_t GenerateInteger(const IntegerSpec& spec, const std::string& rule_name);
  virtual int32_t GenerateNumber(const NumberSpec& spec, const std::string& rule_name);
  virtual int32_t GenerateString(const StringSpec& spec, const std::string& rule_name);
  virtual int32_t GenerateBoolean(const BooleanSpec& spec, const std::string& rule_name);
  virtual int32_t GenerateNull(const NullSpec& spec, const std::string& rule_name);
  virtual int32_t GenerateArray(const ArraySpec& spec, const std::string& rule_name);
  virtual int32_t GenerateObject(
      const ObjectSpec& spec, const std::string& rule_name, bool need_brace = true
  );
  virtual int32_t GenerateAny(const AnySpec& spec, const std::string& rule_name);
  virtual int32_t GenerateConst(const ConstSpec& spec, const std::string& rule_name);
  virtual int32_t GenerateEnum(const EnumSpec& spec, const std::string& rule_name);
  virtual int32_t GenerateRef(const RefSpec& spec, const std::string& rule_name);
  /*! \brief Key reference rules by their output encoding context. */
  virtual std::string RefCacheKey(const std::string& uri) const;
  virtual int32_t GenerateAnyOf(const AnyOfSpec& spec, const std::string& rule_name);
  virtual int32_t GenerateOneOf(const OneOfSpec& spec, const std::string& rule_name);
  virtual int32_t GenerateAllOf(const AllOfSpec& spec, const std::string& rule_name);
  virtual int32_t GenerateTypeArray(const TypeArraySpec& spec, const std::string& rule_name);

  // ==================== Hooks for customization ====================

  /*!
   * \brief Format a property key. Override for different formats.
   * \param schema The schema of the property's value. Formats that encode the value's type
   * next to the key (e.g. the Kimi-K3 `type` attribute) need it; the JSON format ignores it.
   */
  virtual int32_t FormatPropertyKey(const std::string& key, const SchemaSpecPtr& schema);

  /*! \brief Format a property (key + value). Override for different formats. */
  virtual int32_t FormatProperty(
      const std::string& key,
      int32_t value_rule_id,
      const std::string& rule_name,
      int64_t idx,
      const SchemaSpecPtr& schema
  );

  /*!
   * \brief Format an "other" property (additional/unevaluated). Override for different formats.
   * \param schema The schema selected for the property's value, or nullptr when the dynamic key
   * has no single schema. Formats that encode the value's type next to the key need it.
   */
  virtual int32_t FormatOtherProperty(
      int32_t key_pattern_expr,
      int32_t value_rule_id,
      const std::string& rule_name,
      const std::string& rule_name_suffix,
      const SchemaSpecPtr& schema
  );

  /*! \brief Get the basic string rule name. Override for different formats. */
  virtual std::string GetKeyPattern() const;

  /*! \brief Get a key pattern that excludes specific property names. */
  virtual int32_t GetKeyPatternExcluding(
      const std::vector<ObjectSpec::Property>& properties, const std::string& rule_name
  );

  /*! \brief Get the basic any rule name. Override for different formats. */
  virtual std::string GetBasicAnyRuleName() const;

  /*! \brief Add basic rules for the format. Override for different formats. */
  virtual void AddBasicRules();
  void AddBasicRules(const std::vector<std::string>& additional_rule_names);

  /*! \brief Add a key-value pair to the generation cache. Override for custom cache behavior. */
  virtual void AddCache(const std::string& key, int32_t rule_id);

  /*! \brief Get cached value by key. Returns std::nullopt if not found. */
  virtual std::optional<int32_t> GetCache(const std::string& key) const;

  // ==================== Helper methods (for subclasses to use) ====================

  /*! \brief Dispatch to the appropriate Generate method based on spec type. */
  int32_t GenerateFromSpec(const SchemaSpecPtr& spec, const std::string& rule_name_hint);

  /*! \brief Create a rule and return the rule id (handles caching). */
  int32_t CreateRule(const SchemaSpecPtr& spec, const std::string& rule_name_hint);

  /*! \brief Resolve a reference to its parsed schema. */
  SchemaSpecPtr ResolveRefSchema(const RefSpec& spec, const std::string& rule_name_hint);

  /*! \brief Get next separator from indent manager. */
  virtual std::string NextSeparator(bool is_end = false);

  /*! \brief Get whitespace pattern. */
  std::string GetWhitespacePattern() const;

  int32_t Empty();
  /*! \brief An expression that matches nothing, for alternatives ruled out by excludes_. */
  int32_t Unsatisfiable();
  int32_t ByteString(const std::string& value);
  int32_t TagDispatch(bool loop_after_dispatch, std::vector<std::string> excludes);
  int32_t RuleRef(int32_t rule_id);
  int32_t RuleRef(const std::string& rule_name);
  int32_t Sequence(const std::vector<int32_t>& elements);
  int32_t Choice(const std::vector<int32_t>& choices);
  int32_t Repeat(
      const std::string& rule_name_hint, int32_t expr_id, int32_t min_count, int32_t max_count
  );
  int32_t AddSubGrammar(const Grammar& grammar);

  int32_t WhitespaceExpression();
  int32_t FormattingExpression(const std::string& expression);
  int32_t NextSeparatorExpression(bool is_end = false);
  int32_t KeyPatternExpression();

  int32_t RegexExpression(
      const std::string& regex, bool json_string = false, bool force_cfg_expansion = false
  );
  /*!
   * \brief Rules matching the regex (one of the converter's own ASCII string bodies) minus
   * every string containing one of excludes_, and minus excluded_keys as whole strings. With
   * close_json_string the closing quote is appended after the filtering, so it is never part of
   * an exclusion.
   */
  int32_t ExcludingString(
      const std::string& regex,
      const std::string& rule_name,
      bool close_json_string,
      const std::vector<std::string>& excluded_keys = {}
  );
  bool IsAllowedString(const std::string& text) const;
  bool IsAllowedLiteral(const picojson::value& value, bool raw_string = false) const;
  /*! \brief Whether the JSON literal contains none of excludes_ (see IsAllowedLiteral). */
  bool IsAllowedJSONLiteral(const std::string& json_value, bool raw_string = false) const;
  /*! \brief Log that the string's minLength/maxLength are ignored because excludes_ is set. */
  void WarnDroppedLengthConstraints(const StringSpec& spec, const std::string& rule_name) const;

  /*! \brief Helper to create rule with repetition constraints. */
  int32_t GetPropertyWithNumberConstraints(
      int32_t pattern,
      int min_properties,
      int max_properties,
      int already_repeated_times,
      const std::string& rule_name
  );

  /*! \brief Generate partial rule for object properties.
   *  \param additional_property_override When set, used as the additional property
   *         pattern instead of the default GetKeyPattern() : value. This supports patternProperties
   *         and propertyNames constraints on additional keys.
   */
  int32_t GetPartialRuleForProperties(
      const std::vector<ObjectSpec::Property>& properties,
      const std::unordered_set<std::string>& required,
      const SchemaSpecPtr& additional,
      const std::string& rule_name,
      const std::string& additional_suffix,
      int min_properties,
      int max_properties,
      const std::optional<int32_t>& additional_property_override = std::nullopt
  );

  /*! \brief Generate the object rule in "any order" mode: an "item" alternation over all property
   *  keys, repeated between max(min_properties, required.size()) and max_properties times. Only the
   *  entry count is bounded, not which keys appear.
   */
  int32_t GetAnyOrderRuleForProperties(
      const std::vector<ObjectSpec::Property>& properties,
      const std::unordered_set<std::string>& required,
      const SchemaSpecPtr& additional,
      const std::string& rule_name,
      const std::string& additional_suffix,
      int min_properties,
      int max_properties,
      const std::optional<int32_t>& additional_property_override = std::nullopt
  );

  // ==================== Protected members ====================

  GrammarBuilder builder_;
  IndentManager indent_manager_;
  int32_t colon_expr_id_;
  bool any_whitespace_;
  std::optional<int> max_whitespace_cnt_;
  // When true, object properties may appear in any order (see GetAnyOrderRuleForProperties).
  // Applies to all objects (including nested ones). Default false preserves the fixed-order
  // behavior.
  bool any_order_ = false;
  std::vector<std::string> excludes_;

 public:
  // Basic rule names
  static const std::string kBasicAny;
  static const std::string kBasicInteger;
  static const std::string kBasicNumber;
  static const std::string kBasicString;
  static const std::string kBasicBoolean;
  static const std::string kBasicNull;
  static const std::string kBasicArray;
  static const std::string kBasicObject;
  static const std::string kBasicEscape;
  static const std::string kBasicStringSub;

 protected:
  GenerateCacheManager rule_cache_manager_;

 private:
  void AddHelperRules();

  std::unordered_map<std::string, int32_t> uri_to_rule_id_;  // For circular reference handling
  RefResolver ref_resolver_;  // Resolves $ref URI to SchemaSpecPtr at generate time

  // Trie over property names, for key patterns that exclude specific properties
  struct TrieNode {
    bool is_terminal = false;
    std::map<uint8_t, TrieNode> children;
  };
  int32_t BuildTrieBody(const TrieNode& node, const std::string& rule_name);

  // Reused grammar expression ids
  std::optional<int32_t> empty_expr_id_;
  std::optional<int32_t> unsatisfiable_expr_id_;
  std::unordered_map<std::string, int32_t> byte_string_expr_ids_;
  std::unordered_map<int32_t, int32_t> rule_ref_expr_ids_;
  std::optional<int32_t> whitespace_expr_id_;

  // Helper for integer/number range regex generation
  static std::string GenerateRangeRegex(std::optional<int64_t> start, std::optional<int64_t> end);
  int32_t GenerateIntegerMultipleOfDFA(int64_t multiple_of, const std::string& rule_name);
  static std::string GenerateFloatRangeRegex(
      std::optional<double> start,
      std::optional<double> end,
      int precision = 6,
      bool exclusive_start = false,
      bool exclusive_end = false
  );

 protected:
  static std::optional<std::string> JSONFormatToRegexPattern(const std::string& format);

  // Expose for testing
  friend std::string GenerateRangeRegex(std::optional<int64_t> start, std::optional<int64_t> end);
  friend std::string GenerateFloatRangeRegex(
      std::optional<double> start,
      std::optional<double> end,
      bool exclusive_start,
      bool exclusive_end
  );
};

/*!
 * \brief Convert a JSON Schema string directly to an unnormalized grammar AST.
 *
 * Callers that need normalized grammar should apply GrammarNormalizer after composing any
 * subgrammars.
 */
Grammar JSONSchemaToGrammar(
    const std::string& schema,
    bool any_whitespace = true,
    std::optional<int> indent = std::nullopt,
    std::optional<std::pair<std::string, std::string>> separators = std::nullopt,
    bool strict_mode = true,
    std::optional<int> max_whitespace_cnt = std::nullopt,
    bool any_order = false,
    JSONFormat json_format = JSONFormat::kJSON,
    std::vector<std::string> excludes = {}
);

// ==================== Public API functions (backward compatible) ====================

/*!
 * \brief Convert JSON schema string to EBNF grammar string.
 * \param schema The JSON schema string.
 * \param any_whitespace Whether to ignore the indentation restrictions, and allow any whitespace.
 * Default: true.
 * \param indent The number of spaces for indentation. If set to std::nullopt, the output will be
 * in one line. Default: 2.
 * \param separators Two separators used in the schema: comma and colon. Examples: {",", ":"},
 * {", ", ": "}. If std::nullopt, the default separators will be used: {",", ": "} when the
 * indent is not -1, and {", ", ": "} otherwise. This follows the convention in python
 * json.dumps(). Default: std::nullopt.
 * \param strict_mode Whether to use strict mode. In strict
 * mode, the generated grammar will not allow properties and items that is not specified in the
 * schema. This is equivalent to setting unevaluatedProperties and unevaluatedItems to false.
 * This helps LLM to generate accurate output in the grammar-guided generation with JSON
 * schema. Default: true.
 * \param max_whitespace_cnt The maximum number of whitespace characters for the whitespace
 * which is used for indentation or JSON elements separation when any_whitespace is True. If
 * std::nullopt, it means unlimited. Default: std::nullopt.
 * \param json_format Define the root
 * format of the object. If it's JSONFormat::kJSON, then it will generate a fully JSON-style
 * grammar. If it's JSONFormat::kXML, then it will generate a grammar with the root format is
 * XML-style, while the inner format is JSON-style. Default: JSONFormat::kJSON.
 * \returns The EBNF grammar string.
 */

std::string JSONSchemaToEBNF(
    const std::string& schema,
    bool any_whitespace = true,
    std::optional<int> indent = std::nullopt,
    std::optional<std::pair<std::string, std::string>> separators = std::nullopt,
    bool strict_mode = true,
    std::optional<int> max_whitespace_cnt = std::nullopt,
    JSONFormat json_format = JSONFormat::kJSON,
    bool any_order = false
);

/*!
 * \brief Convert JSON schema string to EBNF grammar string.
 * \param schema The JSON schema object.
 * \param any_whitespace Whether to ignore the indentation restrictions, and allow any whitespace.
 * Default: true.
 * \param indent The number of spaces for indentation. If set to std::nullopt, the output will be
 * in one line. Default: 2.
 * \param separators Two separators used in the schema: comma and colon. Examples: {",", ":"},
 * {", ", ": "}. If std::nullopt, the default separators will be used: {",", ": "} when the
 * indent is not -1, and {", ", ": "} otherwise. This follows the convention in python
 * json.dumps(). Default: std::nullopt.
 * \param strict_mode Whether to use strict mode. In strict
 * mode, the generated grammar will not allow properties and items that is not specified in the
 * schema. This is equivalent to setting unevaluatedProperties and unevaluatedItems to false.
 * This helps LLM to generate accurate output in the grammar-guided generation with JSON
 * schema. Default: true.
 * \param max_whitespace_cnt The maximum number of whitespace characters for the whitespace
 * which is used for indentation or JSON elements separation when any_whitespace is True. If
 * std::nullopt, it means unlimited. Default: std::nullopt.
 * \param json_format Define the root format of the object. If it's JSONFormat::kJSON,
 * then it will generate a fully JSON-style grammar. If it's JSONFormat::kXML, then it will
 * generate a grammar with the root format is XML-style, while the inner format is JSON-style.
 * Default: JSONFormat::kJSON.
 * \returns The EBNF grammar string.
 */
std::string JSONSchemaToEBNF(
    const picojson::value& schema,
    bool any_whitespace = true,
    std::optional<int> indent = std::nullopt,
    std::optional<std::pair<std::string, std::string>> separators = std::nullopt,
    bool strict_mode = true,
    std::optional<int> max_whitespace_cnt = std::nullopt,
    JSONFormat json_format = JSONFormat::kJSON,
    bool any_order = false
);

/*!
 * \brief Generate regex pattern for integer/float range.
 * \param start The start of the range (inclusive). If null assume negative infinity.
 * \param end The end of the range (inclusive). If null assume infinity.
 * \returns The regex pattern that matches integers/floats in the given range.
 */
std::string GenerateRangeRegex(std::optional<int64_t> start, std::optional<int64_t> end);

std::string GenerateFloatRangeRegex(
    std::optional<double> start,
    std::optional<double> end,
    bool exclusive_start = false,
    bool exclusive_end = false
);

}  // namespace xgrammar

#endif  // XGRAMMAR_JSON_SCHEMA_CONVERTER_H_
