/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/converter_ext/deepseek.cc
 * \brief DeepSeek XML parameter formats and value rendering.
 */
#include <type_traits>

#include "../json_schema_converter_ext.h"

namespace xgrammar {
namespace converter_ext {

XMLWrapper GetDeepSeekXMLWrapper() {
  return {"<｜DSML｜parameter name=\"", "", "", "</｜DSML｜parameter>"};
}

XMLWrapper GetDeepSeekV41XMLWrapper() {
  return {"<｜DSML｜ parameter name=\"", "", "", "</｜DSML｜ parameter>"};
}

}  // namespace converter_ext

int32_t XMLToolCallingConverter::FormatDeepSeekParamSuffix(
    const SchemaSpecPtr& schema, int32_t value_rule_id, const std::string& rule_name_hint
) {
  // Copy the name: creating alternative rules can reallocate the builder's rule storage.
  std::string value_rule_name =
      value_rule_id >= 0 ? builder_.GetRule(value_rule_id).name : rule_name_hint;
  if (schema != nullptr) {
    if (const auto* ref = std::get_if<RefSpec>(&schema->spec); ref != nullptr) {
      auto cached = deepseek_param_ref_rules_.find(ref->uri);
      if (cached != deepseek_param_ref_rules_.end()) {
        return RuleRef(cached->second);
      }
      // Cache the rule before descending through references or alternatives. A recursive
      // branch then refers back to this rule, and shared acyclic subgraphs are built only once.
      int32_t param_rule_id = builder_.AddEmptyRuleWithHint(value_rule_name + "_dsml_param");
      deepseek_param_ref_rules_.emplace(ref->uri, param_rule_id);
      auto resolved = ResolveRefSchema(*ref, value_rule_name);
      builder_.UpdateRuleBody(
          param_rule_id, FormatDeepSeekParamSuffix(resolved, value_rule_id, value_rule_name)
      );
      return RuleRef(param_rule_id);
    }
  }

  // string="true" wraps a raw string whose whitespace is part of the value; string="false"
  // wraps a JSON value that may be padded with whitespace.
  auto wrap = [&](int32_t value_expr, bool is_string) {
    std::vector<int32_t> elements = {
        ByteString(is_string ? "\" string=\"true\">" : "\" string=\"false\">")
    };
    if (!is_string) elements.push_back(WhitespaceExpression());
    elements.push_back(value_expr);
    if (!is_string) elements.push_back(WhitespaceExpression());
    elements.push_back(ByteString(xml_wrapper_.parameter_suffix));
    return Sequence(elements);
  };

  // A schema rendered with a single type keeps the value rule built by the caller.
  std::optional<std::string> pinned_type = GetRenderedJSONType(schema);
  if (pinned_type.has_value()) {
    if (value_rule_id < 0) {
      value_rule_id = CreateRule(schema, value_rule_name);
    }
    return wrap(RuleRef(value_rule_id), *pinned_type == "string");
  }

  // Unions and mixed enums get one alternative per option so each carries its own attribute.
  std::vector<SchemaSpecPtr> options;
  if (schema != nullptr) {
    std::visit(
        [&](const auto& spec) {
          using T = std::decay_t<decltype(spec)>;
          if constexpr (std::is_same_v<T, AnyOfSpec> || std::is_same_v<T, OneOfSpec>) {
            options = spec.options;
          } else if constexpr (std::is_same_v<T, TypeArraySpec>) {
            options = spec.type_schemas;
          } else if constexpr (std::is_same_v<T, AllOfSpec>) {
            if (spec.schemas.size() == 1) options = spec.schemas;
          } else if constexpr (std::is_same_v<T, EnumSpec>) {
            for (const auto& value : spec.json_values) {
              options.push_back(SchemaSpec::Make(ConstSpec{value}));
            }
          }
        },
        schema->spec
    );
  }
  if (options.empty()) {
    // No schema, {} and allOf with several schemas all render any value.
    return Choice(
        {wrap(RuleRef(kXMLString), true),
         wrap(
             Choice(
                 {RuleRef(kBasicNumber),
                  RuleRef(kBasicBoolean),
                  RuleRef(kBasicNull),
                  RuleRef(kBasicArray),
                  RuleRef(kBasicObject)}
             ),
             false
         )}
    );
  }
  std::vector<int32_t> choices;
  for (size_t index = 0; index < options.size(); ++index) {
    // Only leaves need value rules. Rebuilding a nested union's whole value subtree at each
    // level would create quadratically many rules before assembling its typed alternatives.
    choices.push_back(FormatDeepSeekParamSuffix(
        options[index], -1, value_rule_name + "_dsml_case_" + std::to_string(index)
    ));
  }
  return Choice(choices);
}

}  // namespace xgrammar
