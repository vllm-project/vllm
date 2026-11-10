/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar.cc
 */

#include <xgrammar/grammar.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>

#include "grammar_functor.h"
#include "grammar_parser.h"
#include "grammar_printer.h"
#include "json_schema_converter.h"
#include "lark_converter.h"
#include "regex_converter.h"
#include "structural_tag.h"
#include "support/json_serializer.h"
#include "support/logging.h"
#include "xgrammar/exception.h"

namespace xgrammar {

/******************* Grammar::Impl *******************/

std::size_t MemorySize(const Grammar::Impl& impl) {
  /// TODO: Now, we evaluate the memory size of each rule as sizeof(Rule), which counts its
  /// string members as sizeof(std::string), with an assumption that the strings are small.
  /// This should be improved in the future.
  return impl.rules_.size() * sizeof(Grammar::Impl::Rule) +
         impl.suffix_stop_infos_.size() * sizeof(Grammar::Impl::SuffixStopInfo) +
         MemorySize(impl.grammar_expr_data_) + MemorySize(impl.grammar_expr_indptr_) +
         MemorySize(impl.complete_fsm) + MemorySize(impl.per_rule_fsms) +
         MemorySize(impl.allow_empty_rule_ids);
}

/******************* Grammar *******************/

std::string Grammar::ToString() const { return GrammarPrinter(*this).ToString(); }

Grammar Grammar::FromEBNF(const std::string& ebnf_string, const std::string& root_rule_name) {
  auto grammar = ParseEBNF(ebnf_string, root_rule_name);
  grammar = GrammarNormalizer().Apply(grammar);
  return grammar;
}

Grammar Grammar::FromJSONSchema(
    const std::string& schema,
    bool any_whitespace,
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool strict_mode,
    std::optional<int> max_whitespace_cnt,
    bool print_converted_ebnf,
    bool any_order
) {
  auto grammar = GrammarNormalizer::Apply(JSONSchemaToGrammar(
      schema, any_whitespace, indent, separators, strict_mode, max_whitespace_cnt, any_order
  ));
  if (print_converted_ebnf) {
    XGRAMMAR_LOG(INFO) << "Converted EBNF: " << grammar.ToString() << std::endl;
  }
  return grammar;
}

Grammar Grammar::FromRegex(const std::string& regex, bool print_converted_ebnf) {
  auto ebnf_string = RegexToEBNF(regex);
  if (print_converted_ebnf) {
    XGRAMMAR_LOG(INFO) << "Converted EBNF: " << ebnf_string << std::endl;
  }
  return FromEBNF(ebnf_string);
}

Grammar Grammar::FromLark(
    const std::string& lark_string,
    const std::optional<TokenizerInfo>& tokenizer_info,
    const std::vector<NamedGrammar>& named_grammars
) {
  return LarkToGrammar(lark_string, tokenizer_info, named_grammars);
}

std::variant<Grammar, StructuralTagError> Grammar::FromStructuralTag(
    const std::string& structural_tag_json, const std::optional<TokenizerInfo>& tokenizer_info
) {
  return StructuralTagToGrammar(structural_tag_json, tokenizer_info).ToVariant();
}

// Optimized json grammar for the speed of the grammar matcher
const std::string kJSONGrammarString = R"(
root ::= (
    "{" [ \n\r\t]* members_and_embrace |
    "[" [ \n\r\t]* elements_or_embrace
)
value_non_str ::= (
    "{" [ \n\r\t]* members_and_embrace |
    "[" [ \n\r\t]* elements_or_embrace |
    "0" fraction exponent |
    [1-9] [0-9]* fraction exponent |
    "-" [0-9] fraction exponent |
    "-" [1-9] [0-9]* fraction exponent |
    "true" |
    "false" |
    "null"
) (= [ \n\r\t]* member_suffix_suffix)
members_and_embrace ::= ("\"" characters_and_colon [ \n\r\t]* members_suffix | "}") (= [ \n\r\t,}\]])
members_suffix ::= (
    value_non_str [ \n\r\t]* member_suffix_suffix |
    "\"" characters_and_embrace |
    "\"" characters_and_comma [ \n\r\t]* "\"" characters_and_colon [ \n\r\t]* members_suffix
) (= [ \n\r\t,}\]])
member_suffix_suffix ::= (
    "}" |
    "," [ \n\r\t]* "\"" characters_and_colon [ \n\r\t]* members_suffix
) (= [ \n\r\t,}\]])
elements_or_embrace ::= (
    "{" [ \n\r\t]* members_and_embrace elements_rest [ \n\r\t]* "]" |
    "[" [ \n\r\t]* elements_or_embrace elements_rest [ \n\r\t]* "]" |
    "\"" characters_item elements_rest [ \n\r\t]* "]" |
    "0" fraction exponent elements_rest [ \n\r\t]* "]" |
    [1-9] [0-9]* fraction exponent elements_rest [ \n\r\t]* "]" |
    "-" "0" fraction exponent elements_rest [ \n\r\t]* "]" |
    "-" [1-9] [0-9]* fraction exponent elements_rest [ \n\r\t]* "]" |
    "true" elements_rest [ \n\r\t]* "]" |
    "false" elements_rest [ \n\r\t]* "]" |
    "null" elements_rest [ \n\r\t]* "]" |
    "]"
)
elements ::= (
    "{" [ \n\r\t]* members_and_embrace elements_rest |
    "[" [ \n\r\t]* elements_or_embrace elements_rest |
    "\"" characters_item elements_rest |
    "0" fraction exponent elements_rest |
    [1-9] [0-9]* fraction exponent elements_rest |
    "-" [0-9] fraction exponent elements_rest |
    "-" [1-9] [0-9]* fraction exponent elements_rest |
    "true" elements_rest |
    "false" elements_rest |
    "null" elements_rest
)
elements_rest ::= (
    "" |
    [ \n\r\t]* "," [ \n\r\t]* elements
)
characters_and_colon ::= (
    "\"" [ \n\r\t]* ":" |
    [^"\\\x00-\x1F] characters_and_colon |
    "\\" escape characters_and_colon
) (=[ \n\r\t]* [\"{[0-9tfn-])
characters_and_comma ::= (
    "\"" [ \n\r\t]* "," |
    [^"\\\x00-\x1F] characters_and_comma |
    "\\" escape characters_and_comma
) (=[ \n\r\t]* "\"")
characters_and_embrace ::= (
    "\"" [ \n\r\t]* "}" |
    [^"\\\x00-\x1F] characters_and_embrace |
    "\\" escape characters_and_embrace
) (=[ \n\r\t]* [},])
characters_item ::= (
    "\"" |
    [^"\\\x00-\x1F] characters_item |
    "\\" escape characters_item
) (= [ \n\r\t]* [,\]])
escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
fraction ::= "" | "." [0-9] [0-9]*
exponent ::= "" |  "e" sign [0-9] [0-9]* | "E" sign [0-9] [0-9]*
sign ::= "" | "+" | "-"
)";

Grammar Grammar::BuiltinJSONGrammar() {
  static const Grammar grammar = FromEBNF(kJSONGrammarString);
  return grammar;
}

Grammar Grammar::Union(const std::vector<Grammar>& grammars) {
  return GrammarUnionFunctor::Apply(grammars);
}

Grammar Grammar::Concat(const std::vector<Grammar>& grammars) {
  return GrammarConcatFunctor::Apply(grammars);
}

std::ostream& operator<<(std::ostream& os, const Grammar& grammar) {
  os << grammar.ToString();
  return os;
}

std::optional<std::string> Grammar::Impl::Validate() const {
  const int64_t num_rules = rules_.size();
  const int64_t num_exprs = grammar_expr_indptr_.size();
  const int64_t data_size = grammar_expr_data_.size();
  auto rule_ok = [&](int64_t id) { return id >= 0 && id < num_rules; };
  auto expr_ok = [&](int64_t id) { return id >= 0 && id < num_exprs; };

  // Pass 1: every expr must fit in grammar_expr_data_ before any expr can be read.
  for (int64_t expr_id = 0; expr_id < num_exprs; ++expr_id) {
    const int64_t start = grammar_expr_indptr_[expr_id];
    if (start < 0 || start + 2 > data_size) {
      return "grammar_expr_indptr[" + std::to_string(expr_id) + "] is out of range";
    }
    const int64_t type = grammar_expr_data_[start];
    const int64_t len = grammar_expr_data_[start + 1];
    if (len < 0 || start + 2 + len > data_size) {
      return "The length of grammar expr " + std::to_string(expr_id) + " is out of range";
    }
    if (type < 0 || type > static_cast<int64_t>(GrammarExprType::kSubstring)) {
      return "Unknown type of grammar expr " + std::to_string(expr_id);
    }
  }

  // Pass 2: the ids stored inside each expr must refer to existing rules and exprs.
  for (int64_t expr_id = 0; expr_id < num_exprs; ++expr_id) {
    const auto expr = GetGrammarExpr(expr_id);
    const int64_t size = expr.size();
    bool ok = true;
    switch (expr.type) {
      case GrammarExprType::kByteString:
      case GrammarExprType::kEmptyStr:
      case GrammarExprType::kToken:
      case GrammarExprType::kExcludeToken:
        break;
      case GrammarExprType::kCharacterClass:
      case GrammarExprType::kCharacterClassStar:
        // [is_negative, lower0, upper0, ...]
        ok = size >= 1 && size % 2 == 1;
        break;
      case GrammarExprType::kRuleRef:
        ok = size == 1 && rule_ok(expr[0]);
        break;
      case GrammarExprType::kSequence:
      case GrammarExprType::kChoices:
        ok = std::all_of(expr.begin(), expr.end(), expr_ok);
        break;
      case GrammarExprType::kTagDispatch: {
        // [tag_expr0, rule_id0, ..., loop_after_dispatch, excluded_str_expr_id]
        const int64_t extra = TagDispatch::kTagDispatchExtraParameter;
        ok = size >= extra && (size - extra) % 2 == 0;
        for (int64_t i = 0; ok && i < size - extra; i += 2) {
          ok = expr_ok(expr[i]) && rule_ok(expr[i + 1]);
        }
        // The excluded strings are read as a kChoices expr of byte string exprs.
        ok = ok && expr_ok(expr[size - 1]) &&
             GetGrammarExpr(expr[size - 1]).type == GrammarExprType::kChoices;
        break;
      }
      case GrammarExprType::kRepeat:
        ok = size == 3 && rule_ok(expr[0]);
        break;
      case GrammarExprType::kTokenTagDispatch: {
        // [trigger_cnt, (token_id, rule_id) x N, loop_after_dispatch, exclude_cnt, token_id x M]
        const int64_t trigger_cnt = size >= 1 ? expr[0] : -1;
        ok = trigger_cnt >= 0 && 1 + 2 * trigger_cnt + 2 <= size;
        for (int64_t i = 0; ok && i < trigger_cnt; ++i) {
          ok = rule_ok(expr[2 + 2 * i]);
        }
        if (ok) {
          const int64_t exclude_cnt = expr[1 + 2 * trigger_cnt + 1];
          ok = exclude_cnt >= 0 && 1 + 2 * trigger_cnt + 2 + exclude_cnt == size;
        }
        break;
      }
      case GrammarExprType::kRegex:
        ok = size >= 1;
        break;
      case GrammarExprType::kSubstring:
        // [chunk0_len, byte0_0, ..., chunk1_len, ...]
        for (int64_t i = 0; ok && i < size;) {
          const int64_t chunk_len = expr[i++];
          ok = chunk_len >= 0 && i + chunk_len <= size;
          i += chunk_len;
        }
        break;
    }
    if (!ok) {
      return "Grammar expr " + std::to_string(expr_id) + " is malformed";
    }
  }

  for (int64_t rule_id = 0; rule_id < num_rules; ++rule_id) {
    const auto& rule = rules_[rule_id];
    if (!expr_ok(rule.body_expr_id) ||
        (rule.lookahead_assertion_id != -1 && !expr_ok(rule.lookahead_assertion_id))) {
      return "Rule " + std::to_string(rule_id) + " refers to a grammar expr out of range";
    }
  }
  if (!rule_ok(root_rule_id_)) {
    return "root_rule_id " + std::to_string(root_rule_id_) + " is out of range";
  }
  for (const auto& info : suffix_stop_infos_) {
    if (!rule_ok(info.rule_id) || (info.body_rule_id != -1 && !rule_ok(info.body_rule_id)) ||
        (info.marker_rule_id != -1 && !rule_ok(info.marker_rule_id))) {
      return "suffix_stop_infos refers to a rule out of range";
    }
  }
  if (!std::all_of(allow_empty_rule_ids.begin(), allow_empty_rule_ids.end(), rule_ok)) {
    return "allow_empty_rule_ids refers to a rule out of range";
  }

  // The FSMs are internally consistent (see CompactFSM::Impl::Validate); check the rule ids they
  // refer to. The parser reads the repeat info of a per-rule FSM edge from complete_fsm, so the
  // aux indices of per-rule repeat edges must also be valid for complete_fsm.
  auto fsm_rules_ok = [&](const CompactFSM& fsm) {
    const auto& aux = fsm.GetEdgeAuxData();
    for (int state = 0; state < fsm.NumStates(); ++state) {
      for (const auto& edge : fsm.GetEdges(state)) {
        if ((edge.IsRuleRef() && !rule_ok(edge.max)) ||
            (edge.IsRepeatRef() && !rule_ok(aux[edge.max]))) {
          return false;
        }
      }
    }
    return true;
  };
  if (!complete_fsm.IsNull() && !fsm_rules_ok(complete_fsm)) {
    return "complete_fsm refers to a rule out of range";
  }
  if (optimized &&
      (complete_fsm.IsNull() || static_cast<int64_t>(per_rule_fsms.size()) != num_rules)) {
    return "An optimized grammar must have complete_fsm and one FSM per rule";
  }
  for (int64_t rule_id = 0; rule_id < static_cast<int64_t>(per_rule_fsms.size()); ++rule_id) {
    if (!per_rule_fsms[rule_id].has_value()) {
      if (optimized) {
        return "Rule " + std::to_string(rule_id) + " has no FSM in an optimized grammar";
      }
      continue;
    }
    const CompactFSM& fsm = per_rule_fsms[rule_id]->GetFsm().GetFsm();
    if (!fsm_rules_ok(fsm)) {
      return "The FSM of rule " + std::to_string(rule_id) + " refers to a rule out of range";
    }
    const int64_t complete_aux_size =
        complete_fsm.IsNull() ? 0 : complete_fsm.GetEdgeAuxData().size();
    for (int state = 0; state < fsm.NumStates(); ++state) {
      for (const auto& edge : fsm.GetEdges(state)) {
        if (edge.IsRepeatRef() && (static_cast<int64_t>(edge.max) + 3 > complete_aux_size ||
                                   !rule_ok(complete_fsm.GetEdgeAuxData()[edge.max]))) {
          return "The FSM of rule " + std::to_string(rule_id) +
                 " has a repeat edge that is out of range of complete_fsm";
        }
      }
    }
  }
  return std::nullopt;
}

std::string Grammar::SerializeJSON() const { return AutoSerializeJSON(*this, true); }

std::variant<Grammar, SerializationError> Grammar::DeserializeJSON(const std::string& json_string) {
  Grammar result{NullObj()};
  if (auto err = AutoDeserializeJSON(&result, json_string, true, "Grammar")) {
    return err.value();
  }
  return result;
}

}  // namespace xgrammar
