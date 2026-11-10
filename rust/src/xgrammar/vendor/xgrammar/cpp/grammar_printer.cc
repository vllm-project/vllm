/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar_printer.cc
 */

#include "grammar_printer.h"

#include <picojson.h>

#include <iomanip>
#include <limits>
#include <sstream>

#include "support/encoding.h"

namespace xgrammar {

std::string GrammarPrinter::PrintRule(const Rule& rule, const SuffixStopInfo* suffix_stop_info) {
  std::string res = rule.name;
  // Print the attributes as one comma-separated bracket group, re-parseable by the EBNF lexer.
  if (rule.max_tokens >= 0 || rule.max_chars >= 0 || !rule.capture_name.empty() ||
      suffix_stop_info != nullptr || rule.is_lazy || rule.temperature.has_value()) {
    std::string attributes;
    auto append_attribute = [&](const std::string& attribute) {
      if (!attributes.empty()) {
        attributes += ", ";
      }
      attributes += attribute;
    };
    if (rule.max_tokens >= 0) {
      append_attribute("max_tokens=" + std::to_string(rule.max_tokens));
    }
    if (rule.max_chars >= 0) {
      append_attribute("max_chars=" + std::to_string(rule.max_chars));
    }
    if (!rule.capture_name.empty()) {
      append_attribute("capture=\"" + rule.capture_name + "\"");
    }
    if (suffix_stop_info != nullptr && suffix_stop_info->hidden_suffix_bytes > 0) {
      append_attribute(
          "capture_hidden_suffix_bytes=" + std::to_string(suffix_stop_info->hidden_suffix_bytes)
      );
    }
    if (suffix_stop_info != nullptr && suffix_stop_info->hidden_stop_bytes > 0) {
      append_attribute(
          "capture_hidden_stop_bytes=" + std::to_string(suffix_stop_info->hidden_stop_bytes)
      );
    }
    if (suffix_stop_info != nullptr && suffix_stop_info->body_rule_id >= 0) {
      append_attribute(
          "capture_hidden_body_rule_id=" + std::to_string(suffix_stop_info->body_rule_id)
      );
      append_attribute(
          "capture_hidden_marker_rule_id=" + std::to_string(suffix_stop_info->marker_rule_id)
      );
    }
    if (suffix_stop_info != nullptr && !suffix_stop_info->stop_capture_name.empty()) {
      append_attribute("stop_capture=\"" + suffix_stop_info->stop_capture_name + "\"");
    }
    if (rule.is_lazy) {
      append_attribute("lazy");
    }
    if (rule.temperature.has_value()) {
      std::ostringstream temperature;
      temperature << std::setprecision(std::numeric_limits<float>::max_digits10)
                  << rule.temperature.value();
      append_attribute("temperature=" + temperature.str());
    }
    res += "[" + attributes + "]";
  }
  res += " ::= " + PrintGrammarExpr(rule.body_expr_id);
  if (rule.lookahead_assertion_id != -1) {
    res += " (=" + PrintGrammarExpr(rule.lookahead_assertion_id) + ")";
  }
  return res;
}

std::string GrammarPrinter::PrintRule(int32_t rule_id) {
  return PrintRule(grammar_->GetRule(rule_id), grammar_->GetSuffixStopInfo(rule_id));
}

std::string GrammarPrinter::PrintGrammarExpr(const GrammarExpr& grammar_expr) {
  std::string result;
  switch (grammar_expr.type) {
    case GrammarExprType::kByteString:
      return PrintByteString(grammar_expr);
    case GrammarExprType::kCharacterClass:
      return PrintCharacterClass(grammar_expr);
    case GrammarExprType::kCharacterClassStar:
      return PrintCharacterClassStar(grammar_expr);
    case GrammarExprType::kEmptyStr:
      return PrintEmptyStr(grammar_expr);
    case GrammarExprType::kRuleRef:
      return PrintRuleRef(grammar_expr);
    case GrammarExprType::kSequence:
      return PrintSequence(grammar_expr);
    case GrammarExprType::kChoices:
      return PrintChoices(grammar_expr);
    case GrammarExprType::kTagDispatch:
      return PrintTagDispatch(grammar_expr);
    case GrammarExprType::kRepeat:
      return PrintRepeat(grammar_expr);
    case GrammarExprType::kToken:
      return PrintToken(grammar_expr);
    case GrammarExprType::kExcludeToken:
      return PrintExcludeToken(grammar_expr);
    case GrammarExprType::kTokenTagDispatch:
      return PrintTokenTagDispatch(grammar_expr);
    case GrammarExprType::kRegex:
      return PrintRegex(grammar_expr);
    case GrammarExprType::kSubstring:
      return PrintSubstring(grammar_expr);
    default:
      XGRAMMAR_LOG(FATAL) << "Unexpected GrammarExpr type: " << static_cast<int>(grammar_expr.type);
      XGRAMMAR_UNREACHABLE();
  }
}

std::string GrammarPrinter::PrintGrammarExpr(int32_t grammar_expr_id) {
  return PrintGrammarExpr(grammar_->GetGrammarExpr(grammar_expr_id));
}

std::string GrammarPrinter::PrintByteString(const GrammarExpr& grammar_expr) {
  std::string internal_str;
  internal_str.reserve(grammar_expr.data_len);
  for (int i = 0; i < grammar_expr.data_len; ++i) {
    internal_str += static_cast<char>(grammar_expr[i]);
  }
  return "\"" + EscapeString(internal_str) + "\"";
}

std::string GrammarPrinter::PrintCharacterClass(const GrammarExpr& grammar_expr) {
  static const std::unordered_map<TCodepoint, std::string> kCustomEscapeMap = {
      {'-', "\\-"}, {']', "\\]"}
  };
  std::string result = "[";
  bool is_negative = static_cast<bool>(grammar_expr[0]);
  if (is_negative) {
    result += "^";
  }
  for (auto i = 1; i < grammar_expr.data_len; i += 2) {
    result += EscapeString(grammar_expr[i], kCustomEscapeMap);
    if (grammar_expr[i] == grammar_expr[i + 1]) {
      continue;
    }
    result += "-";
    result += EscapeString(grammar_expr[i + 1], kCustomEscapeMap);
  }
  result += "]";
  return result;
}

std::string GrammarPrinter::PrintCharacterClassStar(const GrammarExpr& grammar_expr) {
  return PrintCharacterClass(grammar_expr) + "*";
}

std::string GrammarPrinter::PrintEmptyStr(const GrammarExpr& grammar_expr) { return "\"\""; }

std::string GrammarPrinter::PrintRuleRef(const GrammarExpr& grammar_expr) {
  return grammar_->GetRule(grammar_expr[0]).name;
}

std::string GrammarPrinter::PrintSequence(const GrammarExpr& grammar_expr) {
  std::string result;
  result += "(";
  for (int i = 0; i < grammar_expr.data_len; ++i) {
    result += PrintGrammarExpr(grammar_expr[i]);
    if (i + 1 != grammar_expr.data_len) {
      result += " ";
    }
  }
  result += ")";
  return result;
}

std::string GrammarPrinter::PrintChoices(const GrammarExpr& grammar_expr) {
  std::string result;

  result += "(";
  for (int i = 0; i < grammar_expr.data_len; ++i) {
    result += PrintGrammarExpr(grammar_expr[i]);
    if (i + 1 != grammar_expr.data_len) {
      result += " | ";
    }
  }
  result += ")";
  return result;
}

std::string GrammarPrinter::PrintRegex(const GrammarExpr& grammar_expr) {
  std::string result = "Regex(" + PrintString(grammar_->GetRegexString(grammar_expr));
  if (grammar_->GetRegexIsJSONString(grammar_expr)) {
    result += ", json_string=true";
  }
  return result + ")";
}

std::string GrammarPrinter::PrintSubstring(const GrammarExpr& grammar_expr) {
  // EscapeString(std::string) stops at embedded NUL bytes, so escape codepoint by codepoint to
  // keep NUL chunks (allowed by substring expressions) re-parseable.
  auto escape_chunk = [](const std::string& chunk) {
    std::string result = "\"";
    size_t offset = 0;
    while (offset < chunk.size()) {
      if (chunk[offset] == '\0') {
        result += "\\0";
        ++offset;
        continue;
      }
      auto [codepoint, length] = ParseNextUTF8(chunk.c_str() + offset);
      if (codepoint == CharHandlingError::kInvalidUTF8) {
        result += EscapeString(static_cast<uint8_t>(chunk[offset]));
        ++offset;
        continue;
      }
      result += EscapeString(codepoint);
      offset += static_cast<size_t>(length);
    }
    return result + "\"";
  };

  auto chunks = grammar_->GetSubstringChunks(grammar_expr);
  std::string result = "Substring(";
  for (size_t i = 0; i < chunks.size(); ++i) {
    if (i > 0) {
      result += ", ";
    }
    result += escape_chunk(chunks[i]);
  }
  return result + ")";
}

std::string GrammarPrinter::PrintString(const std::string& str) {
  return "\"" + EscapeString(str) + "\"";
}

std::string GrammarPrinter::PrintBoolean(bool value) { return value ? "true" : "false"; }

std::string GrammarPrinter::PrintTagDispatch(const GrammarExpr& grammar_expr) {
  auto tag_dispatch = grammar_->GetTagDispatch(grammar_expr);
  std::string result = "TagDispatch(\n";
  std::string indent = "  ";
  for (const auto& [trigger, rule_id] : tag_dispatch.tag_rule_pairs) {
    result += indent + "(" + PrintString(trigger) + ", " + grammar_->GetRule(rule_id).name + "),\n";
  }
  result +=
      indent + "loop_after_dispatch=" + PrintBoolean(tag_dispatch.loop_after_dispatch) + ",\n";
  result += indent + "excludes=(";
  for (int i = 0; i < static_cast<int>(tag_dispatch.excludes.size()); ++i) {
    if (i > 0) result += ", ";
    result += PrintString(tag_dispatch.excludes[i]);
  }
  result += ")\n)";
  return result;
}

std::string GrammarPrinter::PrintRepeat(const GrammarExpr& grammar_expr) {
  int32_t lower_bound = grammar_expr[1];
  int32_t upper_bound = grammar_expr[2];
  std::string result = grammar_->GetRule(grammar_expr[0]).name + "{";
  result += std::to_string(lower_bound);
  result += ", ";
  result += std::to_string(upper_bound);
  result += "}";
  return result;
}

std::string GrammarPrinter::PrintToken(const GrammarExpr& grammar_expr) {
  std::string result = "Token(";
  for (int i = 0; i < grammar_expr.data_len; ++i) {
    if (i > 0) result += ", ";
    result += std::to_string(grammar_expr[i]);
  }
  result += ")";
  return result;
}

std::string GrammarPrinter::PrintExcludeToken(const GrammarExpr& grammar_expr) {
  std::string result = "ExcludeToken(";
  for (int i = 0; i < grammar_expr.data_len; ++i) {
    if (i > 0) result += ", ";
    result += std::to_string(grammar_expr[i]);
  }
  result += ")";
  return result;
}

std::string GrammarPrinter::PrintTokenTagDispatch(const GrammarExpr& grammar_expr) {
  auto ttd = grammar_->GetTokenTagDispatch(grammar_expr);
  std::string result = "TokenTagDispatch(\n";
  std::string indent = "  ";
  for (const auto& [token_id, rule_id] : ttd.trigger_rule_pairs) {
    result +=
        indent + "(" + std::to_string(token_id) + ", " + grammar_->GetRule(rule_id).name + "),\n";
  }
  result += indent + "loop_after_dispatch=" + PrintBoolean(ttd.loop_after_dispatch) + ",\n";
  result += indent + "excludes=(";
  for (int i = 0; i < static_cast<int>(ttd.excludes.size()); ++i) {
    if (i > 0) result += ", ";
    result += std::to_string(ttd.excludes[i]);
  }
  result += ")\n)";
  return result;
}

std::string GrammarPrinter::ToString() {
  std::string result;
  int num_rules = grammar_->NumRules();
  for (auto i = 0; i < num_rules; ++i) {
    result += PrintRule(i) + "\n";
  }
  return result;
}

}  // namespace xgrammar
