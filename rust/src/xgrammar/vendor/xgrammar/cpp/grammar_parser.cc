/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar_parser.cc
 */

#include "grammar_parser.h"

#include <picojson.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <variant>
#include <vector>

#include "fsm_builder.h"
#include "grammar_builder.h"
#include "grammar_impl.h"
#include "support/encoding.h"
#include "support/logging.h"
#include "xgrammar/grammar.h"

namespace xgrammar {

class EBNFLexer::Impl {
 public:
  using Token = EBNFLexer::Token;
  using TokenType = EBNFLexer::TokenType;

  std::vector<Token> Tokenize(const std::string& input);

 private:
  std::string input_;
  const char* cur_ = nullptr;
  int cur_line_ = 1;
  int cur_column_ = 1;

  constexpr static int64_t kMaxIntegerInGrammar = 1e15;

  // Helper functions

  /*!
   * \brief Consume a character sequence and return the next token. Return a token if it's a
   * single token, or a vector of tokens if it's a sequence of tokens.
   *
   * \return std::variant<Token, std::vector<Token>>
   */
  std::variant<Token, std::vector<Token>> NextToken();
  Token ParseIdentifierOrBooleanToken();
  Token ParseStringToken();
  std::vector<Token> ParseCharClassToken();
  Token ParseIntegerToken();
  [[noreturn]] void ReportLexerError(const std::string& msg, int line = -1, int column = -1);
  char Peek(int delta = 0) const;
  void Consume(int cnt = 1);
  void ConsumeSpace();
  std::string ParseIdentifierToken();
  void ConvertIdentifierToRuleName(std::vector<Token>* tokens);
  static bool IsNameChar(char c, bool is_first = false);
};

// Look at the next character
inline char EBNFLexer::Impl::Peek(int delta) const { return *(cur_ + delta); }

// Consume characters and update position information
inline void EBNFLexer::Impl::Consume(int cnt) {
  for (int i = 0; i < cnt; ++i) {
    // Newline\n \r \r\n
    if (*cur_ == '\n' || (*cur_ == '\r' && *(cur_ + 1) != '\n')) {
      ++cur_line_;
      cur_column_ = 1;
    } else {
      ++cur_column_;
    }
    ++cur_;
  }
}

// Skip whitespace and comments
void EBNFLexer::Impl::ConsumeSpace() {
  while (Peek() &&
         (Peek() == ' ' || Peek() == '\t' || Peek() == '#' || Peek() == '\n' || Peek() == '\r')) {
    Consume();
    if (Peek(-1) == '#') {
      while (Peek() && Peek() != '\n' && Peek() != '\r') {
        Consume();
      }
      if (!Peek()) {
        return;
      }
      Consume();
      if (Peek(-1) == '\r' && Peek() == '\n') {
        Consume();
      }
    }
  }
}

// Report parsing error
void EBNFLexer::Impl::ReportLexerError(const std::string& msg, int line, int column) {
  int line_to_print = line == -1 ? cur_line_ : line;
  int column_to_print = column == -1 ? cur_column_ : column;
  XGRAMMAR_LOG(FATAL) << "EBNF lexer error at line " + std::to_string(line_to_print) + ", column " +
                             std::to_string(column_to_print) + ": " + msg;
  XGRAMMAR_UNREACHABLE();
}

// Check if a character can be part of an identifier
bool EBNFLexer::Impl::IsNameChar(char c, bool is_first) {
  return c == '_' || c == '-' || c == '.' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (!is_first && c >= '0' && c <= '9');
}

// Parse identifier
std::string EBNFLexer::Impl::ParseIdentifierToken() {
  const char* start = cur_;
  bool first_char = true;
  while (*cur_ && IsNameChar(*cur_, first_char)) {
    Consume();
    first_char = false;
  }
  if (start == cur_) {
    ReportLexerError("Expect identifier");
  }
  return std::string(start, cur_ - start);
}

// Parse identifier or boolean value
EBNFLexer::Token EBNFLexer::Impl::ParseIdentifierOrBooleanToken() {
  int start_line = cur_line_;
  int start_column = cur_column_;

  std::string identifier = ParseIdentifierToken();

  // Check if it's a boolean value
  if (identifier == "true" || identifier == "false") {
    return {
        TokenType::BooleanLiteral,
        identifier,
        identifier == "true" ? true : false,
        start_line,
        start_column
    };
  }

  // A rule definition may carry an attribute block before ::=, e.g.
  // name[max_tokens=10] ::= ..., name[max_chars=10] ::= ..., name[capture] ::= ...,
  // name[capture="x"] ::= ...,
  // name[capture_hidden_suffix_bytes=3] ::= ..., name[capture_hidden_stop_bytes=3] ::= ...,
  // name[capture_hidden_body_rule_id=1, capture_hidden_marker_rule_id=2] ::= ...,
  // name[stop_capture="marker"] ::= ..., name[lazy] ::= ..., name[temperature=0.7] ::= ..., or a
  // comma-separated combination.
  // The bracket group is treated as an attribute block only when it is followed by "::=";
  // otherwise it is left to be lexed as a character class.
  if (*cur_ == '[') {
    int delta = 1;
    auto skip_space = [&]() {
      while (Peek(delta) == ' ' || Peek(delta) == '\t') {
        ++delta;
      }
    };
    // Match the keyword at the current position and advance delta past it on success.
    auto match_keyword = [&](const char* keyword) {
      int len = 0;
      while (keyword[len] != '\0') {
        if (Peek(delta + len) != keyword[len]) {
          return false;
        }
        ++len;
      }
      delta += len;
      return true;
    };
    bool matched = true;
    bool has_max_tokens = false;
    bool has_max_chars = false;
    bool has_capture = false;
    bool has_capture_hidden_suffix_bytes = false;
    bool has_capture_hidden_stop_bytes = false;
    bool has_capture_hidden_body_rule_id = false;
    bool has_capture_hidden_marker_rule_id = false;
    bool has_stop_capture = false;
    bool has_lazy = false;
    bool has_temperature = false;
    double temperature_value = 0;
    int64_t max_tokens_value = -1;
    int64_t max_chars_value = -1;
    int64_t capture_hidden_suffix_bytes_value = 0;
    int64_t capture_hidden_stop_bytes_value = 0;
    int64_t capture_hidden_body_rule_id_value = -1;
    int64_t capture_hidden_marker_rule_id_value = -1;
    std::string capture_value;
    std::string stop_capture_value;
    auto parse_integer_value = [&](int64_t* value) {
      skip_space();
      if (Peek(delta) != '=') {
        return false;
      }
      ++delta;
      skip_space();
      *value = 0;
      int digits = 0;
      while (Peek(delta) >= '0' && Peek(delta) <= '9' && digits < 10) {
        *value = *value * 10 + (Peek(delta) - '0');
        ++delta;
        ++digits;
      }
      return digits > 0 && !(Peek(delta) >= '0' && Peek(delta) <= '9');
    };
    auto parse_string_value = [&](std::string* value) {
      skip_space();
      if (Peek(delta) != '=') {
        return false;
      }
      ++delta;
      skip_space();
      if (Peek(delta) != '"') {
        return false;
      }
      ++delta;
      while (Peek(delta) != '"') {
        char c = Peek(delta);
        if (c == '\0' || c == '\n' || c == '\r' || c == '\\') {
          return false;
        }
        value->push_back(c);
        ++delta;
      }
      ++delta;
      return true;
    };
    auto parse_float_value = [&](double* value) {
      skip_space();
      if (Peek(delta) != '=') {
        return false;
      }
      ++delta;
      skip_space();
      std::string text;
      while (true) {
        char c = Peek(delta);
        if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
          text.push_back(c);
          ++delta;
        } else {
          break;
        }
      }
      if (text.empty()) {
        return false;
      }
      try {
        *value = std::stod(text);
      } catch (const std::out_of_range&) {
        *value = std::numeric_limits<double>::infinity();
      } catch (const std::exception&) {
        return false;
      }
      return true;
    };
    // Parse a comma-separated attribute list. Each attribute may appear at most once.
    while (matched) {
      skip_space();
      if (!has_max_tokens && match_keyword("max_tokens")) {
        has_max_tokens = true;
        matched = parse_integer_value(&max_tokens_value);
      } else if (!has_max_chars && match_keyword("max_chars")) {
        has_max_chars = true;
        matched = parse_integer_value(&max_chars_value);
      } else if (!has_capture_hidden_suffix_bytes && match_keyword("capture_hidden_suffix_bytes")) {
        has_capture_hidden_suffix_bytes = true;
        matched = parse_integer_value(&capture_hidden_suffix_bytes_value);
      } else if (!has_capture_hidden_stop_bytes && match_keyword("capture_hidden_stop_bytes")) {
        has_capture_hidden_stop_bytes = true;
        matched = parse_integer_value(&capture_hidden_stop_bytes_value);
      } else if (!has_capture_hidden_body_rule_id && match_keyword("capture_hidden_body_rule_id")) {
        has_capture_hidden_body_rule_id = true;
        matched = parse_integer_value(&capture_hidden_body_rule_id_value);
      } else if (!has_capture_hidden_marker_rule_id &&
                 match_keyword("capture_hidden_marker_rule_id")) {
        has_capture_hidden_marker_rule_id = true;
        matched = parse_integer_value(&capture_hidden_marker_rule_id_value);
      } else if (!has_stop_capture && match_keyword("stop_capture")) {
        has_stop_capture = true;
        matched = parse_string_value(&stop_capture_value);
      } else if (!has_capture && match_keyword("capture")) {
        has_capture = true;
        skip_space();
        if (Peek(delta) == '=') {
          matched = parse_string_value(&capture_value);
        } else {
          capture_value = identifier;
        }
      } else if (!has_lazy && match_keyword("lazy")) {
        has_lazy = true;
      } else if (!has_temperature && match_keyword("temperature")) {
        has_temperature = true;
        matched = parse_float_value(&temperature_value);
      } else {
        matched = false;
      }
      if (!matched) {
        break;
      }
      skip_space();
      if (Peek(delta) == ',') {
        ++delta;
        continue;
      }
      break;
    }
    if (matched) {
      skip_space();
      if (Peek(delta) == ']') {
        ++delta;
      } else {
        matched = false;
      }
    }
    if (matched) {
      int after_bracket = delta;
      while (Peek(after_bracket) == ' ' || Peek(after_bracket) == '\t') {
        ++after_bracket;
      }
      if (!(Peek(after_bracket) == ':' && Peek(after_bracket + 1) == ':' &&
            Peek(after_bracket + 2) == '=')) {
        matched = false;
      }
    }
    if (matched) {
      if (has_capture && capture_value.empty()) {
        ReportLexerError("The capture name must not be empty", start_line, start_column);
      }
      if (has_stop_capture && stop_capture_value.empty()) {
        ReportLexerError("The stop capture name must not be empty", start_line, start_column);
      }
      if (has_capture_hidden_body_rule_id != has_capture_hidden_marker_rule_id) {
        ReportLexerError(
            "The capture-hidden body and marker rule ids must be specified together",
            start_line,
            start_column
        );
      }
      if ((has_capture_hidden_suffix_bytes && capture_hidden_suffix_bytes_value <= 0) ||
          (has_capture_hidden_stop_bytes && capture_hidden_stop_bytes_value <= 0)) {
        ReportLexerError(
            "The number of capture-hidden bytes must be positive", start_line, start_column
        );
      }
      constexpr int64_t kMaxInt32 = std::numeric_limits<int32_t>::max();
      if (has_max_chars && max_chars_value < 0) {
        ReportLexerError(
            "The max_chars rule attribute must be non-negative", start_line, start_column
        );
      }
      if ((has_max_chars && max_chars_value > kMaxInt32) ||
          (has_capture_hidden_suffix_bytes && capture_hidden_suffix_bytes_value > kMaxInt32) ||
          (has_capture_hidden_stop_bytes && capture_hidden_stop_bytes_value > kMaxInt32) ||
          (has_capture_hidden_body_rule_id && capture_hidden_body_rule_id_value > kMaxInt32) ||
          (has_capture_hidden_marker_rule_id && capture_hidden_marker_rule_id_value > kMaxInt32)) {
        ReportLexerError("The rule attribute value is too large", start_line, start_column);
      }
      if (has_temperature && !(std::isfinite(temperature_value) && temperature_value >= 0 &&
                               temperature_value <= std::numeric_limits<float>::max())) {
        ReportLexerError(
            "The temperature must be a finite non-negative number", start_line, start_column
        );
      }
      Consume(delta);
      Token token{TokenType::Identifier, identifier, identifier, start_line, start_column};
      token.max_tokens = static_cast<int32_t>(max_tokens_value);
      token.max_chars = static_cast<int32_t>(max_chars_value);
      token.capture_name = capture_value;
      token.capture_hidden_suffix_bytes = static_cast<int32_t>(capture_hidden_suffix_bytes_value);
      token.capture_hidden_stop_bytes = static_cast<int32_t>(capture_hidden_stop_bytes_value);
      token.capture_hidden_body_rule_id = static_cast<int32_t>(capture_hidden_body_rule_id_value);
      token.capture_hidden_marker_rule_id =
          static_cast<int32_t>(capture_hidden_marker_rule_id_value);
      token.stop_capture_name = stop_capture_value;
      token.is_lazy = has_lazy;
      if (has_temperature) {
        token.temperature = static_cast<float>(temperature_value);
      }
      return token;
    }
  }

  // Otherwise it's an identifier
  return {TokenType::Identifier, identifier, identifier, start_line, start_column};
}

// Parse string literal
EBNFLexer::Token EBNFLexer::Impl::ParseStringToken() {
  int start_line = cur_line_;
  int start_column = cur_column_;
  const char* start_pos = cur_;

  Consume();  // Skip opening quote

  std::vector<int32_t> codepoints;
  while (Peek() && Peek() != '"' && Peek() != '\n' && Peek() != '\r') {
    auto [codepoint, len] = ParseNextUTF8OrEscaped(cur_);
    if (codepoint == CharHandlingError::kInvalidUTF8) {
      ReportLexerError("Invalid UTF8 sequence");
    }
    if (codepoint == CharHandlingError::kInvalidEscape) {
      ReportLexerError("Invalid escape sequence");
    }
    Consume(len);
    codepoints.push_back(codepoint);
  }

  if (Peek() != '"') {
    ReportLexerError("Expect \" in string literal");
  }
  Consume();  // Skip closing quote

  // Extract original lexeme
  std::string lexeme(start_pos, cur_ - start_pos);

  // Convert codepoints to UTF-8 string value
  std::string value;
  for (auto codepoint : codepoints) {
    value += CharToUTF8(codepoint);
  }

  return {TokenType::StringLiteral, lexeme, value, start_line, start_column};
}

// Parse character class.
std::vector<EBNFLexer::Token> EBNFLexer::Impl::ParseCharClassToken() {
  std::vector<Token> tokens;

  tokens.push_back({TokenType::LBracket, "[", "", cur_line_, cur_column_});
  Consume();  // Skip '['

  if (Peek() == '^') {
    tokens.push_back({TokenType::Caret, "^", "", cur_line_, cur_column_});
    Consume();
  }

  static const std::unordered_map<char, TCodepoint> kRegexEscapeChars = {
      // clang-format off
      {'^', '^'}, {'$', '$'}, {'\\', '\\'}, {'.', '.'}, {'*', '*'}, {'+', '+'}, {'?', '?'},
      {'(', '('}, {')', ')'}, {'[', '['}, {']', ']'}, {'{', '{'}, {'}', '}'}, {'|', '|'},
      {'/', '/'}, {'-', '-'}  // clang-format on
  };

  static const std::unordered_set<char> kRegexSpecialEscapes = {'d', 'D', 's', 'S', 'w', 'W'};

  while (Peek() && Peek() != ']') {
    if (Peek() == '\r' || Peek() == '\n') {
      ReportLexerError("Character class should not contain newline");
    } else if (Peek() == '-') {
      // Handle dash; this dash could be a range expression or a normal dash.
      // It will further be handled in EBNFParser::ParseCharClass.
      tokens.push_back({TokenType::Dash, "-", "", cur_line_, cur_column_});
      Consume();
    } else if (Peek() == '\\' && kRegexSpecialEscapes.count(Peek(1))) {
      // Handle escaped characters with special function
      tokens.push_back(
          {TokenType::EscapeInCharClass,
           std::string(cur_, cur_ + 2),
           std::string(cur_ + 1, cur_ + 2),
           cur_line_,
           cur_column_}
      );
      Consume(2);
    } else {
      // Handle normal characters
      auto [codepoint, len] = ParseNextUTF8OrEscaped(cur_, kRegexEscapeChars);
      if (codepoint == CharHandlingError::kInvalidUTF8) {
        ReportLexerError("Invalid UTF8 sequence");
      }

      if (codepoint == CharHandlingError::kInvalidEscape) {
        ReportLexerError("Invalid escape sequence" + std::string(cur_, cur_ + 2));
      }

      tokens.push_back(
          {TokenType::CharInCharClass,
           std::string(cur_, cur_ + len),
           codepoint,
           cur_line_,
           cur_column_}
      );
      Consume(len);
    }
  }

  if (!Peek()) {
    ReportLexerError("Unterminated character class");
  }

  tokens.push_back({TokenType::RBracket, "]", "", cur_line_, cur_column_});
  Consume();  // Skip ']'

  return tokens;
}

// Parse integer
EBNFLexer::Token EBNFLexer::Impl::ParseIntegerToken() {
  int start_line = cur_line_;
  int start_column = cur_column_;
  const char* start_pos = cur_;
  bool is_negative = false;

  if (Peek() == '-') {
    is_negative = true;
    Consume();
  } else if (Peek() == '+') {
    Consume();
  }

  int64_t num = 0;
  while (Peek() && isdigit(Peek())) {
    num = num * 10 + (Peek() - '0');
    Consume();
    if (num > kMaxIntegerInGrammar) {
      ReportLexerError(
          "Integer is too large: parsed " + std::to_string(num) + ", max allowed is " +
          std::to_string(kMaxIntegerInGrammar)
      );
    }
  }

  std::string lexeme(start_pos, cur_ - start_pos);
  return {TokenType::IntegerLiteral, lexeme, is_negative ? -num : num, start_line, start_column};
}

// Get the next token
std::variant<EBNFLexer::Token, std::vector<EBNFLexer::Token>> EBNFLexer::Impl::NextToken() {
  ConsumeSpace();  // Skip whitespace and comments

  auto start_line = cur_line_;
  auto start_column = cur_column_;

  if (!Peek()) {
    return EBNFLexer::Token{TokenType::EndOfFile, "", "", start_line, start_column};
  }

  // Determine token type based on current character
  switch (Peek()) {
    case '(':
      if (Peek(1) == '=') {
        Consume(2);
        return EBNFLexer::Token{TokenType::LookaheadLParen, "(=", "", start_line, start_column};
      } else {
        Consume();
        return EBNFLexer::Token{TokenType::LParen, "(", "", start_line, start_column};
      }
    case ')':
      Consume();
      return EBNFLexer::Token{TokenType::RParen, ")", "", start_line, start_column};
    case '{':
      Consume();
      return EBNFLexer::Token{TokenType::LBrace, "{", "", start_line, start_column};
    case '}':
      Consume();
      return EBNFLexer::Token{TokenType::RBrace, "}", "", start_line, start_column};
    case '|':
      Consume();
      return EBNFLexer::Token{TokenType::Pipe, "|", "", start_line, start_column};
    case ',':
      Consume();
      return EBNFLexer::Token{TokenType::Comma, ",", "", start_line, start_column};
    case '*':
      Consume();
      return EBNFLexer::Token{TokenType::Star, "*", "", start_line, start_column};
    case '+':
      Consume();
      return EBNFLexer::Token{TokenType::Plus, "+", "", start_line, start_column};
    case '?':
      Consume();
      return EBNFLexer::Token{TokenType::Question, "?", "", start_line, start_column};
    case '=':
      Consume();
      return EBNFLexer::Token{TokenType::Equal, "=", "", start_line, start_column};
    case ':':
      if (Peek(1) == ':' && Peek(2) == '=') {
        Consume(3);
        return EBNFLexer::Token{TokenType::Assign, "::=", "", start_line, start_column};
      }
      ReportLexerError("Unexpected character: ':'");
      break;
    case '"':
      return ParseStringToken();
    case '[':
      return ParseCharClassToken();
    default:
      if (IsNameChar(*cur_, true)) {
        return ParseIdentifierOrBooleanToken();
      } else if (isdigit(*cur_) || *cur_ == '-' || *cur_ == '+') {
        return ParseIntegerToken();
      }

      // Unrecognized character, report error
      ReportLexerError("Unexpected character: " + std::string(1, *cur_));
  }

  // Should not reach here
  XGRAMMAR_UNREACHABLE();
}

void EBNFLexer::Impl::ConvertIdentifierToRuleName(std::vector<Token>* tokens) {
  for (int i = 0; i < static_cast<int>(tokens->size()); ++i) {
    if (tokens->at(i).type == TokenType::Assign) {
      if (i == 0) {
        ReportLexerError(
            "Assign should not be the first token", tokens->at(i).line, tokens->at(i).column
        );
      }
      if (tokens->at(i - 1).type != TokenType::Identifier) {
        ReportLexerError(
            "Assign should be preceded by an identifier",
            tokens->at(i - 1).line,
            tokens->at(i - 1).column
        );
      }
      if (i >= 2 && tokens->at(i - 2).line == tokens->at(i - 1).line) {
        ReportLexerError(
            "The rule name should be at the beginning of the line",
            tokens->at(i - 1).line,
            tokens->at(i - 1).column
        );
      }
      tokens->at(i - 1).type = TokenType::RuleName;
    }
  }
}

// Tokenize the entire input and return a vector of tokens
std::vector<EBNFLexer::Token> EBNFLexer::Impl::Tokenize(const std::string& input) {
  // Reset position to the beginning
  input_ = input;
  cur_ = input_.c_str();
  cur_line_ = 1;
  cur_column_ = 1;

  // Collect all tokens
  std::vector<Token> tokens;

  while (true) {
    auto token = NextToken();

    if (auto* token_value = std::get_if<Token>(&token)) {
      tokens.push_back(*token_value);
      // Stop when we reach the end of file
      if (token_value->type == TokenType::EndOfFile) {
        break;
      }
    } else {
      auto vec = std::get_if<std::vector<Token>>(&token);
      XGRAMMAR_DCHECK(vec != nullptr);
      tokens.insert(tokens.end(), vec->begin(), vec->end());
    }
  }

  ConvertIdentifierToRuleName(&tokens);

  return tokens;
}

EBNFLexer::EBNFLexer() : pimpl_(std::make_shared<Impl>()) {}

std::vector<EBNFLexer::Token> EBNFLexer::Tokenize(const std::string& input) {
  return pimpl_->Tokenize(input);
}

class EBNFParser {
 public:
  /*! \brief The logic of parsing the grammar string. */
  Grammar Parse(
      const std::vector<EBNFLexer::Token>& tokens,
      const std::string& root_rule_name,
      const int& max_nest_layer = 1000
  );

 private:
  using Rule = Grammar::Impl::Rule;
  using SuffixStopInfo = Grammar::Impl::SuffixStopInfo;
  using GrammarExprType = Grammar::Impl::GrammarExprType;
  using Token = EBNFLexer::Token;
  using TokenType = EBNFLexer::TokenType;

  struct ParsedRule {
    Rule rule;
    SuffixStopInfo suffix_stop_info;
  };

  // Parsing different parts of the grammar
  std::string ParseIdentifier();
  int32_t ParseCharClass();
  int32_t ParseString();
  int32_t ParseRuleRef();
  int32_t ParseElement();
  int64_t ParseInteger();
  std::pair<int64_t, int64_t> ParseRepetitionRange();
  int32_t ParseElementWithQuantifier();
  int32_t ParseLookaheadAssertion();
  int32_t ParseSequence();
  int32_t ParseChoices();
  ParsedRule ParseRule();

  // Parser for macro
  class MacroIR {
   public:
    struct StringNode;
    struct IntegerNode;
    struct BooleanNode;
    struct IdentifierNode;
    struct TupleNode;

    using Node = std::variant<StringNode, IntegerNode, BooleanNode, IdentifierNode, TupleNode>;
    using NodePtr = std::unique_ptr<Node>;

    struct StringNode {
      std::string value;
    };
    struct IntegerNode {
      int64_t value;
    };
    struct BooleanNode {
      bool value;
    };
    struct IdentifierNode {
      std::string name;
    };
    struct TupleNode {
      std::vector<NodePtr> elements;
    };

    struct Arguments {
      std::vector<NodePtr> arguments;
      std::unordered_map<std::string, NodePtr> named_arguments;
    };
  };
  MacroIR::Arguments ParseMacroArguments();
  MacroIR::NodePtr ParseMacroValue();

  int32_t ParseTagDispatch();
  int32_t ParseTokenSet();
  int32_t ParseExcludeToken();
  int32_t ParseTokenTagDispatch();
  int32_t ParseRegexMacro();
  int32_t ParseSubstringMacro();

  // Helper functions

  // Helper for ParseElementWithQuantifier
  int32_t HandleStarQuantifier(int32_t grammar_expr_id);
  int32_t HandlePlusQuantifier(int32_t grammar_expr_id);
  int32_t HandleQuestionQuantifier(int32_t grammar_expr_id);

  // When parsing, we first find the names of all rules, and build the mapping from name to rule id.
  void InitRuleNames();

  // Consume a token and advance to the next
  void Consume(int cnt = 1);

  // Peek at the current token with optional offset
  const Token& Peek(int delta = 0) const;

  // Consume token if it matches expected type, otherwise report error
  void PeekAndConsume(TokenType type, const std::string& message);

  // Report a parsing error with the given message
  [[noreturn]] void ReportParseError(const std::string& msg, int delta_element = 0);

  // The grammar builder
  GrammarBuilder builder_;

  // The current token pointer
  const Token* current_token_ = nullptr;

  // Tokens from lexer
  std::vector<Token> tokens_;

  // The current rule name. Help to generate a name for a new rule.
  std::string cur_rule_name_;

  // The name of the root rule
  std::string root_rule_name_;

  int nest_layer_guard_ = 0;

  int max_nest_layer_ = 1000;  // Max nest layer of the grammar

  static const std::unordered_map<std::string, std::function<int32_t(EBNFParser*)>> kMacroFunctions;
};

const std::unordered_map<std::string, std::function<int32_t(EBNFParser*)>>
    EBNFParser::kMacroFunctions = {
        {"TagDispatch", [](EBNFParser* parser) { return parser->ParseTagDispatch(); }},
        {"Token", [](EBNFParser* parser) { return parser->ParseTokenSet(); }},
        {"ExcludeToken", [](EBNFParser* parser) { return parser->ParseExcludeToken(); }},
        {"TokenTagDispatch", [](EBNFParser* parser) { return parser->ParseTokenTagDispatch(); }},
        {"Regex", [](EBNFParser* parser) { return parser->ParseRegexMacro(); }},
        {"Substring", [](EBNFParser* parser) { return parser->ParseSubstringMacro(); }},
};

const EBNFParser::Token& EBNFParser::Peek(int delta) const { return *(current_token_ + delta); }

void EBNFParser::Consume(int cnt) { current_token_ += cnt; }

void EBNFParser::PeekAndConsume(TokenType type, const std::string& message) {
  if (Peek().type != type) {
    ReportParseError(message);
  }
  Consume();
}

void EBNFParser::ReportParseError(const std::string& msg, int delta_element) {
  XGRAMMAR_DCHECK(current_token_ + delta_element < tokens_.data() + tokens_.size());
  int line_to_print = Peek(delta_element).line;
  int column_to_print = Peek(delta_element).column;
  XGRAMMAR_LOG(FATAL) << "EBNF parser error at line " + std::to_string(line_to_print) +
                             ", column " + std::to_string(column_to_print) + ": " + msg;
  XGRAMMAR_UNREACHABLE();
}

std::string EBNFParser::ParseIdentifier() {
  if (Peek().type != TokenType::Identifier) {
    ReportParseError("Expect identifier");
  }
  std::string identifier = std::any_cast<std::string>(Peek().value);
  Consume();
  return identifier;
}

int32_t EBNFParser::ParseCharClass() {
  PeekAndConsume(TokenType::LBracket, "Expect [ in character class");

  std::vector<GrammarBuilder::CharacterClassElement> elements;
  bool is_negated = false;

  if (Peek().type == TokenType::Caret) {
    is_negated = true;
    Consume();
  }

  while (Peek().type != TokenType::RBracket && Peek().type != TokenType::EndOfFile) {
    if (Peek().type == TokenType::EscapeInCharClass) {
      ReportParseError("Character class escape is not supported yet in EBNF");
    }

    TCodepoint codepoint;
    if (Peek().type == TokenType::CharInCharClass) {
      codepoint = std::any_cast<TCodepoint>(Peek().value);
    } else if (Peek().type == TokenType::Dash) {
      codepoint = static_cast<TCodepoint>(static_cast<uint8_t>('-'));
    } else {
      ReportParseError("Unexpected character in character class: " + Peek().lexeme);
    }
    Consume();

    if (Peek().type == TokenType::Dash &&
        (Peek(1).type == TokenType::CharInCharClass || Peek(1).type == TokenType::Dash)) {
      // Range expression
      TCodepoint codepoint2;
      if (Peek(1).type == TokenType::CharInCharClass) {
        codepoint2 = std::any_cast<TCodepoint>(Peek(1).value);
      } else {
        XGRAMMAR_DCHECK(Peek(1).type == TokenType::Dash);
        codepoint2 = static_cast<TCodepoint>(static_cast<uint8_t>('-'));
      }

      if (codepoint > codepoint2) {
        ReportParseError("Invalid character class: lower bound is larger than upper bound", -1);
      }
      elements.push_back({codepoint, codepoint2});
      Consume(2);
    } else {
      // Single character
      elements.push_back({codepoint, codepoint});
    }
  }

  PeekAndConsume(TokenType::RBracket, "Expect ] in character class");

  return builder_.AddCharacterClass(elements, is_negated);
}

int32_t EBNFParser::ParseString() {
  if (Peek().type != TokenType::StringLiteral) {
    ReportParseError("Expect string literal");
  }

  std::string str_value = std::any_cast<std::string>(Peek().value);
  Consume();

  if (str_value.empty()) {
    return builder_.AddEmptyStr();
  }

  return builder_.AddByteString(str_value);
}

int32_t EBNFParser::ParseRuleRef() {
  std::string name = ParseIdentifier();
  auto rule_id = builder_.GetRuleId(name);
  if (rule_id == -1) {
    ReportParseError("Rule \"" + name + "\" is not defined", -1);
  }
  return builder_.AddRuleRef(rule_id);
}

int32_t EBNFParser::ParseElement() {
  if (Peek().type == TokenType::LParen) {
    nest_layer_guard_++;
    if (nest_layer_guard_ > max_nest_layer_) {
      ReportParseError("Nest layer exceeded the maximum limit", -1);
    }
    Consume();
    if (Peek().type == TokenType::RParen) {
      // Special case: ( )
      Consume();
      nest_layer_guard_--;
      return builder_.AddEmptyStr();
    }
    auto grammar_expr_id = ParseChoices();
    PeekAndConsume(TokenType::RParen, "Expect )");
    nest_layer_guard_--;
    return grammar_expr_id;
  } else if (Peek().type == TokenType::LBracket) {
    return ParseCharClass();
  } else if (Peek().type == TokenType::StringLiteral) {
    return ParseString();
  } else if (Peek().type == TokenType::Identifier) {
    auto id = std::any_cast<std::string>(Peek().value);
    if (kMacroFunctions.count(id)) {
      return kMacroFunctions.at(id)(this);
    } else {
      return ParseRuleRef();
    }
  } else {
    ReportParseError("Expect element, but got " + Peek().lexeme);
  }
}

int64_t EBNFParser::ParseInteger() {
  if (Peek().type != TokenType::IntegerLiteral) {
    ReportParseError("Expect integer, but got " + Peek().lexeme);
  }
  int64_t num = std::any_cast<int64_t>(Peek().value);
  Consume();
  return num;
}

std::pair<int64_t, int64_t> EBNFParser::ParseRepetitionRange() {
  PeekAndConsume(TokenType::LBrace, "Expect {");

  int64_t lower = ParseInteger();

  if (lower < 0) {
    ReportParseError("Lower bound cannot be negative", -1);
  }

  if (Peek().type == TokenType::Comma) {
    Consume();
    if (Peek().type == TokenType::RBrace) {
      Consume();
      return {lower, -1};
    }
    // The grammar printer emits {n, -1} for unbounded upper bounds, and
    // '-' is a valid identifier-start char (IsNameChar), so the lexer
    // produces Identifier("-1") rather than IntegerLiteral.  Accept it
    // as equivalent to {n,}.
    if (Peek().type == TokenType::Identifier && Peek().lexeme == "-1") {
      Consume();
      PeekAndConsume(TokenType::RBrace, "Expect }");
      return {lower, -1};
    }
    int64_t upper = ParseInteger();
    if (upper < lower) {
      ReportParseError(
          "Lower bound is larger than upper bound: " + std::to_string(lower) + " > " +
              std::to_string(upper),
          -1
      );
    }
    PeekAndConsume(TokenType::RBrace, "Expect }");
    return {lower, upper};
  } else if (Peek().type == TokenType::RBrace) {
    Consume();
    return {lower, lower};
  }

  ReportParseError("Expect ',' or '}' in repetition range");
}

int32_t EBNFParser::HandleStarQuantifier(int32_t grammar_expr_id) {
  Grammar::Impl::GrammarExpr grammar_expr = builder_.GetGrammarExpr(grammar_expr_id);
  if (grammar_expr.type == GrammarBuilder::GrammarExprType::kCharacterClass) {
    // We have special handling for character class star, e.g. [a-z]*
    grammar_expr.type = GrammarBuilder::GrammarExprType::kCharacterClassStar;
    // Copy grammar expr because the grammar may change during insertion, and grammar_expr is in the
    // grammar, so it may become invalid
    std::vector<int32_t> grammar_expr_data(grammar_expr.begin(), grammar_expr.end());
    return builder_.AddGrammarExpr(
        {grammar_expr.type, grammar_expr_data.data(), grammar_expr.data_len}
    );
  } else {
    // For other star quantifiers, we transform it into a rule:
    // a*  -->  rule ::= a rule | ""
    auto new_rule_name = builder_.GetNewRuleName(cur_rule_name_);
    auto new_rule_id = builder_.AddEmptyRule(new_rule_name);
    auto ref_to_new_rule = builder_.AddRuleRef(new_rule_id);
    auto new_grammar_expr_id = builder_.AddChoices(
        {builder_.AddEmptyStr(), builder_.AddSequence({grammar_expr_id, ref_to_new_rule})}
    );
    builder_.UpdateRuleBody(new_rule_id, new_grammar_expr_id);

    // Return the reference to the new rule
    return builder_.AddRuleRef(new_rule_id);
  }
}

int32_t EBNFParser::HandlePlusQuantifier(int32_t grammar_expr_id) {
  // a+  -->  rule ::= a rule | a
  auto new_rule_name = builder_.GetNewRuleName(cur_rule_name_);
  auto new_rule_id = builder_.AddEmptyRule(new_rule_name);
  auto ref_to_new_rule = builder_.AddRuleRef(new_rule_id);
  auto new_grammar_expr_id = builder_.AddChoices(
      {builder_.AddSequence({grammar_expr_id, ref_to_new_rule}), grammar_expr_id}
  );
  builder_.UpdateRuleBody(new_rule_id, new_grammar_expr_id);

  // Return the reference to the new rule
  return builder_.AddRuleRef(new_rule_id);
}

int32_t EBNFParser::HandleQuestionQuantifier(int32_t grammar_expr_id) {
  // a?  -->  rule ::= a | empty
  auto new_rule_name = builder_.GetNewRuleName(cur_rule_name_);
  auto new_grammar_expr_id = builder_.AddChoices({builder_.AddEmptyStr(), grammar_expr_id});
  auto new_rule_id = builder_.AddRule({new_rule_name, new_grammar_expr_id});
  return builder_.AddRuleRef(new_rule_id);
}

int32_t EBNFParser::ParseElementWithQuantifier() {
  int32_t grammar_expr_id = ParseElement();

  if (Peek().type == TokenType::Star) {
    Consume();
    return HandleStarQuantifier(grammar_expr_id);
  } else if (Peek().type == TokenType::Plus) {
    Consume();
    return HandlePlusQuantifier(grammar_expr_id);
  } else if (Peek().type == TokenType::Question) {
    Consume();
    return HandleQuestionQuantifier(grammar_expr_id);
  } else if (Peek().type == TokenType::LBrace) {
    auto [lower, upper] = ParseRepetitionRange();
    return builder_.AddRepeatFromExpr(
        cur_rule_name_,
        grammar_expr_id,
        static_cast<int32_t>(lower),
        upper == -1 ? -1 : static_cast<int32_t>(upper)
    );
  }

  return grammar_expr_id;
}

int32_t EBNFParser::ParseSequence() {
  std::vector<int32_t> elements;

  do {
    elements.push_back(ParseElementWithQuantifier());
  } while (Peek().type != TokenType::Pipe && Peek().type != TokenType::RParen &&
           Peek().type != TokenType::LookaheadLParen && Peek().type != TokenType::RuleName &&
           Peek().type != TokenType::EndOfFile);

  return builder_.AddSequence(elements);
}

int32_t EBNFParser::ParseChoices() {
  std::vector<int32_t> choices;

  choices.push_back(ParseSequence());

  while (Peek().type == TokenType::Pipe) {
    Consume();
    choices.push_back(ParseSequence());
  }

  return builder_.AddChoices(choices);
}

// Parse macro arguments and return a MacroIR::Arguments structure
EBNFParser::MacroIR::Arguments EBNFParser::ParseMacroArguments() {
  MacroIR::Arguments args;

  PeekAndConsume(TokenType::LParen, "Expect ( after macro function name");

  // Parse arguments
  if (Peek().type != TokenType::RParen) {
    while (true) {
      // Check if it's a named argument (identifier = value)
      if (Peek().type == TokenType::Identifier && Peek(1).type == TokenType::Equal) {
        std::string name = std::any_cast<std::string>(Peek().value);
        Consume();  // Consume identifier
        Consume();  // Consume =

        // Parse the value
        args.named_arguments[name] = ParseMacroValue();
      } else {
        // Regular positional argument
        args.arguments.push_back(ParseMacroValue());
      }

      // Check for comma or end of arguments
      if (Peek().type == TokenType::Comma) {
        Consume();
      } else if (Peek().type == TokenType::RParen) {
        break;
      } else {
        ReportParseError("Expect , or ) in macro arguments");
      }
    }
  }

  PeekAndConsume(TokenType::RParen, "Expect ) after macro arguments");
  return args;
}

// Parse a single macro value (string, integer, boolean, or tuple)
EBNFParser::MacroIR::NodePtr EBNFParser::ParseMacroValue() {
  if (Peek().type == TokenType::StringLiteral) {
    // String value
    std::string value = std::any_cast<std::string>(Peek().value);
    Consume();
    return std::make_unique<MacroIR::Node>(MacroIR::StringNode{value});
  } else if (Peek().type == TokenType::IntegerLiteral) {
    // Integer value
    int64_t value = std::any_cast<int64_t>(Peek().value);
    Consume();
    return std::make_unique<MacroIR::Node>(MacroIR::IntegerNode{value});
  } else if (Peek().type == TokenType::BooleanLiteral) {
    // Boolean value
    bool value = std::any_cast<bool>(Peek().value);
    Consume();
    return std::make_unique<MacroIR::Node>(MacroIR::BooleanNode{value});
  } else if (Peek().type == TokenType::Identifier) {
    // Identifier value
    std::string name = std::any_cast<std::string>(Peek().value);
    Consume();
    return std::make_unique<MacroIR::Node>(MacroIR::IdentifierNode{name});
  } else if (Peek().type == TokenType::LParen) {
    // Tuple value. Nested tuples recurse once per layer, so guard the depth the same way the
    // ordinary parenthesis path does to avoid a stack overflow on deeply nested input.
    nest_layer_guard_++;
    if (nest_layer_guard_ > max_nest_layer_) {
      ReportParseError("Nest layer exceeded the maximum limit", -1);
    }
    Consume();  // Consume (

    MacroIR::TupleNode tuple;

    // Parse tuple elements (supports trailing comma)
    if (Peek().type != TokenType::RParen) {
      while (true) {
        tuple.elements.push_back(ParseMacroValue());

        if (Peek().type == TokenType::Comma) {
          Consume();
          if (Peek().type == TokenType::RParen) {
            break;
          }
        } else if (Peek().type == TokenType::RParen) {
          break;
        } else {
          ReportParseError("Expect , or ) in tuple");
        }
      }
    }

    Consume();  // Consume )
    nest_layer_guard_--;
    return std::make_unique<MacroIR::Node>(std::move(tuple));
  } else {
    ReportParseError("Expect string, integer, boolean, or tuple in macro argument");
  }
}

int32_t EBNFParser::ParseTagDispatch() {
  Consume();  // Consume TagDispatch operator
  auto start = current_token_;
  auto args = ParseMacroArguments();
  auto delta_element = start - current_token_;  // Used to report parse errors

  Grammar::Impl::TagDispatch tag_dispatch;

  static const std::unordered_set<std::string> kValidNamedArgs = {
      "loop_after_dispatch", "excludes"
  };
  for (const auto& [name, _] : args.named_arguments) {
    if (kValidNamedArgs.count(name) == 0) {
      ReportParseError("Unknown named argument for TagDispatch: " + name, delta_element);
    }
  }

  // Positional parameters: ("tag_string", rule_name) — string triggers only
  for (const auto& arg : args.arguments) {
    auto tuple_node = std::get_if<MacroIR::TupleNode>(arg.get());
    if (tuple_node == nullptr) {
      ReportParseError("Each tag dispatch element must be a tuple", delta_element);
    }

    if (tuple_node->elements.size() != 2) {
      ReportParseError("Each tag dispatch element must be a pair (tag, rule)", delta_element);
    }

    auto tag_str_node = std::get_if<MacroIR::StringNode>(tuple_node->elements[0].get());
    if (tag_str_node == nullptr || tag_str_node->value.empty()) {
      ReportParseError("Tag must be a non-empty string literal", delta_element);
    }

    auto rule_name_node = std::get_if<MacroIR::IdentifierNode>(tuple_node->elements[1].get());
    if (rule_name_node == nullptr) {
      ReportParseError("Rule reference must be an identifier", delta_element);
    }

    auto rule_id = builder_.GetRuleId(rule_name_node->name);
    if (rule_id == -1) {
      ReportParseError("Rule \"" + rule_name_node->name + "\" is not defined", delta_element);
    }
    tag_dispatch.tag_rule_pairs.push_back({tag_str_node->value, rule_id});
  }

  // loop_after_dispatch
  tag_dispatch.loop_after_dispatch = true;
  if (auto it = args.named_arguments.find("loop_after_dispatch");
      it != args.named_arguments.end()) {
    auto bool_node = std::get_if<MacroIR::BooleanNode>(it->second.get());
    if (bool_node == nullptr) {
      ReportParseError("loop_after_dispatch must be a boolean literal", delta_element);
    }
    tag_dispatch.loop_after_dispatch = bool_node->value;
  }

  // excludes — string only
  if (auto it = args.named_arguments.find("excludes"); it != args.named_arguments.end()) {
    auto tuple_node = std::get_if<MacroIR::TupleNode>(it->second.get());
    if (tuple_node == nullptr) {
      ReportParseError("excludes must be a tuple", delta_element);
    }
    for (const auto& element : tuple_node->elements) {
      auto str_node = std::get_if<MacroIR::StringNode>(element.get());
      if (str_node == nullptr || str_node->value.empty()) {
        ReportParseError("Exclude must be a non-empty string literal", delta_element);
      }
      tag_dispatch.excludes.push_back(str_node->value);
    }
  }

  // Well-formedness checks: string excludes vs string triggers
  for (const auto& excl_str : tag_dispatch.excludes) {
    for (const auto& [trigger_str, _] : tag_dispatch.tag_rule_pairs) {
      if (trigger_str.rfind(excl_str, 0) == 0) {
        ReportParseError(
            "Exclude string must not be a prefix of trigger string: " + excl_str, delta_element
        );
      }
    }
  }

  return builder_.AddTagDispatch(tag_dispatch);
}

int32_t EBNFParser::ParseRegexMacro() {
  Consume();  // Consume Regex operator
  auto start = current_token_;
  auto args = ParseMacroArguments();
  auto delta_element = start - current_token_;  // Used to report parse errors

  if (args.arguments.size() != 1) {
    ReportParseError("Regex expects exactly one string argument", delta_element);
  }
  auto pattern_node = std::get_if<MacroIR::StringNode>(args.arguments[0].get());
  if (pattern_node == nullptr) {
    ReportParseError("Regex pattern must be a string literal", delta_element);
  }

  bool json_string = false;
  for (const auto& [name, _] : args.named_arguments) {
    if (name != "json_string" && name != "flags") {
      ReportParseError("Regex does not support the named argument " + name, delta_element);
    }
  }
  if (auto it = args.named_arguments.find("json_string"); it != args.named_arguments.end()) {
    auto bool_node = std::get_if<MacroIR::BooleanNode>(it->second.get());
    if (bool_node == nullptr) {
      ReportParseError("json_string must be a boolean", delta_element);
    }
    json_string = bool_node->value;
  }
  std::string pattern = pattern_node->value;
  if (auto it = args.named_arguments.find("flags"); it != args.named_arguments.end()) {
    auto flags_node = std::get_if<MacroIR::StringNode>(it->second.get());
    if (flags_node == nullptr) {
      ReportParseError("flags must be a string", delta_element);
    }
    bool case_insensitive = false;
    bool dot_all = false;
    for (char flag : flags_node->value) {
      if (flag == 'i') {
        case_insensitive = true;
      } else if (flag == 's') {
        dot_all = true;
      } else if (flag == 'u') {
        // XGrammar regular expressions use Unicode codepoint semantics by default.
      } else {
        ReportParseError(
            "regular-expression flag '" + std::string(1, flag) + "' is not supported", delta_element
        );
      }
    }
    // The flags argument opts into the standard dot semantics: '.' does not match '\n' unless
    // the 's' flag is given. Without the argument the pattern is stored verbatim, where the
    // engine's '.' matches every codepoint.
    pattern = RewriteRegexDots(pattern, dot_all);
    if (case_insensitive && pattern.compare(0, 4, "(?i)") != 0) {
      pattern = "(?i)" + pattern;
    }
  }
  return builder_.AddRegex(pattern, json_string);
}

int32_t EBNFParser::ParseSubstringMacro() {
  Consume();  // Consume Substring identifier
  auto start = current_token_;
  auto args = ParseMacroArguments();
  auto delta_element = start - current_token_;

  if (!args.named_arguments.empty()) {
    ReportParseError("Substring() does not accept named arguments", delta_element);
  }

  std::vector<std::string> chunks;
  chunks.reserve(args.arguments.size());
  for (const auto& arg : args.arguments) {
    auto string_node = std::get_if<MacroIR::StringNode>(arg.get());
    if (string_node == nullptr) {
      ReportParseError("Substring() arguments must be strings", delta_element);
    }
    chunks.push_back(string_node->value);
  }

  return builder_.AddSubstring(chunks);
}

int32_t EBNFParser::ParseTokenSet() {
  Consume();  // Consume Token identifier
  auto start = current_token_;
  auto args = ParseMacroArguments();
  auto delta_element = start - current_token_;

  if (!args.named_arguments.empty()) {
    ReportParseError("Token() does not accept named arguments", delta_element);
  }

  if (args.arguments.empty()) {
    ReportParseError("Token() requires at least one integer argument", delta_element);
  }

  std::vector<int32_t> token_ids;
  for (const auto& arg : args.arguments) {
    auto int_node = std::get_if<MacroIR::IntegerNode>(arg.get());
    if (int_node == nullptr || int_node->value < 0) {
      ReportParseError("Token() arguments must be non-negative integers", delta_element);
    }
    token_ids.push_back(static_cast<int32_t>(int_node->value));
  }

  std::sort(token_ids.begin(), token_ids.end());
  token_ids.erase(std::unique(token_ids.begin(), token_ids.end()), token_ids.end());

  return builder_.AddTokenSet(token_ids);
}

int32_t EBNFParser::ParseExcludeToken() {
  Consume();
  auto start = current_token_;
  auto args = ParseMacroArguments();
  auto delta_element = start - current_token_;

  if (!args.named_arguments.empty()) {
    ReportParseError("ExcludeToken() does not accept named arguments", delta_element);
  }
  if (args.arguments.empty()) {
    ReportParseError("ExcludeToken() requires at least one integer argument", delta_element);
  }

  std::vector<int32_t> token_ids;
  for (const auto& arg : args.arguments) {
    auto int_node = std::get_if<MacroIR::IntegerNode>(arg.get());
    if (int_node == nullptr || int_node->value < 0) {
      ReportParseError("ExcludeToken() arguments must be non-negative integers", delta_element);
    }
    token_ids.push_back(static_cast<int32_t>(int_node->value));
  }
  std::sort(token_ids.begin(), token_ids.end());
  token_ids.erase(std::unique(token_ids.begin(), token_ids.end()), token_ids.end());

  return builder_.AddExcludeTokenSet(token_ids);
}

int32_t EBNFParser::ParseTokenTagDispatch() {
  Consume();
  auto start = current_token_;
  auto args = ParseMacroArguments();
  auto delta_element = start - current_token_;

  Grammar::Impl::TokenTagDispatch ttd;

  static const std::unordered_set<std::string> kValidNamedArgs = {
      "loop_after_dispatch", "excludes"
  };
  for (const auto& [name, _] : args.named_arguments) {
    if (kValidNamedArgs.count(name) == 0) {
      ReportParseError("Unknown named argument for TokenTagDispatch: " + name, delta_element);
    }
  }

  for (const auto& arg : args.arguments) {
    auto tuple_node = std::get_if<MacroIR::TupleNode>(arg.get());
    if (tuple_node == nullptr || tuple_node->elements.size() != 2) {
      ReportParseError(
          "Each TokenTagDispatch element must be a pair (token_id, rule)", delta_element
      );
    }
    auto id_node = std::get_if<MacroIR::IntegerNode>(tuple_node->elements[0].get());
    if (id_node == nullptr || id_node->value < 0) {
      ReportParseError("Token trigger ID must be a non-negative integer", delta_element);
    }
    auto rule_node = std::get_if<MacroIR::IdentifierNode>(tuple_node->elements[1].get());
    if (rule_node == nullptr) {
      ReportParseError("Rule reference must be an identifier", delta_element);
    }
    auto rule_id = builder_.GetRuleId(rule_node->name);
    if (rule_id == -1) {
      ReportParseError("Rule \"" + rule_node->name + "\" is not defined", delta_element);
    }
    ttd.trigger_rule_pairs.push_back({static_cast<int32_t>(id_node->value), rule_id});
  }

  ttd.loop_after_dispatch = true;
  if (auto it = args.named_arguments.find("loop_after_dispatch");
      it != args.named_arguments.end()) {
    auto bool_node = std::get_if<MacroIR::BooleanNode>(it->second.get());
    if (bool_node == nullptr) {
      ReportParseError("loop_after_dispatch must be a boolean", delta_element);
    }
    ttd.loop_after_dispatch = bool_node->value;
  }

  if (auto it = args.named_arguments.find("excludes"); it != args.named_arguments.end()) {
    auto tuple_node = std::get_if<MacroIR::TupleNode>(it->second.get());
    if (tuple_node == nullptr) {
      ReportParseError("excludes must be a tuple", delta_element);
    }
    for (const auto& element : tuple_node->elements) {
      auto int_node = std::get_if<MacroIR::IntegerNode>(element.get());
      if (int_node == nullptr || int_node->value < 0) {
        ReportParseError("Exclude token ID must be a non-negative integer", delta_element);
      }
      ttd.excludes.push_back(static_cast<int32_t>(int_node->value));
    }
  }

  for (auto excl_id : ttd.excludes) {
    for (const auto& [tid, _] : ttd.trigger_rule_pairs) {
      if (tid == excl_id) {
        ReportParseError(
            "Token trigger ID " + std::to_string(tid) + " must not overlap with exclude token ID",
            delta_element
        );
      }
    }
  }

  return builder_.AddTokenTagDispatch(ttd);
}

int32_t EBNFParser::ParseLookaheadAssertion() {
  PeekAndConsume(TokenType::LookaheadLParen, "Expect (= in lookahead assertion");
  auto result = ParseChoices();
  PeekAndConsume(TokenType::RParen, "Expect )");
  return result;
}

EBNFParser::ParsedRule EBNFParser::ParseRule() {
  if (Peek().type != TokenType::RuleName) {
    ReportParseError("Expect rule name");
  }
  cur_rule_name_ = std::any_cast<std::string>(Peek().value);
  int32_t max_tokens = Peek().max_tokens;
  int32_t max_chars = Peek().max_chars;
  std::string capture_name = Peek().capture_name;
  int32_t capture_hidden_suffix_bytes = Peek().capture_hidden_suffix_bytes;
  int32_t capture_hidden_stop_bytes = Peek().capture_hidden_stop_bytes;
  int32_t capture_hidden_body_rule_id = Peek().capture_hidden_body_rule_id;
  int32_t capture_hidden_marker_rule_id = Peek().capture_hidden_marker_rule_id;
  std::string stop_capture_name = Peek().stop_capture_name;
  bool is_lazy = Peek().is_lazy;
  std::optional<float> temperature = Peek().temperature;
  Consume();

  PeekAndConsume(TokenType::Assign, "Expect ::=");

  auto body_id = ParseChoices();

  int32_t lookahead_id = -1;
  if (Peek().type == TokenType::LookaheadLParen) {
    lookahead_id = ParseLookaheadAssertion();
  }

  ParsedRule result;
  result.rule = Rule{cur_rule_name_, body_id, lookahead_id};
  result.rule.max_tokens = max_tokens;
  result.rule.max_chars = max_chars;
  result.rule.capture_name = capture_name;
  result.rule.is_lazy = is_lazy;
  result.rule.temperature = temperature;
  result.suffix_stop_info.hidden_suffix_bytes = capture_hidden_suffix_bytes;
  result.suffix_stop_info.hidden_stop_bytes = capture_hidden_stop_bytes;
  result.suffix_stop_info.body_rule_id = capture_hidden_body_rule_id;
  result.suffix_stop_info.marker_rule_id = capture_hidden_marker_rule_id;
  result.suffix_stop_info.stop_capture_name = std::move(stop_capture_name);
  return result;
}

void EBNFParser::InitRuleNames() {
  int delta_element = 0;
  for (auto& token : tokens_) {
    if (token.type == TokenType::RuleName) {
      auto name = std::any_cast<std::string>(token.value);
      if (builder_.GetRuleId(name) != -1) {
        ReportParseError("Rule \"" + name + "\" is defined multiple times", delta_element);
      }
      builder_.AddEmptyRule(name);
    }
    ++delta_element;
  }
  if (builder_.GetRuleId(root_rule_name_) == -1) {
    ReportParseError("The root rule with name \"" + root_rule_name_ + "\" is not found", 0);
  }
}

Grammar EBNFParser::Parse(
    const std::vector<EBNFLexer::Token>& tokens,
    const std::string& root_rule_name,
    const int& max_nest_layer
) {
  max_nest_layer_ = max_nest_layer;
  nest_layer_guard_ = 0;
  tokens_ = tokens;
  current_token_ = tokens_.data();
  root_rule_name_ = root_rule_name;

  // First collect rule names
  InitRuleNames();

  // Then parse all the rules
  while (Peek().type != TokenType::EndOfFile) {
    auto parsed_rule = ParseRule();
    const auto& rule = parsed_rule.rule;
    builder_.UpdateRuleBody(rule.name, rule.body_expr_id);
    builder_.UpdateLookaheadAssertion(rule.name, rule.lookahead_assertion_id);
    builder_.UpdateMaxTokens(rule.name, rule.max_tokens);
    builder_.UpdateMaxChars(rule.name, rule.max_chars);
    builder_.UpdateCaptureName(rule.name, rule.capture_name);
    builder_.UpdateSuffixStopInfo(rule.name, parsed_rule.suffix_stop_info);
    builder_.UpdateLazy(rule.name, rule.is_lazy);
    builder_.UpdateRuleTemperature(builder_.GetRuleId(rule.name), rule.temperature);
  }

  return builder_.Get(root_rule_name);
}

Grammar ParseEBNF(const std::string& ebnf_string, const std::string& root_rule_name) {
  EBNFLexer lexer;
  auto tokens = lexer.Tokenize(ebnf_string);
  EBNFParser parser;
  return parser.Parse(std::move(tokens), root_rule_name);
}

}  // namespace xgrammar
