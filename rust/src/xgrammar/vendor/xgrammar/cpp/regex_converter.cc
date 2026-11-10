/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/regex_converter.cc
 */
#include "regex_converter.h"

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "support/encoding.h"
#include "support/logging.h"
#include "support/utils.h"

namespace xgrammar {

/*!
 * \brief Convert a regex to EBNF.
 * \details The implementation refers to the regex described in
 * https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Regular_expressions
 */
class RegexConverter {
 public:
  explicit RegexConverter(const std::string& regex) : regex_(regex) {
    if (!regex.empty()) {
      // ParseUTF8 takes a C string and stops at the first NUL, so a regex
      // containing one would either yield an empty codepoint vector (leading
      // NUL -- an out-of-bounds read below) or be silently truncated to the
      // prefix. A NUL is never meaningful in a regex, so reject it here.
      if (regex.find('\0') != std::string::npos) {
        XGRAMMAR_LOG(FATAL) << "The regex must not contain null characters.";
        XGRAMMAR_UNREACHABLE();
      }
      regex_codepoints_ = ParseUTF8(regex_.c_str(), false);
      if (regex_codepoints_[0] == kInvalidUTF8) {
        XGRAMMAR_LOG(FATAL) << "The regex is not a valid UTF-8 string.";
        XGRAMMAR_UNREACHABLE();
      }
    }
    regex_codepoints_.push_back(0);  // Add a null terminator
  }
  std::string Convert();

 private:
  /**
   * \brief Add a segment string to the result EBNF string. It especially adds a space if needed
   * and add_space is true.
   */
  void AddEBNFSegment(const std::string& element);

  [[noreturn]] void RaiseError(const std::string& message);
  void RaiseWarning(const std::string& message);

  std::string HandleCharacterClass();
  std::string HandleRepetitionRange();
  std::string HandleCharEscape();
  std::string HandleEscape();
  std::string HandleEscapeInCharClass();
  /**
   * \brief Handle group modifier. The general format is "(?" + modifier + content + ")". E.g.
   * "(?:abc)" is a non-capturing group.
   */
  void HandleGroupModifier();

  std::string regex_;
  std::vector<TCodepoint> regex_codepoints_;
  TCodepoint* start_;
  TCodepoint* current_;
  TCodepoint* end_;
  std::string result_ebnf_;
  int parenthesis_level_ = 0;
};

void RegexConverter::AddEBNFSegment(const std::string& element) {
  if (!result_ebnf_.empty()) {
    result_ebnf_ += ' ';
  }
  result_ebnf_ += element;
}

void RegexConverter::RaiseError(const std::string& message) {
  XGRAMMAR_LOG(FATAL) << "Regex parsing error at position " << current_ - start_ + 1 << ": "
                      << message;
  XGRAMMAR_UNREACHABLE();
}

void RegexConverter::RaiseWarning(const std::string& message) {
  XGRAMMAR_LOG(WARNING) << "Regex parsing warning at position " << current_ - start_ + 1 << ": "
                        << message;
}

std::string RegexConverter::HandleCharacterClass() {
  std::string char_class = "[";
  ++current_;
  if (*current_ == ']') {
    RaiseError("Empty character class is not allowed in regex.");
  }
  while (*current_ != ']' && current_ != end_) {
    if (*current_ == '\\') {
      char_class += HandleEscapeInCharClass();
    } else {
      char_class += CharToUTF8(*current_);
      ++current_;
    }
  }
  if (current_ == end_) {
    RaiseError("Unclosed '['");
  }
  char_class += ']';
  ++current_;
  return char_class;
}

// {x}: Match exactly x occurrences of the preceding regular expression.
// {x,}
// {x,y}
std::string RegexConverter::HandleRepetitionRange() {
  std::string result = "{";
  ++current_;
  if (!isdigit(*current_)) {
    RaiseError("Invalid repetition count.");
  }
  while (isdigit(*current_)) {
    result += static_cast<char>(*current_);
    ++current_;
  }
  if (*current_ != ',' && *current_ != '}') {
    RaiseError("Invalid repetition count.");
  }
  result += static_cast<char>(*current_);
  ++current_;
  if (current_[-1] == '}') {
    // Matches {x}
    return result;
  }
  if (!isdigit(*current_) && *current_ != '}') {
    RaiseError("Invalid repetition count.");
  }
  while (isdigit(*current_)) {
    result += static_cast<char>(*current_);
    ++current_;
  }
  if (*current_ != '}') {
    RaiseError("Invalid repetition count.");
  }
  result += '}';
  ++current_;
  return result;
}

std::string RegexConverter::HandleCharEscape() {
  // clang-format off
  static const std::unordered_map<char, TCodepoint> CUSTOM_ESCAPE_MAP = {
      {'^', '^'}, {'$', '$'}, {'.', '.'}, {'*', '*'}, {'+', '+'}, {'?', '?'}, {'\\', '\\'},
      {'(', '('}, {')', ')'}, {'[', '['}, {']', ']'}, {'{', '{'}, {'}', '}'}, {'|', '|'},
      {'/', '/'}, {'-', '-'}
  };
  // clang-format on
  if (end_ - current_ < 2 || (current_[1] == 'u' && end_ - current_ < 5) ||
      (current_[1] == 'x' && end_ - current_ < 4) || (current_[1] == 'c' && end_ - current_ < 3)) {
    RaiseError("Escape sequence is not finished.");
  }
  auto [codepoint, len] = ParseNextEscaped(current_, CUSTOM_ESCAPE_MAP);
  if (codepoint != CharHandlingError::kInvalidEscape) {
    current_ += len;
    return EscapeString(codepoint);
  } else if (current_[1] == 'u' && current_[2] == '{') {
    current_ += 3;
    int len = 0;
    TCodepoint value = 0;
    while (HexCharToInt(current_[len]) != -1 && len <= 6) {
      value = value * 16 + HexCharToInt(current_[len]);
      ++len;
    }
    if (len == 0 || len > 6 || current_[len] != '}') {
      RaiseError("Invalid Unicode escape sequence.");
    }
    current_ += len + 1;
    return EscapeString(value);
  } else if (current_[1] == 'c') {
    current_ += 2;
    if (!std::isalpha(*current_)) {
      RaiseError("Invalid control character escape sequence.");
    }
    ++current_;
    return EscapeString((*(current_ - 1)) % 32);
  } else {
    RaiseWarning(
        "Escape sequence '\\" + EscapeString(current_[1]) +
        "' is not recognized. The character itself will be matched"
    );
    current_ += 2;
    return EscapeString(current_[-1]);
  }
}

std::string RegexConverter::HandleEscapeInCharClass() {
  if (end_ - current_ < 2) {
    RaiseError("Escape sequence is not finished.");
  }
  if (current_[1] == 'd') {
    current_ += 2;
    return "0-9";
  } else if (current_[1] == 'D') {
    current_ += 2;
    return R"(\x00-\x2F\x3A-\U0010FFFF)";
  } else if (current_[1] == 'w') {
    current_ += 2;
    return "a-zA-Z0-9_";
  } else if (current_[1] == 'W') {
    current_ += 2;
    return R"(\x00-\x2F\x3A-\x40\x5B-\x5E\x60\x7B-\U0010FFFF)";
  } else if (current_[1] == 's') {
    current_ += 2;
    return R"(\f\n\r\t\v\u0020\u00a0)";
  } else if (current_[1] == 'S') {
    current_ += 2;
    return R"(\x00-\x08\x0E-\x1F\x21-\x9F\xA1-\U0010FFFF)";
  } else {
    auto res = HandleCharEscape();
    if (res == "]" || res == "-") {
      return "\\" + res;
    } else {
      return res;
    }
  }
}

std::string RegexConverter::HandleEscape() {
  // clang-format off
  static const std::unordered_map<char, TCodepoint> CUSTOM_ESCAPE_MAP = {
      {'^', '^'}, {'$', '$'}, {'.', '.'}, {'*', '*'}, {'+', '+'}, {'?', '?'}, {'\\', '\\'},
      {'(', '('}, {')', ')'}, {'[', '['}, {']', ']'}, {'{', '{'}, {'}', '}'}, {'|', '|'},
      {'/', '/'}
  };
  // clang-format on
  if (end_ - current_ < 2) {
    RaiseError("Escape sequence is not finished.");
  }
  if (current_[1] == 'd') {
    current_ += 2;
    return "[0-9]";
  } else if (current_[1] == 'D') {
    current_ += 2;
    return "[^0-9]";
  } else if (current_[1] == 'w') {
    current_ += 2;
    return "[a-zA-Z0-9_]";
  } else if (current_[1] == 'W') {
    current_ += 2;
    return "[^a-zA-Z0-9_]";
  } else if (current_[1] == 's') {
    current_ += 2;
    return R"([\f\n\r\t\v\u0020\u00a0])";
  } else if (current_[1] == 'S') {
    current_ += 2;
    return R"([^\f\n\r\t\v\u0020\u00a0])";
  } else if ((current_[1] >= '1' && current_[1] <= '9') || current_[1] == 'k') {
    RaiseError("Backreference is not supported yet.");
  } else if (current_[1] == 'p' || current_[1] == 'P') {
    RaiseError("Unicode character class escape sequence is not supported yet.");
  } else if (current_[1] == 'b' || current_[1] == 'B') {
    RaiseError("Word boundary is not supported yet.");
  } else {
    return "\"" + HandleCharEscape() + "\"";
  }
}

void RegexConverter::HandleGroupModifier() {
  if (current_ == end_) {
    RaiseError("Group modifier is not finished.");
  }
  if (*current_ == ':') {
    // Non-capturing group.
    ++current_;
  } else if (*current_ == '=' || *current_ == '!') {
    // Positive or negative lookahead.
    RaiseError("Lookahead is not supported yet.");
  } else if (*current_ == '<' && current_ + 1 != end_ &&
             (current_[1] == '=' || current_[1] == '!')) {
    // Positive or negative lookbehind.
    RaiseError("Lookbehind is not supported yet.");
  } else if (*current_ == '<') {
    ++current_;
    while (current_ != end_ && isalpha(*current_)) {
      ++current_;
    }
    if (current_ == end_ || *current_ != '>') {
      RaiseError("Invalid named capturing group.");
    }
    // Just ignore the named of the group.
    ++current_;
  } else {
    // Group modifier flag.
    RaiseError("Group modifier flag is not supported yet.");
  }
}

std::string RegexConverter::Convert() {
  start_ = regex_codepoints_.data();
  current_ = start_;
  end_ = start_ + regex_codepoints_.size() - 1;
  bool is_empty = true;
  while (current_ != end_) {
    if (*current_ == '^') {
      if (current_ != start_) {
        RaiseWarning(
            "'^' should be at the start of the regex, but found in the middle. It is ignored."
        );
      }
      ++current_;
    } else if (*current_ == '$') {
      if (current_ != end_ - 1) {
        RaiseWarning(
            "'$' should be at the end of the regex, but found in the middle. It is ignored."
        );
      }
      ++current_;
    } else if (*current_ == '[') {
      is_empty = false;
      AddEBNFSegment(HandleCharacterClass());
    } else if (*current_ == '(') {
      is_empty = false;
      ++current_;
      ++parenthesis_level_;
      AddEBNFSegment("(");
      if (current_ != end_ && *current_ == '?') {
        ++current_;
        HandleGroupModifier();
      }
    } else if (*current_ == ')') {
      is_empty = false;
      if (parenthesis_level_ == 0) {
        RaiseError("Unmatched ')'");
      }
      // Empty alternative before ')' (e.g. "(a|)" or "(a|$)"): emit "" so it isn't a bare '|'.
      if (!result_ebnf_.empty() && result_ebnf_.back() == '|') {
        AddEBNFSegment("\"\"");
      }
      --parenthesis_level_;
      AddEBNFSegment(")");
      ++current_;
    } else if (*current_ == '*' || *current_ == '+' || *current_ == '?') {
      is_empty = false;
      result_ebnf_ += static_cast<char>(*current_);
      ++current_;
      if (current_ != end_ && *current_ == '?') {
        // Ignore the non-greedy modifier because our grammar handles all repetition numbers
        // non-deterministically.
        ++current_;
      }
      if (current_ != end_ &&
          (*current_ == '{' || *current_ == '*' || *current_ == '+' || *current_ == '?')) {
        RaiseError("Two consecutive repetition modifiers are not allowed.");
      }
    } else if (*current_ == '{') {
      is_empty = false;
      result_ebnf_ += HandleRepetitionRange();
      if (current_ != end_ && *current_ == '?') {
        // Still ignore the non-greedy modifier.
        ++current_;
      }
      if (current_ != end_ &&
          (*current_ == '{' || *current_ == '*' || *current_ == '+' || *current_ == '?')) {
        RaiseError("Two consecutive repetition modifiers are not allowed.");
      }
    } else if (*current_ == '|') {
      is_empty = false;
      // Empty alternative before '|': emit "" so there's no bare '|' on the left.
      // Covers leading ("^$|abc"), consecutive ("a||b") and group-start ("(|a)") cases.
      if (result_ebnf_.empty() || result_ebnf_.back() == '|' || result_ebnf_.back() == '(') {
        AddEBNFSegment("\"\"");
      }
      AddEBNFSegment("|");
      ++current_;
    } else if (*current_ == '\\') {
      is_empty = false;
      AddEBNFSegment(HandleEscape());
    } else if (*current_ == '.') {
      is_empty = false;
      AddEBNFSegment(R"([\u0000-\U0010FFFF])");
      ++current_;
    } else {
      is_empty = false;
      // Non-special characters are matched literally.
      AddEBNFSegment("\"" + EscapeString(*current_) + "\"");
      ++current_;
    }
  }
  if (parenthesis_level_ != 0) {
    RaiseError("The parenthesis is not closed.");
  }
  // Trailing empty alternative, e.g. "abc|": emit "" so it doesn't end with a bare '|'.
  if (!result_ebnf_.empty() && result_ebnf_.back() == '|') {
    AddEBNFSegment("\"\"");
  }
  if (is_empty) {
    AddEBNFSegment("\"\"");
  }
  return result_ebnf_;
}

std::string RegexToEBNF(const std::string& regex, bool with_rule_name) {
  RegexConverter converter(regex);
  if (with_rule_name) {
    return "root ::= " + converter.Convert() + "\n";
  } else {
    return converter.Convert();
  }
}

}  // namespace xgrammar
