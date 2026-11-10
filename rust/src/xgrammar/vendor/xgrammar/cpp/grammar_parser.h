/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar_parser.h
 * \brief The header for the parser of BNF/EBNF grammar into BNF AST.
 */

#ifndef XGRAMMAR_GRAMMAR_PARSER_H_
#define XGRAMMAR_GRAMMAR_PARSER_H_

#include <xgrammar/xgrammar.h>

#include <any>
#include <optional>

namespace xgrammar {

class EBNFLexer {
 public:
  // Token types
  enum class TokenType {
    RuleName,        // the name of a rule definition, e.g.: root, rule1
    Identifier,      // reference to a rule, or a Macro name, e.g.: root, rule1, TagDispatch
    StringLiteral,   // e.g.: "tag1", "hello"
    BooleanLiteral,  // true, false
    IntegerLiteral,  // 123
    LParen,          // (
    RParen,          // )
    LBrace,          // {
    RBrace,          // }
    Pipe,            // |
    Comma,           // ,
    EndOfFile,       // End of file

    // Symbols and quantifiers
    Assign,    // ::=
    Equal,     // =
    Star,      // *
    Plus,      // +
    Question,  // ?

    // Character class
    LBracket,           // [
    RBracket,           // ]
    Dash,               // -
    Caret,              // ^
    CharInCharClass,    // a character in a character class, e.g. a and z in [a-z]; escaped chars
                        // with no special meaning are also included, e.g. . in [a\.z]
    EscapeInCharClass,  // Escaped sequence with special function, e.g. \S in [\S]

    // Special structures
    LookaheadLParen,  // (=
  };

  // Token structure
  struct Token {
    TokenType type;
    std::string lexeme;  // original text
    std::any value;  // The processed value. Can be a int for integer literal, a string for string
                     // literal, etc.
    int line;
    int column;
    // The token budget attached to a rule-definition identifier via name[max_tokens=N], or -1.
    int32_t max_tokens = -1;
    // The character budget attached to a rule-definition identifier via name[max_chars=N], or -1.
    int32_t max_chars = -1;
    // The capture name attached to a rule-definition identifier via name[capture="x"], or empty.
    std::string capture_name = {};
    // Trailing bytes hidden only from the rule's own capture.
    int32_t capture_hidden_suffix_bytes = 0;
    // Trailing bytes hidden from the rule and every enclosing capture.
    int32_t capture_hidden_stop_bytes = 0;
    // Helper rule ids used to recover a variable-length suffix/stop marker boundary.
    int32_t capture_hidden_body_rule_id = -1;
    int32_t capture_hidden_marker_rule_id = -1;
    // Capture name for the bytes matched by a suffix/stop marker.
    std::string stop_capture_name = {};
    // Whether the identifier is a rule name carrying the [lazy] attribute, e.g. r[lazy] ::= ...
    bool is_lazy = false;
    // The sampling temperature attached to a rule-definition identifier via name[temperature=T].
    std::optional<float> temperature = std::nullopt;
  };

  EBNFLexer();
  std::vector<Token> Tokenize(const std::string& input);

  XGRAMMAR_DEFINE_PIMPL_METHODS(EBNFLexer);
};

/*!
 * \brief This class parses a BNF/EBNF grammar string into an BNF abstract syntax tree (AST).
 * \details This function accepts the EBNF notation defined in the W3C XML Specification
 * (https://www.w3.org/TR/xml/#sec-notation), which is a popular standard, with the following
 * changes:
 * - Using # as comment mark instead of C-style comments
 * - Accept C-style unicode escape sequence \u01AB, \U000001AB, \xAB instead of #x0123
 * - Rule A-B (match A and not match B) is not supported yet
 *
 * See tests/python/serve/json.ebnf for an example.
 * \param ebnf_string The grammar string.
 * \param root_rule_name The name of the root rule. Default is "root".
 * \return The parsed grammar.
 */
Grammar ParseEBNF(const std::string& ebnf_string, const std::string& root_rule_name = "root");

}  // namespace xgrammar

#endif  // XGRAMMAR_GRAMMAR_PARSER_H_
