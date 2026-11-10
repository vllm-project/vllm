/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar.h
 * \brief The header for the definition and construction of BNF grammar.
 */

#ifndef XGRAMMAR_GRAMMAR_H_
#define XGRAMMAR_GRAMMAR_H_

#include <xgrammar/object.h>
#include <xgrammar/tokenizer_info.h>

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "xgrammar/exception.h"

namespace xgrammar {

struct StructuralTagItem {
  std::string begin;
  std::string schema;
  std::string end;

  bool operator==(const StructuralTagItem& other) const {
    return begin == other.begin && schema == other.schema && end == other.end;
  }
};

struct NamedGrammar;

/*!
 * \brief This class stores the abstract syntax tree (AST) of the Backus-Naur Form (BNF) grammar.
 * The BNF definition here is standard BNF, and the characters are represented using regex-style
 * character classes (e.g. [a-z], [^a-z]).
 *
 * \details
 * ### Rules
 * The BNF grammar AST consists of a set of rules. Each rule contains a name and a definition, and
 * corresponds to a production in the grammar. The definition of a rule is a GrammarExpr. Each rule
 * has a rule_id for reference.
 *
 * ### GrammarExprs
 * GrammarExpr is the definition of a rule or part of the definition of a rule. It can contain
 * elements, empty string, reference to other GrammarExprs, or reference to other rules. Each
 * GrammarExpr corresponds to a grammar_expr_id for reference.
 *
 * For example, in the following rule: rule ::= ("a" "b") | "c"
 * ("a" "b"), "c", ("a" "b") | "c" are all GrammarExprs.
 *
 * #### Types of GrammarExprs
 * Every GrammarExpr is represented by a type as well as a variable-length array containing its
 * data. GrammarExpr has several types:
 * - Byte string: a string of bytes (0~255). Supports UTF-8 strings.
 * - Character class: a range of characters (each character is a unicode codepoint), e.g. [a-z],
 *   [ac-z]. Can be negated: [^a-z], [^ac-z]. Now only ascii chars is allowed in [], but this
 *   expression can accept/reject unicode chars.
 * - Character class star: a star quantifier of a character class. e.g. [a-z]*, [^a-z]*.
 * - EmptyStr: an empty string, i.e. ""
 * - Rule reference: a reference to another rule
 * - Sequence: a sequence of grammar_exprs, e.g. ("a" "b"). These grammar_exprs are concatenated
 * together.
 * - Choices: a choice of grammar_exprs, e.g. ("a" "b") | "c". Each grammar_expr can be matched.
 *
 * #### Storage of GrammarExprs
 * Each type of GrammarExpr has a different data format. For the format of each type of GrammarExpr,
 * see docs in Grammar::Impl::GrammarExprType.
 *
 * We store all GrammarExprs in csr_matrix style. That is, they are stored consecutively in one
 * vector (data vector) and the starting position of each GrammarExpr is recorded in the indptr
 * vector.
 *
 * \remark The character class star GrammarExpr is for the special support for elements like [a-z]*
 * in the grammar. We add it to make the matching more efficient, as we can avoid recursion into
 * rules when matching a sequence of characters. It should be used like:
 * rule1 ::= ((element1 element2 rule2 ...) | ...)
 * rule2 ::= character_class_star_grammar_expr(id_of_a_character_class_grammar_expr)
 */
class Grammar {
  static const std::vector<NamedGrammar>& EmptyNamedGrammars();

 public:
  /*!
   * \brief Get the EBNF string of the grammar.
   */
  std::string ToString() const;

  /*!
   * \brief Construct a BNF grammar with a EBNF-formatted string. The grammar will be normalized
   * (simplified) by default.
   * \param ebnf_string The EBNF-formatted string.
   * \param root_rule_name The name of the root rule.
   */
  static Grammar FromEBNF(
      const std::string& ebnf_string, const std::string& root_rule_name = "root"
  );

  /*!
   * \brief Construct a BNF grammar from the json schema string. The schema string should be in the
   * format of the schema of a JSON file. We will parse the schema and generate a BNF grammar.
   * \param schema The schema string.
   * \param indent The number of spaces for indentation. If set to std::nullopt, the output will be
   * in one line. Default: 2.
   * \param separators Two separators used in the schema: comma and colon. Examples: {",", ":"},
   * {", ", ": "}. If std::nullopt, the default separators will be used: {",", ": "} when the
   * indent is not nullopt, and {", ", ": "} otherwise. This follows the convention in python
   * json.dumps(). Default: std::nullopt.
   * \param strict_mode Whether to use strict mode. In strict mode, the generated grammar will not
   * allow properties and items that is not specified in the schema. This is equivalent to
   * setting unevaluatedProperties and unevaluatedItems to false.
   *
   * This helps LLM to generate accurate output in the grammar-guided generation with JSON
   * schema. Default: true.
   */
  static Grammar FromJSONSchema(
      const std::string& schema,
      bool any_whitespace = true,
      std::optional<int> indent = std::nullopt,
      std::optional<std::pair<std::string, std::string>> separators = std::nullopt,
      bool strict_mode = true,
      std::optional<int> max_whitespace_cnt = std::nullopt,
      bool print_converted_ebnf = false,
      bool any_order = false
  );

  /*!
   * \brief Construct a grammar from a regular expression string.
   * \param regex The regular expression string.
   * \param print_converted_ebnf This method will convert the regex to EBNF first. If this is true,
   * the converted EBNF string will be printed. For debugging purpose. Default: false.
   */
  static Grammar FromRegex(const std::string& regex, bool print_converted_ebnf = false);

  /*!
   * \brief Construct a grammar from Lark syntax.
   * \param lark_string The Lark grammar. The root rule must be named "start".
   * \param tokenizer_info Optional tokenizer metadata used to resolve named special tokens.
   * \param named_grammars Grammar objects or Lark sources that can be referenced with `@name`.
   */
  static Grammar FromLark(
      const std::string& lark_string,
      const std::optional<TokenizerInfo>& tokenizer_info = std::nullopt,
      const std::vector<NamedGrammar>& named_grammars = EmptyNamedGrammars()
  );

  /*!
   * \brief Construct a grammar from a structural tag string.
   * \param structural_tag_json The structural tag string.
   * \param tokenizer_info Optional tokenizer info for resolving string token references.
   */
  static std::variant<Grammar, StructuralTagError> FromStructuralTag(
      const std::string& structural_tag_json,
      const std::optional<TokenizerInfo>& tokenizer_info = std::nullopt
  );

  /*!
   * \brief Get the grammar of standard JSON format. We have built-in support for JSON.
   * \return The grammar of standard JSON format.
   */
  static Grammar BuiltinJSONGrammar();

  /*!
   * \brief Create a grammar that matches any of the grammars in the list. That is equivalent to
   * using the `|` operator to concatenate the grammars in the list.
   * \param grammars The grammars to create the union of.
   * \returns The union of the grammars.
   */
  static Grammar Union(const std::vector<Grammar>& grammars);

  /*!
   * \brief Create a grammar that matches the concatenation of the grammars in the list. That is
   * equivalent to using the `+` operator to concatenate the grammars in the list.
   * \param grammars The grammars to create the concatenation of.
   * \returns The concatenation of the grammars.
   */
  static Grammar Concat(const std::vector<Grammar>& grammars);

  /*!
   * \brief Print a BNF grammar.
   * \param os The output stream.
   * \param grammar The grammar to print.
   * \return The output stream.
   */
  friend std::ostream& operator<<(std::ostream& os, const Grammar& grammar);

  /*!
   * \brief Return the serialized JSON string of the grammar.
   * \return The serialized JSON string.
   */
  std::string SerializeJSON() const;

  /*!
   * \brief Deserialize a grammar from a JSON string.
   * \param json_string The JSON string to deserialize.
   * \return If the deserialization is successful, return the grammar. Otherwise, return a runtime
   * error with the error message.
   */
  static std::variant<Grammar, SerializationError> DeserializeJSON(const std::string& json_string);

  // Allow the builder to bind to this grammar's impl for in-place editing (FromMutableGrammar).
  friend class GrammarBuilder;

  XGRAMMAR_DEFINE_PIMPL_METHODS(Grammar);
};

/*!
 * \brief A grammar or Lark source that can be referenced by name from another Lark grammar.
 */
struct NamedGrammar {
  /*! \brief The reference name, without the leading `@`. */
  std::string name;

  /*! \brief An existing grammar or a Lark source with its own `start` rule. */
  std::variant<Grammar, std::string> grammar;
};

inline const std::vector<NamedGrammar>& Grammar::EmptyNamedGrammars() {
  static const std::vector<NamedGrammar> empty;
  return empty;
}

}  // namespace xgrammar

#endif  // XGRAMMAR_GRAMMAR_H_
