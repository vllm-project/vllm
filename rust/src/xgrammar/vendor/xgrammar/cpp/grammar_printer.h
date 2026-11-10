/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar_printer.h
 * \brief The header for printing the AST of a BNF grammar.
 */

#ifndef XGRAMMAR_GRAMMAR_PRINTER_H_
#define XGRAMMAR_GRAMMAR_PRINTER_H_

#include <xgrammar/xgrammar.h>

#include <string>

#include "grammar_impl.h"

namespace xgrammar {

/*!
 * \brief Prints the BNF AST with standard BNF format.
 */
class GrammarPrinter {
 private:
  using Rule = Grammar::Impl::Rule;
  using SuffixStopInfo = Grammar::Impl::SuffixStopInfo;
  using GrammarExprType = Grammar::Impl::GrammarExprType;
  using GrammarExpr = Grammar::Impl::GrammarExpr;

 public:
  /*!
   * \brief Constructor.
   * \param grammar The grammar to print.
   */
  explicit GrammarPrinter(const Grammar& grammar) : grammar_(grammar) {}

  /*! \brief Print the complete grammar. */
  std::string ToString();

  /*! \brief Print a rule. */
  std::string PrintRule(const Rule& rule, const SuffixStopInfo* suffix_stop_info);
  /*! \brief Print a rule corresponding to the given id. */
  std::string PrintRule(int32_t rule_id);
  /*! \brief Print a GrammarExpr. */
  std::string PrintGrammarExpr(const GrammarExpr& grammar_expr);
  /*! \brief Print a GrammarExpr corresponding to the given id. */
  std::string PrintGrammarExpr(int32_t grammar_expr_id);

 private:
  /*! \brief Print a GrammarExpr for byte string. */
  std::string PrintByteString(const GrammarExpr& grammar_expr);
  /*! \brief Print a GrammarExpr for character class. */
  std::string PrintCharacterClass(const GrammarExpr& grammar_expr);
  /*! \brief Print a GrammarExpr for a star quantifier of a character class. */
  std::string PrintCharacterClassStar(const GrammarExpr& grammar_expr);
  /*! \brief Print a GrammarExpr for empty string. */
  std::string PrintEmptyStr(const GrammarExpr& grammar_expr);
  /*! \brief Print a GrammarExpr for rule reference. */
  std::string PrintRuleRef(const GrammarExpr& grammar_expr);
  /*! \brief Print a GrammarExpr for grammar_expr sequence. */
  std::string PrintSequence(const GrammarExpr& grammar_expr);
  /*! \brief Print a GrammarExpr for grammar_expr choices. */
  std::string PrintChoices(const GrammarExpr& grammar_expr);
  /*! \brief Print a GrammarExpr for tag dispatch. */
  std::string PrintTagDispatch(const GrammarExpr& grammar_expr);
  /*! \brief Print a GrammarExpr for repeat. */
  std::string PrintRepeat(const GrammarExpr& grammar_expr);
  /*! \brief Print a GrammarExpr for token. */
  std::string PrintToken(const GrammarExpr& grammar_expr);
  /*! \brief Print a GrammarExpr for exclude token. */
  std::string PrintExcludeToken(const GrammarExpr& grammar_expr);
  /*! \brief Print a GrammarExpr for token tag dispatch. */
  std::string PrintTokenTagDispatch(const GrammarExpr& grammar_expr);
  /*! \brief Print a GrammarExpr for regex. */
  std::string PrintRegex(const GrammarExpr& grammar_expr);
  /*! \brief Print a GrammarExpr for substring. */
  std::string PrintSubstring(const GrammarExpr& grammar_expr);
  /*! \brief Print a string. */
  std::string PrintString(const std::string& str);
  /*! \brief Print a boolean. */
  std::string PrintBoolean(bool value);

  Grammar grammar_;
};

}  // namespace xgrammar

#endif  // XGRAMMAR_GRAMMAR_PRINTER_H_
