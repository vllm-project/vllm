/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar_functor.h
 * \brief The header for the simplification of the BNF AST.
 */

#ifndef XGRAMMAR_GRAMMAR_FUNCTOR_H_
#define XGRAMMAR_GRAMMAR_FUNCTOR_H_

#include <xgrammar/xgrammar.h>

#include <bitset>
#include <cstddef>
#include <cstdint>
#include <string>

#include "compiled_grammar_impl.h"
#include "grammar_builder.h"
#include "grammar_impl.h"
#include "support/utils.h"
#include "xgrammar/grammar.h"

namespace xgrammar {

/*!
 * \brief Base class for visitors and mutators of the BNF grammar.
 * \tparam T The type of the return value of visitor functions. Typical values:
 * - int32_t: the id of the new grammar_expr
 * - void: no return value
 * \tparam ReturnType The type of the return value of the transform function Apply(). Typical values
 * are void (for visitor) and Grammar (for mutator).
 */
template <typename T = int32_t, typename ReturnType = Grammar>
class GrammarFunctor {
 public:
  /*!
   * \brief Constructor.
   * \param grammar The grammar to visit or mutate.
   */
  explicit GrammarFunctor() {}

  /*!
   * \brief Apply the transformation to the grammar, or visit the grammar.
   * \return The transformed grammar, or the visiting result, or void.
   */
  virtual ReturnType Apply(const Grammar& grammar) {
    // The initializer MUST be called at first when overriding the Apply() function.
    InitGrammar(grammar);
    if constexpr (std::is_same<T, void>::value) {
      for (int i = 0; i < static_cast<int>(base_grammar_->NumRules()); ++i) {
        auto rule = base_grammar_->GetRule(i);
        cur_rule_name_ = rule.name;
        VisitExpr(rule.body_expr_id);
        VisitLookaheadAssertion(rule.lookahead_assertion_id);
      }
      return ReturnType();
    } else if constexpr (std::is_same<T, int32_t>::value &&
                         std::is_same<ReturnType, Grammar>::value) {
      InitBuilder();
      // First add empty rules to ensure the new rule ids the same as the old ones, then update
      // the rule bodies
      for (int i = 0; i < static_cast<int>(base_grammar_->NumRules()); ++i) {
        builder_->AddEmptyRule(base_grammar_->GetRule(i).name);
      }
      for (int i = 0; i < static_cast<int>(base_grammar_->NumRules()); ++i) {
        auto rule = base_grammar_->GetRule(i);
        cur_rule_name_ = rule.name;
        auto new_body_expr_id = VisitExpr(rule.body_expr_id);
        builder_->UpdateRuleBody(i, new_body_expr_id);
        // Handle lookahead assertion
        builder_->UpdateLookaheadAssertion(i, VisitLookaheadAssertion(rule.lookahead_assertion_id));
        builder_->UpdateMaxTokens(i, rule.max_tokens);
        builder_->UpdateMaxChars(i, rule.max_chars);
        builder_->UpdateCaptureName(i, rule.capture_name);
        if (const auto* suffix_stop_info = base_grammar_->GetSuffixStopInfo(i)) {
          builder_->UpdateSuffixStopInfo(i, *suffix_stop_info);
        }
        builder_->UpdateLazy(i, rule.is_lazy);
        builder_->UpdateRuleTemperature(i, rule.temperature);
      }
      return builder_->Get(base_grammar_->GetRootRule().name);
    } else {
      return ReturnType();
    }
  }

  /*! \brief Virtual destructor. */
  virtual ~GrammarFunctor() = default;

 protected:
  using Rule = Grammar::Impl::Rule;
  using GrammarExpr = Grammar::Impl::GrammarExpr;
  using GrammarExprType = Grammar::Impl::GrammarExprType;

  /*! \brief Initialize the functor. Should be called at the beginning of Apply(). */
  virtual void InitGrammar() {}

  virtual void InitGrammar(const Grammar& grammar) { base_grammar_ = grammar; }

  virtual void InitBuilder() {
    owned_builder_ = GrammarBuilder();
    builder_ = &owned_builder_;
  }

  virtual void InitBuilder(const Grammar& grammar) {
    owned_builder_ = GrammarBuilder(grammar);
    builder_ = &owned_builder_;
  }

  virtual void InitBuilder(GrammarBuilder* builder) { builder_ = builder; }

  /*! \brief Visit a lookahead assertion expr referred by id. */
  virtual T VisitLookaheadAssertion(int32_t lookahead_assertion_id) {
    if (lookahead_assertion_id == -1) {
      if constexpr (std::is_same<T, int32_t>::value) {
        return -1;
      } else {
        return T();
      }
    }
    return VisitExpr(lookahead_assertion_id);
  }

  /*! \brief Visit a GrammarExpr by id. */
  virtual T VisitExpr(int32_t old_grammar_expr_id) {
    return VisitExpr(base_grammar_->GetGrammarExpr(old_grammar_expr_id));
  }

  /*! \brief Visit a GrammarExpr. Dispatch to the corresponding Visit function. */
  virtual T VisitExpr(const GrammarExpr& grammar_expr) {
    switch (grammar_expr.type) {
      case GrammarExprType::kSequence:
        return VisitSequence(grammar_expr);
      case GrammarExprType::kChoices:
        return VisitChoices(grammar_expr);
      case GrammarExprType::kEmptyStr:
        return VisitEmptyStr(grammar_expr);
      case GrammarExprType::kByteString:
        return VisitByteString(grammar_expr);
      case GrammarExprType::kCharacterClass:
        return VisitCharacterClass(grammar_expr);
      case GrammarExprType::kCharacterClassStar:
        return VisitCharacterClassStar(grammar_expr);
      case GrammarExprType::kRuleRef:
        return VisitRuleRef(grammar_expr);
      case GrammarExprType::kTagDispatch:
        return VisitTagDispatch(grammar_expr);
      case GrammarExprType::kRepeat:
        return VisitRepeat(grammar_expr);
      case GrammarExprType::kToken:
        return VisitToken(grammar_expr);
      case GrammarExprType::kExcludeToken:
        return VisitExcludeToken(grammar_expr);
      case GrammarExprType::kTokenTagDispatch:
        return VisitTokenTagDispatch(grammar_expr);
      case GrammarExprType::kRegex:
        return VisitRegex(grammar_expr);
      case GrammarExprType::kSubstring:
        return VisitSubstring(grammar_expr);
      default:
        XGRAMMAR_LOG(FATAL) << "Unexpected sequence type: " << static_cast<int>(grammar_expr.type);
        XGRAMMAR_UNREACHABLE();
    }
  }

  /*! \brief Visit a choices GrammarExpr. */
  virtual T VisitChoices(const GrammarExpr& grammar_expr) {
    if constexpr (std::is_same<T, void>::value) {
      for (auto i : grammar_expr) {
        VisitExpr(i);
      }
    } else if constexpr (std::is_same<T, int32_t>::value) {
      std::vector<int32_t> choice_ids;
      for (int32_t i : grammar_expr) {
        choice_ids.push_back(VisitExpr(i));
      }
      return builder_->AddChoices(choice_ids);
    } else {
      return T();
    }
  }

  /*! \brief Visit a sequence GrammarExpr. */
  virtual T VisitSequence(const GrammarExpr& grammar_expr) {
    if constexpr (std::is_same<T, void>::value) {
      for (auto i : grammar_expr) {
        VisitExpr(i);
      }
    } else if constexpr (std::is_same<T, int32_t>::value) {
      std::vector<T> sequence_ids;
      for (int32_t i : grammar_expr) {
        sequence_ids.push_back(VisitExpr(i));
      }
      return builder_->AddSequence(sequence_ids);
    } else {
      return T();
    }
  }

  virtual T VisitTagDispatch(const GrammarExpr& grammar_expr) {
    if constexpr (std::is_same<T, void>::value) {
      return;
    } else if constexpr (std::is_same<T, int32_t>::value) {
      Grammar::Impl::TagDispatch tag_dispatch = base_grammar_->GetTagDispatch(grammar_expr);
      return builder_->AddTagDispatch(tag_dispatch);
    } else {
      return T();
    }
  }

  /*! \brief Visit an element GrammarExpr, including empty string, character class, and rule ref. */
  virtual T VisitElement(const GrammarExpr& grammar_expr) {
    if constexpr (std::is_same<T, void>::value) {
      return;
    } else if constexpr (std::is_same<T, int32_t>::value) {
      return builder_->AddGrammarExpr(grammar_expr);
    } else {
      return T();
    }
  }

  /*! \brief Visit an empty string GrammarExpr. */
  virtual T VisitEmptyStr(const GrammarExpr& grammar_expr) { return VisitElement(grammar_expr); }

  /*! \brief Visit a character class GrammarExpr. */
  virtual T VisitByteString(const GrammarExpr& grammar_expr) { return VisitElement(grammar_expr); }

  /*! \brief Visit a character class GrammarExpr. */
  virtual T VisitCharacterClass(const GrammarExpr& grammar_expr) {
    return VisitElement(grammar_expr);
  }

  /*! \brief Visit a star quantifier GrammarExpr. */
  virtual T VisitCharacterClassStar(const GrammarExpr& grammar_expr) {
    return VisitElement(grammar_expr);
  }

  /*! \brief Visit a rule reference GrammarExpr. */
  virtual T VisitRuleRef(const GrammarExpr& grammar_expr) { return VisitElement(grammar_expr); }

  /*! \brief Visit a repeat GrammarExpr. */
  virtual T VisitRepeat(const GrammarExpr& grammar_expr) { return VisitElement(grammar_expr); }

  virtual T VisitToken(const GrammarExpr& grammar_expr) { return VisitElement(grammar_expr); }

  virtual T VisitExcludeToken(const GrammarExpr& grammar_expr) {
    return VisitElement(grammar_expr);
  }

  virtual T VisitTokenTagDispatch(const GrammarExpr& grammar_expr) {
    return VisitElement(grammar_expr);
  }

  /*! \brief Visit a regex GrammarExpr. It is a leaf: the pattern string is carried as-is. */
  virtual T VisitRegex(const GrammarExpr& grammar_expr) { return VisitElement(grammar_expr); }

  /*! \brief Visit a substring GrammarExpr. It is a leaf: the chunk list is carried as-is. */
  virtual T VisitSubstring(const GrammarExpr& grammar_expr) { return VisitElement(grammar_expr); }

  /*! \brief The grammar to visit or mutate. */
  Grammar base_grammar_{NullObj{}};

  /*!
   * \brief The builder to build the new grammar. It is empty when the mutator is constructed, and
   * can be used to build a new grammar in subclasses.
   */
  GrammarBuilder* builder_ = nullptr;

  GrammarBuilder owned_builder_;

  /*! \brief The name of the current rule being visited. */
  std::string cur_rule_name_;
};

/*!
 * \brief Visitor of Grammar.
 * \tparam ReturnType The return type of the Apply() function. Denotes the collected information.
 */
template <typename ReturnType>
using GrammarVisitor = GrammarFunctor<void, ReturnType>;

/*!
 * \brief Mutator of Grammar. The Apply() function returns the updated grammar.
 */
using GrammarMutator = GrammarFunctor<int32_t, Grammar>;

/****** All below methods are implemented as functor to hide the implementation ******/

/*************************** Grammar Constructor ***************************/
/*!
 * \brief Find the union of multiple grammars as a new grammar.
 */
class GrammarUnionFunctor {
 public:
  static Grammar Apply(const std::vector<Grammar>& grammars);
};

/*!
 * \brief Find the concatenation of multiple grammars as a new grammar.
 */
class GrammarConcatFunctor {
 public:
  static Grammar Apply(const std::vector<Grammar>& grammars);
};

/*!
 * \brief Add a sub grammar to the current builder. The return value
 * of Apply is the new rule id of the sub grammar's root rule.
 */
class SubGrammarAdder {
 public:
  static int32_t Apply(GrammarBuilder* builder, const Grammar& sub_grammar);
};

/*************************** Grammar Normalizer ***************************/

/*!
 * \brief Normalize a Grammar: expand the nested rules, combine consequent sequences and strings,
 * etc.
 */
class GrammarNormalizer {
 public:
  static Grammar Apply(const Grammar& grammar);
};

/*!
 * \brief Normalize the structure of the grammar. It will ensure each rule is a choices of
 * sequences of elements, or a tag dispatch. The expanded context will be a sequence of elements.
 */
class StructureNormalizer {
 public:
  static Grammar Apply(const Grammar& grammar);
};

/*************************** Grammar Optimizer ***************************/

/*!
 * \brief Fuse adjacent byte string elements in sequences.
 * \details Rewrites *grammar in place. Only sequences that actually contain adjacent byte strings
 * are rewritten; if nothing needs fusing, the grammar is left untouched. The caller must own the
 * grammar, as the input is mutated directly.
 */
class ByteStringFuser {
 public:
  static void Apply(Grammar* grammar);
};

/*!
 * \brief Analyze the grammar to find the rules that are allowed to be empty.
 */
class AllowEmptyRuleAnalyzer {
 public:
  static std::vector<int32_t> Apply(const Grammar& grammar);
};

/*!
 * \brief Inline the rule references in the grammar.
 * \details Rewrites *grammar in place. Only choices with an inlinable leading rule reference are
 * rewritten; if nothing can be inlined, the grammar is left untouched. The caller must own the
 * grammar, as the input is mutated directly.
 */
class RuleInliner {
 public:
  static void Apply(Grammar* grammar);
};

/*!
 * \brief Eliminate the not referenced rules in the grammar.
 */
class DeadCodeEliminator {
 public:
  static Grammar Apply(const Grammar& grammar);
};

/*!
 * \brief Analyze and add lookahead assertions in the grammar.
 */
class LookaheadAssertionAnalyzer {
 public:
  static Grammar Apply(const Grammar& grammar);
};

/*!
 * \brief Build the FSMs of the grammar.
 */
class GrammarFSMBuilder {
  using GrammarExpr = Grammar::Impl::GrammarExpr;

 public:
  static void Apply(Grammar* grammar);
  static FSMWithStartEnd RuleRef(const GrammarExpr& expr);
  static FSMWithStartEnd CharacterClass(const GrammarExpr& expr);
  static FSMWithStartEnd ByteString(const GrammarExpr& expr);
  static FSMWithStartEnd Token(const GrammarExpr& expr);
  static FSMWithStartEnd ExcludeToken(const GrammarExpr& expr);
  static std::optional<FSMWithStartEnd> TokenTagDispatch(
      const Grammar::Impl::TokenTagDispatch& token_tag_dispatch
  );
  static std::optional<FSMWithStartEnd> Sequence(const GrammarExpr& expr, const Grammar& grammar);
  static std::optional<FSMWithStartEnd> Choices(const GrammarExpr& expr, const Grammar& grammar);
  static std::optional<FSMWithStartEnd> TagDispatch(const Grammar::Impl::TagDispatch& tag_dispatch);
  /*!
   * \brief Build the automaton of a regex pattern. Returns the error message on failure.
   * \param regex The regex pattern string.
   * \param json_string Whether the regex matches the body of a JSON string literal. If true,
   * the characters in JSONStringForbiddenChars() are excluded from every character match.
   */
  static Result<FSMWithStartEnd> Regex(const std::string& regex, bool json_string = false);
  /*!
   * \brief Flatten a regular EBNF grammar without reinterpreting its terminal expressions.
   * Supports finite rule calls, direct tail self-recursion, and repetition. Each call gets
   * separate return states. Other recursion, non-byte terminals, and rule annotations return
   * an error. Intended for the existing RegexToEBNF fallback, not arbitrary CFGs.
   * \param max_num_states Bound on the intermediate automaton size.
   */
  static Result<FSMWithStartEnd> FromRegularGrammar(
      const Grammar& grammar, int max_num_states = 100000
  );
  /*! \brief The characters that must be escaped inside a JSON string literal: the control
   * characters 0x00-0x1F, the quote '"' and the backslash '\\'. */
  static const std::bitset<256>& JSONStringForbiddenChars();
};

/*!
 * \brief Normalize the repetition expression. If the context of
 * repetition expression is nullable, then the repetition range will be
 * normalized from {m, n} to {0, n} to reduce uncertainty.
 */
class RepetitionNormalizer {
 public:
  static void Apply(Grammar* grammar);
};

/*!
 * \brief Expand kRepeat grammar expressions using HandleRepetitionRange logic.
 * Transforms repetition structures into explicit sequences and choices.
 */
class RepetitionRangeExpander {
 public:
  static Grammar Apply(const Grammar& grammar);
};

/*!
 * \brief Optimize the grammar when compiling.
 * \note No matter whether the grammar is optimized, grammar optimizer will
 * return a new grammar. The following optimization will be applied:
 * 1. Byte fuser.
 * 2. Rule inliner.
 * 3. Dead code eliminator.
 * 4. Lookahead assertion analyzer.
 * 5. Allow-empty rule analyzer.
 * 6. Repetition normalizer.
 * 7. FSM builder.
 */
class GrammarOptimizer {
 public:
  static Grammar Apply(const Grammar& grammar);
};

/*!
 * \brief Rename the root rule of the grammar to "root".
 */
class RootRuleRenamer {
 public:
  static Grammar Apply(const Grammar& grammar);
};

/*!
 * \brief Hash the fsms in the grammar,
 * and get the new state ids of each fsm's states.
 */
class GrammarFSMHasher {
 public:
  static void Apply(Grammar* grammar);
  static std::optional<uint64_t> HashSequence(const Grammar& grammar, int32_t sequence_id);
};

/*!
 * \brief Store the crossing cache for different grammars.
 * \param max_cache_size The maximum size of the cache numbers.
 * \details LRU algorithm is implemented.
 */
class RuleLevelCache {
 public:
  static const size_t kUnlimitedSize = static_cast<size_t>(-1);

  std::optional<AdaptiveTokenMask> GetCache(
      const uint64_t& fsm_hash,
      int32_t fsm_new_node_id,
      const int32_t& state_cnt,
      const int32_t edge_cnt
  );
  bool AddCache(
      const uint64_t& fsm_hash,
      int32_t fsm_new_node_id,
      const int32_t& state_cnt,
      const int32_t edge_cnt,
      const AdaptiveTokenMask& token_mask
  );
  bool AddCache(
      const uint64_t& fsm_hash,
      int32_t fsm_new_node_id,
      const int32_t& state_cnt,
      const int32_t edge_cnt,
      AdaptiveTokenMask&& token_mask
  );
  RuleLevelCache(size_t max_cache_memory_size = kUnlimitedSize);

  void ClearCache();

  size_t GetMaxSize() const;

  friend size_t MemorySize(const RuleLevelCache& manager);

  XGRAMMAR_DEFINE_PIMPL_METHODS(RuleLevelCache);
};

}  // namespace xgrammar

#endif  // XGRAMMAR_GRAMMAR_FUNCTOR_H_
