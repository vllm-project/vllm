/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/matcher.h
 * \brief The header for the matcher.
 */

#ifndef XGRAMMAR_MATCHER_H_
#define XGRAMMAR_MATCHER_H_

#include <dlpack/dlpack.h>
#include <xgrammar/compiler.h>
#include <xgrammar/object.h>

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace xgrammar {

int32_t GetBitmaskSize(int vocab_size);

DLDataType GetBitmaskDLType();

void _DebugGetMaskedTokensFromBitmask(
    std::vector<int>* rejected_tokens, const DLTensor& token_bitmask, int vocab_size, int index = 0
);

std::pair<bool, int> _IsSingleTokenBitmask(const DLTensor& bitmask, int vocab_size, int index);

void ApplyTokenBitmaskInplaceCPU(
    DLTensor* logits,
    const DLTensor& bitmask,
    int vocab_size = -1,
    std::optional<std::vector<int>> indices = std::nullopt
);

/*!
 * \brief A stateful matcher to match tokens to the specified BNF grammar. This class is the core
 * logic of the grammar-guided generation.
 *
 * \details This class implements the non-deterministic pushdown automaton (NPDA) matching algorithm
 * to match characters to a BNF grammar. It keep track of the current state of the matching process
 * by maintaining several stacks internally as possible paths in the NPDA. It also supports
 * backtracking.
 *
 * It is particularly capable of finding the set of tokens that are acceptable for the next step
 * and storing them in a bitmask. This aids in grammar-guided generation.
 *
 * \example
 * \code
 * Tokenizer tokenizer = ...;
 * auto compiled_grammar = GrammarMatcher::CreateCompiledGrammar(grammar,
 *                                                        tokenizer->PostProcessedVocab());
 * GrammarMatcher matcher(compiled_grammar, 10);
 * matcher->AcceptToken(67);
 *
 * // Construct a DLTensor with shape (tokenizer.GetVocabSize() + 31) / 32, and dtype int32.
 * DLTensor next_token_bitmask = ...;
 * matcher->FillNextTokenBitmask(&next_token_bitmask);
 *
 * // Rollback is supported
 * matcher->Rollback(1);
 * \endcode
 */
class GrammarMatcher {
 public:
  /*!
   * \brief Construct a GrammarMatcher from the preprocessing result of type
   * CompiledGrammar.
   * \param compiled_grammar The compiled grammar. It is obtained through
   * CreateCompiledGrammar as a result of preprocessing the grammar and tokenizer.
   * \param override_stop_tokens Optional stop token ids that replace those from the tokenizer.
   * \param terminate_without_stop_token Whether to terminate when the root rule is complete.
   * \param max_rollback_tokens Deprecated and unused.
   * \param default_temperature The temperature used when no active rule specifies one.
   */
  GrammarMatcher(
      const CompiledGrammar& compiled_grammar,
      std::optional<std::vector<int>> override_stop_tokens = std::nullopt,
      bool terminate_without_stop_token = false,
      int max_rollback_tokens = -1,
      std::optional<float> default_temperature = std::nullopt
  );

  /*!
   * \brief Accept one token and update the state of the matcher.
   * \param token_id The id of the token to accept.
   * \return Whether the token is accepted.
   * \note Termination state.
   * When the end of the root rule is reached, the matcher can only accept the stop token.
   * The matcher is terminated after accepting the stop token, i.e. no AcceptToken or
   * FindNextTokenMask operations can be performed. The termination state can be canceled
   * using Rollback().
   */
  bool AcceptToken(int32_t token_id, bool debug_print = false);

  /*!
   * \brief Accept a string and update the state of the matcher. The whole string is considered
   * as one step in rollback. It is used to complement the functionality of AcceptToken, and
   * AcceptToken should always be used to accept tokens.
   * \param input_str The string to be accepted.
   * \param debug_print Whether to print information about the internal state of the matcher.
   * \return Whether the string is accepted.
   */
  bool AcceptString(const std::string& input_str, bool debug_print = false);

  /*!
   * \brief Get the set of tokens that are acceptable for the next step and store them in a
   * bitmask.
   * \param next_token_bitmask The bitmask to store the result. The bitmask must be pre-allocated
   * and with shape (GetBitmaskSize(),) and dtype int32.
   * \return Whether the bitmask need to be applied (not all-true).
   */
  bool FillNextTokenBitmask(DLTensor* next_token_bitmask, int index = 0, bool debug_print = false);

  /*!
   * \brief Traverse a draft token tree and fill the token bitmask for each node.
   *
   * This function performs a DFS traversal of the speculative decoding tree and fills
   * the token bitmask for each node based on grammar constraints.
   *
   * \param retrieve_next_token DLTensor where retrieve_next_token[i] gives the index of
   *        the child node of node i, or -1 if no child exists.
   * \param retrieve_next_sibling DLTensor where retrieve_next_sibling[i] gives the index of
   *        the sibling node of node i, or -1 if no sibling exists.
   * \param draft_tokens DLTensor of draft token ids at each node.
   * \param token_bitmask DLTensor to store the bitmask (2D: num_nodes x bitmask_size).
   * \param time_threshold Maximum allowed time in seconds for the DFS traversal.
   *        If the traversal exceeds this threshold, it returns false.
   *        A value <= 0 disables the timeout (default: -1.0).
   * \param temperatures Optional DLTensor to store the effective temperature for each node
   *        (1D float32 with num_nodes elements). -1 represents no effective temperature; it is
   *        also written for nodes that were not visited or were rejected.
   * \return true if the traversal completed successfully, false if it timed out.
   */
  bool TraverseDraftTree(
      const DLTensor* retrieve_next_token,
      const DLTensor* retrieve_next_sibling,
      const DLTensor* draft_tokens,
      DLTensor* token_bitmask,
      double time_threshold = -1.0,
      DLTensor* temperatures = nullptr
  );

  /*!
   * \brief Find the jump-forward string for jump-forward decoding. This is the longest string that
   will be valid according to the current syntax.
   * \note This method does not change the grammar state.
   */
  std::string FindJumpForwardString();

  /*!
   * \brief Rollback the matcher to a previous state.
   * \param num_tokens The number of tokens to rollback. It cannot exceed the current number of
   * steps, nor can it exceed the specified maximum number of rollback tokens.
   */
  void Rollback(int num_tokens = 1);

  /*!
   * \brief Get the capture groups recorded so far, ordered by completion position.
   * \param deduplicate The Earley parser explores parse hypotheses in parallel, so one
   * occurrence of a captured rule may complete at several candidate end positions (e.g. a
   * /[0-9]+/ body completes after every digit). If true (default), only the last (longest)
   * completion of each occurrence is kept. Distinct occurrences — repeated matches of the same
   * rule at different positions — are always kept. If false, the raw completion events are
   * returned.
   * \return A list of (capture_name, matched_bytes) pairs. Rules gain a capture name via the
   * Lark attribute rule[capture] / rule[capture="name"] or the EBNF form rule[capture="name"]
   * ::= ... The bytes are the input span that the rule matched.
   * \note Captures are recorded when a rule is completed during AcceptToken / AcceptString, and
   * are rolled back together with Rollback. Mask computation never records captures. Completions
   * on parse paths that are later abandoned may still be recorded; for unambiguous grammars
   * whose captured rules have unambiguous boundaries, the deduplicated result is exact.
   */
  std::vector<std::pair<std::string, std::string>> GetCaptures(bool deduplicate = true) const;

  /*!
   * \brief Check if the matcher has accepted the stop token and terminated.
   * \sa AcceptToken
   */
  bool IsTerminated() const;

  /*!
   * \brief Check if the grammar's root rule has been fully matched by the input accepted so far.
   * Unlike IsTerminated(), this does not require the stop token to have been accepted.
   * \sa IsTerminated, AcceptToken
   */
  bool IsCompleted() const;

  /*! \brief Reset the matcher to the initial state. */
  void Reset();

  /*!
   * \brief Fork the matcher. Returns a new GrammarMatcher with a deep copy of all state except
   * compiled_grammar and tokenizer_info, which are shared with this matcher.
   */
  GrammarMatcher Fork() const;

  /*! \brief Get the maximum number of rollback tokens allowed. */
  int GetMaxRollbackTokens() const;

  /*! \brief Get the effective sampling temperature for the next token. */
  std::optional<float> GetTemperature() const;

  const std::vector<int>& GetStopTokenIds() const;

  /*! \brief Print the internal state of the matcher. This is only used for debugging. The
   * representation of the internal state is subject to change.
   */
  std::string _DebugPrintInternalState() const;

  XGRAMMAR_DEFINE_PIMPL_METHODS(GrammarMatcher);
};

/*!
 * \brief A batched version of GrammarMatcher for better efficiency. It supports batch processing
 * of multiple GrammarMatcher objects in parallel.
 *
 * \details This class provides batched versions of the core methods of GrammarMatcher, including
 * FillNextTokenBitmask, AcceptString, and AcceptToken. It utilizes multi-threading to process
 * multiple GrammarMatcher objects simultaneously, significantly improving efficiency when dealing
 * with a large number of matchers.
 */
class BatchGrammarMatcher {
 public:
  BatchGrammarMatcher(std::variant<std::string, int32_t> max_threads = "auto");

  /*!
    \brief A batched version of FillNextTokenBitmask for better efficiency.
    \param matchers The array of GrammarMatcher objects.
    \param next_token_bitmask The pre-allocated DLTensor to store the result bitmasks.
    \param indices The optional array of indices to specify which matcher corresponds to which slice
    of the bitmask tensor. If not provided, all matchers will write to the corresponding
    indices(matchers[i] to next_token_bitmask[i]).
    \param debug_print Whether to print debug information. Default is false.
  */
  void BatchFillNextTokenBitmask(
      std::vector<GrammarMatcher>* matchers,
      DLTensor* next_token_bitmask,
      const std::optional<std::vector<int32_t>>& indices = std::nullopt,
      bool debug_print = false
  );

  /*!
   * \brief Fill the effective sampling temperature of each matcher into a tensor.
   * \param matchers The array of GrammarMatcher objects.
   * \param temperatures The pre-allocated 1D float32 CPU tensor to store the result. The entry of
   * a matcher without an effective temperature is set to -1.
   * \param indices The optional array of indices to specify which element each matcher writes to.
   * If not provided, matchers[i] writes to temperatures[i].
   */
  static void BatchFillTemperature(
      const std::vector<GrammarMatcher>& matchers,
      DLTensor* temperatures,
      const std::optional<std::vector<int32_t>>& indices = std::nullopt
  );

  /*!
   * \brief A batched version of AcceptString for better efficiency.
   * \param matchers The array of GrammarMatcher objects.
   * \param input_strs The array of input strings to be accepted.
   * \param debug_print Whether to print debug information. Default is false.
   * \return A vector of bytes indicating whether each string is accepted.
   */
  static std::vector<uint8_t> BatchAcceptString(
      std::vector<GrammarMatcher>* matchers,
      const std::vector<std::string>& input_strs,
      bool debug_print = false
  );

  /*!
   * \brief A batched version of AcceptToken for better efficiency.
   * \param matchers The array of GrammarMatcher objects.
   * \param token_ids The array of token ids to be accepted.
   * \param debug_print Whether to print debug information. Default is false.
   * \return A vector of bytes indicating whether each token is accepted.
   */
  static std::vector<uint8_t> BatchAcceptToken(
      std::vector<GrammarMatcher>* matchers,
      const std::vector<int32_t>& token_ids,
      bool debug_print = false
  );

  /*!
   * \brief A batched version of Rollback for better efficiency.
   * \param matchers The array of GrammarMatcher objects.
   * \param num_tokens The array of the number of tokens to rollback for each matcher.
   */
  static void BatchRollback(
      std::vector<GrammarMatcher>* matchers, const std::vector<int>& num_tokens
  );

  XGRAMMAR_DEFINE_PIMPL_METHODS(BatchGrammarMatcher);
};

}  // namespace xgrammar

#endif  // XGRAMMAR_MATCHER_H_
