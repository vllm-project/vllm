/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/compiler.cc
 */

#include <xgrammar/compiler.h>

#include <algorithm>
#include <bitset>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "compiled_grammar_impl.h"
#include "earley_parser.h"
#include "fsm.h"
#include "grammar_functor.h"
#include "grammar_impl.h"
#include "support/dynamic_bitset.h"
#include "support/int_set.h"
#include "support/logging.h"
#include "support/thread_pool.h"
#include "support/thread_safe_cache.h"
#include "support/utils.h"
#include "tokenizer_info_impl.h"
#include "xgrammar/grammar.h"
#include "xgrammar/tokenizer_info.h"

namespace xgrammar {

/************** AdaptiveTokenMaskCache Generator **************/

/*! \brief The concrete implementation of GrammarMatcherNode. */
class GrammarMatcherForTokenMaskCache : public EarleyParser {
 public:
  GrammarMatcherForTokenMaskCache(
      const Grammar& grammar,
      const ParserState& init_state,
      const std::unordered_map<int32_t, DynamicBitset>&
          tag_dispatch_rule_id_to_second_slicing_bitset,
      const TokenizerInfo& tokenizer_info,
      std::optional<RuleLevelCache>& rule_level_cache
  )
      : EarleyParser(grammar, init_state),
        init_rule_id_(init_state.rule_id),
        initial_state_(init_state),
        tag_dispatch_rule_id_to_second_slicing_bitset_(tag_dispatch_rule_id_to_second_slicing_bitset
        ),
        tokenizer_info_(tokenizer_info),
        rule_level_cache_(rule_level_cache) {}
  /*!
   * \brief Get the adaptive token mask for the given ParserState.
   * \param is_root_rule Whether to consider the parent rule. If false, there will be
   * no uncertain tokens. Useful for the root rule.
   */
  AdaptiveTokenMask GetAdaptiveTokenMask(bool is_root_rule);

  /*!
   * \brief Get the token mask for the given ParserState.
   * \param first_char_mask The first character mask.
   * \param is_root_rule Whether to consider the parent rule. If false, there will be
   * no uncertain tokens. Useful for the root rule.
   * \returns True if the rejected indices are filled as usual, False otherwise.
   * It's used to determine which construction function will be used.
   */
  bool GetTokenMaskWithFirstCharacterCheck(
      const std::bitset<256>& first_char_mask,
      bool is_root_rule,
      const std::vector<int32_t>& token_edge_accepted
  );

  /*!
   * \brief Adapt the cache with lookahead assertion.
   * \param cache The adaptive token mask to be adapted.
   * \param is_root_rule Whether to consider the parent rule.
   */
  void AdaptCacheWithLookahead(AdaptiveTokenMask* cache, bool is_root_rule);

 private:
  /*! \brief Check if a token can pass the lookahead assertion. */
  std::pair</*acceptable*/ bool, /*can reach end*/ bool> IsTokenPassLookaheadAssertion(
      const std::string& token, const std::vector<bool>& can_reach_end_stack
  );

  /*!
   * \brief Check if speculative calculation will be applied.
   * \return first: whether speculative calculation is applicable.
   * \return second: part of the first character mask,
   * which can be used in speculative calculation.
   */
  std::pair<bool, std::bitset<256>> GetSpeculativeCalculation();

  /*!
   * \brief Get the first character mask.
   * \param first_character_mask the bitset to store the first character mask.
   */
  void GetFirstCharacterMask(std::bitset<256>& first_character_mask);

  /*!
   * \brief Compute sorted vocab indices accepted by token edges at the current FSM state.
   * Token(ids) edges accept listed token IDs.
   * ExcludeToken(ids) edges accept all tokens except listed IDs.
   * \return Sorted, deduplicated vector of accepted sorted vocab indices.
   */
  const std::vector<int32_t>& GetTokenEdgeAcceptedIndices();

  // The id of the initial rule.
  int32_t init_rule_id_;

  // The initial state of the parser.
  ParserState initial_state_;

  /*!
   * \brief This is a mapping from TagDispatch rule id to the bitset used for second slicing.
   * \note If a rule is a TagDispatch rule, then there will be an AC automaton for its triggers.
   *  Which means that it can accept a lot of tokens. However, it will be slow to check a lot of
   *  tokens. The DynamicBitset here is used to do a second slicing: if a token's substr(1, n - 1)
   *  can be accepted by the start state of the AC automaton, then it will be True in the bitset.
   *  When we check a token, we first check if its first character can transit to the start state.
   *  If yes, then we check if it is in the bitset. If yes, then we accept it directly.
   */
  const std::unordered_map<int32_t, DynamicBitset>& tag_dispatch_rule_id_to_second_slicing_bitset_;

  const TokenizerInfo& tokenizer_info_;

  std::optional<RuleLevelCache> rule_level_cache_;

  // Temporary data for GetAdaptiveTokenMask.
  std::vector<int32_t> tmp_accepted_indices_;
  std::vector<int32_t> tmp_rejected_indices_;
  std::vector<int32_t> tmp_uncertain_indices_;
  std::vector<int32_t> tmp_rejected_by_lookahead_indices_;
  std::vector<int32_t> tmp_accepted_by_lookahead_indices_;
  std::vector<bool> tmp_can_reach_end_stack_;
  std::vector<bool> tmp_can_reach_end_prefix_or_stack_;
  // Temporary data for GetTokenEdgeAcceptedIndices.
  std::vector<int32_t> tmp_token_edge_accepted_;
  std::vector<int32_t> tmp_token_edge_excluded_;
};

void GrammarMatcherForTokenMaskCache::AdaptCacheWithLookahead(
    AdaptiveTokenMask* cache_ptr, bool is_root_rule
) {
  AdaptiveTokenMask& cache = *cache_ptr;
  const auto& sorted_decoded_vocab = tokenizer_info_.GetSortedDecodedVocab();
  const auto& subtree_nodes_range = tokenizer_info_.GetTrieSubtreeNodesRange();
  const std::string* prev_token = nullptr;
  bool is_exact_lookahead = grammar_->GetRule(init_rule_id_).is_exact_lookahead;
  int prev_matched_size = 0;
  int last_rejected_range = 0;
  int last_uncertain_range = 0;
  if (is_root_rule) {
    tmp_rejected_indices_ = cache.uncertain_indices;
  } else {
    const auto& lookahead_id = grammar_->GetRule(init_rule_id_).lookahead_assertion_id;
    if (lookahead_id == -1) {
      return;
    }
    for (const auto& uncertain_index : cache.uncertain_indices) {
      const auto& token = sorted_decoded_vocab[uncertain_index].second;
      // Many tokens may contain the same prefix, so we will avoid unnecessary matching
      // by finding the longest common prefix with the previous token.
      bool accepted = true;
      if (uncertain_index < last_rejected_range) {
        tmp_rejected_indices_.push_back(uncertain_index);
        continue;
      }
      if (uncertain_index < last_uncertain_range) {
        // This token is already marked as uncertain.
        continue;
      }
      if (prev_token != nullptr) {
        int lcp_len =
            std::mismatch(token.begin(), token.end(), prev_token->begin(), prev_token->end())
                .first -
            token.begin();
        if (lcp_len > prev_matched_size) {
          // Case 1. The common prefix is rejected by the matcher in the last token. Reject
          // directly.
          accepted = false;
        } else if (lcp_len < prev_matched_size) {
          // Case 2. The common prefix is shorter than the previous matched size. Rollback
          // the non-common part.
          PopLastStates(prev_matched_size - lcp_len);
          tmp_can_reach_end_stack_.erase(
              tmp_can_reach_end_stack_.end() - (prev_matched_size - lcp_len),
              tmp_can_reach_end_stack_.end()
          );
          tmp_can_reach_end_prefix_or_stack_.erase(
              tmp_can_reach_end_prefix_or_stack_.end() - (prev_matched_size - lcp_len),
              tmp_can_reach_end_prefix_or_stack_.end()
          );
        }
        prev_matched_size = std::min(prev_matched_size, lcp_len);
      }

      prev_token = &token;

      if (accepted) {
        // Accept the rest chars one by one.
        for (int j = prev_matched_size; j < static_cast<int>(token.size()); ++j) {
          if (!Advance(token[j])) {
            accepted = false;
            break;
          }
          tmp_can_reach_end_stack_.push_back(IsCompleted());
          tmp_can_reach_end_prefix_or_stack_.push_back(
              tmp_can_reach_end_stack_.back() || tmp_can_reach_end_prefix_or_stack_.back()
          );
          prev_matched_size = j + 1;
        }
      }

      XGRAMMAR_DCHECK(!tmp_can_reach_end_prefix_or_stack_.empty());
      bool can_reach_end = tmp_can_reach_end_prefix_or_stack_.back();

      XGRAMMAR_DCHECK(!accepted) << "All the tokens are at least uncertain!";
      if (can_reach_end && prev_matched_size > 0) {
        auto [lookahead_accepted, lookahead_completed] =
            IsTokenPassLookaheadAssertion(token, tmp_can_reach_end_stack_);
        if ((!is_root_rule) && lookahead_accepted) {
          if (lookahead_completed || !is_exact_lookahead) {
            tmp_uncertain_indices_.push_back(uncertain_index);
          } else {
            tmp_accepted_indices_.push_back(uncertain_index);
          }
        } else {
          tmp_rejected_indices_.push_back(uncertain_index);
          last_rejected_range = subtree_nodes_range[uncertain_index];
        }
      } else {
        tmp_rejected_indices_.push_back(uncertain_index);
        last_rejected_range = subtree_nodes_range[uncertain_index];
      }
    }
  }

  // This strategy ensures the consistency of the cache storage type in most cases.
  // However, in this case, the storage type is inconsistent:
  // 1. The original cache is accepted_indices, and rejected_indices is also small.
  // After adapting with lookahead, |accepted_indices| + |accepted_by_lookahead_indices| >
  // |rejected_indices| + |rejected_by_lookahead_indices|, and |rejected_indices| +
  // |rejected_by_lookahead_indices| < AdaptiveTokenMask::USE_BITSET_THRESHOLD. In this case, it
  // should be kRejected, but ignored.
  // 2. The original cache is rejected_indices, and accepted_indices is also small.
  // After adapting with lookahead, |accepted_indices| + |accepted_by_lookahead_indices| <
  // |rejected_indices| + |rejected_by_lookahead_indices|, and |accepted_indices| +
  // |accepted_by_lookahead_indices| < AdaptiveTokenMask::USE_BITSET_THRESHOLD. In this case, it
  // should be kAccepted, but ignored. These two cases are very rare in practice, and the impact is
  // very limited, so we ignore them for simplicity.
  cache.uncertain_indices = tmp_uncertain_indices_;
  switch (cache.store_type) {
    case AdaptiveTokenMask::StoreType::kAccepted: {
      if (cache.accepted_indices.size() + tmp_accepted_indices_.size() <
          AdaptiveTokenMask::USE_BITSET_THRESHOLD) {
        IntsetUnion(&cache.accepted_indices, tmp_accepted_indices_);
        break;
      }
      // Transform to bitset.
      cache.store_type = AdaptiveTokenMask::StoreType::kAcceptedBitset;
      cache.accepted_bitset = DynamicBitset(tokenizer_info_.GetVocabSize());
      for (const auto& accepted_index : cache.accepted_indices) {
        cache.accepted_bitset.Set(sorted_decoded_vocab[accepted_index].first);
      }
      for (const auto& accepted_index : tmp_accepted_indices_) {
        cache.accepted_bitset.Set(sorted_decoded_vocab[accepted_index].first);
      }
      cache.accepted_indices.clear();
      break;
    }
    case AdaptiveTokenMask::StoreType::kRejected: {
      if (cache.rejected_indices.size() + tmp_rejected_indices_.size() <
          AdaptiveTokenMask::USE_BITSET_THRESHOLD) {
        IntsetUnion(&cache.rejected_indices, tmp_rejected_indices_);
        break;
      }
      // Transform to bitset.
      cache.store_type = AdaptiveTokenMask::StoreType::kAcceptedBitset;
      cache.accepted_bitset = DynamicBitset(tokenizer_info_.GetVocabSize());
      cache.accepted_bitset.Set();
      for (const auto& special_index : tokenizer_info_.GetSpecialTokenIds()) {
        cache.accepted_bitset.Reset(special_index);
      }
      for (const auto& uncertain_index : cache.uncertain_indices) {
        cache.accepted_bitset.Reset(sorted_decoded_vocab[uncertain_index].first);
      }
      for (const auto& rejected_index : cache.rejected_indices) {
        cache.accepted_bitset.Reset(sorted_decoded_vocab[rejected_index].first);
      }
      for (const auto& rejected_index : tmp_rejected_indices_) {
        cache.accepted_bitset.Reset(sorted_decoded_vocab[rejected_index].first);
      }
      cache.rejected_indices.clear();
      break;
    }
    case AdaptiveTokenMask::StoreType::kAcceptedBitset: {
      for (const auto& accepted_index : tmp_accepted_indices_) {
        cache.accepted_bitset.Set(sorted_decoded_vocab[accepted_index].first);
      }
      break;
    }
  }
}

std::pair<bool, bool> GrammarMatcherForTokenMaskCache::IsTokenPassLookaheadAssertion(
    const std::string& token, const std::vector<bool>& can_reach_end_stack
) {
  bool accepted = true;
  bool can_reach_end = true;
  auto lookahead_assertion_id = grammar_->GetRule(init_rule_id_).lookahead_assertion_id;
  if (lookahead_assertion_id == -1) {
    return {accepted, can_reach_end};
  }
  auto lookahead_state =
      ParserState(/*rule_id*/ -1, lookahead_assertion_id, 0, ParserState::kNoPrevInputPos, 0);
  PushStateAndExpand(lookahead_state);
  int token_len = token.size();
  if (IsCompleted()) {
    // If the lookahead assertion is already completed, we can accept the token.
    PopLastStates(1);
    return {accepted, can_reach_end};
  }

  // Find all positions that can come to and end. Then check if the suffix from that position
  // can be accepted by the lookahead assertion.
  for (int i = static_cast<int>(can_reach_end_stack.size()) - 1; i >= 0; --i) {
    if (!can_reach_end_stack[i]) {
      continue;
    }
    int last_accept_pos = i - 1;
    for (int pos = i; pos < token_len; ++pos) {
      if (!Advance(token[pos])) {
        break;
      }
      last_accept_pos = pos;
      // Case 1. The whole rule is finished.
      if (IsCompleted()) {
        // accepted chars: pos - i + 1
        // we need to rollback the pushed initial state as well
        PopLastStates(pos - i + 2);
        return {accepted, can_reach_end};
      }
    }
    // Case 2. The whole token is accepted
    if (last_accept_pos == token_len - 1) {
      PopLastStates(last_accept_pos - i + 2);
      can_reach_end = false;
      return {accepted, can_reach_end};
    }
    // Case 3. The token is not accepted. Check the next position.
    PopLastStates(last_accept_pos - i + 1);
  }

  PopLastStates(1);
  can_reach_end = false;
  accepted = false;
  return {accepted, can_reach_end};
}

// Comparator for std::pair<int32_t, std::string> based on the string value.
class IntStringPairComparator {
 public:
  bool operator()(
      const std::pair<int32_t, std::string>& lhs, const std::pair<int32_t, std::string>& rhs
  ) const {
    return lhs.second < rhs.second;
  }
};

int GetPossibleTokenIntervals(
    const std::vector<std::pair<int32_t, std::string>>& sorted_decoded_vocab,
    const std::bitset<256>& first_char_mask,
    std::vector<std::pair<int32_t, int32_t>>& possible_intervals
) {
  int possible_token_num = 0;
  int matched_size = 0;
  int last_interval_end = -1;
  for (int32_t i = 0; i < 256; i++) {
    if (first_char_mask[i]) {
      if (last_interval_end == -1) {
        last_interval_end = i;
      }
    } else {
      if (last_interval_end != -1) {
        int32_t interval_left_end =
            std::lower_bound(
                sorted_decoded_vocab.begin() + matched_size,
                sorted_decoded_vocab.end(),
                std::make_pair(0, std::string(1, static_cast<uint8_t>(last_interval_end))),
                IntStringPairComparator()
            ) -
            sorted_decoded_vocab.begin();
        int32_t interval_right_end = std::lower_bound(
                                         sorted_decoded_vocab.begin() + interval_left_end,
                                         sorted_decoded_vocab.end(),
                                         std::make_pair(0, std::string(1, static_cast<uint8_t>(i))),
                                         IntStringPairComparator()
                                     ) -
                                     sorted_decoded_vocab.begin();
        possible_intervals.emplace_back(interval_left_end, interval_right_end);
        possible_token_num += interval_right_end - interval_left_end;
        last_interval_end = -1;
        matched_size = interval_right_end;
      }
    }
  }

  if (last_interval_end != -1) {
    // If the last interval is not closed, we need to close it.
    int32_t interval_left_end =
        std::lower_bound(
            sorted_decoded_vocab.begin() + matched_size,
            sorted_decoded_vocab.end(),
            std::make_pair(0, std::string(1, static_cast<uint8_t>(last_interval_end))),
            IntStringPairComparator()
        ) -
        sorted_decoded_vocab.begin();
    possible_intervals.emplace_back(interval_left_end, sorted_decoded_vocab.size());
    possible_token_num += sorted_decoded_vocab.size() - interval_left_end;
  }
  return possible_token_num;
}

std::pair<bool, std::bitset<256>> GrammarMatcherForTokenMaskCache::GetSpeculativeCalculation() {
  using GrammarExprType = Grammar::Impl::GrammarExprType;
  // If the initial rule is a tag dispatch, we will check if it can achieve its initial state.
  const auto& rule = grammar_->GetRule(init_rule_id_);
  if (rule.is_lazy) {
    // The fast path assumes greedy self-loop extension is always legal, which does not hold for
    // committed-shortest rules; they must go through the full per-token simulation.
    return {false, std::bitset<256>()};
  }
  const auto& rule_body = grammar_->GetGrammarExpr(rule.body_expr_id);
  if (rule_body.type == GrammarExprType::kTagDispatch) {
    std::bitset<256> speculative_mask;
    XGRAMMAR_DCHECK(grammar_->per_rule_fsms[init_rule_id_].has_value());
    const auto& fsm = grammar_->per_rule_fsms[init_rule_id_].value();
    for (const auto& edge : fsm.GetFsm().GetFsm().GetEdges(initial_state_.element_id)) {
      if (edge.target != fsm.GetFsm().GetStart()) {
        continue;
      }
      if (!edge.IsCharRange()) {
        continue;
      }
      for (int32_t ch = edge.min; ch <= edge.max; ++ch) {
        speculative_mask.set(ch);
      }
    }
    return {true, speculative_mask};
  }

  // Check if the initial state is self-recursive-like via FSM.
  XGRAMMAR_DCHECK(grammar_->per_rule_fsms[init_rule_id_].has_value());
  bool can_be_applied = false;
  std::bitset<256> speculative_mask;
  const auto& fsm = grammar_->per_rule_fsms[init_rule_id_].value();
  XGRAMMAR_DCHECK(initial_state_.element_id < fsm.GetFsm().NumStates())
      << "Initial State's element id cannot exceed the whole FSM's number of states.";
  for (const auto& edge : fsm.GetFsm().GetFsm().GetEdges(initial_state_.element_id)) {
    if (edge.IsCharRange()) {
      // Case A: The edge is towards itself.
      if (edge.target == initial_state_.element_id) {
        can_be_applied = true;
        for (int ch = edge.min; ch <= edge.max; ++ch) {
          speculative_mask.set(ch);
        }
        continue;
      }

      // Case B: The state is the start state, and there's an edge to another state,
      // which calls the fsm itself.
      if (fsm.GetFsm().GetStart() == initial_state_.element_id) {
        for (const auto& next_edge : fsm.GetFsm().GetFsm().GetEdges(edge.target)) {
          if ((next_edge.IsRuleRef() && next_edge.GetRefRuleId() == init_rule_id_) ||
              (next_edge.IsRepeatRef() &&
               fsm.GetFsm().GetFsm().GetRepeatEdgeInfo(next_edge.GetAuxIndex()).RuleId() ==
                   init_rule_id_)) {
            can_be_applied = true;
            for (int ch = edge.min; ch <= edge.max; ++ch) {
              speculative_mask.set(ch);
            }
            break;
          }
        }
      }
    }
  }
  return {can_be_applied, speculative_mask};
}

bool GrammarMatcherForTokenMaskCache::GetTokenMaskWithFirstCharacterCheck(
    const std::bitset<256>& first_char_mask,
    bool is_root_rule,
    const std::vector<int32_t>& token_edge_accepted
) {
  const auto& sorted_decoded_vocab = tokenizer_info_.GetSortedDecodedVocab();
  const auto& subtree_nodes_range = tokenizer_info_.GetTrieSubtreeNodesRange();
  // the pair (a, b) means [a, b). Intialize the possible intervals.
  std::vector<std::pair<int32_t, int32_t>> possible_intervals;
  int possible_token_num =
      GetPossibleTokenIntervals(sorted_decoded_vocab, first_char_mask, possible_intervals);

  // Check if the type of the mask can be rejected.
  tmp_accepted_indices_.reserve(possible_token_num);
  bool fill_reject_indices =
      (sorted_decoded_vocab.size() - possible_token_num) < AdaptiveTokenMask::USE_BITSET_THRESHOLD;

  XGRAMMAR_DCHECK(possible_intervals.size() > 0)
      << "There should be at least one possible interval for the first character mask.";

  if (possible_intervals[0].first != 0 && fill_reject_indices) {
    for (int i = 0; i < possible_intervals[0].first; ++i) {
      tmp_rejected_indices_.push_back(i);
    }
  }

  XGRAMMAR_DCHECK(init_rule_id_ != -1 && grammar_->per_rule_fsms[init_rule_id_].has_value());
  auto [speculative_calculation, speculative_mask] = GetSpeculativeCalculation();
  if (has_char_budget_rules_) {
    speculative_calculation = false;
  }

  int prev_matched_size = 0;
  int last_rejected_range = 0;
  const bool& is_exact_lookahead = grammar_->GetRule(init_rule_id_).is_exact_lookahead;
  std::optional<const DynamicBitset*> definite_accepted_bitset = std::nullopt;
  const bool is_tag_dispatch_rule =
      grammar_->GetGrammarExpr(grammar_->GetRule(init_rule_id_).body_expr_id).type ==
      Grammar::Impl::GrammarExprType::kTagDispatch;
  if (is_tag_dispatch_rule) {
    XGRAMMAR_DCHECK(tag_dispatch_rule_id_to_second_slicing_bitset_.count(init_rule_id_) > 0);
    definite_accepted_bitset = &tag_dispatch_rule_id_to_second_slicing_bitset_.at(init_rule_id_);
  }

  const std::string* prev_token = nullptr;
  int32_t skip_ptr = 0;
  const int32_t skip_size = static_cast<int32_t>(token_edge_accepted.size());
  for (size_t interval_idx = 0; interval_idx < possible_intervals.size(); ++interval_idx) {
    const auto& interval = possible_intervals[interval_idx];
    for (int i = interval.first; i < interval.second; ++i) {
      // Skip tokens already accepted by token edges (avoid expensive Earley simulation).
      while (skip_ptr < skip_size && token_edge_accepted[skip_ptr] < i) ++skip_ptr;
      if (skip_ptr < skip_size && token_edge_accepted[skip_ptr] == i) continue;

      // Check if the current token is in the rejected range. i.e. check if the current token
      // is on the subtree of the rejected token.
      if (i < last_rejected_range) {
        if (fill_reject_indices) {
          tmp_rejected_indices_.push_back(i);
          fill_reject_indices =
              tmp_rejected_indices_.size() >= AdaptiveTokenMask::USE_BITSET_THRESHOLD
                  ? false
                  : fill_reject_indices;
        } else {
          i = last_rejected_range - 1;
        }
        continue;
      }
      const auto& token = sorted_decoded_vocab[i].second;
      // This optimization is useful for simple self-recursive rules, like string content.
      if (speculative_calculation) {
        // Optimization for tag dispatch rules.
        if (definite_accepted_bitset.has_value()) {
          // If the token is empty, it must be accepted.
          if (token.empty()) {
            tmp_accepted_indices_.push_back(i);
            continue;
          }
          // If the token doesn't contain tags or stop strings since the second character, and it
          // will transit to the start state after consuming the first character, it must be
          // accepted.
          if (speculative_mask[static_cast<uint8_t>(token[0])] &&
              (*definite_accepted_bitset.value())[i]) {
            tmp_accepted_indices_.push_back(i);
            continue;
          }
        } else {
          bool all_accepted = true;
          for (char ch : token) {
            // If the first character is not the ascii character or can't be accepted by the
            // first character mask, we need to check them in the parser.
            if (isascii(ch) == 0 || !speculative_mask[static_cast<uint8_t>(ch)]) {
              all_accepted = false;
              break;
            }
          }
          if (all_accepted) {
            tmp_accepted_indices_.push_back(i);
            continue;
          }
        }
      }
      // Many tokens may contain the same prefix, so we will avoid unnecessary matching
      // by finding the longest common prefix with the previous token.
      bool accepted = true;
      if (prev_token != nullptr) {
        int lcp_len =
            std::mismatch(token.begin(), token.end(), prev_token->begin(), prev_token->end())
                .first -
            token.begin();
        if (lcp_len > prev_matched_size) {
          // Case 1. The common prefix is rejected by the matcher in the last token. Reject
          // directly.
          accepted = false;
        } else if (lcp_len < prev_matched_size) {
          // Case 2. The common prefix is shorter than the previous matched size. Rollback
          // the non-common part.
          PopLastStates(prev_matched_size - lcp_len);
          tmp_can_reach_end_stack_.erase(
              tmp_can_reach_end_stack_.end() - (prev_matched_size - lcp_len),
              tmp_can_reach_end_stack_.end()
          );
          tmp_can_reach_end_prefix_or_stack_.erase(
              tmp_can_reach_end_prefix_or_stack_.end() - (prev_matched_size - lcp_len),
              tmp_can_reach_end_prefix_or_stack_.end()
          );
        }
        prev_matched_size = std::min(prev_matched_size, lcp_len);
      }

      prev_token = &token;

      if (accepted) {
        // Accept the rest chars one by one.
        for (int j = prev_matched_size; j < static_cast<int>(token.size()); ++j) {
          if (!Advance(token[j])) {
            accepted = false;
            break;
          }
          tmp_can_reach_end_stack_.push_back(IsCompleted());
          tmp_can_reach_end_prefix_or_stack_.push_back(
              tmp_can_reach_end_stack_.back() || tmp_can_reach_end_prefix_or_stack_.back()
          );
          prev_matched_size = j + 1;
        }
      }

      bool can_reach_end = tmp_can_reach_end_prefix_or_stack_.back();

      if (accepted) {
        if (HasEnteredCharBudget()) {
          tmp_uncertain_indices_.push_back(i);
        } else {
          tmp_accepted_indices_.push_back(i);
        }
      } else if (can_reach_end && prev_matched_size > 0) {
        auto [lookahead_accepted, lookahead_completed] =
            IsTokenPassLookaheadAssertion(token, tmp_can_reach_end_stack_);
        if ((!is_root_rule) && lookahead_accepted) {
          if (lookahead_completed || !is_exact_lookahead) {
            tmp_uncertain_indices_.push_back(i);
          } else if (HasEnteredCharBudget()) {
            tmp_uncertain_indices_.push_back(i);
          } else {
            tmp_accepted_indices_.push_back(i);
            tmp_accepted_by_lookahead_indices_.push_back(i);
          }
        } else {
          for (int j = i; j < subtree_nodes_range[i]; j++) {
            tmp_rejected_indices_.push_back(j);
            tmp_rejected_by_lookahead_indices_.push_back(j);
          }
          i = subtree_nodes_range[i] - 1;  // Skip the subtree nodes.
        }
      } else {
        tmp_rejected_indices_.push_back(i);
        last_rejected_range = subtree_nodes_range[i];
        fill_reject_indices =
            tmp_rejected_indices_.size() >= AdaptiveTokenMask::USE_BITSET_THRESHOLD
                ? false
                : fill_reject_indices;
      }
    }
    if (interval_idx != possible_intervals.size() - 1 && fill_reject_indices) {
      const auto& next_interval = possible_intervals[interval_idx + 1];
      for (int i = interval.second; i < next_interval.first; ++i) {
        tmp_rejected_indices_.push_back(i);
      }
      fill_reject_indices = tmp_rejected_indices_.size() >= AdaptiveTokenMask::USE_BITSET_THRESHOLD
                                ? false
                                : fill_reject_indices;
    }
  }

  // Rollback the last matched part.
  PopLastStates(prev_matched_size);

  if (possible_intervals.back().second != static_cast<int>(sorted_decoded_vocab.size()) &&
      fill_reject_indices) {
    // If the last interval is not closed, we need to reject the rest tokens.
    for (int i = possible_intervals.back().second;
         i < static_cast<int>(sorted_decoded_vocab.size());
         ++i) {
      tmp_rejected_indices_.push_back(i);
    }
  }

  return fill_reject_indices;
}

void GrammarMatcherForTokenMaskCache::GetFirstCharacterMask(std::bitset<256>& first_character_mask
) {
  first_character_mask.reset();
  XGRAMMAR_DCHECK(grammar_->per_rule_fsms[init_rule_id_].has_value());
  const auto& fsm = grammar_->per_rule_fsms[init_rule_id_].value();
  const auto& edges = fsm.GetFsm().GetFsm().GetEdges(initial_state_.element_id);
  for (const auto& edge : edges) {
    if (edge.IsCharRange()) {
      for (int c = edge.min; c <= edge.max; ++c) {
        first_character_mask[c] = true;
      }
    }
  }
}

const std::vector<int32_t>& GrammarMatcherForTokenMaskCache::GetTokenEdgeAcceptedIndices() {
  // Compute sorted vocab indices accepted by Token(ids) and ExcludeToken(ids) edges.
  // Result is stored in tmp_token_edge_accepted_.

  tmp_token_edge_accepted_.clear();
  tmp_token_edge_excluded_.clear();

  XGRAMMAR_DCHECK(grammar_->per_rule_fsms[init_rule_id_].has_value());
  const auto& fsm = grammar_->per_rule_fsms[init_rule_id_].value();
  const auto& edges = fsm.GetFsm().GetFsm().GetEdges(initial_state_.element_id);

  const auto& sorted_decoded_vocab = tokenizer_info_.GetSortedDecodedVocab();
  int32_t sorted_size = static_cast<int32_t>(sorted_decoded_vocab.size());
  const auto& tid_to_sorted = tokenizer_info_.ImplPtr()->GetTokenIdToSortedVocabIndex();
  // Out-of-vocabulary ids are rejected before compilation starts (CheckTokenIdsInVocab); guard
  // here as well because this runs on worker threads, where an out-of-bounds read cannot be
  // reported as an error.
  auto sorted_index = [&](int32_t tid) {
    return tid >= 0 && tid < static_cast<int32_t>(tid_to_sorted.size()) ? tid_to_sorted[tid] : -1;
  };

  bool has_exclude_token = false;

  for (const auto& edge : edges) {
    if (edge.IsToken()) {
      auto info = fsm.GetFsm().GetFsm().GetTokenEdgeInfo(edge.GetAuxIndex());
      for (int32_t i = 0; i < info.Count(); ++i) {
        if (int32_t index = sorted_index(info.TokenIds()[i]); index >= 0) {
          tmp_token_edge_accepted_.push_back(index);
        }
      }
    } else if (edge.IsExcludeToken()) {
      has_exclude_token = true;
      auto info = fsm.GetFsm().GetFsm().GetExcludeTokenEdgeInfo(edge.GetAuxIndex());
      for (int32_t i = 0; i < info.Count(); ++i) {
        if (int32_t index = sorted_index(info.TokenIds()[i]); index >= 0) {
          tmp_token_edge_excluded_.push_back(index);
        }
      }
    }
  }

  // Token-only: result = token_accepted
  if (!has_exclude_token) {
    if (!tmp_token_edge_accepted_.empty()) {
      std::sort(tmp_token_edge_accepted_.begin(), tmp_token_edge_accepted_.end());
      tmp_token_edge_accepted_.erase(
          std::unique(tmp_token_edge_accepted_.begin(), tmp_token_edge_accepted_.end()),
          tmp_token_edge_accepted_.end()
      );
    }
    return tmp_token_edge_accepted_;
  }

  // ExcludeToken: result = [0, sorted_size) - (excluded - token_accepted)
  // Token(ids) overrides ExcludeToken(ids) when both present.
  if (!tmp_token_edge_accepted_.empty()) {
    std::sort(tmp_token_edge_accepted_.begin(), tmp_token_edge_accepted_.end());
    tmp_token_edge_accepted_.erase(
        std::unique(tmp_token_edge_accepted_.begin(), tmp_token_edge_accepted_.end()),
        tmp_token_edge_accepted_.end()
    );
  }
  std::sort(tmp_token_edge_excluded_.begin(), tmp_token_edge_excluded_.end());
  tmp_token_edge_excluded_.erase(
      std::unique(tmp_token_edge_excluded_.begin(), tmp_token_edge_excluded_.end()),
      tmp_token_edge_excluded_.end()
  );
  IntsetDifference(&tmp_token_edge_excluded_, tmp_token_edge_accepted_);
  IntsetComplement(&tmp_token_edge_accepted_, sorted_size, tmp_token_edge_excluded_);
  return tmp_token_edge_accepted_;
}

AdaptiveTokenMask GrammarMatcherForTokenMaskCache::GetAdaptiveTokenMask(bool is_root_rule) {
  tmp_accepted_indices_.clear();
  tmp_rejected_indices_.clear();
  tmp_uncertain_indices_.clear();
  tmp_rejected_by_lookahead_indices_.clear();
  tmp_accepted_by_lookahead_indices_.clear();
  tmp_can_reach_end_prefix_or_stack_.clear();
  tmp_can_reach_end_stack_.clear();
  // For every character in the current token, stores whether it is possible to reach the end of
  // the rule when matching until this character. Store it in a stack for later rollback.
  tmp_can_reach_end_stack_.push_back(false);
  tmp_can_reach_end_prefix_or_stack_.push_back(false);

  // Try to get the crossing cache.
  bool rule_level_cache_is_available = !has_char_budget_rules_ && rule_level_cache_.has_value() &&
                                       grammar_->per_rule_fsm_hashes[init_rule_id_].has_value();
  std::optional<uint64_t> fsm_hash = std::nullopt;
  int32_t new_state_id = -1;
  std::optional<AdaptiveTokenMask> crossing_cache = std::nullopt;
  int lookahead_id = grammar_->GetRule(initial_state_.rule_id).lookahead_assertion_id;
  bool is_exact_lookahead = grammar_->GetRule(initial_state_.rule_id).is_exact_lookahead;
  std::optional<uint64_t> lookahead_hash = std::nullopt;
  if (rule_level_cache_is_available) {
    lookahead_hash = GrammarFSMHasher::HashSequence(grammar_, lookahead_id);
    const auto& original_to_new_id = grammar_->per_rule_fsm_new_state_ids[init_rule_id_];
    fsm_hash = grammar_->per_rule_fsm_hashes[init_rule_id_].value();
    for (const auto& original_new_pair : original_to_new_id) {
      if (original_new_pair.first == initial_state_.element_id) {
        new_state_id = original_new_pair.second;
        break;
      }
    }
    XGRAMMAR_DCHECK(new_state_id != -1);
    const auto& fsm = grammar_->per_rule_fsms[init_rule_id_].value();
    if (lookahead_hash.has_value()) {
      crossing_cache = rule_level_cache_->GetCache(
          HashCombine(fsm_hash.value(), lookahead_hash.value(), is_exact_lookahead),
          new_state_id,
          fsm.GetNodeNum(),
          fsm.GetEdgeNum()
      );
      if (crossing_cache.has_value()) {
        // A perfect match.
        return crossing_cache.value();
      }
    }
    crossing_cache = rule_level_cache_->GetCache(
        fsm_hash.value(), new_state_id, fsm.GetNodeNum(), fsm.GetEdgeNum()
    );
    // If the rule doesn't have a lookahead, then it is exactly the same fsm.
    if (crossing_cache.has_value()) {
      AdaptCacheWithLookahead(&crossing_cache.value(), is_root_rule);
      return std::move(crossing_cache.value());
    }
  }

  std::bitset<256> first_character_mask;
  GetFirstCharacterMask(first_character_mask);

  // Token edge accepted indices (for byte path skip + merge).
  const auto& token_edge_accepted = GetTokenEdgeAcceptedIndices();

  // Byte path: skip tokens already accepted by token edges.
  bool rejected_filled;
  if (first_character_mask.none()) {
    rejected_filled = false;
  } else {
    rejected_filled = GetTokenMaskWithFirstCharacterCheck(
        first_character_mask, is_root_rule, token_edge_accepted
    );
  }

  // Token edges are rechecked at runtime when a character budget is present because accepting
  // one can enter a budgeted rule within the same token.
  if (!token_edge_accepted.empty()) {
    if (has_char_budget_rules_) {
      IntsetUnion(&tmp_uncertain_indices_, token_edge_accepted);
    } else {
      IntsetUnion(&tmp_accepted_indices_, token_edge_accepted);
      IntsetDifference(&tmp_uncertain_indices_, token_edge_accepted);
    }
    IntsetDifference(&tmp_rejected_indices_, token_edge_accepted);
  }
  if (rejected_filled) {
    auto return_value = AdaptiveTokenMask(
        tokenizer_info_.GetVocabSize(),
        tokenizer_info_.GetSortedDecodedVocab(),
        tmp_accepted_indices_,
        tmp_rejected_indices_,
        tmp_uncertain_indices_
    );
    if (rule_level_cache_is_available) {
      if (lookahead_id == -1 && !is_root_rule) {
        // If the rule doesn't have a lookahead, then it is exactly the same fsm.
        auto& fsm = grammar_->per_rule_fsms[init_rule_id_].value();
        rule_level_cache_->AddCache(
            fsm_hash.value(), new_state_id, fsm.GetNodeNum(), fsm.GetEdgeNum(), return_value
        );
        return return_value;
      }

      // We can add a cache for basic fsm, and a better one for lookahead.
      // All the tokens rejected by lookahead should be uncertain.
      IntsetUnion(&tmp_uncertain_indices_, tmp_rejected_by_lookahead_indices_);
      IntsetUnion(&tmp_uncertain_indices_, tmp_accepted_by_lookahead_indices_);
      std::vector<int32_t> rejected_indices_without_lookahead;
      std::vector<int32_t> accepted_indices_without_lookahead;
      rejected_indices_without_lookahead.reserve(
          tmp_rejected_indices_.size() - tmp_rejected_by_lookahead_indices_.size()
      );
      accepted_indices_without_lookahead.reserve(
          tmp_accepted_indices_.size() - tmp_accepted_by_lookahead_indices_.size()
      );
      std::set_difference(
          tmp_rejected_indices_.begin(),
          tmp_rejected_indices_.end(),
          tmp_rejected_by_lookahead_indices_.begin(),
          tmp_rejected_by_lookahead_indices_.end(),
          std::back_inserter(rejected_indices_without_lookahead)
      );
      std::set_difference(
          tmp_accepted_indices_.begin(),
          tmp_accepted_indices_.end(),
          tmp_accepted_by_lookahead_indices_.begin(),
          tmp_accepted_by_lookahead_indices_.end(),
          std::back_inserter(accepted_indices_without_lookahead)
      );
      auto& fsm = grammar_->per_rule_fsms[init_rule_id_].value();
      rule_level_cache_->AddCache(
          fsm_hash.value(),
          new_state_id,
          fsm.GetNodeNum(),
          fsm.GetEdgeNum(),
          AdaptiveTokenMask(
              tokenizer_info_.GetVocabSize(),
              tokenizer_info_.GetSortedDecodedVocab(),
              accepted_indices_without_lookahead,
              rejected_indices_without_lookahead,
              tmp_uncertain_indices_
          )
      );
      if (lookahead_hash.has_value()) {
        auto& fsm = grammar_->per_rule_fsms[init_rule_id_].value();
        rule_level_cache_->AddCache(
            HashCombine(fsm_hash.value(), lookahead_hash.value(), is_exact_lookahead),
            new_state_id,
            fsm.GetNodeNum(),
            fsm.GetEdgeNum(),
            return_value
        );
      }
    }
    return return_value;
  } else {
    auto return_value = AdaptiveTokenMask(
        tokenizer_info_.GetVocabSize(),
        tokenizer_info_.GetSortedDecodedVocab(),
        tmp_accepted_indices_,
        tmp_uncertain_indices_
    );

    if (rule_level_cache_is_available) {
      // Prepare for cache.
      auto& fsm = grammar_->per_rule_fsms[init_rule_id_].value();
      if (lookahead_id == -1 && !is_root_rule) {
        // If the rule doesn't have a lookahead, then it is exactly the same fsm.
        rule_level_cache_->AddCache(
            fsm_hash.value(), new_state_id, fsm.GetNodeNum(), fsm.GetEdgeNum(), return_value
        );
        return return_value;
      }

      // Add 2 caches.
      IntsetUnion(&tmp_uncertain_indices_, tmp_rejected_by_lookahead_indices_);
      IntsetUnion(&tmp_uncertain_indices_, tmp_accepted_by_lookahead_indices_);
      std::vector<int32_t> accepted_indices_without_lookahead;
      accepted_indices_without_lookahead.reserve(
          tmp_accepted_indices_.size() - tmp_accepted_by_lookahead_indices_.size()
      );
      std::set_difference(
          tmp_accepted_indices_.begin(),
          tmp_accepted_indices_.end(),
          tmp_accepted_by_lookahead_indices_.begin(),
          tmp_accepted_by_lookahead_indices_.end(),
          std::back_inserter(accepted_indices_without_lookahead)
      );
      rule_level_cache_->AddCache(
          fsm_hash.value(),
          new_state_id,
          fsm.GetNodeNum(),
          fsm.GetEdgeNum(),
          AdaptiveTokenMask(
              tokenizer_info_.GetVocabSize(),
              tokenizer_info_.GetSortedDecodedVocab(),
              accepted_indices_without_lookahead,
              tmp_uncertain_indices_
          )
      );

      if (lookahead_hash.has_value()) {
        rule_level_cache_->AddCache(
            HashCombine(fsm_hash.value(), lookahead_hash.value(), is_exact_lookahead),
            new_state_id,
            fsm.GetNodeNum(),
            fsm.GetEdgeNum(),
            return_value
        );
      }
    }
    return return_value;
  }
}

/******************* GrammarCompilerNoCache *******************/

/*!
 * \brief The base class for the grammar compiler. Handles the compilation logic without cache.
 */
class GrammarCompilerSub {
 public:
  GrammarCompilerSub(
      const TokenizerInfo& tokenizer_info,
      int max_threads,
      std::optional<RuleLevelCache> rule_level_cache
  )
      : tokenizer_info_(tokenizer_info),
        max_threads_(max_threads),
        rule_level_cache_(rule_level_cache) {}

  CompiledGrammar CompileBuiltinJSONGrammar();

  CompiledGrammar CompileJSONSchema(
      const std::string& schema,
      bool any_whitespace,
      std::optional<int> indent,
      std::optional<std::pair<std::string, std::string>> separators,
      bool strict_mode,
      std::optional<int> max_whitespace_cnt,
      bool any_order
  );

  CompiledGrammar CompileRegex(const std::string& regex);

  CompiledGrammar CompileLark(
      const std::string& lark_string, const std::vector<NamedGrammar>& named_grammars
  );

  CompiledGrammar CompileStructuralTag(const std::string& structural_tag_json);

  CompiledGrammar CompileGrammar(const Grammar& grammar);

  CompiledGrammar CompileGrammar(const std::string& ebnf_str, std::string root_rule_name);

 private:
  /*! \brief The main logic. Compile the grammar with multi-threading. */
  CompiledGrammar MultiThreadCompileGrammar(Grammar grammar);
  /*! \brief Optimization for TagDispatch.
   *  \param compiled_grammar_impl the compiled_grammar to be optimized.
   *  \param tag_dispatch_rule_id_to_second_slicing_bitset Return value. Mapping from the rule_id to
   * the definite accepted token mask.
   */
  void TagDispatchOptimization(
      std::shared_ptr<CompiledGrammar::Impl> compiled_grammar_impl,
      std::unordered_map<int32_t, DynamicBitset>* tag_dispatch_rule_id_to_second_slicing_bitset
  );

  /*! \brief The vocabulary associated with this storage class. */
  const TokenizerInfo tokenizer_info_;
  /*! \brief The maximum number of threads to use. */
  const int max_threads_;

  /*! \brief The manager of the rule level cache.*/
  std::optional<RuleLevelCache> rule_level_cache_;
};

/*!
 * \brief Check that every token id written in Token(...), ExcludeToken(...) and TokenTagDispatch
 * exists in the vocabulary. They are used as indices into per-token arrays during compilation,
 * which runs on worker threads where an error cannot be reported.
 */
static void CheckTokenIdsInVocab(const Grammar& grammar, int vocab_size) {
  const CompactFSM& fsm = grammar->complete_fsm;
  for (int state = 0; state < fsm.NumStates(); ++state) {
    for (const auto& edge : fsm.GetEdges(state)) {
      if (!edge.IsToken() && !edge.IsExcludeToken()) {
        continue;
      }
      // Token and exclude-token edges share the [count, token_id_0, ...] aux layout.
      const auto info = fsm.GetTokenEdgeInfo(edge.GetAuxIndex());
      for (int32_t i = 0; i < info.Count(); ++i) {
        XGRAMMAR_CHECK(info.TokenIds()[i] >= 0 && info.TokenIds()[i] < vocab_size)
            << "Token id " << info.TokenIds()[i]
            << " in the grammar is out of the vocabulary range [0, " << vocab_size << ")";
      }
    }
  }
}

CompiledGrammar GrammarCompilerSub::MultiThreadCompileGrammar(Grammar grammar_unoptimized) {
  auto compiled_grammar_impl = std::make_shared<CompiledGrammar::Impl>();
  compiled_grammar_impl->grammar = GrammarOptimizer::Apply(grammar_unoptimized);
  compiled_grammar_impl->tokenizer_info = tokenizer_info_;
  CheckTokenIdsInVocab(compiled_grammar_impl->grammar, tokenizer_info_.GetVocabSize());
  if (tokenizer_info_.GetVocabSize() == 0) {
    return CompiledGrammar(compiled_grammar_impl);
  }
  std::unordered_map<int32_t, DynamicBitset> tag_dispatch_rule_id_to_second_slicing_bitset;
  TagDispatchOptimization(compiled_grammar_impl, &tag_dispatch_rule_id_to_second_slicing_bitset);

  // If the compiler is cache-enabled, then we hash the grammars for crossing-grammar caching.
  if (rule_level_cache_.has_value()) {
    GrammarFSMHasher().Apply(&compiled_grammar_impl->grammar);
  }
  // Step 3. Compute the adaptive token mask cache
  // The token mask cache is computed for these positions in the grammar:
  // 1. All character class or character class star (with last_utf8_bytes=0, 1, 2, 3)
  // 2. All byte strings (with element_in_string=0, 1, 2, ...)
  // since other positions will be expanded to the above positions

  // TODO(Charlie): Figure out how to support ThreadPool and std::mutex in WebAssembly.
  // Only declare ThreadPool and mutex if max_threads > 1, so when max_threads = 1, we do
  // not need ThreadPool or std::mutex, which throws error in runtime in WebAssembly.
  std::optional<ThreadPool> thread_pool;
  std::optional<std::mutex> adaptive_token_mask_cache_mutex;
  if (max_threads_ > 1) {
    thread_pool.emplace(max_threads_);
    adaptive_token_mask_cache_mutex.emplace();
  }

  auto add_adaptive_token_mask = [&](const ParserState& state, bool is_root_rule) {
    auto grammar_matcher = GrammarMatcherForTokenMaskCache(
        compiled_grammar_impl->grammar,
        state,
        tag_dispatch_rule_id_to_second_slicing_bitset,
        tokenizer_info_,
        rule_level_cache_
    );
    auto cur_adaptive_token_mask_cache = grammar_matcher.GetAdaptiveTokenMask(is_root_rule);
    if (max_threads_ > 1) {
      std::lock_guard<std::mutex> lock(adaptive_token_mask_cache_mutex.value());
      compiled_grammar_impl->adaptive_token_mask_cache[state] = cur_adaptive_token_mask_cache;
    } else {
      compiled_grammar_impl->adaptive_token_mask_cache[state] = cur_adaptive_token_mask_cache;
    }
  };

  auto add_task_adaptive_token_mask = [&](const ParserState& state, bool is_root_rule) {
    // Execute depending on whether we use thread_pool
    if (max_threads_ > 1) {
      thread_pool->Execute([add_adaptive_token_mask, state, is_root_rule]() {
        add_adaptive_token_mask(state, is_root_rule);
      });
    } else {
      add_adaptive_token_mask(state, is_root_rule);
    }
  };

  auto root_rule_id = compiled_grammar_impl->grammar->GetRootRuleId();

  for (int32_t rule_id = 0; rule_id < static_cast<int>(compiled_grammar_impl->grammar->NumRules());
       ++rule_id) {
    auto rule = compiled_grammar_impl->grammar->GetRule(rule_id);
    const auto& rule_fsm = compiled_grammar_impl->grammar->per_rule_fsms[rule_id];
    XGRAMMAR_DCHECK(rule_fsm.has_value());
    auto cur_stack_element =
        ParserState(rule_id, rule.body_expr_id, 0, ParserState::kNoPrevInputPos, 0);
    std::unordered_set<int> reachable_states;
    rule_fsm->GetFsm().GetReachableStates(&reachable_states);
    for (int i : reachable_states) {
      cur_stack_element.element_id = i;
      if (!rule_fsm->GetFsm().IsScanableState(i)) {
        continue;
      }
      add_task_adaptive_token_mask(cur_stack_element, rule_id == root_rule_id);
    }
  }

  if (max_threads_ > 1) {
    thread_pool->Join();
  }

  return CompiledGrammar(compiled_grammar_impl);
}

CompiledGrammar GrammarCompilerSub::CompileBuiltinJSONGrammar() {
  return MultiThreadCompileGrammar(Grammar::BuiltinJSONGrammar());
}

CompiledGrammar GrammarCompilerSub::CompileJSONSchema(
    const std::string& schema,
    bool any_whitespace,
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool strict_mode,
    std::optional<int> max_whitespace_cnt,
    bool any_order
) {
  return MultiThreadCompileGrammar(Grammar::FromJSONSchema(
      schema,
      any_whitespace,
      indent,
      separators,
      strict_mode,
      max_whitespace_cnt,
      /*print_converted_ebnf=*/false,
      any_order
  ));
}

CompiledGrammar GrammarCompilerSub::CompileStructuralTag(const std::string& structural_tag_json) {
  auto result = Grammar::FromStructuralTag(structural_tag_json, tokenizer_info_);
  XGRAMMAR_CHECK(std::holds_alternative<Grammar>(result))
      << GetMessageFromVariantError(std::get<1>(result));
  return MultiThreadCompileGrammar(std::get<0>(result));
}

CompiledGrammar GrammarCompilerSub::CompileRegex(const std::string& regex) {
  return MultiThreadCompileGrammar(Grammar::FromRegex(regex));
}

CompiledGrammar GrammarCompilerSub::CompileLark(
    const std::string& lark_string, const std::vector<NamedGrammar>& named_grammars
) {
  return MultiThreadCompileGrammar(Grammar::FromLark(lark_string, tokenizer_info_, named_grammars));
}

CompiledGrammar GrammarCompilerSub::CompileGrammar(const Grammar& grammar) {
  return MultiThreadCompileGrammar(grammar);
}

CompiledGrammar GrammarCompilerSub::CompileGrammar(
    const std::string& ebnf_str, std::string root_rule_name
) {
  return MultiThreadCompileGrammar(Grammar::FromEBNF(ebnf_str, root_rule_name));
}

void GrammarCompilerSub::TagDispatchOptimization(
    std::shared_ptr<CompiledGrammar::Impl> compiled_grammar_impl,
    std::unordered_map<int32_t, DynamicBitset>* tag_dispatch_rule_id_to_second_slicing_bitset
) {
  using GrammarExprType = Grammar::Impl::GrammarExprType;
  tag_dispatch_rule_id_to_second_slicing_bitset->clear();

  // Optimization for TagDispatch: Precompute the definitely accepted tokens.
  for (int i = 0; i < compiled_grammar_impl->grammar->NumRules(); i++) {
    const auto& rule = compiled_grammar_impl->grammar->GetRule(i);
    const auto& rule_body = compiled_grammar_impl->grammar->GetGrammarExpr(rule.body_expr_id);
    if (rule_body.type != GrammarExprType::kTagDispatch) {
      continue;
    }
    XGRAMMAR_DCHECK(rule_body.type == GrammarExprType::kTagDispatch);
    Grammar::Impl::TagDispatch tag_dispatch =
        compiled_grammar_impl->GetGrammar()->GetTagDispatch(rule.body_expr_id);
    const auto& sorted_decoded_vocab = tokenizer_info_.GetSortedDecodedVocab();
    DynamicBitset definite_accepted_tokens_since_second_char(sorted_decoded_vocab.size());
    for (int j = 0; j < static_cast<int32_t>(sorted_decoded_vocab.size()); j++) {
      bool definite_accept_since_second_char = true;
      const auto& token = sorted_decoded_vocab[j].second;
      if (token.empty()) {
        definite_accepted_tokens_since_second_char.Set(j);
        continue;
      }

      // Check if the token contains any string trigger or exclude string after first char.
      for (const auto& [trigger, rule_id] : tag_dispatch.tag_rule_pairs) {
        if (token.find(trigger, 1) != std::string::npos) {
          definite_accept_since_second_char = false;
          break;
        }
      }
      if (definite_accept_since_second_char) {
        for (const auto& excl : tag_dispatch.excludes) {
          if (token.find(excl, 1) != std::string::npos) {
            definite_accept_since_second_char = false;
            break;
          }
        }
      }

      if (definite_accept_since_second_char) {
        definite_accepted_tokens_since_second_char.Set(j);
      }
    }
    (*tag_dispatch_rule_id_to_second_slicing_bitset)[i] =
        definite_accepted_tokens_since_second_char;
  }
}

/******************* GrammarCompiler::Impl *******************/

/*!
 * \brief The keys for the cache. This is defined here instead of inside the GrammarCompiler::Impl
 * class due C++ template specialization and hash specialization rules.
 */
class GrammarCompilerCacheKeys {
 public:
  struct SchemaKey {
    std::string schema;
    bool any_whitespace;
    std::optional<int> indent;
    std::optional<std::pair<std::string, std::string>> separators;
    bool strict_mode;
    std::optional<int> max_whitespace_cnt;
    bool any_order;

    XGRAMMAR_EQUAL_BY_MEMBERS(
        SchemaKey,
        &SchemaKey::schema,
        &SchemaKey::any_whitespace,
        &SchemaKey::indent,
        &SchemaKey::separators,
        &SchemaKey::strict_mode,
        &SchemaKey::max_whitespace_cnt,
        &SchemaKey::any_order
    );
  };

  struct StructuralTagKey {
    std::string structural_tag_json;

    XGRAMMAR_EQUAL_BY_MEMBERS(StructuralTagKey, &StructuralTagKey::structural_tag_json);
  };

  struct GrammarKey {
    std::string ebnf_str;
    std::string root_rule_name;

    XGRAMMAR_EQUAL_BY_MEMBERS(GrammarKey, &GrammarKey::ebnf_str, &GrammarKey::root_rule_name);
  };

  struct RegexKey {
    std::string regex;

    XGRAMMAR_EQUAL_BY_MEMBERS(RegexKey, &RegexKey::regex);
  };

  struct LarkNamedGrammarKey {
    std::string name;
    bool is_lark_source;
    std::string source_or_ebnf;
    std::string root_rule_name;
    std::variant<Grammar, std::string> value;

    bool operator==(const LarkNamedGrammarKey& other) const {
      return std::tie(name, is_lark_source, source_or_ebnf, root_rule_name) ==
             std::tie(other.name, other.is_lark_source, other.source_or_ebnf, other.root_rule_name);
    }
  };

  struct LarkKey {
    std::string lark_string;
    std::vector<LarkNamedGrammarKey> named_grammars;

    XGRAMMAR_EQUAL_BY_MEMBERS(LarkKey, &LarkKey::lark_string, &LarkKey::named_grammars);
  };

  struct BuiltinJSONGrammarKey {
    XGRAMMAR_EQUAL_BY_MEMBERS_EMPTY(BuiltinJSONGrammarKey);
  };

  using UnionKey = std::
      variant<SchemaKey, StructuralTagKey, GrammarKey, RegexKey, LarkKey, BuiltinJSONGrammarKey>;
};

}  // namespace xgrammar

XGRAMMAR_HASH_BY_MEMBERS(
    xgrammar::GrammarCompilerCacheKeys::SchemaKey,
    &xgrammar::GrammarCompilerCacheKeys::SchemaKey::schema,
    &xgrammar::GrammarCompilerCacheKeys::SchemaKey::any_whitespace,
    &xgrammar::GrammarCompilerCacheKeys::SchemaKey::indent,
    &xgrammar::GrammarCompilerCacheKeys::SchemaKey::separators,
    &xgrammar::GrammarCompilerCacheKeys::SchemaKey::strict_mode,
    &xgrammar::GrammarCompilerCacheKeys::SchemaKey::max_whitespace_cnt,
    &xgrammar::GrammarCompilerCacheKeys::SchemaKey::any_order
);

XGRAMMAR_HASH_BY_MEMBERS(
    xgrammar::GrammarCompilerCacheKeys::StructuralTagKey,
    &xgrammar::GrammarCompilerCacheKeys::StructuralTagKey::structural_tag_json
);

XGRAMMAR_HASH_BY_MEMBERS(
    xgrammar::GrammarCompilerCacheKeys::GrammarKey,
    &xgrammar::GrammarCompilerCacheKeys::GrammarKey::ebnf_str,
    &xgrammar::GrammarCompilerCacheKeys::GrammarKey::root_rule_name
);

XGRAMMAR_HASH_BY_MEMBERS(
    xgrammar::GrammarCompilerCacheKeys::RegexKey,
    &xgrammar::GrammarCompilerCacheKeys::RegexKey::regex
);

XGRAMMAR_HASH_BY_MEMBERS(
    xgrammar::GrammarCompilerCacheKeys::LarkNamedGrammarKey,
    &xgrammar::GrammarCompilerCacheKeys::LarkNamedGrammarKey::name,
    &xgrammar::GrammarCompilerCacheKeys::LarkNamedGrammarKey::is_lark_source,
    &xgrammar::GrammarCompilerCacheKeys::LarkNamedGrammarKey::source_or_ebnf,
    &xgrammar::GrammarCompilerCacheKeys::LarkNamedGrammarKey::root_rule_name
);

namespace std {
template <>
struct hash<xgrammar::GrammarCompilerCacheKeys::LarkKey> {
  std::size_t operator()(const xgrammar::GrammarCompilerCacheKeys::LarkKey& key) const noexcept {
    uint64_t seed = std::hash<std::string>{}(key.lark_string);
    for (const auto& named_grammar : key.named_grammars) {
      xgrammar::HashCombineBinary(
          seed, std::hash<xgrammar::GrammarCompilerCacheKeys::LarkNamedGrammarKey>{}(named_grammar)
      );
    }
    return seed;
  }
};
}  // namespace std

XGRAMMAR_HASH_BY_MEMBERS_EMPTY(xgrammar::GrammarCompilerCacheKeys::BuiltinJSONGrammarKey);

namespace xgrammar {

/*!
 * \brief The implementation of the grammar compiler with cache. It calls the no cache compiler
 * to compile the grammar, and implements the cache logic upon it.
 */
class GrammarCompiler::Impl {
 public:
  Impl(
      const TokenizerInfo& tokenizer_info,
      int max_threads,
      bool cache_enabled,
      int64_t max_memory_bytes
  )
      : cache_enabled_(cache_enabled),
        rule_level_cache_(
            cache_enabled
                ? std::optional<RuleLevelCache>(
                      max_memory_bytes == -1
                          ? static_cast<std::size_t>(-1)
                          : static_cast<std::size_t>(max_memory_bytes - max_memory_bytes / 3 * 2)
                  )
                : std::nullopt
        ),
        no_cache_compiler_(tokenizer_info, max_threads, rule_level_cache_),
        grammar_level_cache_(
            max_memory_bytes == -1 ? static_cast<std::size_t>(-1)
                                   : static_cast<std::size_t>(max_memory_bytes / 3 * 2),
            Computer(*this)
        ) {
    if (max_memory_bytes < -1) {
      XGRAMMAR_LOG(FATAL) << "Invalid max_memory_bytes: " << max_memory_bytes << ". "
                          << "It should be -1 (unlimited) or a non-negative integer.";
    }
  }

  CompiledGrammar CompileBuiltinJSONGrammar();

  CompiledGrammar CompileJSONSchema(
      const std::string& schema,
      bool any_whitespace,
      std::optional<int> indent,
      std::optional<std::pair<std::string, std::string>> separators,
      bool strict_mode,
      std::optional<int> max_whitespace_cnt,
      bool any_order
  );

  CompiledGrammar CompileStructuralTag(const std::string& structural_tag_json);

  CompiledGrammar CompileRegex(const std::string& regex);

  CompiledGrammar CompileLark(
      const std::string& lark_string, const std::vector<NamedGrammar>& named_grammars
  );

  CompiledGrammar CompileGrammar(const Grammar& grammar);

  CompiledGrammar CompileGrammar(const std::string& ebnf_str, std::string root_rule_name);

  void ClearCache();

  int64_t GetCacheSizeBytes() const;

  int64_t CacheLimitBytes() const;

 private:
  using SchemaKey = GrammarCompilerCacheKeys::SchemaKey;
  using StructuralTagKey = GrammarCompilerCacheKeys::StructuralTagKey;
  using GrammarKey = GrammarCompilerCacheKeys::GrammarKey;
  using RegexKey = GrammarCompilerCacheKeys::RegexKey;
  using LarkNamedGrammarKey = GrammarCompilerCacheKeys::LarkNamedGrammarKey;
  using LarkKey = GrammarCompilerCacheKeys::LarkKey;
  using BuiltinJSONGrammarKey = GrammarCompilerCacheKeys::BuiltinJSONGrammarKey;
  using UnionKey = GrammarCompilerCacheKeys::UnionKey;

  CompiledGrammar Compute(const UnionKey& key);

  struct Computer {
    Computer(Impl& compiler) : compiler(compiler) {}
    // Forward the key to GrammarCompiler::Impl::Compute(key)
    CompiledGrammar operator()(const UnionKey& key) const { return compiler.Compute(key); }
    GrammarCompiler::Impl& compiler;
  };

  struct SizeEstimator {
    std::size_t operator()(const CompiledGrammar& value) const { return value.MemorySizeBytes(); }
  };

  /*! \brief Whether the cache is enabled. */
  const bool cache_enabled_;

  /*! \brief The crossing cache manager for compiled grammars. */
  std::optional<RuleLevelCache> rule_level_cache_ = std::nullopt;

  /*! \brief The no cache compiler. */
  GrammarCompilerSub no_cache_compiler_;

  /*! \brief The cache for compiled grammars. */
  ThreadSafeLRUCache<UnionKey, CompiledGrammar, Computer, SizeEstimator> grammar_level_cache_;
};

CompiledGrammar GrammarCompiler::Impl::Compute(const UnionKey& key) {
  return std::visit(
      [this](const auto& key) -> CompiledGrammar {
        using KeyType = std::decay_t<decltype(key)>;
        if constexpr (std::is_same_v<KeyType, GrammarKey>) {
          const auto& [ebnf_str, root_rule_name] = key;
          return this->no_cache_compiler_.CompileGrammar(ebnf_str, root_rule_name);
        } else if constexpr (std::is_same_v<KeyType, SchemaKey>) {
          const auto& [schema, any_whitespace, indent, separators, strict_mode, max_whitespace_cnt, any_order] =
              key;
          return this->no_cache_compiler_.CompileJSONSchema(
              schema, any_whitespace, indent, separators, strict_mode, max_whitespace_cnt, any_order
          );
        } else if constexpr (std::is_same_v<KeyType, StructuralTagKey>) {
          const auto& [structural_tag_json] = key;
          return this->no_cache_compiler_.CompileStructuralTag(structural_tag_json);
        } else if constexpr (std::is_same_v<KeyType, RegexKey>) {
          const auto& [regex] = key;
          return this->no_cache_compiler_.CompileRegex(regex);
        } else if constexpr (std::is_same_v<KeyType, LarkKey>) {
          std::vector<NamedGrammar> named_grammars;
          named_grammars.reserve(key.named_grammars.size());
          for (const auto& named_grammar : key.named_grammars) {
            named_grammars.push_back({named_grammar.name, named_grammar.value});
          }
          return this->no_cache_compiler_.CompileLark(key.lark_string, named_grammars);
        } else if constexpr (std::is_same_v<KeyType, BuiltinJSONGrammarKey>) {
          return this->no_cache_compiler_.CompileBuiltinJSONGrammar();
        } else {
          XGRAMMAR_UNREACHABLE();
        }
      },
      key
  );
}

CompiledGrammar GrammarCompiler::Impl::CompileBuiltinJSONGrammar() {
  if (!cache_enabled_) {
    return no_cache_compiler_.CompileBuiltinJSONGrammar();
  }
  return grammar_level_cache_.Get(BuiltinJSONGrammarKey{});
}

CompiledGrammar GrammarCompiler::Impl::CompileJSONSchema(
    const std::string& schema,
    bool any_whitespace,
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool strict_mode,
    std::optional<int> max_whitespace_cnt,
    bool any_order
) {
  if (!cache_enabled_) {
    return no_cache_compiler_.CompileJSONSchema(
        schema, any_whitespace, indent, separators, strict_mode, max_whitespace_cnt, any_order
    );
  }
  return grammar_level_cache_.Get(SchemaKey{
      schema, any_whitespace, indent, separators, strict_mode, max_whitespace_cnt, any_order
  });
}

CompiledGrammar GrammarCompiler::Impl::CompileStructuralTag(const std::string& structural_tag_json
) {
  if (!cache_enabled_) {
    return no_cache_compiler_.CompileStructuralTag(structural_tag_json);
  }
  return grammar_level_cache_.Get(StructuralTagKey{structural_tag_json});
}

CompiledGrammar GrammarCompiler::Impl::CompileRegex(const std::string& regex) {
  if (!cache_enabled_) {
    return no_cache_compiler_.CompileRegex(regex);
  }
  return grammar_level_cache_.Get(RegexKey{regex});
}

CompiledGrammar GrammarCompiler::Impl::CompileLark(
    const std::string& lark_string, const std::vector<NamedGrammar>& named_grammars
) {
  if (!cache_enabled_) {
    return no_cache_compiler_.CompileLark(lark_string, named_grammars);
  }

  std::vector<LarkNamedGrammarKey> named_grammar_keys;
  named_grammar_keys.reserve(named_grammars.size());
  for (const auto& named_grammar : named_grammars) {
    if (std::holds_alternative<std::string>(named_grammar.grammar)) {
      const auto& source = std::get<std::string>(named_grammar.grammar);
      named_grammar_keys.push_back(
          {named_grammar.name, /*is_lark_source=*/true, source, "", named_grammar.grammar}
      );
    } else {
      const auto& grammar = std::get<Grammar>(named_grammar.grammar);
      named_grammar_keys.push_back(
          {named_grammar.name,
           /*is_lark_source=*/false,
           grammar.ToString(),
           grammar->GetRootRule().name,
           named_grammar.grammar}
      );
    }
  }
  std::sort(
      named_grammar_keys.begin(),
      named_grammar_keys.end(),
      [](const LarkNamedGrammarKey& lhs, const LarkNamedGrammarKey& rhs) {
        return std::tie(lhs.name, lhs.is_lark_source, lhs.source_or_ebnf, lhs.root_rule_name) <
               std::tie(rhs.name, rhs.is_lark_source, rhs.source_or_ebnf, rhs.root_rule_name);
      }
  );
  return grammar_level_cache_.Get(LarkKey{lark_string, std::move(named_grammar_keys)});
}

CompiledGrammar GrammarCompiler::Impl::CompileGrammar(const Grammar& grammar) {
  if (!cache_enabled_) {
    return no_cache_compiler_.CompileGrammar(grammar);
  }
  return grammar_level_cache_.Get(GrammarKey{grammar.ToString(), grammar->GetRootRule().name});
}

CompiledGrammar GrammarCompiler::Impl::CompileGrammar(
    const std::string& ebnf_str, std::string root_rule_name
) {
  if (!cache_enabled_) {
    return no_cache_compiler_.CompileGrammar(ebnf_str, root_rule_name);
  }
  return grammar_level_cache_.Get(GrammarKey{ebnf_str, root_rule_name});
}

void GrammarCompiler::Impl::ClearCache() {
  grammar_level_cache_.Clear();
  if (rule_level_cache_.has_value()) {
    rule_level_cache_->ClearCache();
  }
}

int64_t GrammarCompiler::Impl::GetCacheSizeBytes() const {
  return static_cast<int64_t>(grammar_level_cache_.MemorySize()) +
         static_cast<int64_t>(MemorySize(rule_level_cache_));
}

int64_t GrammarCompiler::Impl::CacheLimitBytes() const {
  const auto size = grammar_level_cache_.MaxMemorySize();
  if (size == grammar_level_cache_.kUnlimitedSize) return -1;
  return static_cast<int64_t>(size) + (rule_level_cache_.has_value()
                                           ? static_cast<int64_t>(rule_level_cache_->GetMaxSize())
                                           : 0);
}

/******************* GrammarCompiler *******************/

GrammarCompiler::GrammarCompiler(
    const TokenizerInfo& tokenizer_info,
    int max_threads,
    bool cache_enabled,
    int64_t max_memory_bytes
)
    : pimpl_(std::make_shared<Impl>(tokenizer_info, max_threads, cache_enabled, max_memory_bytes)) {
}

CompiledGrammar GrammarCompiler::CompileJSONSchema(
    const std::string& schema,
    bool any_whitespace,
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool strict_mode,
    std::optional<int> max_whitespace_cnt,
    bool any_order
) {
  return pimpl_->CompileJSONSchema(
      schema, any_whitespace, indent, separators, strict_mode, max_whitespace_cnt, any_order
  );
}

CompiledGrammar GrammarCompiler::CompileBuiltinJSONGrammar() {
  return pimpl_->CompileBuiltinJSONGrammar();
}

CompiledGrammar GrammarCompiler::CompileStructuralTag(const std::string& structural_tag_json) {
  return pimpl_->CompileStructuralTag(structural_tag_json);
}

CompiledGrammar GrammarCompiler::CompileRegex(const std::string& regex) {
  return pimpl_->CompileRegex(regex);
}

CompiledGrammar GrammarCompiler::CompileLark(
    const std::string& lark_string, const std::vector<NamedGrammar>& named_grammars
) {
  return pimpl_->CompileLark(lark_string, named_grammars);
}

CompiledGrammar GrammarCompiler::CompileGrammar(const Grammar& grammar) {
  return pimpl_->CompileGrammar(grammar);
}

CompiledGrammar GrammarCompiler::CompileGrammar(
    const std::string& ebnf_str, const std::string& root_rule_name
) {
  return pimpl_->CompileGrammar(ebnf_str, root_rule_name);
}

void GrammarCompiler::ClearCache() { pimpl_->ClearCache(); }

int64_t GrammarCompiler::GetCacheSizeBytes() const { return pimpl_->GetCacheSizeBytes(); }

int64_t GrammarCompiler::CacheLimitBytes() const { return pimpl_->CacheLimitBytes(); }

}  // namespace xgrammar
