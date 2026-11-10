/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/earley_parser.cc
 */

#include "earley_parser.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <utility>
#include <vector>

#include "fsm.h"
#include "grammar_impl.h"
#include "support/encoding.h"
#include "support/logging.h"
#include "xgrammar/grammar.h"

namespace xgrammar {

using GrammarExprType = Grammar::Impl::GrammarExprType;

using GrammarExpr = Grammar::Impl::GrammarExpr;

bool EarleyParser::IsCompleted() const { return is_completed_.back(); }

bool EarleyParser::CompletionConsumedMarker(const ParserState& state) const {
  const auto& body = grammar_->GetGrammarExpr(grammar_->GetRule(state.rule_id).body_expr_id);
  if (body.type != GrammarExprType::kTagDispatch) {
    return true;
  }
  XGRAMMAR_DCHECK(grammar_->per_rule_fsms[state.rule_id].has_value());
  const auto& fsm = grammar_->per_rule_fsms[state.rule_id]->GetFsm().GetFsm();
  return fsm.GetEdges(state.element_id).size() == 0;
}

std::vector<CaptureOccurrence> EarleyParser::CollectStopCaptureTargets(const ParserState& state
) const {
  // Follow only the parent links of this concrete rule occurrence. A byte-overlap test at
  // materialization time cannot distinguish an actual captured ancestor from an unrelated
  // Earley branch that happens to cover the same input.
  std::vector<CaptureOccurrence> targets;
  std::vector<CaptureOccurrence> pending{{state.rule_id, state.rule_start_pos}};
  std::unordered_set<int64_t> visited;
  while (!pending.empty()) {
    CaptureOccurrence occurrence = pending.back();
    pending.pop_back();
    int64_t occurrence_key = (static_cast<int64_t>(occurrence.rule_id) << 32) |
                             static_cast<uint32_t>(occurrence.start_pos);
    if (!visited.insert(occurrence_key).second) {
      continue;
    }
    if (RuleHasCapture(occurrence.rule_id)) {
      targets.push_back(occurrence);
    }
    if (occurrence.start_pos == ParserState::kNoPrevInputPos) {
      continue;
    }
    const auto& parent_states = rule_id_to_completable_states_[occurrence.start_pos];
    for (const auto& [ref_rule_id, parent_state] : parent_states) {
      if (ref_rule_id != occurrence.rule_id || parent_state.rule_id < 0) {
        continue;
      }
      pending.push_back({parent_state.rule_id, parent_state.rule_start_pos});
    }
  }
  return targets;
}

void EarleyParser::RecordCaptureEvent(const ParserState& state, bool marker_present) {
  const auto* suffix_stop_info = grammar_->GetSuffixStopInfo(state.rule_id);
  bool marker_consumed =
      marker_present && suffix_stop_info != nullptr &&
      (suffix_stop_info->hidden_suffix_bytes > 0 || suffix_stop_info->hidden_stop_bytes > 0) &&
      CompletionConsumedMarker(state);
  const int32_t hidden_suffix_bytes = marker_consumed ? suffix_stop_info->hidden_suffix_bytes : 0;
  const int32_t hidden_stop_bytes = marker_consumed ? suffix_stop_info->hidden_stop_bytes : 0;

  int32_t event_start_pos = state.rule_start_pos;
  if (marker_consumed && suffix_stop_info->body_rule_id == state.rule_id) {
    // A self-referencing body helper marks the zero-width event inserted immediately after a
    // dynamic string trigger. Its capture span is the fixed-length marker that precedes it.
    XGRAMMAR_DCHECK(event_start_pos != ParserState::kNoPrevInputPos);
    event_start_pos -= std::max(hidden_suffix_bytes, hidden_stop_bytes);
    XGRAMMAR_DCHECK(event_start_pos >= 0);
  }

  std::vector<CaptureOccurrence> stop_capture_targets =
      hidden_stop_bytes > 0 ? CollectStopCaptureTargets(state) : std::vector<CaptureOccurrence>{};

  capture_event_history_.PushBackInLatestRow(
      {state.rule_id,
       event_start_pos,
       state.rule_start_pos,
       hidden_suffix_bytes,
       hidden_stop_bytes,
       std::move(stop_capture_targets)}
  );
}

int32_t EarleyParser::ResolveActiveTemperatureRule(int32_t rule_id, int32_t inherited_rule_id)
    const {
  return grammar_->GetRule(rule_id).temperature.has_value() ? rule_id : inherited_rule_id;
}

void EarleyParser::PopLastStates(int32_t cnt) {
  stop_token_is_accepted_ = false;
  if (cnt >= static_cast<int32_t>(rule_id_to_completable_states_.size())) {
    XGRAMMAR_LOG(FATAL) << "The number of states to be popped is larger than the size of states.";
  }
  rule_id_to_completable_states_.PopBack(cnt);
  is_completed_.erase(is_completed_.end() - cnt, is_completed_.end());
  scanable_state_history_.PopBack(cnt);
  if (capture_tracking_) {
    capture_event_history_.PopBack(cnt);
  }
  if (has_char_budget_rules_) {
    char_count_history_.erase(char_count_history_.end() - cnt, char_count_history_.end());
    char_budget_entry_history_.erase(
        char_budget_entry_history_.end() - cnt, char_budget_entry_history_.end()
    );
  }
}

void EarleyParser::Complete(const ParserState& state, bool debug_print, bool marker_present) {
  // Record capture and hidden-span events. This is only enabled during definitive advances;
  // speculative completions (mask computation, lookahead) never record events.
  if (capture_recording_ && RuleNeedsCaptureEvent(state.rule_id)) {
    RecordCaptureEvent(state, marker_present);
  }
  if (state.rule_id != -1 && grammar_->GetRule(state.rule_id).is_lazy) {
    tmp_completed_lazy_occurrences_.emplace_back(state.rule_id, state.rule_start_pos);
  }
  // Check if a rule is completed.
  if (state.rule_start_pos == ParserState::kNoPrevInputPos) {
    // assert: if a root rule can achieve here, then it must be completed.
    if (debug_print) {
      XGRAMMAR_LOG(INFO) << "The root rule is completed.";
    }
    tmp_accept_stop_token_ = true;
    return;
  }
  if (debug_print) {
    XGRAMMAR_LOG(INFO) << "The rule " << state.rule_id << ": "
                       << grammar_->GetRule(state.rule_id).name
                       << " is completed, trying to complete its parent states.";
  }

  // Check all the possible parent states.
  const auto& parent_states_map = rule_id_to_completable_states_[state.rule_start_pos];
  for (const auto& [ref_id, parent_state] : parent_states_map) {
    if (ref_id != state.rule_id) {
      continue;
    }
    XGRAMMAR_DCHECK(
        parent_state.rule_id == -1 || grammar_->per_rule_fsms[parent_state.rule_id].has_value()
    );
    if (parent_state.rule_id == -1) {
      const auto& parent_expr = grammar_->GetGrammarExpr(parent_state.sequence_id);
      const auto& element_expr = grammar_->GetGrammarExpr(parent_expr[parent_state.element_id]);
      // The new rule is not referenced by a fsm.
      XGRAMMAR_DCHECK(
          element_expr.type == GrammarExprType::kRuleRef ||
          element_expr.type == GrammarExprType::kRepeat
      );
      if (element_expr.type == GrammarExprType::kRuleRef) {
        Enqueue(ParserState{
            parent_state.rule_id,
            parent_state.sequence_id,
            parent_state.element_id + 1,
            parent_state.rule_start_pos,
            parent_state.budget_deadline,
            0,
            0,
            0,
            parent_state.active_temperature_rule_id,
            parent_state.char_budget_deadline
        });
        continue;
      }
      XGRAMMAR_DCHECK(element_expr.type == GrammarExprType::kRepeat);
      // The parent state is a repeat, we need to increase the repeat count.
      auto new_state = parent_state;
      const int32_t& min_repeat_count = element_expr[1];
      const int32_t& max_repeat_count = element_expr[2];
      new_state.repeat_count++;
      // The repeat rule can be completed, and we advance the state. Don't forget to
      // reset the repeat count.
      if (new_state.repeat_count >= min_repeat_count) {
        Enqueue(ParserState{
            parent_state.rule_id,
            parent_state.sequence_id,
            parent_state.element_id + 1,
            parent_state.rule_start_pos,
            parent_state.budget_deadline,
            0,
            0,
            0,
            parent_state.active_temperature_rule_id,
            parent_state.char_budget_deadline
        });
      }
      // If the repeat count is less than the max repeat count, we can continue to
      // visit the repeat state for another round.
      if (new_state.repeat_count < max_repeat_count) {
        Enqueue(new_state);
      }
      continue;
    }
    // If the rule is referenced by a fsm, we need to advance the fsm.
    XGRAMMAR_DCHECK(grammar_->per_rule_fsms[parent_state.rule_id].has_value());

    // Check if the parent_state sits on a kRepeatRef edge
    bool handled_as_repeat = false;
    const auto& parent_fsm = grammar_->per_rule_fsms[parent_state.rule_id].value();
    for (const auto& edge : parent_fsm.GetFsm().GetFsm().GetEdges(parent_state.element_id)) {
      // Because of invariance, a state with a kRepeatRef edge has exactly one outgoing edge.
      if (!edge.IsRepeatRef()) continue;
      auto info = grammar_->complete_fsm.GetRepeatEdgeInfo(edge.GetAuxIndex());
      if (info.RuleId() != ref_id) continue;
      handled_as_repeat = true;
      int32_t new_count = parent_state.repeat_count + 1;
      if (new_count >= info.Lower()) {
        Enqueue(ParserState{
            parent_state.rule_id,
            parent_state.sequence_id,
            edge.target,
            parent_state.rule_start_pos,
            parent_state.budget_deadline,
            0,
            0,
            0,
            parent_state.active_temperature_rule_id,
            parent_state.char_budget_deadline
        });
      }
      if (new_count < info.Upper()) {
        Enqueue(ParserState{
            parent_state.rule_id,
            parent_state.sequence_id,
            parent_state.element_id,
            parent_state.rule_start_pos,
            parent_state.budget_deadline,
            0,
            new_count,
            0,
            parent_state.active_temperature_rule_id,
            parent_state.char_budget_deadline
        });
      }
      break;
    }
    if (!handled_as_repeat) {
      Enqueue(parent_state);
    }
  }
}

std::pair</* scanable */ bool, /* completable */ bool> EarleyParser::Predict(
    const ParserState& state, bool debug_print
) {
  // Check if the rule has a corresponding FSM.
  if (state.rule_id != -1) {
    XGRAMMAR_DCHECK(grammar_->per_rule_fsms[state.rule_id].has_value());
    const uint8_t flags = GetFsmStateFlags(state.rule_id, state.element_id);
    if (flags & kFsmStateNonTerminal) {
      ExpandNextRuleRefElementOnFSM(state, debug_print);
    }
    return std::make_pair(flags & kFsmStateScanable, flags & kFsmStateEnd);
  }
  const GrammarExpr& grammar_expr = grammar_->GetGrammarExpr(state.sequence_id);
  XGRAMMAR_DCHECK(
      grammar_expr.type == GrammarExprType::kSequence ||
      grammar_expr.type == GrammarExprType::kEmptyStr
  );
  if (state.element_id == grammar_expr.size()) {
    // The rule is completed.
    return std::make_pair(false, true);
  }
  const auto& element_expr = grammar_->GetGrammarExpr(grammar_expr[state.element_id]);
  switch (element_expr.type) {
    case GrammarExprType::kRuleRef: {
      ExpandNextRuleRefElement(state, grammar_expr, &element_expr, debug_print);
      return std::make_pair(false, false);
    }
    case GrammarExprType::kCharacterClassStar: {
      if (state.sub_element_id == 0) {
        Enqueue(ParserState{
            state.rule_id,
            state.sequence_id,
            state.element_id + 1,
            state.rule_start_pos,
            state.budget_deadline,
            0,
            0,
            0,
            state.active_temperature_rule_id,
            state.char_budget_deadline
        });
      }
      return std::make_pair(true, false);
    }
    case GrammarExprType::kRepeat: {
      const int32_t& min_repeat_count = element_expr[1];
      const int32_t& max_repeat_count = element_expr[2];
      // If the current repeat count is less than the max repeat count,
      // we can expand the next rule reference element.
      XGRAMMAR_DCHECK(state.repeat_count <= max_repeat_count);
      ExpandNextRuleRefElement(state, grammar_expr, &element_expr, debug_print);
      if (state.repeat_count >= min_repeat_count) {
        Enqueue(ParserState{
            state.rule_id,
            state.sequence_id,
            state.element_id + 1,
            state.rule_start_pos,
            state.budget_deadline,
            0,
            0,
            0,
            state.active_temperature_rule_id,
            state.char_budget_deadline
        });
      }
      return std::make_pair(false, false);
    }
    case GrammarExprType::kByteString:
    case GrammarExprType::kCharacterClass: {
      return std::make_pair(true, false);  // The element is scanable, but not completable.
    }
    case GrammarExprType::kToken:
    case GrammarExprType::kExcludeToken: {
      return std::make_pair(false, false);
    }
    default: {
      XGRAMMAR_LOG(FATAL) << "The element type is not supported! The type is: "
                          << int(element_expr.type);
      XGRAMMAR_UNREACHABLE();
    }
  }
}

void EarleyParser::Scan(const ParserState& state, const uint8_t ch) {
  XGRAMMAR_DCHECK(state.rule_id == -1 || grammar_->per_rule_fsms[state.rule_id].has_value());
  if (state.rule_id == -1) {
    const auto& cur_rule = grammar_->GetGrammarExpr(state.sequence_id);
    const auto& element_expr = grammar_->GetGrammarExpr(cur_rule[state.element_id]);
    // The element is a rule reference, we do not need to scan it.
    switch (element_expr.type) {
      case (GrammarExprType::kByteString): {
        AdvanceByteString(state, ch, element_expr);
        break;
      }
      case (GrammarExprType::kCharacterClass): {
        AdvanceCharacterClass(state, ch, element_expr);
        break;
      }
      case (GrammarExprType::kCharacterClassStar): {
        AdvanceCharacterClassStar(state, ch, element_expr);
        break;
      }
      default: {
        XGRAMMAR_LOG(FATAL) << "The element type is not supported! The type is: "
                            << int(element_expr.type);
        XGRAMMAR_UNREACHABLE();
      }
    }
  } else {
    AdvanceFsm(state, ch);
  }
}

/*!
  \note The workflow of Advance is as follows:
  1. Scan all the states in the latest states. Add all the possible states
  to the next states.
  2. If the next states are empty, then the character is not accepted.
  3. If the next states are not empty, then the character is accepted. Moreover,
  we need to complete and predict the next states.

  \note Thus, when initializing the Earley parser, we need to add the initial state
  to the history_states[0], and perform prediction and completion on the initial state.
*/
bool EarleyParser::Advance(const uint8_t ch, bool debug_print) {
  // Initialize the containers.
  XGRAMMAR_DCHECK(tmp_process_state_queue_.empty())
      << "The tmp_process_state_queue_ should be empty before the scan.";
  tmp_states_visited_in_queue_.Clear();
  tmp_states_to_be_added_.clear();
  tmp_accept_stop_token_ = false;
  tmp_completed_lazy_occurrences_.clear();
  if (has_char_budget_rules_) {
    tmp_char_budget_entered_ = char_budget_entry_history_.back();
    char_count_history_.push_back(GetCurrentCharIndex() + StartsUTF8Codepoint(ch));
  }
  const auto& latest_states = scanable_state_history_[scanable_state_history_.size() - 1];
  // Scan all the scanable states.
  for (const auto& state : latest_states) {
    if (skip_expired_states_ && IsExpiredState(state)) {
      continue;
    }
    Scan(state, ch);
  }

  // Check if the character is accepted.
  if (tmp_process_state_queue_.empty() && tmp_states_to_be_added_.empty()) {
    if (has_char_budget_rules_) {
      char_count_history_.pop_back();
    }
    return false;
  }

  // execute Predict and Complete for all states in the queue until empty.
  rule_id_to_completable_states_.PushBack(std::vector<std::pair<int32_t, ParserState>>());
  if (capture_tracking_) {
    capture_event_history_.PushBack(std::vector<CaptureEvent>());
  }
  while (!tmp_process_state_queue_.empty()) {
    const auto state = std::move(tmp_process_state_queue_.front());
    tmp_process_state_queue_.pop();
    auto [scanable, completable] = Predict(state, debug_print);
    if (completable) {
      Complete(state, debug_print);
    }
    if (scanable) {
      tmp_states_to_be_added_.push_back(state);
    }
  }

  // Check if the grammar is completed, and add the scannable states to the history.
  if (!tmp_completed_lazy_occurrences_.empty()) {
    RemoveCommittedLazyStates();
  }
  is_completed_.push_back(tmp_accept_stop_token_);
  scanable_state_history_.PushBack(tmp_states_to_be_added_);
  if (has_char_budget_rules_) {
    char_budget_entry_history_.push_back(tmp_char_budget_entered_);
  }
  return true;
}

void EarleyParser::RemoveCommittedLazyStates() {
  auto is_committed = [&](const ParserState& state) {
    for (const auto& [rule_id, rule_start_pos] : tmp_completed_lazy_occurrences_) {
      if (state.rule_id == rule_id && state.rule_start_pos == rule_start_pos) {
        return true;
      }
    }
    return false;
  };
  tmp_states_to_be_added_.erase(
      std::remove_if(tmp_states_to_be_added_.begin(), tmp_states_to_be_added_.end(), is_committed),
      tmp_states_to_be_added_.end()
  );
}

EarleyParser::EarleyParser(const Grammar& grammar, std::optional<ParserState> initial_state)
    : grammar_(grammar),
      fsm_state_flags_cache_(grammar->NumRules()),
      rule_is_nullable_(grammar->NumRules(), 0) {
  if (!grammar->optimized) {
    XGRAMMAR_LOG(FATAL) << "The grammar is not optimized. Please optimize the grammar before using "
                           "the Earley parser.";
  }
  for (int32_t i = 0; i < grammar_->NumRules(); ++i) {
    has_budget_rules_ = has_budget_rules_ || grammar_->GetRule(i).max_tokens >= 0;
    has_char_budget_rules_ = has_char_budget_rules_ || grammar_->GetRule(i).max_chars >= 0;
    if (has_budget_rules_ && has_char_budget_rules_) {
      break;
    }
  }
  for (int32_t i = 0; i < grammar_->NumRules(); ++i) {
    const auto& rule = grammar_->GetRule(i);
    const auto* suffix_stop_info = grammar_->GetSuffixStopInfo(i);
    capture_tracking_ =
        capture_tracking_ || !rule.capture_name.empty() ||
        (suffix_stop_info != nullptr && !suffix_stop_info->stop_capture_name.empty());
    has_hidden_capture_rules_ =
        has_hidden_capture_rules_ ||
        (suffix_stop_info != nullptr &&
         (suffix_stop_info->hidden_suffix_bytes > 0 || suffix_stop_info->hidden_stop_bytes > 0));
    if (capture_tracking_ && has_hidden_capture_rules_) {
      break;
    }
  }
  for (int32_t rule_id : grammar_->allow_empty_rule_ids) {
    rule_is_nullable_[rule_id] = true;
  }
  PushStateAndExpand(initial_state.has_value() ? *initial_state : RootInitialState());
}

uint8_t EarleyParser::InitializeFsmStateFlags(int32_t rule_id, int32_t state_id) {
  XGRAMMAR_DCHECK(grammar_->per_rule_fsms[rule_id].has_value());
  const auto& fsm = grammar_->per_rule_fsms[rule_id]->GetFsm();
  auto& flags_cache = fsm_state_flags_cache_[rule_id];
  if (flags_cache.empty()) {
    flags_cache.resize(fsm.NumStates());
  }
  XGRAMMAR_DCHECK(state_id >= 0 && state_id < static_cast<int32_t>(flags_cache.size()));
  uint8_t& flags = flags_cache[state_id];
  if (flags & kFsmStateInitialized) {
    return flags;
  }

  flags = kFsmStateInitialized;
  if (fsm.IsEndState(state_id)) {
    flags |= kFsmStateEnd;
  }
  const auto& edges = fsm.GetFsm().GetEdges(state_id);
  if (edges.size() != 0) {
    flags |= kFsmStateHasEdges;
  }
  for (const auto& edge : edges) {
    if (edge.IsCharRange() || edge.IsToken() || edge.IsExcludeToken()) {
      flags |= kFsmStateScanable;
    } else if (edge.IsRuleRef() || edge.IsEpsilon() || edge.IsRepeatRef()) {
      flags |= kFsmStateNonTerminal;
    }
  }
  return flags;
}

ParserState EarleyParser::RootInitialState() const {
  const auto root_rule_id = grammar_->GetRootRuleId();
  XGRAMMAR_DCHECK(grammar_->per_rule_fsms[root_rule_id].has_value());
  return ParserState(
      root_rule_id,
      grammar_->GetRule(root_rule_id).body_expr_id,
      grammar_->per_rule_fsms[root_rule_id]->GetFsm().GetStart(),
      ParserState::kNoPrevInputPos,
      DeadlineForRule(root_rule_id, -1),
      0,
      0,
      0,
      ResolveActiveTemperatureRule(root_rule_id, -1),
      CharDeadlineForRule(root_rule_id, -1)
  );
}

void EarleyParser::PushStateAndExpand(const ParserState& state) {
  tmp_states_visited_in_queue_.Clear();
  tmp_accept_stop_token_ = false;
  tmp_states_to_be_added_.clear();
  tmp_completed_lazy_occurrences_.clear();
  Enqueue(state);
  rule_id_to_completable_states_.PushBack(std::vector<std::pair<int32_t, ParserState>>());
  if (capture_tracking_) {
    capture_event_history_.PushBack(std::vector<CaptureEvent>());
  }
  while (!tmp_process_state_queue_.empty()) {
    const auto state = tmp_process_state_queue_.front();
    tmp_process_state_queue_.pop();
    auto [scanable, completable] = Predict(state);
    if (completable) {
      Complete(state);
    }
    if (scanable) {
      tmp_states_to_be_added_.push_back(state);
    }
  }
  if (!tmp_completed_lazy_occurrences_.empty()) {
    RemoveCommittedLazyStates();
  }
  is_completed_.push_back(tmp_accept_stop_token_);
  scanable_state_history_.PushBack(tmp_states_to_be_added_);
  if (has_char_budget_rules_) {
    char_count_history_.push_back(GetCurrentCharIndex());
    char_budget_entry_history_.push_back(tmp_char_budget_entered_);
  }
}

void EarleyParser::Reset() {
  rule_id_to_completable_states_.PopBack(rule_id_to_completable_states_.size());
  scanable_state_history_.PopBack(scanable_state_history_.size());
  is_completed_.clear();
  stop_token_is_accepted_ = false;
  if (capture_tracking_) {
    capture_event_history_.PopBack(capture_event_history_.size());
  }
  char_count_history_.clear();
  char_budget_entry_history_.clear();
  tmp_char_budget_entered_ = false;
  capture_recording_ = false;
  XGRAMMAR_DCHECK(tmp_process_state_queue_.empty());
  PushStateAndExpand(RootInitialState());
}

void EarleyParser::ExpandNextRuleRefElement(
    const ParserState& state,
    const GrammarExpr& grammar_expr,
    const GrammarExpr* sub_grammar_expr,
    bool debug_print
) {
  // Path A. The rule has a corresponding FSM.
  XGRAMMAR_DCHECK(!(state.rule_id != -1 && grammar_->per_rule_fsms[state.rule_id].has_value()));
  XGRAMMAR_DCHECK(grammar_expr.type == GrammarExprType::kSequence);
  XGRAMMAR_DCHECK(
      sub_grammar_expr->type == GrammarExprType::kRuleRef ||
      sub_grammar_expr->type == GrammarExprType::kRepeat
  );
  auto ref_rule_id = (*sub_grammar_expr)[0];

  if (debug_print) {
    XGRAMMAR_LOG(INFO) << "The rule " << state.rule_id << ": "
                       << grammar_->GetRule(state.rule_id).name << " predict the new rule "
                       << ref_rule_id << ": " << grammar_->GetRule(ref_rule_id).name << ".";
  }

  bool right_recursion_to_root = false;
  // The right-recursion optimization elides the completion of the parent rule (and, in the
  // to-root case, corrupts the start position of the child rule), so it must be disabled when
  // either rule produces capture-history events.
  if (state.element_id != grammar_expr.size() - 1 ||
      sub_grammar_expr->type == GrammarExprType::kRepeat ||
      (state.rule_start_pos == rule_id_to_completable_states_.size() - 1) ||
      RuleNeedsCaptureEvent(state.rule_id) || RuleNeedsCaptureEvent(ref_rule_id)) {
    // It's not the right recursion, or it's the root rule.
    rule_id_to_completable_states_.PushBackInLatestRow(std::make_pair(ref_rule_id, state));
  } else {
    if (state.rule_start_pos == ParserState::kNoPrevInputPos) {
      right_recursion_to_root = true;
    } else {
      // If it's the right recursion, we need to add the ancestors of the parent state.
      const auto in_vec = [&](const ParserState& state_) {
        return std::find_if(
                   rule_id_to_completable_states_.Back().begin(),
                   rule_id_to_completable_states_.Back().end(),
                   [&](const auto& s) {
                     return StateEqualForParsing()(s.second, state_) && s.first == ref_rule_id;
                   }
               ) != rule_id_to_completable_states_.Back().end();
      };
      const auto& parent_states_map = rule_id_to_completable_states_[state.rule_start_pos];
      std::vector<std::pair<int32_t, ParserState>> to_added_states;
      for (const auto& parent_state_iter : parent_states_map) {
        if (parent_state_iter.first != state.rule_id) continue;
        const auto& parent_state = parent_state_iter.second;
        if (!in_vec(parent_state)) {
          to_added_states.push_back({ref_rule_id, parent_state});
        }
      }
      for (const auto& to_add_state : to_added_states) {
        rule_id_to_completable_states_.PushBackInLatestRow(to_add_state);
      }
    }
  }

  if (IsRuleNullable(ref_rule_id)) {
    XGRAMMAR_DCHECK(grammar_expr.type == GrammarExprType::kSequence);
    Enqueue(ParserState{
        state.rule_id,
        state.sequence_id,
        state.element_id + 1,
        state.rule_start_pos,
        state.budget_deadline,
        0,
        0,
        0,
        state.active_temperature_rule_id,
        state.char_budget_deadline
    });
  }

  // If the reference rule is not visited, we need to add it to the queue.
  const auto& ref_rule = grammar_->GetRule(ref_rule_id);
  if (ref_rule.max_chars >= 0) {
    tmp_char_budget_entered_ = true;
  }
  const auto& ref_grammar_expr_id = ref_rule.body_expr_id;

  XGRAMMAR_DCHECK(grammar_->per_rule_fsms[ref_rule_id].has_value());
  const auto& ref_fsm = grammar_->per_rule_fsms[ref_rule_id].value();
  Enqueue(ParserState{
      ref_rule_id,
      ref_grammar_expr_id,
      ref_fsm.GetFsm().GetStart(),
      right_recursion_to_root ? ParserState::kNoPrevInputPos
                              : int32_t(rule_id_to_completable_states_.size() - 1),
      DeadlineForRule(ref_rule_id, state.budget_deadline),
      0,
      0,
      0,
      ResolveActiveTemperatureRule(ref_rule_id, state.active_temperature_rule_id),
      CharDeadlineForRule(ref_rule_id, state.char_budget_deadline)
  });
}

void EarleyParser::ExpandNextRuleRefElementOnFSM(const ParserState& state, bool debug_print) {
  XGRAMMAR_DCHECK(state.rule_id != -1 && grammar_->per_rule_fsms[state.rule_id].has_value());
  const auto& fsm = grammar_->per_rule_fsms[state.rule_id].value();

  // Add the rule reference pairs, and enqueue the epsilon edges.
  for (const auto& edge : fsm.GetFsm().GetFsm().GetEdges(state.element_id)) {
    if (edge.IsEpsilon()) {
      Enqueue(ParserState{
          state.rule_id,
          state.sequence_id,
          edge.target,
          state.rule_start_pos,
          state.budget_deadline,
          0,
          0,
          0,
          state.active_temperature_rule_id,
          state.char_budget_deadline
      });
      continue;
    }

    int target;
    int ref_rule_id;
    bool is_repeat = false;
    RepeatEdgeRef repeat_info{nullptr};

    if (edge.IsRuleRef()) {
      target = edge.target;
      ref_rule_id = edge.GetRefRuleId();
    } else if (edge.IsRepeatRef()) {
      is_repeat = true;
      repeat_info = grammar_->complete_fsm.GetRepeatEdgeInfo(edge.GetAuxIndex());
      target = edge.target;
      ref_rule_id = repeat_info.RuleId();

      if (state.repeat_count >= repeat_info.Lower()) {
        Enqueue(ParserState{
            state.rule_id,
            state.sequence_id,
            target,
            state.rule_start_pos,
            state.budget_deadline,
            0,
            0,
            0,
            state.active_temperature_rule_id,
            state.char_budget_deadline
        });
      }
      if (state.repeat_count >= repeat_info.Upper()) {
        continue;
      }
    } else {
      continue;
    }
    bool right_recursion_to_root = false;
    if (debug_print) {
      XGRAMMAR_LOG(INFO) << "The rule " << state.rule_id << ": "
                         << grammar_->GetRule(state.rule_id).name << " predict the new rule "
                         << ref_rule_id << ": " << grammar_->GetRule(ref_rule_id).name << ".";
    }
    const uint8_t target_flags = GetFsmStateFlags(state.rule_id, target);
    if (!is_repeat && !(target_flags & kFsmStateHasEdges) && (target_flags & kFsmStateEnd) &&
        state.rule_start_pos != static_cast<int32_t>(rule_id_to_completable_states_.size() - 1) &&
        !RuleNeedsCaptureEvent(state.rule_id) && !RuleNeedsCaptureEvent(ref_rule_id)) {
      // It's a right recursion. We can optimize it. The optimization elides the completion of
      // the parent rule, so it is disabled when either rule produces capture-history events.
      // If it's the right recursion, we need to add the ancestors of the parent state.
      if (state.rule_start_pos == ParserState::kNoPrevInputPos) {
        // In this case, we can mark the new state as the root state to speed up.
        right_recursion_to_root = true;
      } else {
        const auto in_vec = [&](const ParserState& state_) {
          return std::find_if(
                     rule_id_to_completable_states_.Back().begin(),
                     rule_id_to_completable_states_.Back().end(),
                     [&](const auto& s) {
                       return StateEqualForParsing()(s.second, state_) && s.first == ref_rule_id;
                     }
                 ) != rule_id_to_completable_states_.Back().end();
        };
        const auto& parent_states_map = rule_id_to_completable_states_[state.rule_start_pos];
        std::vector<std::pair<int32_t, ParserState>> to_added_states;
        for (const auto& parent_state_iter : parent_states_map) {
          if (parent_state_iter.first != state.rule_id) continue;
          const auto& parent_state = parent_state_iter.second;
          if (!in_vec(parent_state)) {
            to_added_states.push_back({ref_rule_id, parent_state});
          }
        }
        for (const auto& to_add_state : to_added_states) {
          rule_id_to_completable_states_.PushBackInLatestRow(to_add_state);
        }
      }
    } else {
      if (is_repeat) {
        // For kRepeatRef: store element_id = source state, preserve repeat_count
        rule_id_to_completable_states_.PushBackInLatestRow(
            {ref_rule_id,
             ParserState{
                 state.rule_id,
                 state.sequence_id,
                 state.element_id,
                 state.rule_start_pos,
                 state.budget_deadline,
                 0,
                 state.repeat_count,
                 0,
                 state.active_temperature_rule_id,
                 state.char_budget_deadline
             }}
        );
      } else {
        // For kRuleRef: store element_id = target (post-transition state)
        rule_id_to_completable_states_.PushBackInLatestRow(
            {ref_rule_id,
             ParserState{
                 state.rule_id,
                 state.sequence_id,
                 target,
                 state.rule_start_pos,
                 state.budget_deadline,
                 0,
                 0,
                 0,
                 state.active_temperature_rule_id,
                 state.char_budget_deadline
             }}
        );
      }
    }

    // Check if the reference rule can be empty.
    if (!is_repeat && IsRuleNullable(ref_rule_id)) {
      Enqueue(ParserState{
          state.rule_id,
          state.sequence_id,
          target,
          state.rule_start_pos,
          state.budget_deadline,
          0,
          0,
          0,
          state.active_temperature_rule_id,
          state.char_budget_deadline
      });
    }

    // If the reference rule is not visited, we need to add it to the queue.
    const auto& ref_rule = grammar_->GetRule(ref_rule_id);
    if (ref_rule.max_chars >= 0) {
      tmp_char_budget_entered_ = true;
    }
    const auto& ref_grammar_expr_id = ref_rule.body_expr_id;

    XGRAMMAR_DCHECK(grammar_->per_rule_fsms[ref_rule_id].has_value());
    const auto& ref_fsm = grammar_->per_rule_fsms[ref_rule_id].value();
    Enqueue(ParserState{
        ref_rule_id,
        ref_grammar_expr_id,
        ref_fsm.GetFsm().GetStart(),
        right_recursion_to_root ? ParserState::kNoPrevInputPos
                                : int32_t(rule_id_to_completable_states_.size() - 1),
        DeadlineForRule(ref_rule_id, state.budget_deadline),
        0,
        0,
        0,
        ResolveActiveTemperatureRule(ref_rule_id, state.active_temperature_rule_id),
        CharDeadlineForRule(ref_rule_id, state.char_budget_deadline)
    });
  }
}

void EarleyParser::AdvanceByteString(
    const ParserState& state, const uint8_t ch, const GrammarExpr& sub_rule
) {
  XGRAMMAR_DCHECK(sub_rule.type == GrammarExprType::kByteString);
  XGRAMMAR_DCHECK(sub_rule.size() > state.sub_element_id);
  if (static_cast<uint8_t>(sub_rule[state.sub_element_id]) == ch) {
    auto new_state = state;
    new_state.sub_element_id++;
    if (new_state.sub_element_id == sub_rule.size()) {
      new_state.element_id++;
      new_state.sub_element_id = 0;
      Enqueue(new_state);
      // Assert: In a sequence, the bytestring can't be skipped. So the state can't be repeated.
    } else {
      tmp_states_to_be_added_.push_back(new_state);
    }
  }
  return;
}

void EarleyParser::AdvanceCharacterClass(
    const ParserState& state, const uint8_t ch, const GrammarExpr& sub_sequence
) {
  XGRAMMAR_DCHECK(sub_sequence.type == GrammarExprType::kCharacterClass)
      << "The element type is not supported!";

  bool is_negative = static_cast<bool>(sub_sequence[0]);

  // The state is matching a UTF8 character (continuation bytes).
  if (state.sub_element_id > 0) {
    if ((ch & 0xC0) == 0x80) {
      auto new_state = state;
      new_state.sub_element_id--;
      // Accumulate the codepoint from continuation byte
      new_state.partial_codepoint = (new_state.partial_codepoint << 6) | (ch & 0x3F);

      // Check if the UTF8 character is completed.
      if (new_state.sub_element_id == 0) {
        if (is_negative) {
          // For negative classes, accept if codepoint is NOT in any range
          bool matches_range = false;
          for (int i = 1; i < sub_sequence.size(); i += 2) {
            if (new_state.partial_codepoint >= sub_sequence[i] &&
                new_state.partial_codepoint <= sub_sequence[i + 1]) {
              matches_range = true;
              break;
            }
          }
          if (!matches_range) {
            new_state.element_id++;
            new_state.partial_codepoint = 0;
            Enqueue(new_state);
          }
        } else {
          // For positive classes, accept if codepoint IS in a range
          bool matches_range = false;
          for (int i = 1; i < sub_sequence.size(); i += 2) {
            if (new_state.partial_codepoint >= sub_sequence[i] &&
                new_state.partial_codepoint <= sub_sequence[i + 1]) {
              matches_range = true;
              break;
            }
          }
          if (matches_range) {
            new_state.element_id++;
            new_state.partial_codepoint = 0;
            Enqueue(new_state);
          }
        }
      } else {
        // Check if partial codepoint could still potentially match any range
        int32_t remaining_bytes = new_state.sub_element_id;
        int32_t min_codepoint = new_state.partial_codepoint << (6 * remaining_bytes);
        int32_t max_codepoint = min_codepoint | ((1 << (6 * remaining_bytes)) - 1);

        bool could_match = false;
        for (int i = 1; i < sub_sequence.size(); i += 2) {
          int32_t lower = sub_sequence[i];
          int32_t upper = sub_sequence[i + 1];
          if (max_codepoint >= lower && min_codepoint <= upper) {
            could_match = true;
            break;
          }
        }

        // For negative classes: always continue (will verify on final byte)
        // For positive classes: only continue if some range could match
        bool should_continue = is_negative ? true : could_match;
        if (should_continue) {
          tmp_states_to_be_added_.push_back(new_state);
        }
      }
    }
    return;
  }

  // Handle non-ASCII first bytes
  if (!isascii(ch)) {
    auto [accepted, num_bytes, partial] = HandleUTF8FirstByte(ch);
    if (!accepted) {
      return;
    }

    XGRAMMAR_DCHECK(num_bytes > 1);

    // Compute possible codepoint range for this first byte
    int32_t min_codepoint = partial << (6 * (num_bytes - 1));
    int32_t max_codepoint = min_codepoint | ((1 << (6 * (num_bytes - 1))) - 1);

    // Check if any stored range could potentially match
    bool could_match = false;
    for (int i = 1; i < sub_sequence.size(); i += 2) {
      int32_t lower = sub_sequence[i];
      int32_t upper = sub_sequence[i + 1];
      // Check for overlap between [min_codepoint, max_codepoint] and [lower, upper]
      if (max_codepoint >= lower && min_codepoint <= upper) {
        could_match = true;
        break;
      }
    }

    // For negative classes: accept if no range could match (will verify on final byte)
    // For positive classes: accept if some range could match (will verify on final byte)
    bool should_continue = is_negative ? true : could_match;

    if (should_continue) {
      auto new_state = state;
      new_state.sub_element_id = num_bytes - 1;
      new_state.partial_codepoint = partial;
      tmp_states_to_be_added_.push_back(new_state);
    }
    return;
  }

  // ASCII handling (unchanged)
  for (int i = 1; i < sub_sequence.size(); i += 2) {
    if (static_cast<uint8_t>(sub_sequence[i]) <= ch &&
        ch <= static_cast<uint8_t>(sub_sequence[i + 1])) {
      if (!is_negative) {
        auto new_state = state;
        new_state.element_id++;
        new_state.sub_element_id = 0;
        Enqueue(new_state);
      }
      return;
    }
  }
  if (is_negative) {
    auto new_state = state;
    new_state.element_id++;
    new_state.sub_element_id = 0;
    Enqueue(new_state);
  }
}

void EarleyParser::AdvanceCharacterClassStar(
    const ParserState& state, const uint8_t ch, const GrammarExpr& sub_sequence
) {
  XGRAMMAR_DCHECK(sub_sequence.type == GrammarExprType::kCharacterClassStar)
      << "The element type is not supported!";

  bool is_negative = static_cast<bool>(sub_sequence[0]);

  // The state is matching a UTF8 character (continuation bytes).
  if (state.sub_element_id > 0) {
    if ((ch & 0xC0) == 0x80) {
      auto new_state = state;
      new_state.sub_element_id--;
      // Accumulate the codepoint from continuation byte
      new_state.partial_codepoint = (new_state.partial_codepoint << 6) | (ch & 0x3F);

      // Check if the UTF8 character is completed.
      if (new_state.sub_element_id == 0) {
        if (is_negative) {
          // For negative classes, accept if codepoint is NOT in any range
          bool matches_range = false;
          for (int i = 1; i < sub_sequence.size(); i += 2) {
            if (new_state.partial_codepoint >= sub_sequence[i] &&
                new_state.partial_codepoint <= sub_sequence[i + 1]) {
              matches_range = true;
              break;
            }
          }
          if (!matches_range) {
            new_state.partial_codepoint = 0;
            Enqueue(new_state);
          }
        } else {
          // For positive classes, accept if codepoint IS in a range
          bool matches_range = false;
          for (int i = 1; i < sub_sequence.size(); i += 2) {
            if (new_state.partial_codepoint >= sub_sequence[i] &&
                new_state.partial_codepoint <= sub_sequence[i + 1]) {
              matches_range = true;
              break;
            }
          }
          if (matches_range) {
            new_state.partial_codepoint = 0;
            Enqueue(new_state);
          }
        }
      } else {
        // Check if partial codepoint could still potentially match any range
        int32_t remaining_bytes = new_state.sub_element_id;
        int32_t min_codepoint = new_state.partial_codepoint << (6 * remaining_bytes);
        int32_t max_codepoint = min_codepoint | ((1 << (6 * remaining_bytes)) - 1);

        bool could_match = false;
        for (int i = 1; i < sub_sequence.size(); i += 2) {
          int32_t lower = sub_sequence[i];
          int32_t upper = sub_sequence[i + 1];
          if (max_codepoint >= lower && min_codepoint <= upper) {
            could_match = true;
            break;
          }
        }

        // For negative classes: always continue (will verify on final byte)
        // For positive classes: only continue if some range could match
        bool should_continue = is_negative ? true : could_match;
        if (should_continue) {
          tmp_states_to_be_added_.push_back(new_state);
        }
      }
    }
    return;
  }

  // Handle non-ASCII first bytes
  if (!isascii(ch)) {
    auto [accepted, num_bytes, partial] = HandleUTF8FirstByte(ch);
    if (!accepted) {
      return;
    }

    XGRAMMAR_DCHECK(num_bytes > 1);

    // Compute possible codepoint range for this first byte
    int32_t min_codepoint = partial << (6 * (num_bytes - 1));
    int32_t max_codepoint = min_codepoint | ((1 << (6 * (num_bytes - 1))) - 1);

    // Check if any stored range could potentially match
    bool could_match = false;
    for (int i = 1; i < sub_sequence.size(); i += 2) {
      int32_t lower = sub_sequence[i];
      int32_t upper = sub_sequence[i + 1];
      // Check for overlap between [min_codepoint, max_codepoint] and [lower, upper]
      if (max_codepoint >= lower && min_codepoint <= upper) {
        could_match = true;
        break;
      }
    }

    // For negative classes: accept if no range could match (will verify on final byte)
    // For positive classes: accept if some range could match (will verify on final byte)
    bool should_continue = is_negative ? true : could_match;

    if (should_continue) {
      auto new_state = state;
      new_state.sub_element_id = num_bytes - 1;
      new_state.partial_codepoint = partial;
      tmp_states_to_be_added_.push_back(new_state);
    }
    return;
  }

  // ASCII handling (unchanged)
  for (int i = 1; i < sub_sequence.size(); i += 2) {
    if (static_cast<uint8_t>(sub_sequence[i]) <= ch &&
        ch <= static_cast<uint8_t>(sub_sequence[i + 1])) {
      if (!is_negative) {
        Enqueue(state);
      }
      return;
    }
  }
  if (is_negative) {
    Enqueue(state);
  }
}

void EarleyParser::AdvanceFsm(const ParserState& state, const uint8_t ch) {
  XGRAMMAR_DCHECK(state.rule_id != -1 && grammar_->per_rule_fsms[state.rule_id].has_value());
  const auto& current_fsm = grammar_->per_rule_fsms[state.rule_id].value();
  for (const auto& edge : current_fsm.GetFsm().GetFsm().GetEdges(state.element_id)) {
    if ((!edge.IsCharRange()) || ch < edge.min || ch > edge.max) {
      continue;
    }
    auto new_state = state;
    new_state.element_id = edge.target;
    const uint8_t flags = GetFsmStateFlags(state.rule_id, edge.target);
    if (!(flags & kFsmStateNonTerminal) && !(flags & kFsmStateEnd) && (flags & kFsmStateScanable)) {
      EnqueueWithoutProcessing(std::move(new_state));
    } else {
      Enqueue(std::move(new_state));
    }
  }
}

void EarleyParser::ScanAtomicToken(const ParserState& state, int32_t token_id) {
  if (state.rule_id == -1) return;
  XGRAMMAR_DCHECK(grammar_->per_rule_fsms[state.rule_id].has_value());
  const auto& current_fsm = grammar_->per_rule_fsms[state.rule_id].value();
  for (const auto& edge : current_fsm.GetFsm().GetFsm().GetEdges(state.element_id)) {
    bool matched = false;
    if (edge.IsToken()) {
      auto info = current_fsm.GetFsm().GetFsm().GetTokenEdgeInfo(edge.GetAuxIndex());
      matched = info.Contains(token_id);
    } else if (edge.IsExcludeToken()) {
      auto info = current_fsm.GetFsm().GetFsm().GetExcludeTokenEdgeInfo(edge.GetAuxIndex());
      matched = info.Accepts(token_id);
    }
    if (!matched) continue;
    auto new_state = state;
    new_state.element_id = edge.target;
    const uint8_t flags = GetFsmStateFlags(state.rule_id, edge.target);
    if (!(flags & kFsmStateNonTerminal) && !(flags & kFsmStateEnd) && (flags & kFsmStateScanable)) {
      EnqueueWithoutProcessing(std::move(new_state));
    } else {
      Enqueue(std::move(new_state));
    }
  }
}

bool EarleyParser::AdvanceAtomicToken(
    int32_t token_id, bool debug_print, int32_t token_char_count
) {
  XGRAMMAR_DCHECK(tmp_process_state_queue_.empty())
      << "The tmp_process_state_queue_ should be empty before AdvanceAtomicToken.";
  tmp_states_visited_in_queue_.Clear();
  tmp_states_to_be_added_.clear();
  tmp_accept_stop_token_ = false;
  tmp_completed_lazy_occurrences_.clear();
  if (has_char_budget_rules_) {
    tmp_char_budget_entered_ = char_budget_entry_history_.back();
    char_count_history_.push_back(GetCurrentCharIndex() + token_char_count);
  }
  const auto& latest_states = scanable_state_history_[scanable_state_history_.size() - 1];
  for (const auto& state : latest_states) {
    if (skip_expired_states_ && IsExpiredState(state)) {
      continue;
    }
    ScanAtomicToken(state, token_id);
  }
  if (tmp_process_state_queue_.empty() && tmp_states_to_be_added_.empty()) {
    if (has_char_budget_rules_) {
      char_count_history_.pop_back();
    }
    return false;
  }
  rule_id_to_completable_states_.PushBack(std::vector<std::pair<int32_t, ParserState>>());
  if (capture_tracking_) {
    capture_event_history_.PushBack(std::vector<CaptureEvent>());
  }
  while (!tmp_process_state_queue_.empty()) {
    const auto state = std::move(tmp_process_state_queue_.front());
    tmp_process_state_queue_.pop();
    auto [scanable, completable] = Predict(state, debug_print);
    if (completable) {
      Complete(state, debug_print);
    }
    if (scanable) {
      tmp_states_to_be_added_.push_back(state);
    }
  }
  if (!tmp_completed_lazy_occurrences_.empty()) {
    RemoveCommittedLazyStates();
  }
  is_completed_.push_back(tmp_accept_stop_token_);
  scanable_state_history_.PushBack(tmp_states_to_be_added_);
  if (has_char_budget_rules_) {
    char_budget_entry_history_.push_back(tmp_char_budget_entered_);
  }
  return true;
}

bool RepeatDetector::IsVisited(const ParserState& state) const {
  // If the size is larger than the threshold, then we use the set to check.
  if (size_ > transition_threshold_) {
    return visited_set_.find(state) != visited_set_.end();
  }
  return std::find_if(
             visited_vector_.begin(),
             visited_vector_.begin() + size_,
             [&state](const ParserState& s) { return StateEqualForParsing()(state, s); }
         ) != visited_vector_.begin() + size_;
}

void RepeatDetector::Insert(const ParserState& state) {
  if (size_ == transition_threshold_) {
    for (const auto& s : visited_vector_) {
      visited_set_.insert(s);
    }
  }
  size_++;
  if (size_ > transition_threshold_) {
    visited_set_.insert(state);
  } else {
    visited_vector_[size_ - 1] = state;
  }
}

void RepeatDetector::Clear() {
  if (size_ > transition_threshold_) {
    visited_set_.clear();
  }
  size_ = 0;
}

}  // namespace xgrammar
