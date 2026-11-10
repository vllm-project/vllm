/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/fsm.h
 * \note For functions accepting a pointer to a container as result, the container will be cleared
 * before the result is stored.
 */
#ifndef XGRAMMAR_FSM_H_
#define XGRAMMAR_FSM_H_

#include <picojson.h>
#include <xgrammar/object.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "support/compact_2d_array.h"
#include "support/logging.h"
#include "support/reflection.h"
#include "support/utils.h"
#include "xgrammar/exception.h"

namespace xgrammar {

/*!
 * \brief The edge of a FSM.
 */
struct FSMEdge {
  /*!
   * \brief Edge type is encoded in the `min` field. When min >= 0, the edge is a character range
   * [min, max]. When min < 0, it is a special edge type identified by the enum values below.
   *
   * For each type, `max` has a type-specific meaning (see comments on each enumerator).
   */
  enum EdgeType : int32_t {
    //! Character range [min, max]. min >= 0.
    kCharRange = 0,
    //! Epsilon transition. max is unused.
    kEpsilon = -1,
    //! Rule reference. max = rule_id.
    kRuleRef = -2,
    //! Accepts the EOS token. max is unused.
    kEOS = -3,
    //! Repeated rule reference. max = aux index into edge_aux_data
    //! (layout: [rule_id, lower, upper]).
    //! Invariant: a state with a kRepeatRef edge has exactly one outgoing edge.
    kRepeatRef = -4,
    //! Accepts a set of token IDs. max = aux index into edge_aux_data
    //! (layout: [count, token_id_0, token_id_1, ...]).
    kToken = -5,
    //! Accepts any token NOT in the given set. max = aux index into edge_aux_data
    //! (layout: [count, token_id_0, token_id_1, ...]).
    kExcludeToken = -6,
  };

  inline static constexpr int kMaxChar = 255;

  int32_t min, max;

  /*!
   * \brief The target state id of the edge.
   */
  int32_t target;

  // for serialization only
  FSMEdge() = default;

  FSMEdge(int32_t min, int32_t max, int32_t target) : min(min), max(max), target(target) {
    XGRAMMAR_DCHECK(!IsCharRange() || min <= max)
        << "Invalid FSMEdge: min > max. min=" << min << ", max=" << max;
  }

  /*!
   * \brief Compare the edges. Used to sort the edges in the FSM.
   */
  // TODO(yixin): consider combining the fields to a single int64_t for better efficiency
  friend bool operator==(const FSMEdge& lhs, const FSMEdge& rhs) {
    return std::make_tuple(lhs.min, lhs.max, lhs.target) ==
           std::make_tuple(rhs.min, rhs.max, rhs.target);
  }

  /*!
   * \brief Compare the edges. Used to sort the edges in the FSM.
   */
  friend bool operator<(const FSMEdge& lhs, const FSMEdge& rhs) {
    return std::make_tuple(lhs.min, lhs.max, lhs.target) <
           std::make_tuple(rhs.min, rhs.max, rhs.target);
  }

  /*!
   * \brief Check if the edge is a character range.
   */
  bool IsCharRange() const { return min >= 0; }

  /*!
   * \brief Check if the edge is an epsilon transition.
   */
  bool IsEpsilon() const { return min == EdgeType::kEpsilon; }

  /*!
   * \brief Check if the edge is a rule reference.
   */
  bool IsRuleRef() const { return min == EdgeType::kRuleRef; }

  /*!
   * \brief Check if the edge is an EOS transition.
   */
  bool IsEOS() const { return min == EdgeType::kEOS; }

  /*!
   * \brief Check if the edge is a repeat reference.
   */
  bool IsRepeatRef() const { return min == EdgeType::kRepeatRef; }

  bool IsToken() const { return min == EdgeType::kToken; }

  bool IsExcludeToken() const { return min == EdgeType::kExcludeToken; }

  /*!
   * \brief Get the rule id of the edge.
   * \return The rule id of the edge. -1 if the edge is not a rule reference.
   */
  int32_t GetRefRuleId() const { return IsRuleRef() ? max : -1; }

  /*!
   * \brief Get the auxiliary data index for repeat reference edges.
   * \return The index into the owning FSM's edge_aux_data. -1 if not a repeat reference.
   */
  int32_t GetAuxIndex() const {
    return (IsRepeatRef() || IsToken() || IsExcludeToken()) ? max : -1;
  }

  /*! \brief Check if the edge uses auxiliary data. */
  bool IsAuxEdge() const { return IsRepeatRef() || IsToken() || IsExcludeToken(); }

  friend struct member_trait<FSMEdge>;
};

/*! \brief View into edge_aux_data for a repeat edge (layout: [rule_id, lower, upper]). */
struct RepeatEdgeRef {
  const int32_t* data;
  int32_t RuleId() const { return data[0]; }
  int32_t Lower() const { return data[1]; }
  int32_t Upper() const { return data[2]; }
};

/*! \brief View into edge_aux_data for a token edge (layout: [count, token_id_0, ...]). */
struct TokenEdgeRef {
  const int32_t* data;
  int32_t Count() const { return data[0]; }
  const int32_t* TokenIds() const { return data + 1; }
  bool Contains(int32_t token_id) const {
    for (int32_t i = 0; i < Count(); ++i) {
      if (TokenIds()[i] == token_id) return true;
    }
    return false;
  }
};

/*! \brief View into edge_aux_data for an exclude-token edge (layout: [count, token_id_0, ...]). */
struct ExcludeTokenEdgeRef {
  const int32_t* data;
  int32_t Count() const { return data[0]; }
  const int32_t* TokenIds() const { return data + 1; }
  bool Contains(int32_t token_id) const {
    for (int32_t i = 0; i < Count(); ++i) {
      if (TokenIds()[i] == token_id) return true;
    }
    return false;
  }
  bool Accepts(int32_t token_id) const { return !Contains(token_id); }
};

/*!
 * \brief Comparator for FSMEdge. Only compare the min and max.
 */
struct FSMEdgeRangeComparator {
  bool operator()(const FSMEdge& lhs, const FSMEdge& rhs) const {
    return std::make_tuple(lhs.min, lhs.max) < std::make_tuple(rhs.min, rhs.max);
  }
};

XGRAMMAR_MEMBER_ARRAY(FSMEdge, &FSMEdge::min, &FSMEdge::max, &FSMEdge::target);

}  // namespace xgrammar

XGRAMMAR_HASH_BY_MEMBERS(
    xgrammar::FSMEdge, &xgrammar::FSMEdge::min, &xgrammar::FSMEdge::max, &xgrammar::FSMEdge::target
);

namespace xgrammar {

class CompactFSM;

/*!
 * \brief FSM is a class that represents a finite state machine, could be a DFA or an NFA.
 * \details It's mutable, which means you can add edges and states to it.
 */
class FSM {
 public:
  /*!
   * \brief Construct an FSM with a given number of states.
   * \param num_states The number of states in the FSM.
   */
  FSM(int num_states = 0);

  /*!
   * \brief Construct an FSM with a given set of edges.
   */
  FSM(const std::vector<std::vector<FSMEdge>>& edges, std::vector<int32_t> edge_aux_data = {});

  /*!
   * \brief Construct an FSM with a given set of edges.
   */
  FSM(std::vector<std::vector<FSMEdge>>&& edges, std::vector<int32_t> edge_aux_data = {});

  /****************** FSM Visitors ******************/

  /*!
   * \brief Get the number of states in the FSM.
   * \return The number of states in the FSM.
   */
  int NumStates() const;

  /*!
   * \brief Get the edges of the FSM.
   * \return The edges of the FSM.
   */
  const std::vector<std::vector<FSMEdge>>& GetEdges() const;

  /*!
   * \brief Get the edges of the FSM.
   * \return The edges of the FSM.
   */
  std::vector<std::vector<FSMEdge>>& GetEdges();

  /*!
   * \brief Get the edges of the FSM.
   * \param state The state to get the edges from.
   * \return The edges of the FSM.
   */
  std::vector<FSMEdge>& GetEdges(int state);

  /*!
   * \brief Get the edges of the FSM.
   * \param state The state to get the edges from.
   * \return The edges of the FSM.
   */
  const std::vector<FSMEdge>& GetEdges(int state) const;

  /*!
   * \brief Convert the edges of the FSM to a string. Used in printing the FSM.
   * \return The string representation of the edges of the FSM.
   */
  std::string EdgesToString(std::optional<std::vector<int>> states = std::nullopt) const;

  /****************** FSM Traversal Visitors ******************/

  inline static constexpr int kNoNextState = -1;

  /*!
   * \brief Advance the FSM from a given state based on an input character. If there are multiple
   * transitions, the first one will be returned.
   * \param from The source state to transition from.
   * \param character The input character.
   * \return The target state if a valid transition exists, kNoNextState otherwise.
   */
  int GetNextState(int from, int value, FSMEdge::EdgeType edge_type = FSMEdge::EdgeType::kCharRange)
      const;

  /*!
   * \brief Advance the FSM to the next state.
   * \param from The current states.
   * \param value The input value.
   * \param result The possible next states. The result is cleared at the beginning.
   * \param value_is_rule Whether the input value is a rule id.
   * \param from_is_closure Whether from is an epsilon closure.
   */
  void Advance(
      const std::unordered_set<int>& from,
      int value,
      std::unordered_set<int>* result,
      FSMEdge::EdgeType edge_type = FSMEdge::EdgeType::kCharRange,
      bool from_is_closure = false
  ) const;

  /*!
   * \brief Get all the possible rule numbers for a given state.
   * \param state_num The state number.
   * \param rules The set of possible rule numbers. The result is cleared at the beginning.
   */
  void GetPossibleRules(int state_num, std::unordered_set<int>* rules) const;

  /*!
   * \brief Get the epsilon closure of a set of states, i.e. those can be reached by epsilon
   * transitions.
   * \param state_set The states in the epsilon closure. The result is not cleared.
   */
  void GetEpsilonClosure(std::unordered_set<int>* state_set) const;

  /*!
   * \brief Get the reachable states from a set of states.
   * \param from The current states.
   * \param result The reachable states. The result is cleared at the beginning.
   */
  void GetReachableStates(const std::vector<int>& from, std::unordered_set<int>* result) const;

  /****************** FSM Mutators ******************/

  /*!
   * \brief Adds a new state to the FSM.
   * \return The index of the newly added state.
   */
  int AddState();

  /*!
   * \brief Adds a transition edge between states with given min and max values. For character
   * transitions, it accepts any character in range [min, max].
   * \param from The source state.
   * \param to The target state.
   * \param min The min value of the range.
   * \param max The max value of the range.
   */
  void AddEdge(int from, int to, int32_t min, int32_t max);

  /*! \brief Add a raw edge with explicit type and value. */
  void AddEdge(int from, int to, FSMEdge::EdgeType type, int32_t value);

  /*!
   * \brief Add an epsilon transition between two states.
   * \param from The source state.
   * \param to The target state.
   */
  void AddEpsilonEdge(int from, int to);

  /*!
   * \brief Add a rule reference edge between states.
   * \param from The source state.
   * \param to The target state.
   * \param rule_id The rule id to reference.
   */
  void AddRuleEdge(int from, int to, int32_t rule_id);

  /*!
   * \brief Add an EOS transition between two states.
   * \param from The source state.
   * \param to The target state.
   */
  void AddEOSEdge(int from, int to);

  /*!
   * \brief Add a repeat reference edge between states, allocating auxiliary data.
   * \param from The source state.
   * \param to The target state.
   * \param rule_id The rule to repeat.
   * \param lower Minimum repeat count.
   * \param upper Maximum repeat count (-1 for unlimited).
   */
  void AddRepeatEdge(int from, int to, int32_t rule_id, int32_t lower, int32_t upper);

  void AddTokenEdge(int from, int to, const std::vector<int32_t>& token_ids);

  void AddExcludeTokenEdge(int from, int to, const std::vector<int32_t>& token_ids);

  /*! \brief Get the edge auxiliary data. */
  const std::vector<int32_t>& GetEdgeAuxData() const;

  /*! \brief Set the edge auxiliary data (used during FSM construction). */
  void SetEdgeAuxData(std::vector<int32_t> data);

  /*! \brief Get repeat edge info by aux index. */
  RepeatEdgeRef GetRepeatEdgeInfo(int32_t idx) const;

  /*! \brief Get token edge info by aux index. */
  TokenEdgeRef GetTokenEdgeInfo(int32_t idx) const;

  /*! \brief Get exclude-token edge info by aux index. */
  ExcludeTokenEdgeRef GetExcludeTokenEdgeInfo(int32_t idx) const;

  /*!
   * \brief Add a whole FSM to the current FSM.
   * \param fsm The FSM to be added.
   * \param state_mapping The mapping from the state ids of the added FSM to the new ids in the
   * current FSM. The result is cleared at the beginning. If the fsm's state id starts from 0, use
   * it for efficiency.
   */
  void AddFSM(const FSM& fsm, std::vector<int>* state_mapping = nullptr);

  /****************** FSM Construction Methods ******************/

  /*!
    \brief Return a copy of the FSM.
  */
  FSM Copy() const;

  /*!
   * \brief Rebuild the FSM with the new state ids.
   * \param state_mapping The mapping from the old state ids to the new state ids.
   * \param new_num_states The new number of states.
   * \return The rebuilt FSM.
   */
  FSM RebuildWithMapping(const std::vector<int>& state_mapping, int new_num_states) const;

  /*!
   * \brief Sort the edges of the FSM by their min, max and target.
   */
  void SortEdges();

  /*!
   * \brief Transform a FSM to a compact FSM. This method will first sort the edges of the FSM,
   * then put all the edges into a compact array.
   * \return The compact FSM.
   */
  CompactFSM ToCompact();

  XGRAMMAR_DEFINE_PIMPL_METHODS(FSM);
};

/*!
 * \brief CompactFSM is the compact from of FSM.
 * \details It uses Compact2DArray to store the edges, ensuring memory contiguity. It sorts all
 * outgoing edges from a node according to their min and max values, so traversal can be faster.
 *
 * CompactFSM is immutable. If you need to modify a CompactFSM, you need to convert it to a FSM
 * first, and convert it back after modification.
 *
 * It share the same set of visitor methods with FSM.
 */

class CompactFSM {
 public:
  // for serialization only
  CompactFSM() = default;

  explicit CompactFSM(
      const Compact2DArray<FSMEdge>& edges, std::vector<int32_t> edge_aux_data = {}
  );

  explicit CompactFSM(Compact2DArray<FSMEdge>&& edges, std::vector<int32_t> edge_aux_data = {});

  /****************** CompactFSM Visitors ******************/

  /*!
   * \brief Get the number of states in the FSM.
   * \return The number of states in the FSM.
   */
  int NumStates() const;

  /*!
   * \brief Get the edges of the CompactFSM.
   * \return The edges of the CompactFSM.
   */
  const Compact2DArray<FSMEdge>& GetEdges() const;

  /*!
   * \brief Get the edges of the CompactFSM.
   * \param state The state to get the edges from.
   * \return The edges of the CompactFSM.
   */
  Compact2DArray<FSMEdge>::Row GetEdges(int state) const;

  /*!
   * \brief Convert the edges of the CompactFSM to a string. Used in printing the CompactFSM.
   * \return The string representation of the edges of the CompactFSM.
   */
  std::string EdgesToString(std::optional<std::vector<int>> states = std::nullopt) const;

  /*!
   * \brief Get the memory size of the CompactFSM.
   * \param self The CompactFSM.
   * \return The memory size of the CompactFSM.
   */
  friend std::size_t MemorySize(const CompactFSM& self);

  /****************** CompactFSM Traversal Visitors ******************/

  inline static constexpr int kNoNextState = -1;

  /*!
   * \brief Advance the FSM from a given state based on an input character. If there are multiple
   * transitions, the first one will be returned.
   * \param from The source state to transition from.
   * \param character The input character.
   * \param targets The target states to be filled with the possible next states.
   * \return The target state if a valid transition exists, kNoNextState otherwise.
   */
  void GetNextStates(
      int from,
      int value,
      FSMEdge::EdgeType edge_type = FSMEdge::EdgeType::kCharRange,
      std::vector<int>* targets = nullptr
  ) const;

  /*!
   * \brief Advance the FSM to the next state.
   * \param from The current states.
   * \param value The input value.
   * \param result The possible next states. The result is cleared at the beginning.
   * \param value_is_rule Whether the input value is a rule id.
   * \param from_is_closure Whether from is an epsilon closure.
   */
  void Advance(
      const std::unordered_set<int>& from,
      int value,
      std::unordered_set<int>* result,
      FSMEdge::EdgeType edge_type = FSMEdge::EdgeType::kCharRange,
      bool from_is_closure = false
  ) const;

  /*!
   * \brief Get all the possible rule numbers for a given state.
   * \param state_num The state number.
   * \param rules The set of possible rule numbers. The result is cleared at the beginning.
   */
  void GetPossibleRules(int state_num, std::unordered_set<int>* rules) const;

  /*!
   * \brief Get the epsilon closure of a set of states, i.e. those can be reached by epsilon
   * transitions.
   * \param state_set The states in the epsilon closure. The result is not cleared.
   */
  void GetEpsilonClosure(std::unordered_set<int>* state_set) const;

  /*!
   * \brief Get the reachable states from a set of states.
   * \param from The current states.
   * \param result The reachable states. The result is cleared at the beginning.
   */
  void GetReachableStates(const std::vector<int>& from, std::unordered_set<int>* result) const;

  /*!
   * \brief Get the number of edges in the compact FSM.
   * \return The number of edges.
   */
  size_t GetNumEdges() const;

  /****************** CompactFSM Auxiliary Data ******************/

  /*! \brief Get the edge auxiliary data. */
  const std::vector<int32_t>& GetEdgeAuxData() const;

  /*! \brief Set the edge auxiliary data (used during FSM construction). */
  void SetEdgeAuxData(std::vector<int32_t> data);

  /*! \brief Get repeat edge info by aux index. */
  RepeatEdgeRef GetRepeatEdgeInfo(int32_t idx) const;

  /*! \brief Get token edge info by aux index. */
  TokenEdgeRef GetTokenEdgeInfo(int32_t idx) const;

  /*! \brief Get exclude-token edge info by aux index. */
  ExcludeTokenEdgeRef GetExcludeTokenEdgeInfo(int32_t idx) const;

  /****************** CompactFSM Construction Methods ******************/

  /*!
   * \brief Transform the compact FSM to a FSM.
   * \return The FSM.
   */
  FSM ToFSM() const;

  friend picojson::value SerializeJSONValue(const CompactFSM& value);
  friend std::optional<SerializationError> DeserializeJSONValue(
      CompactFSM* result, const picojson::value& value, const std::string& type_name
  );

  XGRAMMAR_DEFINE_PIMPL_METHODS(CompactFSM);
};

std::optional<SerializationError> DeserializeJSONValue(
    CompactFSM* result, const picojson::value& value, const std::string& type_name = ""
);

class FSMWithStartEnd;
class FSMWithStartEndWithSize;
class CompactFSMWithStartEnd;
class CompactFSMWithStartEndWithSize;
struct CompactFSMWithStartEndWithSizeSerializeHelper;

/*!
 * \brief The base class for FSMWithStartEnd and CompactFSMWithStartEnd. It defines the
 * common constructor and visitor methods.
 */
template <typename FSMType>
class FSMWithStartEndBase {
  static_assert(
      std::is_same_v<FSMType, FSM> || std::is_same_v<FSMType, CompactFSM>,
      "FSMType must be FSM or CompactFSM"
  );

 public:
  // For serialization only
  FSMWithStartEndBase() = default;

  FSMWithStartEndBase(const FSMType& fsm, int start, std::vector<int32_t> ends, bool is_dfa = false)
      : fsm_(fsm), start_(start), ends_(std::move(ends)), is_dfa_(is_dfa) {
    NormalizeEnds();
  }

  /****************** Member Accessors and Mutators ******************/

  /*! \brief Returns the underlying FSM. */
  const FSMType& GetFsm() const { return fsm_; }

  /*! \brief Returns the start state of the FSM. */
  int GetStart() const { return start_; }

  /*!
   * \brief Returns the end states of the FSM as a sorted, deduplicated list of state ids.
   * \note End states are stored sparsely because an FSM view over a large shared FSM (e.g. the
   * per-rule FSMs over the complete FSM of a grammar) usually has only a few end states, while
   * the state id space can be very large.
   */
  const std::vector<int32_t>& GetEnds() const { return ends_; }

  /*!
   * \brief Checks if a given state is an end/accepting state.
   * \param state The state to check.
   * \return True if the state is an end state, false otherwise.
   */
  bool IsEndState(int state) const { return std::binary_search(ends_.begin(), ends_.end(), state); }

  /*! \brief Check if a state is scanable.
   *  \param state The state to check.
   *  \return True if the state is scanable, false otherwise.
   */
  bool IsScanableState(int state) const {
    for (const auto& edge : fsm_.GetEdges(state)) {
      if (edge.IsCharRange() || edge.IsToken() || edge.IsExcludeToken()) {
        return true;
      }
    }
    return false;
  }

  /*!
   * \brief Check if a state is not terminal.
   * \param state The state to check.
   * \return True if the state is scanable, false otherwise.
   */
  bool IsNonTerminalState(int state) const {
    for (const auto& edge : fsm_.GetEdges(state)) {
      if (edge.IsRuleRef() || edge.IsEpsilon() || edge.IsRepeatRef()) {
        return true;
      }
    }
    return false;
  }

  /*!
   * \brief Sets the start state of the FSM.
   * \param state The state to set as the start state.
   */
  void SetStartState(int state) {
    XGRAMMAR_DCHECK(state < NumStates());
    start_ = state;
  }

  /*!
   * \brief Adds an end/accepting state to the FSM.
   * \param state The state to add as an end state.
   */
  void AddEndState(int state) {
    XGRAMMAR_DCHECK(state < NumStates());
    auto it = std::lower_bound(ends_.begin(), ends_.end(), state);
    if (it == ends_.end() || *it != state) {
      ends_.insert(it, state);
    }
  }

  /*!
   * \brief Adds a new state to the FSM and marks it as non-end.
   * \return The index of the newly added state.
   */
  int AddState() { return fsm_.AddState(); }

  /*!
   * \brief Sets the end states of the FSM.
   * \param ends The new end states, as a list of state ids. Need not be sorted or deduplicated.
   */
  void SetEndStates(std::vector<int32_t> ends) {
    ends_ = std::move(ends);
    NormalizeEnds();
  }

  /*! \brief Returns the total number of states in the FSM. */
  int NumStates() const { return fsm_.NumStates(); }

  /*!
   * \brief Access the methods of the underlying FSM.
   */
  FSMType& GetFsm() { return fsm_; }

  /****************** FSM Traversal Algorithms ******************/

  /*!
   * \brief Check if the FSM accepts the string.
   * \param str The input string.
   * \return True if the FSM accepts the string, false otherwise.
   */
  bool AcceptString(const std::string& str) const;

  /*!
   * \brief Get the reachable states from the start state.
   * \param result The reachable states. The result is cleared at the beginning.
   */
  void GetReachableStates(std::unordered_set<int>* result) const;

  /*!
   * \brief Check if the FSM is a leaf FSM.
   * \return True if the FSM is a leaf FSM, false otherwise.
   */
  bool IsLeaf() const;

 protected:
  /*! \brief Sort and deduplicate the end states to maintain the sorted invariant. */
  void NormalizeEnds() {
    std::sort(ends_.begin(), ends_.end());
    ends_.erase(std::unique(ends_.begin(), ends_.end()), ends_.end());
  }

  /*! \brief The underlying finite state machine. */
  FSMType fsm_;
  /*! \brief The start state of the FSM. */
  int start_;

  /*!
   * \brief The set of accepting/end states, stored as a sorted, deduplicated list of state ids.
   * \note Stored sparsely (rather than as a bitvector over all states) because FSM views over a
   * large shared FSM usually have only a few end states, while the state id space can be huge.
   */
  std::vector<int32_t> ends_;

 protected:
  /*! \brief Whether this FSM is a deterministic finite automaton. */
  bool is_dfa_ = false;
};

/*!
 * \brief FSMWithStartEnd represents a FSM with start and end states.
 * \details It stores a pointer to a FSM, a start state, and a set of end states. Multiple
 * FSMWithStartEnd can share the same FSM. It also provides a set of methods to construct FSMs.
 */
class FSMWithStartEnd : public FSMWithStartEndBase<FSM> {
 public:
  using FSMWithStartEndBase<FSM>::FSMWithStartEndBase;

  /*!
   * \brief Convert the FSMWithStartEnd to a string. Only considers the nodes approachable from the
   * start state.
   * \return The string representation of the FSMWithStartEnd.
   */
  std::string ToString() const;

  friend std::ostream& operator<<(std::ostream& os, const FSMWithStartEnd& fsm);

  /****************** FSM Construction Methods ******************/

  /*!
   * \brief Return a copy of the FSMWithStartEnd.
   */
  FSMWithStartEnd Copy() const;

  /*!
   * \brief Rebuild the FSM with the new state ids.
   * \param state_mapping The mapping from old state ids to new state ids.
   * \param new_num_states The new number of states.
   */
  FSMWithStartEnd RebuildWithMapping(const std::vector<int>& state_mapping, int new_num_states)
      const;

  /*!
   * \brief Add the underlying FSM to another complete FSM that could contain multiple FSMs.
   * Return a new FSMWithStartEnd that points to the complete FSM and whose start and ends are
   * mapped to the states in the complete FSM.
   * \param complete_fsm The complete FSM.
   * \param state_mapping The mapping from the old state ids to the new state ids. The result is
   * cleared at the beginning. Should not be nullptr.
   * \return The FSMWithStartEnd that points to the complete FSM.
   */
  FSMWithStartEndWithSize AddToCompleteFSM(FSM* complete_fsm, std::vector<int>* state_mapping);

  /*!
   * \brief Transform the FSMWithStartEnd to a CompactFSMWithStartEnd.
   * \return The CompactFSMWithStartEnd.
   */
  CompactFSMWithStartEnd ToCompact();

  /****************** FSM Algorithms ******************/

  /*!
   * \brief Return a new FSM representing FSM*
   * \return The FSM that accepts FSM*.
   */
  FSMWithStartEnd Star() const;

  /*!
   * \brief Return a new FSM representing rule1+.
   * \return The FSM that accepts rule1+.
   */
  FSMWithStartEnd Plus() const;

  /*!
   * \brief Return a new FSM representing rule1?.
   * \return The FSM that accepts rule1?.
   */
  FSMWithStartEnd Optional() const;

  /*!
   * \brief Return a new FSM representing the complement of the language.
   * \return The complement FSM.
   */
  Result<FSMWithStartEnd> Not(int max_result_num_states = 1e6) const;

  /*!
   * \brief Intersect the FSMs.
   * \param lhs The left FSM.
   * \param rhs The right FSM.
   * \return The intersection of the FSMs.
   */
  static Result<FSMWithStartEnd> Intersect(
      const FSMWithStartEnd& lhs, const FSMWithStartEnd& rhs, int max_result_num_states = 1e6
  );

  /*!
   * \brief Union the FSMs.
   * \param fsms The FSMs to be unioned.
   * \return The union of the FSMs.
   */
  static FSMWithStartEnd Union(const std::vector<FSMWithStartEnd>& fsms);

  /*!
   * \brief Concatenate the FSMs.
   * \param fsms The FSMs to be concatenated, which should be in order.
   * \return The concatenation of the FSMs.
   */
  static FSMWithStartEnd Concat(const std::vector<FSMWithStartEnd>& fsms);

  /*!
   * \brief Check if the FSM is a DFA.
   * \return True if the FSM is a DFA, false otherwise.
   */
  bool IsDFA();

  /*!
   * \brief Merge some states by removing some epsilon transitions.
   * \details If a --\epsilon--> b, and either 1) b doesn't have any other inward edges, or
   * 2) a doesn't have any other outward edges, we can merge a and b.
   */
  FSMWithStartEnd SimplifyEpsilon(int max_num_states = 1e8) const;

  /*!
   * \brief Merge equivalent states in the FSM.
   * \details If two states are 1) pointed to by edges with the same label from the same state, and
   * 2) they are not pointed to by other edges, then we can merge them.
   * \example n0 --(c)--> n1, n0 --(c)--> n2, then we can merge n1 and n2.
   */
  FSMWithStartEnd MergeEquivalentStates(int max_num_states = 1e5) const;

  /*!
   * \brief Transform the FSM to a DFA.
   * \param max_result_num_states The maximum number of states in the DFA.
   * \return The DFA.
   */
  Result<FSMWithStartEnd> ToDFA(int max_num_states = 1e3) const;

  /*!
   * \brief Minimize the DFA.
   * \param max_result_num_states The maximum number of states in the DFA.
   * \return The minimized DFA.
   */
  Result<FSMWithStartEnd> MinimizeDFA(int max_num_states = 1e3) const;
};

/*!
 * \brief Wrapper that bundles an FSMWithStartEnd with explicit size metadata. It is
 * used when we want to store the number of edges and nodes in the part of the FSM, instead
 * of the completed FSMWithStartEnd.
 */
class FSMWithStartEndWithSize {
 public:
  // For serialization only
  FSMWithStartEndWithSize() = default;

  explicit FSMWithStartEndWithSize(FSMWithStartEnd fsm, int edge_num, int node_num)
      : fsm_(std::move(fsm)), edge_num_(edge_num), node_num_(node_num) {}

  const FSMWithStartEnd& GetFsm() const { return fsm_; }
  int GetEdgeNum() const { return edge_num_; }
  int GetNodeNum() const { return node_num_; }

 private:
  FSMWithStartEnd fsm_;
  int edge_num_ = 0;
  int node_num_ = 0;
};

/*!
 * \brief A class that represents a compact-form FSM with a start state and a set of end states.
 * \details CompactFSMWithStartEnd stores a pointer to a CompactFSM, a start state, and a set of end
 * states. Multiple CompactFSMWithStartEnd can share the same CompactFSM. It share the same set of
 * visitor methods with FSMWithStartEnd.
 */
class CompactFSMWithStartEnd : public FSMWithStartEndBase<CompactFSM> {
 public:
  // For serialization only
  CompactFSMWithStartEnd() = default;

  explicit CompactFSMWithStartEnd(const CompactFSM& fsm, int start, std::vector<int32_t> ends)
      : FSMWithStartEndBase<CompactFSM>(fsm, start, std::move(ends)),
        edge_num_(fsm.GetNumEdges()) {}

  using FSMWithStartEndBase<CompactFSM>::FSMWithStartEndBase;

  /*!
   * \brief Convert the FSMWithStartEnd to a string. Only considers the nodes approachable from the
   * start state.
   * \return The string representation of the FSMWithStartEnd.
   */
  std::string ToString() const;

  /*!
   * \brief Transform the CompactFSMWithStartEnd to a FSMWithStartEnd.
   * \return The FSMWithStartEnd.
   */
  FSMWithStartEnd ToFSM() const;

 private:
  size_t edge_num_ = 0;

  /*!
   * \brief Print the CompactFSMWithStartEnd.
   * \param os The output stream.
   * \param fsm The CompactFSMWithStartEnd.
   * \return The output stream.
   */
  friend std::ostream& operator<<(std::ostream& os, const CompactFSMWithStartEnd& fsm);

  /*!
   * \brief Get the memory size of the CompactFSMWithStartEnd.
   * \param self The CompactFSMWithStartEnd.
   * \return The memory size of the CompactFSMWithStartEnd.
   */
  friend std::size_t MemorySize(const CompactFSMWithStartEnd& self);

  friend struct member_trait<CompactFSMWithStartEnd>;

  friend struct CompactFSMWithStartEndSerializeHelper;

  friend picojson::value SerializeJSONValue(const CompactFSMWithStartEnd& value);
  friend std::optional<SerializationError> DeserializeJSONValue(
      CompactFSMWithStartEnd* result, const picojson::value& value, const std::string& type_name
  );
};

/*!
 * \brief Wrapper that bundles a CompactFSMWithStartEnd with explicit size metadata. It is
 * used when we want to store the number of edges and nodes in the part of the
 * CompactFSMWithStartEnd, instead of the completed CompactFSMWithStartEnd.
 */
class CompactFSMWithStartEndWithSize {
 public:
  // For serialization only
  CompactFSMWithStartEndWithSize() = default;

  explicit CompactFSMWithStartEndWithSize(CompactFSMWithStartEnd fsm, int edge_num, int node_num)
      : fsm_(std::move(fsm)), edge_num_(edge_num), node_num_(node_num) {}

  const CompactFSMWithStartEnd& GetFsm() const { return fsm_; }
  int GetEdgeNum() const { return edge_num_; }
  int GetNodeNum() const { return node_num_; }

  friend picojson::value SerializeJSONValue(const CompactFSMWithStartEndWithSize& value);
  friend std::optional<SerializationError> DeserializeJSONValue(
      CompactFSMWithStartEndWithSize* result,
      const picojson::value& value,
      const std::string& type_name
  );

 private:
  CompactFSMWithStartEnd fsm_;
  int edge_num_ = 0;
  int node_num_ = 0;

  friend std::size_t MemorySize(const CompactFSMWithStartEndWithSize& self) {
    return MemorySize(self.fsm_) + sizeof(self.edge_num_) + sizeof(self.node_num_);
  }

  friend struct CompactFSMWithStartEndWithSizeSerializeHelper;
};

std::optional<SerializationError> DeserializeJSONValue(
    CompactFSMWithStartEndWithSize* result,
    const picojson::value& value,
    const std::string& type_name = ""
);

/****************** FSMWithStartEndBase Template Implementation ******************/

template <typename FSMType>
inline bool FSMWithStartEndBase<FSMType>::AcceptString(const std::string& str) const {
  std::unordered_set<int> start_states{start_};
  fsm_.GetEpsilonClosure(&start_states);
  std::unordered_set<int> result_states;
  for (const auto& character : str) {
    result_states.clear();
    fsm_.Advance(
        start_states,
        static_cast<int>(static_cast<unsigned char>(character)),
        &result_states,
        FSMEdge::EdgeType::kCharRange,
        false
    );
    if (result_states.empty()) {
      return false;
    }
    start_states = result_states;
  }
  return std::any_of(start_states.begin(), start_states.end(), [&](int state) {
    return IsEndState(state);
  });
}

template <typename FSMType>
inline void FSMWithStartEndBase<FSMType>::GetReachableStates(std::unordered_set<int>* result
) const {
  return fsm_.GetReachableStates({start_}, result);
}

template <typename FSMType>
inline bool FSMWithStartEndBase<FSMType>::IsLeaf() const {
  std::unordered_set<int> reachable_states;
  GetReachableStates(&reachable_states);
  for (const auto& state : reachable_states) {
    for (const auto& edge : fsm_.GetEdges(state)) {
      if (edge.IsRuleRef() || edge.IsRepeatRef()) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace xgrammar

#endif  // XGRAMMAR_FSM_H_
