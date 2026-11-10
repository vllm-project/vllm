/*!
 *  Copyright (c) 2026 by Contributors
 * \file xgrammar/suffix_automata.cc
 */

#include "suffix_automata.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace xgrammar {

FSMWithStartEnd SuffixAutomata::Build(const std::vector<std::string>& chunks) {
  // Step 1. Build the suffix automaton over the chunk sequence with the standard online
  // construction. Each chunk is treated as one symbol of the alphabet.
  struct State {
    int32_t length = 0;
    int32_t suffix_link = -1;
    std::map<std::string, int32_t> transitions;
  };

  std::vector<State> states(1);
  int32_t last = 0;
  for (const std::string& chunk : chunks) {
    int32_t current = static_cast<int32_t>(states.size());
    states.push_back({states[last].length + 1, -1, {}});

    int32_t parent = last;
    while (parent != -1 && !states[parent].transitions.count(chunk)) {
      states[parent].transitions[chunk] = current;
      parent = states[parent].suffix_link;
    }
    if (parent == -1) {
      states[current].suffix_link = 0;
    } else {
      int32_t target = states[parent].transitions.at(chunk);
      if (states[parent].length + 1 == states[target].length) {
        states[current].suffix_link = target;
      } else {
        int32_t clone = static_cast<int32_t>(states.size());
        states.push_back(states[target]);
        states[clone].length = states[parent].length + 1;
        while (parent != -1) {
          auto transition = states[parent].transitions.find(chunk);
          if (transition == states[parent].transitions.end() || transition->second != target) {
            break;
          }
          transition->second = clone;
          parent = states[parent].suffix_link;
        }
        states[target].suffix_link = clone;
        states[current].suffix_link = clone;
      }
    }
    last = current;
  }

  // Step 2. Expand the chunk-level automaton into a byte-level FSM. Automaton state i maps to
  // FSM state i; every automaton state is accepting. A chunk-labeled transition becomes a chain
  // of byte transitions through fresh intermediate states; an empty chunk becomes an epsilon
  // transition.
  FSM fsm(static_cast<int>(states.size()));
  std::vector<int32_t> end_states;
  end_states.reserve(states.size());
  for (int32_t index = 0; index < static_cast<int32_t>(states.size()); ++index) {
    end_states.push_back(index);
  }
  for (int32_t index = 0; index < static_cast<int32_t>(states.size()); ++index) {
    for (const auto& [chunk, target] : states[index].transitions) {
      if (chunk.empty()) {
        fsm.AddEpsilonEdge(index, target);
        continue;
      }
      int current_state = index;
      for (size_t offset = 0; offset < chunk.size(); ++offset) {
        int next_state = offset + 1 == chunk.size() ? static_cast<int>(target) : fsm.AddState();
        uint8_t byte = static_cast<uint8_t>(chunk[offset]);
        fsm.AddEdge(current_state, next_state, byte, byte);
        current_state = next_state;
      }
    }
  }
  return FSMWithStartEnd(fsm, 0, std::move(end_states));
}

}  // namespace xgrammar
