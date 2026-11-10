/*!
 *  Copyright (c) 2026 by Contributors
 * \file xgrammar/suffix_automata.h
 * \brief Suffix automaton construction for substring expressions.
 */
#ifndef XGRAMMAR_SUFFIX_AUTOMATA_H_
#define XGRAMMAR_SUFFIX_AUTOMATA_H_

#include <string>
#include <vector>

#include "fsm.h"

namespace xgrammar {

/*!
 * \brief Builds the automaton of a substring expression via a chunk-level suffix automaton.
 */
class SuffixAutomata {
 public:
  /*!
   * \brief Build an FSM that accepts exactly the contiguous subsequences of the chunk list,
   * including the empty one.
   * \details A suffix automaton is built over the chunk sequence (each chunk is one symbol), so
   * the number of automaton states grows linearly with the number of chunks. Every automaton
   * state is accepting. Each chunk-labeled transition is then expanded into a chain of byte
   * transitions; an empty chunk becomes an epsilon transition.
   * \param chunks The list of byte string chunks. Chunks may be empty or repeated.
   * \return The FSM with start and end states.
   */
  static FSMWithStartEnd Build(const std::vector<std::string>& chunks);
};

}  // namespace xgrammar

#endif  // XGRAMMAR_SUFFIX_AUTOMATA_H_
