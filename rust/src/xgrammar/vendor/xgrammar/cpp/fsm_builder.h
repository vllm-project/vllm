/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/fsm_builder.h
 */
#ifndef XGRAMMAR_FSM_BUILDER_H_
#define XGRAMMAR_FSM_BUILDER_H_

#include <bitset>
#include <cstdint>
#include <string>
#include <vector>

#include "fsm.h"
#include "support/utils.h"

namespace xgrammar {

class GrammarBuilder;

/*! \brief Bounds of the packed UTF-8 representation for each encoded length. The packed format
 * stores the UTF-8 bytes of one codepoint as (byte0 << 24) | (byte1 << 16) | (byte2 << 8) | byte3,
 * left-aligned to the actual length (e.g. a 2-byte character is (byte0 << 8) | byte1). */
constexpr uint32_t kMax1ByteUnicode = 0x7F;
constexpr uint32_t kMin2BytesUnicode = 0xC080;
constexpr uint32_t kMax2BytesUnicode = 0xDFBF;
constexpr uint32_t kMin3BytesUnicode = 0xE08080;
constexpr uint32_t kMax3BytesUnicode = 0xEFBFBF;
constexpr uint32_t kMin4BytesUnicode = 0xF0808080;
constexpr uint32_t kMax4BytesUnicode = 0xF7BFBFBF;

/*! \brief Convert a Unicode codepoint to the packed UTF-8 format described above. */
uint32_t CodepointToPackedUTF8(uint32_t codepoint);

/*!
 * \brief Add FSM edges (with intermediate states for multi-byte characters) from `from` to `to`
 * accepting every UTF-8 encoded character in the packed range [min, max].
 */
void AddPackedUTF8RangeEdges(FSM& fsm, int from, int to, uint32_t min, uint32_t max);

/*!
 * \brief Rewrite every unescaped '.' outside character classes to "[^\n]" unless
 * `dot_matches_newline` is true. Used to implement the standard regex dot semantics (and the
 * dot-all 's' flag) on top of the regex engine, whose '.' matches every codepoint.
 */
std::string RewriteRegexDots(const std::string& pattern, bool dot_matches_newline);

/*!
 * \brief A builder that converts a regex string to a FSM.
 */
class RegexFSMBuilder {
 public:
  /*!
   * \brief Converts a regex string to a FSM.
   * \param regex The regex string. A leading "(?i)" makes the match ASCII case-insensitive.
   * \param builder If not null, bounded repetitions whose upper bound exceeds the unroll
   * threshold are compiled into a kRepeatRef FSM edge referencing a new rule (with a kRegex
   * body holding the repeated sub-pattern) added through this builder, instead of being
   * physically unrolled.
   * \param rule_hint Name hint for rules created through `builder`.
   * \return The FSM with start and end states.
   */
  static Result<FSMWithStartEnd> Build(
      const std::string& regex, GrammarBuilder* builder = nullptr, const std::string& rule_hint = ""
  );

  /*!
   * \brief Converts a regex string to a FSM, then removes the forbidden characters from every
   * character transition. The result accepts the intersection of the regex language and the
   * set of strings that contain no forbidden character. The result language may be empty.
   * \param regex The regex string.
   * \param forbidden_chars The forbidden characters.
   * \param builder See Build().
   * \param rule_hint See Build().
   * \return The FSM with start and end states.
   */
  static Result<FSMWithStartEnd> BuildWithForbiddenChars(
      const std::string& regex,
      const std::bitset<256>& forbidden_chars,
      GrammarBuilder* builder = nullptr,
      const std::string& rule_hint = ""
  );

  /*!
   * \brief Check whether the regex matches the empty string. Only parses the regex; no FSM is
   * built, so this is cheap even for regexes with huge bounded repetitions.
   */
  static Result<bool> MatchesEmpty(const std::string& regex);
};

/*!
 * \brief A builder that converts a list of patterns to a trie-based FSM.
 */
class TrieFSMBuilder {
 public:
  /*!
   * \brief Build a trie-based FSM from a list of patterns.
   * \param patterns The patterns to be built.
   * \param excluded_patterns The patterns to be excluded.
   * \param end_states The end states of the FSM. This is the terminal state of each pattern and
   * the order follows the order of patterns.
   * \param allow_overlap Whether to allow overlap between patterns (one being a prefix of the
   * other). It does not allow empty patterns either. If false and there is overlap, will return
   * std::nullopt.
   * \param add_back_edges Whether to add back edges to the FSM. This complements the trie to an
   * Aho-Corasick automaton.
   * \return If success, the FSM with start and end states. Otherwise, std::nullopt.
   */
  static std::optional<FSMWithStartEnd> Build(
      const std::vector<std::string>& patterns,
      const std::vector<std::string>& excluded_patterns,
      std::vector<int32_t>* end_states = nullptr,
      bool allow_overlap = true,
      bool add_back_edges = false
  );
};

}  // namespace xgrammar

#endif  // XGRAMMAR_FSM_BUILDER_H_
