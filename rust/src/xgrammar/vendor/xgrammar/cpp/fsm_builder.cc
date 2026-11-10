/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/fsm_builder.cc
 */
#include "fsm_builder.h"

#include <sys/types.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <set>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "fsm.h"
#include "grammar_builder.h"
#include "support/encoding.h"
#include "support/logging.h"
#include "support/utils.h"

namespace xgrammar {

/******************** Packed UTF-8 range helpers ********************/

uint32_t CodepointToPackedUTF8(uint32_t codepoint) {
  if (codepoint <= 0x7F) {
    // 1-byte sequence (ASCII)
    return codepoint;
  } else if (codepoint <= 0x7FF) {
    // 2-byte sequence: byte0 = 110xxxxx, byte1 = 10xxxxxx
    uint8_t byte0 = 0xC0 | ((codepoint >> 6) & 0x1F);
    uint8_t byte1 = 0x80 | (codepoint & 0x3F);
    return (static_cast<uint32_t>(byte0) << 8) | byte1;
  } else if (codepoint <= 0xFFFF) {
    // 3-byte sequence: byte0 = 1110xxxx, byte1 = 10xxxxxx, byte2 = 10xxxxxx
    uint8_t byte0 = 0xE0 | ((codepoint >> 12) & 0x0F);
    uint8_t byte1 = 0x80 | ((codepoint >> 6) & 0x3F);
    uint8_t byte2 = 0x80 | (codepoint & 0x3F);
    return (static_cast<uint32_t>(byte0) << 16) | (static_cast<uint32_t>(byte1) << 8) | byte2;
  } else {
    // 4-byte sequence: byte0 = 11110xxx, byte1-3 = 10xxxxxx
    uint8_t byte0 = 0xF0 | ((codepoint >> 18) & 0x07);
    uint8_t byte1 = 0x80 | ((codepoint >> 12) & 0x3F);
    uint8_t byte2 = 0x80 | ((codepoint >> 6) & 0x3F);
    uint8_t byte3 = 0x80 | (codepoint & 0x3F);
    return (static_cast<uint32_t>(byte0) << 24) | (static_cast<uint32_t>(byte1) << 16) |
           (static_cast<uint32_t>(byte2) << 8) | byte3;
  }
}

// This function will add a range [min, max] of characters to the FSM, and the length
// of the characters are the same.
static void AddSameLengthCharacterRange(FSM& fsm, int from, int to, uint32_t min, uint32_t max) {
  uint8_t byte_min[4] = {
      static_cast<uint8_t>(min & 0xFF),
      static_cast<uint8_t>(min >> 8),
      static_cast<uint8_t>(min >> 16),
      static_cast<uint8_t>(min >> 24)
  };
  uint8_t byte_max[4] = {
      static_cast<uint8_t>(max & 0xFF),
      static_cast<uint8_t>(max >> 8),
      static_cast<uint8_t>(max >> 16),
      static_cast<uint8_t>(max >> 24)
  };

  // ASCII.
  if (byte_max[1] == 0) {
    fsm.AddEdge(from, to, byte_min[0], byte_max[0]);
    return;
  }

  if (byte_max[3] != 0) {
    // 4-byte unicode.
    if (byte_max[3] == byte_min[3]) {
      int tmp_state = fsm.AddState();
      fsm.AddEdge(from, tmp_state, byte_min[3], byte_max[3]);
      min = (min & 0x00FFFFFF);
      max = (max & 0x00FFFFFF);
      AddSameLengthCharacterRange(fsm, tmp_state, to, min, max);
      return;
    }
    if ((min & 0x00FFFFFF) != 0x808080) {
      int tmp_state_min = fsm.AddState();
      fsm.AddEdge(from, tmp_state_min, byte_min[3], byte_min[3]);
      AddSameLengthCharacterRange(fsm, tmp_state_min, to, (min & 0x00FFFFFF), 0x00BFBFBF);
    } else {
      byte_min[3]--;
    }
    if ((max & 0x00FFFFFF) != 0xBFBFBF) {
      int tmp_state_max = fsm.AddState();
      fsm.AddEdge(from, tmp_state_max, byte_max[3], byte_max[3]);
      AddSameLengthCharacterRange(fsm, tmp_state_max, to, 0x00808080, (max & 0x00FFFFFF));
    } else {
      byte_max[3]++;
    }
    if (byte_max[3] - byte_min[3] > 1) {
      int tmp_state_mid = fsm.AddState();
      // First byte.
      fsm.AddEdge(from, tmp_state_mid, byte_min[3] + 1, byte_max[3] - 1);
      int tmp_state_mid2 = fsm.AddState();
      // Second byte.
      fsm.AddEdge(tmp_state_mid, tmp_state_mid2, 0x80, 0xBF);
      int tmp_state_mid3 = fsm.AddState();
      // Third byte.
      fsm.AddEdge(tmp_state_mid2, tmp_state_mid3, 0x80, 0xBF);
      // Last byte.
      fsm.AddEdge(tmp_state_mid3, to, 0x80, 0xBF);
    }
    return;
  }
  if (byte_max[2] != 0) {
    // 3 byte unicode.
    if (byte_max[2] == byte_min[2]) {
      int tmp_state = fsm.AddState();
      fsm.AddEdge(from, tmp_state, byte_min[2], byte_max[2]);
      min = (min & 0x00FFFF);
      max = (max & 0x00FFFF);
      AddSameLengthCharacterRange(fsm, tmp_state, to, min, max);
      return;
    }
    if ((min & 0x00FFFF) != 0x8080) {
      int tmp_state_min = fsm.AddState();
      fsm.AddEdge(from, tmp_state_min, byte_min[2], byte_min[2]);
      AddSameLengthCharacterRange(fsm, tmp_state_min, to, (min & 0x00FFFF), 0x00BFBF);
    } else {
      byte_min[2]--;
    }
    if ((max & 0x00FFFF) != 0xBFBF) {
      int tmp_state_max = fsm.AddState();
      fsm.AddEdge(from, tmp_state_max, byte_max[2], byte_max[2]);
      AddSameLengthCharacterRange(fsm, tmp_state_max, to, 0x008080, (max & 0x00FFFF));
    } else {
      byte_max[2]++;
    }
    if (byte_max[2] - byte_min[2] > 1) {
      int tmp_state_mid = fsm.AddState();
      // First byte.
      fsm.AddEdge(from, tmp_state_mid, byte_min[2] + 1, byte_max[2] - 1);
      int tmp_state_mid2 = fsm.AddState();
      // Second byte.
      fsm.AddEdge(tmp_state_mid, tmp_state_mid2, 0x80, 0xBF);
      // Last byte.
      fsm.AddEdge(tmp_state_mid2, to, 0x80, 0xBF);
    }
    return;
  }

  // 2 byte unicode.
  if (byte_max[1] == byte_min[1]) {
    int tmp_state = fsm.AddState();
    fsm.AddEdge(from, tmp_state, byte_min[1], byte_max[1]);
    min = (min & 0x00FF);
    max = (max & 0x00FF);
    AddSameLengthCharacterRange(fsm, tmp_state, to, min, max);
    return;
  }
  if ((min & 0x00FF) != 0x80) {
    int tmp_state_min = fsm.AddState();
    fsm.AddEdge(from, tmp_state_min, byte_min[1], byte_min[1]);
    AddSameLengthCharacterRange(fsm, tmp_state_min, to, (min & 0x00FF), 0x00BF);
  } else {
    byte_min[1]--;
  }
  if ((max & 0x00FF) != 0xBF) {
    int tmp_state_max = fsm.AddState();
    fsm.AddEdge(from, tmp_state_max, byte_max[1], byte_max[1]);
    AddSameLengthCharacterRange(fsm, tmp_state_max, to, 0x0080, (max & 0x00FF));
  } else {
    byte_max[1]++;
  }
  if (byte_max[1] - byte_min[1] > 1) {
    int tmp_state_mid = fsm.AddState();
    // First byte.
    fsm.AddEdge(from, tmp_state_mid, byte_min[1] + 1, byte_max[1] - 1);
    fsm.AddEdge(tmp_state_mid, to, 0x80, 0xBF);
  }
  return;
}

void AddPackedUTF8RangeEdges(FSM& fsm, int from, int to, uint32_t min, uint32_t max) {
  XGRAMMAR_CHECK(min <= max) << "Invalid character range: min (" << min << ") > max (" << max
                             << ")";
  // Ensure max and min are valid unicode value.
  if (max > kMax4BytesUnicode) {
    max = kMax4BytesUnicode;
  } else if (max > kMax3BytesUnicode) {
    if (max < kMin4BytesUnicode) {
      max = kMax3BytesUnicode;
    }
  } else if (max > kMax2BytesUnicode) {
    if (max < kMin3BytesUnicode) {
      max = kMax2BytesUnicode;
    }
  } else if (max < kMin2BytesUnicode && (max > kMax1ByteUnicode)) {
    max = kMax1ByteUnicode;
  }

  if (min > kMax4BytesUnicode) {
    min = kMax4BytesUnicode;
  } else if (min > kMax3BytesUnicode) {
    if (min < kMin4BytesUnicode) {
      min = kMin4BytesUnicode;
    }
  } else if (min > kMax2BytesUnicode) {
    if (min < kMin3BytesUnicode) {
      min = kMin3BytesUnicode;
    }
  } else if (min < kMin2BytesUnicode && (min > kMax1ByteUnicode)) {
    min = kMin2BytesUnicode;
  }

  // Step2. Divide the range into several ranges, which contain characters with different lengths.
  if (max <= kMax1ByteUnicode) {
    AddSameLengthCharacterRange(fsm, from, to, min, max);
    return;
  }
  if (max <= kMax2BytesUnicode) {
    if (min >= kMin2BytesUnicode) {
      AddSameLengthCharacterRange(fsm, from, to, min, max);
    } else {
      AddSameLengthCharacterRange(fsm, from, to, min, kMax1ByteUnicode);
      AddSameLengthCharacterRange(fsm, from, to, kMin2BytesUnicode, max);
    }
    return;
  }
  if (max <= kMax3BytesUnicode) {
    if (min >= kMin3BytesUnicode) {
      AddSameLengthCharacterRange(fsm, from, to, min, max);
    } else if (min >= kMin2BytesUnicode) {
      AddSameLengthCharacterRange(fsm, from, to, min, kMax2BytesUnicode);
      AddSameLengthCharacterRange(fsm, from, to, kMin3BytesUnicode, max);
    } else {
      AddSameLengthCharacterRange(fsm, from, to, min, kMax1ByteUnicode);
      AddSameLengthCharacterRange(fsm, from, to, kMin2BytesUnicode, kMax2BytesUnicode);
      AddSameLengthCharacterRange(fsm, from, to, kMin3BytesUnicode, max);
    }
    return;
  }
  XGRAMMAR_CHECK(max <= kMax4BytesUnicode);
  if (min >= kMin4BytesUnicode) {
    AddSameLengthCharacterRange(fsm, from, to, min, max);
  } else if (min >= kMin3BytesUnicode) {
    AddSameLengthCharacterRange(fsm, from, to, min, kMax3BytesUnicode);
    AddSameLengthCharacterRange(fsm, from, to, kMin4BytesUnicode, max);
  } else if (min >= kMin2BytesUnicode) {
    AddSameLengthCharacterRange(fsm, from, to, min, kMax2BytesUnicode);
    AddSameLengthCharacterRange(fsm, from, to, kMin3BytesUnicode, kMax3BytesUnicode);
    AddSameLengthCharacterRange(fsm, from, to, kMin4BytesUnicode, max);
  } else {
    AddSameLengthCharacterRange(fsm, from, to, min, kMax1ByteUnicode);
    AddSameLengthCharacterRange(fsm, from, to, kMin2BytesUnicode, kMax2BytesUnicode);
    AddSameLengthCharacterRange(fsm, from, to, kMin3BytesUnicode, kMax3BytesUnicode);
    AddSameLengthCharacterRange(fsm, from, to, kMin4BytesUnicode, max);
  }
  return;
}

std::string RewriteRegexDots(const std::string& pattern, bool dot_matches_newline) {
  if (dot_matches_newline) {
    return pattern;
  }
  std::string result;
  result.reserve(pattern.size());
  bool escaped = false;
  bool in_character_class = false;
  for (char c : pattern) {
    if (escaped) {
      result.push_back(c);
      escaped = false;
      continue;
    }
    if (c == '\\') {
      result.push_back(c);
      escaped = true;
    } else if (c == '[') {
      result.push_back(c);
      in_character_class = true;
    } else if (c == ']' && in_character_class) {
      result.push_back(c);
      in_character_class = false;
    } else if (c == '.' && !in_character_class) {
      result += "[^\\n]";
    } else {
      result.push_back(c);
    }
  }
  return result;
}

/******************** Codepoint range utilities ********************/

namespace {

constexpr uint32_t kMaxCodepoint = 0x10FFFF;

/*! \brief Bounded repetitions above this threshold are compiled into a repeat FSM edge (when a
 * GrammarBuilder is available) instead of being physically unrolled. Matches the unroll threshold
 * of the grammar-level RepetitionRangeExpander. */
constexpr int kLargeRepeatThreshold = 128;

/*! \brief Hard cap of the estimated state count when a bounded repetition has to be unrolled
 * because no GrammarBuilder is available. */
constexpr int64_t kMaxUnrolledRepeatStates = 100000;

using CodepointRange = std::pair<uint32_t, uint32_t>;

/*! \brief Sort the ranges and merge overlapping or adjacent ones. */
void NormalizeRanges(std::vector<CodepointRange>* ranges) {
  std::sort(ranges->begin(), ranges->end());
  std::vector<CodepointRange> result;
  for (const auto& range : *ranges) {
    if (!result.empty() && range.first <= result.back().second + 1 &&
        range.first >= result.back().first) {
      result.back().second = std::max(result.back().second, range.second);
    } else {
      result.push_back(range);
    }
  }
  *ranges = std::move(result);
}

/*! \brief Complement normalized ranges over the codepoint domain [0, kMaxCodepoint]. */
std::vector<CodepointRange> ComplementRanges(const std::vector<CodepointRange>& ranges) {
  std::vector<CodepointRange> result;
  uint32_t next = 0;
  for (const auto& range : ranges) {
    if (range.first > next) {
      result.push_back({next, range.first - 1});
    }
    if (range.second >= kMaxCodepoint) {
      return result;
    }
    next = std::max(next, range.second + 1);
  }
  result.push_back({next, kMaxCodepoint});
  return result;
}

/*! \brief Append the ASCII case-folded counterparts of every letter contained in the ranges. */
void FoldAsciiCaseRanges(std::vector<CodepointRange>* ranges) {
  size_t original_size = ranges->size();
  for (size_t i = 0; i < original_size; ++i) {
    uint32_t low = (*ranges)[i].first;
    uint32_t high = (*ranges)[i].second;
    uint32_t fold_low = std::max<uint32_t>(low, 'a');
    uint32_t fold_high = std::min<uint32_t>(high, 'z');
    if (fold_low <= fold_high) {
      ranges->push_back({fold_low - ('a' - 'A'), fold_high - ('a' - 'A')});
    }
    fold_low = std::max<uint32_t>(low, 'A');
    fold_high = std::min<uint32_t>(high, 'Z');
    if (fold_low <= fold_high) {
      ranges->push_back({fold_low + ('a' - 'A'), fold_high + ('a' - 'A')});
    }
  }
}

/*! \brief Add edges from `from` to `to` accepting the UTF-8 encoding of every codepoint in the
 * normalized ranges. Multi-byte characters get intermediate states. */
void AddCodepointRangesToFSM(
    FSM* fsm, int from, int to, const std::vector<CodepointRange>& ranges
) {
  for (const auto& [low, high] : ranges) {
    if (low <= kMax1ByteUnicode) {
      fsm->AddEdge(from, to, low, std::min<uint32_t>(high, kMax1ByteUnicode));
    }
    if (high > kMax1ByteUnicode) {
      uint32_t multi_byte_low = std::max<uint32_t>(low, kMax1ByteUnicode + 1);
      AddPackedUTF8RangeEdges(
          *fsm, from, to, CodepointToPackedUTF8(multi_byte_low), CodepointToPackedUTF8(high)
      );
    }
  }
}

/*! \brief One parsed regex escape (or literal): either a single codepoint, or a (possibly
 * negated) set of codepoint ranges for class escapes like \d, \D, \w, \W, \s, \S. */
struct RegexEscapeItem {
  std::vector<CodepointRange> ranges;
  bool negated = false;
  bool is_single = false;
  uint32_t codepoint = 0;
};

/*!
 * \brief Parse the escape sequence starting at regex[*pos] == '\\'. On success, *pos is advanced
 * past the escape sequence.
 * \param in_class Whether the escape appears inside a character class ([...]). Inside a class,
 * \b means the backspace character instead of a word boundary assertion.
 */
Result<RegexEscapeItem> ParseRegexEscape(const std::string& regex, size_t* pos, bool in_class) {
  XGRAMMAR_DCHECK(regex[*pos] == '\\');
  if (*pos + 1 >= regex.size()) {
    return ResultErr("Regex ends with a trailing backslash");
  }
  char escaped = regex[*pos + 1];
  *pos += 2;
  RegexEscapeItem item;
  auto single = [&](uint32_t codepoint) {
    item.is_single = true;
    item.codepoint = codepoint;
    return ResultOk(std::move(item));
  };
  switch (escaped) {
    case 'n':
      return single('\n');
    case 't':
      return single('\t');
    case 'r':
      return single('\r');
    case 'f':
      return single('\f');
    case 'v':
      return single('\v');
    case 'a':
      return single('\a');
    case 'e':
      return single(0x1B);
    case '0':
      return single(0);
    case 'b':
      if (in_class) {
        return single(0x08);
      }
      return ResultErr("Word boundary assertion \\b is not supported in regex");
    case 'B':
      return ResultErr("Word boundary assertion \\B is not supported in regex");
    case 'p':
    case 'P':
      return ResultErr("Unicode property escape \\p / \\P is not supported in regex");
    case 'k':
      return ResultErr("Backreference \\k is not supported in regex");
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
      return ResultErr("Backreference \\" + std::string(1, escaped) + " is not supported in regex");
    case 'd':
      item.ranges = {{'0', '9'}};
      return ResultOk(std::move(item));
    case 'D':
      item.ranges = {{'0', '9'}};
      item.negated = true;
      return ResultOk(std::move(item));
    case 'w':
      item.ranges = {{'0', '9'}, {'A', 'Z'}, {'_', '_'}, {'a', 'z'}};
      return ResultOk(std::move(item));
    case 'W':
      item.ranges = {{'0', '9'}, {'A', 'Z'}, {'_', '_'}, {'a', 'z'}};
      item.negated = true;
      return ResultOk(std::move(item));
    case 's':
      item.ranges = {{0x09, 0x0D}, {' ', ' '}};
      return ResultOk(std::move(item));
    case 'S':
      item.ranges = {{0x09, 0x0D}, {' ', ' '}};
      item.negated = true;
      return ResultOk(std::move(item));
    case 'x': {
      if (*pos + 1 >= regex.size()) {
        return ResultErr("\\x must be followed by two hexadecimal digits in regex");
      }
      int high_digit = HexCharToInt(regex[*pos]);
      int low_digit = HexCharToInt(regex[*pos + 1]);
      if (high_digit < 0 || low_digit < 0) {
        return ResultErr("\\x must be followed by two hexadecimal digits in regex");
      }
      *pos += 2;
      return single(high_digit * 16 + low_digit);
    }
    case 'u': {
      if (*pos < regex.size() && regex[*pos] == '{') {
        size_t close = regex.find('}', *pos + 1);
        if (close == std::string::npos || close == *pos + 1 || close > *pos + 7) {
          return ResultErr("\\u{...} must contain one to six hexadecimal digits in regex");
        }
        uint32_t codepoint = 0;
        for (size_t i = *pos + 1; i < close; ++i) {
          int digit = HexCharToInt(regex[i]);
          if (digit < 0) {
            return ResultErr("\\u{...} must contain one to six hexadecimal digits in regex");
          }
          codepoint = codepoint * 16 + digit;
        }
        if (codepoint > kMaxCodepoint) {
          return ResultErr("\\u{...} escape is beyond the Unicode range in regex");
        }
        *pos = close + 1;
        return single(codepoint);
      }
      if (*pos + 3 >= regex.size()) {
        return ResultErr("\\u must be followed by four hexadecimal digits in regex");
      }
      uint32_t codepoint = 0;
      for (size_t i = *pos; i < *pos + 4; ++i) {
        int digit = HexCharToInt(regex[i]);
        if (digit < 0) {
          return ResultErr("\\u must be followed by four hexadecimal digits in regex");
        }
        codepoint = codepoint * 16 + digit;
      }
      *pos += 4;
      return single(codepoint);
    }
    case 'c': {
      if (*pos >= regex.size() || !std::isalpha(static_cast<unsigned char>(regex[*pos]))) {
        return ResultErr("\\c must be followed by a letter in regex");
      }
      uint32_t codepoint = static_cast<unsigned char>(regex[*pos]) & 0x1F;
      *pos += 1;
      return single(codepoint);
    }
    default: {
      if (static_cast<unsigned char>(escaped) >= 0x80) {
        // Multi-byte UTF-8 character after the backslash: match it literally.
        *pos -= 1;
        auto [codepoint, num_bytes] = ParseNextUTF8(regex.c_str() + *pos);
        if (codepoint == CharHandlingError::kInvalidUTF8 || *pos + num_bytes > regex.size()) {
          return ResultErr("Invalid UTF-8 in regex escape sequence");
        }
        *pos += num_bytes;
        return single(static_cast<uint32_t>(codepoint));
      }
      if (std::isalnum(static_cast<unsigned char>(escaped))) {
        XGRAMMAR_LOG(WARNING) << "Escape sequence \\" << escaped
                              << " is not recognized in regex; matching the character literally";
      }
      return single(static_cast<unsigned char>(escaped));
    }
  }
}

/*!
 * \brief Parse a character class leaf "[...]" into the final set of accepted codepoint ranges.
 * ASCII case folding (if requested) is applied before negation.
 */
Result<std::vector<CodepointRange>> ParseCharacterClassLeaf(
    const std::string& regex, bool case_insensitive
) {
  XGRAMMAR_DCHECK(regex.size() >= 2 && regex.front() == '[' && regex.back() == ']');
  size_t pos = 1;
  size_t content_end = regex.size() - 1;
  bool negated = false;
  if (pos < content_end && regex[pos] == '^') {
    negated = true;
    ++pos;
  }
  if (pos == content_end) {
    return ResultErr("Empty character class " + regex + " is not allowed in regex");
  }

  // Parse one class unit: an escape sequence or a literal (possibly multi-byte) character.
  auto parse_unit = [&]() -> Result<RegexEscapeItem> {
    if (regex[pos] == '\\') {
      return ParseRegexEscape(regex, &pos, /*in_class=*/true);
    }
    auto [codepoint, num_bytes] = ParseNextUTF8(regex.c_str() + pos);
    if (codepoint == CharHandlingError::kInvalidUTF8 || pos + num_bytes > content_end) {
      return ResultErr("Invalid UTF-8 in regex character class " + regex);
    }
    pos += num_bytes;
    RegexEscapeItem item;
    item.is_single = true;
    item.codepoint = static_cast<uint32_t>(codepoint);
    return ResultOk(std::move(item));
  };

  std::vector<CodepointRange> ranges;
  while (pos < content_end) {
    auto unit_result = parse_unit();
    if (unit_result.IsErr()) {
      return ResultErr(std::move(unit_result).UnwrapErr());
    }
    auto unit = std::move(unit_result).Unwrap();
    if (!unit.is_single) {
      // Class escapes like \d cannot be a range endpoint; a following '-' is literal.
      if (unit.negated) {
        NormalizeRanges(&unit.ranges);
        auto complement = ComplementRanges(unit.ranges);
        ranges.insert(ranges.end(), complement.begin(), complement.end());
      } else {
        ranges.insert(ranges.end(), unit.ranges.begin(), unit.ranges.end());
      }
      continue;
    }
    if (pos < content_end && regex[pos] == '-' && pos + 1 < content_end) {
      ++pos;
      auto high_result = parse_unit();
      if (high_result.IsErr()) {
        return ResultErr(std::move(high_result).UnwrapErr());
      }
      auto high_unit = std::move(high_result).Unwrap();
      if (!high_unit.is_single) {
        return ResultErr("Invalid character range endpoint in regex character class " + regex);
      }
      if (high_unit.codepoint < unit.codepoint) {
        return ResultErr(
            "Invalid character range (lower bound exceeds upper bound) in regex character class " +
            regex
        );
      }
      ranges.push_back({unit.codepoint, high_unit.codepoint});
    } else {
      ranges.push_back({unit.codepoint, unit.codepoint});
    }
  }

  if (case_insensitive) {
    FoldAsciiCaseRanges(&ranges);
  }
  NormalizeRanges(&ranges);
  if (negated) {
    ranges = ComplementRanges(ranges);
  }
  return ResultOk(std::move(ranges));
}

}  // namespace

/******************** RegexIR ********************/

class RegexIR {
 public:
  struct Leaf;

  struct Symbol;

  struct Union;

  struct Bracket;

  struct Repeat;

  struct RuleRefNode;

  struct RepeatSubrule;

  static constexpr int kRepeatNoUpperBound = -1;

  using State = std::variant<Leaf, Symbol, Union, Bracket, Repeat, RuleRefNode, RepeatSubrule>;

  // This struct is used to store one atom of the regex: the empty string (regex == ""), a
  // character class (regex == "[...]"), or a short sequence of literal characters / escapes.
  struct Leaf {
    std::string regex;
  };

  // This struct is used to store the symbol in regex, i.e.
  // +, *, ?
  enum class RegexSymbol {
    star,
    plus,
    optional,
  };

  struct Bracket {
    std::vector<State> states;
  };

  struct Symbol {
    RegexSymbol symbol;
    std::vector<State> state;
  };

  // This struct is used to represent a union symbol.
  struct Union {
    std::vector<State> states;
  };

  struct Repeat {
    std::vector<State> states;
    int lower_bound = 0;
    int upper_bound = 0;
  };

  // A reference to a grammar rule, compiled into a kRuleRef FSM edge.
  struct RuleRefNode {
    int32_t rule_id;
  };

  // A bounded repetition of a grammar rule, compiled into a kRepeatRef FSM edge. The referenced
  // rule holds the repeated sub-pattern; the Earley parser executes the repetition with a
  // counter at runtime, so no FSM unrolling happens.
  struct RepeatSubrule {
    int32_t rule_id;
    int lower_bound = 0;
    int upper_bound = 0;
  };

  // The top-level sequence of the regex.
  std::vector<State> states;

  // Whether matching is ASCII case-insensitive (enabled by a leading "(?i)").
  bool case_insensitive = false;

  /*!
    \brief Constructs a NFA from the regex IR.
  */
  Result<FSMWithStartEnd> Build() const;

  /*!
    \brief the visit function for the variant.
  */
  Result<FSMWithStartEnd> visit(const Leaf& state) const;

  Result<FSMWithStartEnd> visit(const Symbol& state) const;

  Result<FSMWithStartEnd> visit(const Union& state) const;

  Result<FSMWithStartEnd> visit(const Bracket& state) const;

  Result<FSMWithStartEnd> visit(const Repeat& state) const;

  Result<FSMWithStartEnd> visit(const RuleRefNode& state) const;

  Result<FSMWithStartEnd> visit(const RepeatSubrule& state) const;

  /*! \brief Whether the IR node can match the empty string. Purely syntactic; no FSM is built. */
  static bool IsNullable(const State& state);

  /*! \brief Whether a sequence of IR nodes can match the empty string. */
  static bool IsNullableSequence(const std::vector<State>& states);

  /*!
   * \brief Check repeat in regex. i.e {...} and {...,...}
   * \param regex The regex string.
   * \param start The start position of the repeat. i.e. regex[start] == '{'.
   * After the function, start will be the position of '}'.
   * \return The repeat range.
   */
  static Result<std::pair<int, int>> CheckRepeat(const std::string& regex, int& start);

 private:
  /*!
   * \brief Construct a FSM from a regex leaf.
   * \details The leaf is the empty string, a character class like [a-c0-9], or a sequence of
   * literal characters / escapes like "ab\n". Any symbols like "a|b" or "a*b" are not supported.
   * \param regex The regex string.
   * \return The FSM with start and end states.
   */
  Result<FSMWithStartEnd> BuildLeafFSMFromRegex(const std::string& regex) const;

  /*!
   * \brief Add the transition(s) accepting a single codepoint (with ASCII case folding when
   * case_insensitive is set) from `current` to a new state, and return the new state.
   */
  int AddSingleCodepoint(FSMWithStartEnd& result, int current, uint32_t codepoint) const;
};

Result<std::pair<int, int>> RegexIR::CheckRepeat(const std::string& regex, int& start) {
  // 10^9 fits in an int; longer counts would overflow.
  constexpr size_t kMaxRepeatDigits = 9;
  if (regex[start] != '{') {
    return ResultErr("Invalid repetition: expected '{'");
  }
  int lower_bound = 0;
  int upper_bound = RegexIR::kRepeatNoUpperBound;
  std::string num_str;
  XGRAMMAR_DCHECK(regex[start] == '{');
  start++;
  while (static_cast<size_t>(start) < regex.size() && regex[start] == ' ') {
    start++;
  }
  while (static_cast<size_t>(start) < regex.size() && std::isdigit(regex[start])) {
    num_str += regex[start];
    start++;
  }
  if (num_str.empty()) {
    return ResultErr("Invalid repetition count: expected a number after '{'");
  }
  if (num_str.size() > kMaxRepeatDigits) {
    return ResultErr("Invalid repetition count: the count " + num_str + " is too large");
  }
  lower_bound = std::stoi(num_str);
  while (static_cast<size_t>(start) < regex.size() && regex[start] == ' ') {
    start++;
  }
  // The format is {n}
  if (regex[start] == '}') {
    upper_bound = lower_bound;
    return ResultOk(std::make_pair(lower_bound, upper_bound));
  }
  if (regex[start] != ',') {
    return ResultErr("Invalid repetition count: expected ',' or '}' after the lower bound");
  }
  XGRAMMAR_DCHECK(regex[start] == ',');
  start++;
  while (static_cast<size_t>(start) < regex.size() && regex[start] == ' ') {
    start++;
  }
  // The format is {n,}
  if (regex[start] == '}') {
    return ResultOk(std::make_pair(lower_bound, upper_bound));
  }
  num_str.clear();
  while (static_cast<size_t>(start) < regex.size() && std::isdigit(regex[start])) {
    num_str += regex[start];
    start++;
  }
  if (num_str.empty()) {
    return ResultErr("Invalid repetition count: expected a number or '}' after ','");
  }
  if (num_str.size() > kMaxRepeatDigits) {
    return ResultErr("Invalid repetition count: the count " + num_str + " is too large");
  }
  upper_bound = std::stoi(num_str);
  if (upper_bound < lower_bound) {
    return ResultErr(
        "Invalid repetition count: the lower bound " + std::to_string(lower_bound) +
        " is larger than the upper bound " + std::to_string(upper_bound)
    );
  }
  while (static_cast<size_t>(start) < regex.size() && regex[start] == ' ') {
    start++;
  }
  if (regex[start] != '}') {
    return ResultErr("Invalid repetition count: expected '}' after the upper bound");
  }
  XGRAMMAR_DCHECK(regex[start] == '}');
  return ResultOk(std::make_pair(lower_bound, upper_bound));
}

bool RegexIR::IsNullableSequence(const std::vector<State>& states) {
  return std::all_of(states.begin(), states.end(), [](const State& state) {
    return IsNullable(state);
  });
}

bool RegexIR::IsNullable(const State& state) {
  return std::visit(
      [](const auto& node) -> bool {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, Leaf>) {
          return node.regex.empty();
        } else if constexpr (std::is_same_v<T, Symbol>) {
          return node.symbol != RegexSymbol::plus || IsNullableSequence(node.state);
        } else if constexpr (std::is_same_v<T, Union>) {
          return std::any_of(node.states.begin(), node.states.end(), [](const State& child) {
            return IsNullable(child);
          });
        } else if constexpr (std::is_same_v<T, Bracket>) {
          return IsNullableSequence(node.states);
        } else if constexpr (std::is_same_v<T, Repeat>) {
          return node.lower_bound == 0 || IsNullableSequence(node.states);
        } else if constexpr (std::is_same_v<T, RuleRefNode>) {
          return false;
        } else {
          static_assert(std::is_same_v<T, RepeatSubrule>);
          return node.lower_bound == 0;
        }
      },
      state
  );
}

Result<FSMWithStartEnd> RegexIR::Build() const {
  if (states.empty()) {
    FSM empty_fsm(1);
    FSMWithStartEnd result(empty_fsm, 0, {0}, false);
    return ResultOk(std::move(result));
  }
  std::vector<FSMWithStartEnd> fsm_list;
  for (const auto& state : states) {
    auto visited = std::visit([&](auto&& arg) { return visit(arg); }, state);
    if (visited.IsErr()) {
      return visited;
    }
    fsm_list.push_back(std::move(visited).Unwrap());
  }
  if (fsm_list.size() > 1) {
    return ResultOk(FSMWithStartEnd::Concat(fsm_list));
  } else {
    // If there is only one FSM, return it directly.
    return ResultOk(std::move(fsm_list[0]));
  }
}

Result<FSMWithStartEnd> RegexIR::visit(const RegexIR::Leaf& state) const {
  return BuildLeafFSMFromRegex(state.regex);
}

Result<FSMWithStartEnd> RegexIR::visit(const RegexIR::Union& state) const {
  std::vector<FSMWithStartEnd> fsm_list;
  for (const auto& child : state.states) {
    auto visited = std::visit([&](auto&& arg) { return RegexIR::visit(arg); }, child);
    if (visited.IsErr()) {
      return visited;
    }
    fsm_list.push_back(std::move(visited).Unwrap());
  }
  if (fsm_list.size() <= 1) {
    return ResultErr("Internal error: a union node in the regex IR has fewer than two branches");
  }
  return ResultOk(FSMWithStartEnd::Union(fsm_list));
}

Result<FSMWithStartEnd> RegexIR::visit(const RegexIR::Symbol& state) const {
  if (state.state.size() != 1) {
    return ResultErr("Internal error: a quantifier node in the regex IR must hold exactly one child"
    );
  }
  Result<FSMWithStartEnd> child_result =
      std::visit([&](auto&& arg) { return RegexIR::visit(arg); }, state.state[0]);
  if (child_result.IsErr()) {
    return child_result;
  }
  auto child = std::move(child_result).Unwrap();

  switch (state.symbol) {
    case RegexIR::RegexSymbol::plus: {
      return ResultOk(child.Plus());
    }
    case RegexIR::RegexSymbol::star: {
      return ResultOk(child.Star());
    }
    case RegexIR::RegexSymbol::optional: {
      return ResultOk(child.Optional());
    }
    default: {
      XGRAMMAR_LOG(FATAL) << "Unknown regex symbol: " << static_cast<int>(state.symbol);
      XGRAMMAR_UNREACHABLE();
    }
  }
}

Result<FSMWithStartEnd> RegexIR::visit(const RegexIR::Bracket& state) const {
  if (state.states.empty()) {
    // An empty group or an empty union branch matches the empty string.
    FSM empty_fsm(1);
    return ResultOk(FSMWithStartEnd(empty_fsm, 0, {0}, false));
  }
  std::vector<FSMWithStartEnd> fsm_list;
  for (const auto& child : state.states) {
    auto visited = std::visit([&](auto&& arg) { return RegexIR::visit(arg); }, child);
    if (visited.IsErr()) {
      return visited;
    }
    fsm_list.push_back(std::move(visited).Unwrap());
  }
  return ResultOk(FSMWithStartEnd::Concat(fsm_list));
}

Result<FSMWithStartEnd> RegexIR::visit(const RegexIR::RuleRefNode& state) const {
  FSM fsm(2);
  fsm.AddRuleEdge(0, 1, state.rule_id);
  return ResultOk(FSMWithStartEnd(fsm, 0, {1}, false));
}

Result<FSMWithStartEnd> RegexIR::visit(const RegexIR::RepeatSubrule& state) const {
  // The state holding the repeat edge is padded with epsilon transitions on both sides, so that
  // FSM compositions (Concat / Star / Optional / ...) never add another outgoing edge to it: a
  // state with a kRepeatRef edge must have no other outgoing edges.
  FSM fsm(4);
  fsm.AddEpsilonEdge(0, 1);
  fsm.AddRepeatEdge(1, 2, state.rule_id, state.lower_bound, state.upper_bound);
  fsm.AddEpsilonEdge(2, 3);
  return ResultOk(FSMWithStartEnd(fsm, 0, {3}, false));
}

Result<FSMWithStartEnd> RegexIR::visit(const RegexIR::Repeat& state) const {
  if (state.states.size() != 1) {
    return ResultErr("Internal error: a repetition node in the regex IR must hold exactly one child"
    );
  }
  bool has_upper_bound = state.upper_bound != RegexIR::kRepeatNoUpperBound;
  if (has_upper_bound && state.upper_bound == 0) {
    // {0} / {0,0}: matches exactly the empty string. The general path below cannot express
    // this: it starts from one copy of the child whose end states stay accepting.
    FSM empty_fsm(1);
    return ResultOk(FSMWithStartEnd(empty_fsm, 0, {0}, false));
  }
  Result<FSMWithStartEnd> child_result =
      std::visit([&](auto&& arg) { return RegexIR::visit(arg); }, state.states[0]);
  if (child_result.IsErr()) {
    return child_result;
  }
  FSMWithStartEnd child = std::move(child_result).Unwrap();

  // Guard against FSM state explosion when the repetition has to be physically unrolled. When
  // a GrammarBuilder is available, large repetitions are compiled into repeat edges instead and
  // never reach this point.
  int64_t num_copies = has_upper_bound ? state.upper_bound : std::max(state.lower_bound, 1);
  if (static_cast<int64_t>(child.NumStates()) * num_copies > kMaxUnrolledRepeatStates) {
    return ResultErr(
        "The bounded repetition {" + std::to_string(state.lower_bound) + "," +
        (has_upper_bound ? std::to_string(state.upper_bound) : "") +
        "} is too large to compile into a FSM"
    );
  }

  FSMWithStartEnd result = child.Copy();
  std::unordered_set<int> new_ends;

  if (state.lower_bound <= 1 && (!has_upper_bound || state.upper_bound >= 1)) {
    // A single copy is accepting when the lower bound is at most 1.
    for (int end = 0; end < result.NumStates(); ++end) {
      if (result.IsEndState(end)) {
        new_ends.insert(end);
      }
    }
  }

  // Add a fresh accepting start state so that zero repetitions match. A fresh state is
  // required: making the original start accepting would also accept strings that merely
  // loop back to the start inside the first copy.
  auto allow_zero_repetitions = [](FSMWithStartEnd* fsm) {
    int new_start = fsm->AddState();
    fsm->GetFsm().AddEpsilonEdge(new_start, fsm->GetStart());
    fsm->SetStartState(new_start);
    fsm->AddEndState(new_start);
  };

  // Handling {n,}
  if (!has_upper_bound) {
    for (int i = 2; i < state.lower_bound; i++) {
      result = FSMWithStartEnd::Concat(std::vector<FSMWithStartEnd>{result, child});
    }
    int end_state_of_lower_bound_fsm = -1;
    for (int end = 0; end < result.NumStates(); ++end) {
      if (result.IsEndState(end)) {
        end_state_of_lower_bound_fsm = end;
        break;
      }
    }
    XGRAMMAR_DCHECK(end_state_of_lower_bound_fsm != -1)
        << "No end state found in the lower bound FSM.";
    result = FSMWithStartEnd::Concat(std::vector<FSMWithStartEnd>{result, child});
    for (int end = 0; end < result.NumStates(); ++end) {
      if (result.IsEndState(end)) {
        result.GetFsm().AddEpsilonEdge(end, end_state_of_lower_bound_fsm);
      }
    }
    for (const auto& end : new_ends) {
      result.AddEndState(end);
    }
    if (state.lower_bound == 0) {
      allow_zero_repetitions(&result);
    }
    return ResultOk(std::move(result));
  }
  // Handling {n, m} or {n}
  for (int i = 2; i <= state.upper_bound; i++) {
    result = FSMWithStartEnd::Concat(std::vector<FSMWithStartEnd>{result, child});
    if (i >= state.lower_bound) {
      for (int end = 0; end < result.NumStates(); ++end) {
        if (result.IsEndState(end)) {
          new_ends.insert(end);
        }
      }
    }
  }
  for (const auto& end : new_ends) {
    result.AddEndState(end);
  }
  if (state.lower_bound == 0) {
    allow_zero_repetitions(&result);
  }
  return ResultOk(std::move(result));
}

int RegexIR::AddSingleCodepoint(FSMWithStartEnd& result, int current, uint32_t codepoint) const {
  int next = result.AddState();
  if (codepoint <= kMax1ByteUnicode) {
    result.GetFsm().AddEdge(current, next, codepoint, codepoint);
    if (case_insensitive) {
      if (codepoint >= 'a' && codepoint <= 'z') {
        uint32_t upper = codepoint - ('a' - 'A');
        result.GetFsm().AddEdge(current, next, upper, upper);
      } else if (codepoint >= 'A' && codepoint <= 'Z') {
        uint32_t lower = codepoint + ('a' - 'A');
        result.GetFsm().AddEdge(current, next, lower, lower);
      }
    }
    return next;
  }
  std::string utf8_bytes = CharToUTF8(static_cast<TCodepoint>(codepoint));
  int state = current;
  for (size_t i = 0; i < utf8_bytes.size(); ++i) {
    int target = (i + 1 == utf8_bytes.size()) ? next : result.AddState();
    uint8_t byte = static_cast<uint8_t>(utf8_bytes[i]);
    result.GetFsm().AddEdge(state, target, byte, byte);
    state = target;
  }
  return next;
}

Result<FSMWithStartEnd> RegexIR::BuildLeafFSMFromRegex(const std::string& regex) const {
  FSM initial_fsm(1);
  FSMWithStartEnd result(initial_fsm, 0, {}, false);
  if (regex.empty()) {
    // The empty leaf matches the empty string.
    result.AddEndState(0);
    return ResultOk(std::move(result));
  }
  if (regex[0] == '[') {
    // Character class.
    auto ranges_result = ParseCharacterClassLeaf(regex, case_insensitive);
    if (ranges_result.IsErr()) {
      return ResultErr(std::move(ranges_result).UnwrapErr());
    }
    auto ranges = std::move(ranges_result).Unwrap();
    int end_state = result.AddState();
    AddCodepointRangesToFSM(&result.GetFsm(), 0, end_state, ranges);
    result.AddEndState(end_state);
    return ResultOk(std::move(result));
  }
  // Sequence of literal characters, '.' and escapes.
  int current = 0;
  size_t pos = 0;
  while (pos < regex.size()) {
    if (regex[pos] == '.') {
      ++pos;
      int next = result.AddState();
      AddCodepointRangesToFSM(&result.GetFsm(), current, next, {{0, kMaxCodepoint}});
      current = next;
      continue;
    }
    if (regex[pos] == '\\') {
      auto item_result = ParseRegexEscape(regex, &pos, /*in_class=*/false);
      if (item_result.IsErr()) {
        return ResultErr(std::move(item_result).UnwrapErr());
      }
      auto item = std::move(item_result).Unwrap();
      if (item.is_single) {
        current = AddSingleCodepoint(result, current, item.codepoint);
      } else {
        auto ranges = std::move(item.ranges);
        if (case_insensitive) {
          FoldAsciiCaseRanges(&ranges);
        }
        NormalizeRanges(&ranges);
        if (item.negated) {
          ranges = ComplementRanges(ranges);
        }
        int next = result.AddState();
        AddCodepointRangesToFSM(&result.GetFsm(), current, next, ranges);
        current = next;
      }
      continue;
    }
    auto [codepoint, num_bytes] = ParseNextUTF8(regex.c_str() + pos);
    if (codepoint == CharHandlingError::kInvalidUTF8 || pos + num_bytes > regex.size()) {
      // Be permissive with non-UTF-8 patterns: match the raw byte.
      int next = result.AddState();
      uint8_t byte = static_cast<uint8_t>(regex[pos]);
      result.GetFsm().AddEdge(current, next, byte, byte);
      current = next;
      ++pos;
      continue;
    }
    current = AddSingleCodepoint(result, current, static_cast<uint32_t>(codepoint));
    pos += num_bytes;
  }
  result.AddEndState(current);
  return ResultOk(std::move(result));
}

/******************** RegexFSMBuilder ********************/

namespace {

/*! \brief One entry of the regex parsing stack: an IR node or a marker character ('(' or '|'),
 * together with the source span [span_begin, span_end) of the atom in the regex string. */
struct RegexStackEntry {
  std::variant<RegexIR::State, char> item;
  int span_begin = 0;
  int span_end = 0;
};

/*! \brief Skip a character class starting at regex[pos] == '['. Returns the position right after
 * the closing ']', or std::string::npos if the class is not closed. */
size_t SkipCharacterClass(const std::string& regex, size_t pos) {
  XGRAMMAR_DCHECK(regex[pos] == '[');
  ++pos;
  if (pos < regex.size() && regex[pos] == '^') {
    ++pos;
  }
  while (pos < regex.size()) {
    if (regex[pos] == '\\') {
      pos += 2;
      continue;
    }
    if (regex[pos] == ']') {
      return pos + 1;
    }
    ++pos;
  }
  return std::string::npos;
}

/*!
 * \brief Parse a regex string into a RegexIR.
 * \param regex_with_flags The regex. A leading "(?i)" enables ASCII case-insensitive matching.
 * \param builder If not null, large bounded repetitions are compiled into subrules added through
 * this builder; see RegexFSMBuilder::Build.
 * \param rule_hint Name hint for the created subrules.
 */
Result<RegexIR> ParseRegexToIR(
    const std::string& regex_with_flags, GrammarBuilder* builder, const std::string& rule_hint
) {
  RegexIR ir;
  std::string regex = regex_with_flags;
  int flag_prefix_length = 0;
  if (regex.size() >= 4 && regex.compare(0, 4, "(?i)") == 0) {
    ir.case_insensitive = true;
    regex = regex.substr(4);
    flag_prefix_length = 4;
  }

  // Mirror the error format of the RegexConverter path: a 1-based position in the pattern
  // (including the "(?i)" prefix when present), followed by the description.
  auto error_at = [&](int pos, const std::string& message) {
    return "Regex parsing error at position " + std::to_string(pos + flag_prefix_length + 1) +
           ": " + message;
  };

  std::vector<RegexStackEntry> stack;
  int size = static_cast<int>(regex.size());
  for (int i = 0; i < size; i++) {
    char current_char = regex[i];
    // Handle anchors.
    if (current_char == '^' || current_char == '$') {
      if (!((current_char == '^' && i == 0) || (current_char == '$' && i == size - 1))) {
        XGRAMMAR_LOG(WARNING) << "Anchor '" << current_char
                              << "' in the middle of regex is ignored: " << regex;
      }
      continue;
    }
    // Handle the character class.
    if (current_char == '[') {
      size_t class_end = SkipCharacterClass(regex, i);
      if (class_end == std::string::npos) {
        return ResultErr(error_at(i, "Unclosed '['"));
      }
      size_t content_begin = i + 1;
      if (content_begin < regex.size() && regex[content_begin] == '^') {
        ++content_begin;
      }
      if (content_begin + 1 == class_end) {
        return ResultErr(error_at(i, "Empty character class is not allowed in regex"));
      }
      RegexIR::Leaf leaf;
      leaf.regex = regex.substr(i, class_end - i);
      stack.push_back({leaf, i, static_cast<int>(class_end)});
      i = static_cast<int>(class_end) - 1;
      continue;
    }
    if (current_char == ']') {
      return ResultErr(error_at(i, "Unmatched ']'"));
    }
    // Handle quantifiers.
    if (current_char == '+' || current_char == '*' || current_char == '?') {
      if (stack.empty() || std::holds_alternative<char>(stack.back().item)) {
        return ResultErr(
            error_at(i, std::string("There is nothing to repeat before '") + current_char + "'")
        );
      }
      RegexStackEntry atom = std::move(stack.back());
      stack.pop_back();
      RegexIR::Symbol symbol;
      symbol.state.push_back(std::move(std::get<RegexIR::State>(atom.item)));
      switch (current_char) {
        case '+': {
          symbol.symbol = RegexIR::RegexSymbol::plus;
          break;
        }
        case '*': {
          symbol.symbol = RegexIR::RegexSymbol::star;
          break;
        }
        case '?': {
          symbol.symbol = RegexIR::RegexSymbol::optional;
          break;
        }
      }
      // Skip the non-greedy modifier: greedy and non-greedy quantifiers accept the same
      // language, and the Earley parser explores all derivations anyway.
      if (i + 1 < size && regex[i + 1] == '?') {
        i++;
      }
      stack.push_back({std::move(symbol), atom.span_begin, i + 1});
      continue;
    }
    // Handle groups and alternation.
    if (current_char == '(') {
      if (i + 1 < size && regex[i + 1] == '?') {
        if (i + 2 >= size) {
          return ResultErr(error_at(i, "Group modifier is not finished"));
        }
        char modifier = regex[i + 2];
        if (modifier == ':') {
          stack.push_back({'(', i, i + 1});
          i += 2;
          continue;
        }
        if (modifier == '=' || modifier == '!') {
          // Skip the whole lookahead group and treat it as the empty string.
          XGRAMMAR_LOG(WARNING) << "Lookahead assertion is not supported and is ignored in regex: "
                                << regex;
          int depth = 1;
          size_t j = i + 3;
          while (j < regex.size() && depth > 0) {
            if (regex[j] == '\\') {
              j += 2;
              continue;
            }
            if (regex[j] == '[') {
              size_t class_begin = j;
              j = SkipCharacterClass(regex, j);
              if (j == std::string::npos) {
                return ResultErr(error_at(static_cast<int>(class_begin), "Unclosed '['"));
              }
              continue;
            }
            if (regex[j] == '(') {
              depth++;
            } else if (regex[j] == ')') {
              depth--;
            }
            j++;
          }
          if (depth != 0) {
            return ResultErr(error_at(i, "The parenthesis is not closed"));
          }
          stack.push_back({RegexIR::Leaf{""}, i, static_cast<int>(j)});
          i = static_cast<int>(j) - 1;
          continue;
        }
        if (modifier == '<' || (modifier == 'P' && i + 3 < size && regex[i + 3] == '<')) {
          size_t name_begin = (modifier == '<') ? i + 3 : i + 4;
          if (name_begin < regex.size() && (regex[name_begin] == '=' || regex[name_begin] == '!')) {
            return ResultErr(error_at(i, "Lookbehind assertion is not supported in regex"));
          }
          size_t j = name_begin;
          while (j < regex.size() &&
                 (std::isalnum(static_cast<unsigned char>(regex[j])) || regex[j] == '_')) {
            j++;
          }
          if (j == name_begin || j >= regex.size() || regex[j] != '>') {
            return ResultErr(error_at(i, "Invalid named capturing group"));
          }
          // Ignore the group name and compile the content as a normal group.
          stack.push_back({'(', i, i + 1});
          i = static_cast<int>(j);
          continue;
        }
        return ResultErr(
            error_at(i, "Unsupported group modifier '(?" + std::string(1, modifier) + "'")
        );
      }
      stack.push_back({'(', i, i + 1});
      continue;
    }
    if (current_char == '|') {
      stack.push_back({'|', i, i + 1});
      continue;
    }
    if (current_char == ')') {
      std::vector<RegexStackEntry> popped;
      bool paired = false;
      bool unioned = false;
      int group_begin = 0;
      while (!stack.empty()) {
        RegexStackEntry entry = std::move(stack.back());
        stack.pop_back();
        if (std::holds_alternative<char>(entry.item)) {
          char marker = std::get<char>(entry.item);
          if (marker == '(') {
            paired = true;
            group_begin = entry.span_begin;
            break;
          }
          XGRAMMAR_DCHECK(marker == '|');
          unioned = true;
        }
        popped.push_back(std::move(entry));
      }
      if (!paired) {
        return ResultErr(error_at(i, "Unmatched ')'"));
      }
      // `popped` stores the group content from right to left.
      if (!unioned) {
        RegexIR::Bracket bracket;
        for (auto it = popped.rbegin(); it != popped.rend(); ++it) {
          bracket.states.push_back(std::move(std::get<RegexIR::State>(it->item)));
        }
        // An empty bracket (e.g. "()") matches the empty string.
        stack.push_back({std::move(bracket), group_begin, i + 1});
      } else {
        RegexIR::Union union_state;
        RegexIR::Bracket bracket;
        for (auto it = popped.rbegin(); it != popped.rend(); ++it) {
          if (std::holds_alternative<char>(it->item)) {
            XGRAMMAR_DCHECK(std::get<char>(it->item) == '|');
            // An empty bracket represents an empty alternative, e.g. "(a|)".
            union_state.states.push_back(std::move(bracket));
            bracket = RegexIR::Bracket();
            continue;
          }
          bracket.states.push_back(std::move(std::get<RegexIR::State>(it->item)));
        }
        union_state.states.push_back(std::move(bracket));
        stack.push_back({std::move(union_state), group_begin, i + 1});
      }
      continue;
    }
    // Handle repetitions.
    if (current_char == '{') {
      if (stack.empty() || std::holds_alternative<char>(stack.back().item)) {
        return ResultErr(error_at(i, "There is nothing to repeat before the repetition"));
      }
      RegexStackEntry atom = std::move(stack.back());
      stack.pop_back();
      int repeat_begin = i;
      auto bounds_result = RegexIR::CheckRepeat(regex, i);
      if (bounds_result.IsErr()) {
        return ResultErr(error_at(repeat_begin, std::move(bounds_result).UnwrapErr().what()));
      }
      auto [lower_bound, upper_bound] = std::move(bounds_result).Unwrap();
      // Skip the non-greedy modifier.
      if (i + 1 < size && regex[i + 1] == '?') {
        i++;
      }
      bool is_large_repeat = upper_bound == RegexIR::kRepeatNoUpperBound
                                 ? lower_bound > kLargeRepeatThreshold
                                 : upper_bound > kLargeRepeatThreshold;
      if (is_large_repeat && builder != nullptr) {
        // Compile the repeated sub-pattern into a new rule (with a kRegex body), and represent
        // the repetition as a kRepeatRef FSM edge. The Earley parser executes it with a counter
        // at runtime, so the FSM is not unrolled.
        const auto& atom_state = std::get<RegexIR::State>(atom.item);
        std::string inner_regex = regex.substr(atom.span_begin, atom.span_end - atom.span_begin);
        if (ir.case_insensitive) {
          inner_regex = "(?i)" + inner_regex;
        }
        if (RegexIR::IsNullable(atom_state)) {
          // Mirror RepetitionNormalizer: when the repeated element can match the empty string,
          // the repetition count cannot be enforced, so the lower bound is relaxed to 0.
          lower_bound = 0;
        }
        std::string name_hint = (rule_hint.empty() ? "regex" : rule_hint) + "_repeat";
        int32_t inner_rule_id = builder->AddRuleWithHint(name_hint, builder->AddRegex(inner_regex));
        builder->UpdateLookaheadExact(inner_rule_id, true);
        if (upper_bound == RegexIR::kRepeatNoUpperBound) {
          // {n,} == {n}{0,}: a repeat edge for the mandatory part, then a starred rule
          // reference.
          RegexIR::Symbol star_symbol;
          star_symbol.symbol = RegexIR::RegexSymbol::star;
          star_symbol.state.push_back(RegexIR::RuleRefNode{inner_rule_id});
          if (lower_bound > 0) {
            RegexIR::Bracket bracket;
            bracket.states.push_back(RegexIR::RepeatSubrule{inner_rule_id, lower_bound, lower_bound}
            );
            bracket.states.push_back(std::move(star_symbol));
            stack.push_back({std::move(bracket), atom.span_begin, i + 1});
          } else {
            stack.push_back({std::move(star_symbol), atom.span_begin, i + 1});
          }
        } else {
          stack.push_back(
              {RegexIR::RepeatSubrule{inner_rule_id, lower_bound, upper_bound},
               atom.span_begin,
               i + 1}
          );
        }
      } else {
        RegexIR::Repeat repeat;
        repeat.lower_bound = lower_bound;
        repeat.upper_bound = upper_bound;
        repeat.states.push_back(std::move(std::get<RegexIR::State>(atom.item)));
        stack.push_back({std::move(repeat), atom.span_begin, i + 1});
      }
      continue;
    }
    // Handle literal characters and escapes. Each leaf holds exactly one codepoint or escape
    // sequence, so that a following quantifier applies to the whole character.
    RegexIR::Leaf leaf;
    if (current_char == '\\') {
      size_t escape_end = i;
      auto escape_result = ParseRegexEscape(regex, &escape_end, /*in_class=*/false);
      if (escape_result.IsErr()) {
        return ResultErr(error_at(i, std::move(escape_result).UnwrapErr().what()));
      }
      leaf.regex = regex.substr(i, escape_end - i);
      stack.push_back({std::move(leaf), i, static_cast<int>(escape_end)});
      i = static_cast<int>(escape_end) - 1;
      continue;
    }
    auto [codepoint, num_bytes] = ParseNextUTF8(regex.c_str() + i);
    if (codepoint == CharHandlingError::kInvalidUTF8 || i + num_bytes > size) {
      num_bytes = 1;
    }
    leaf.regex = regex.substr(i, num_bytes);
    stack.push_back({std::move(leaf), i, i + num_bytes});
    i += num_bytes - 1;
    continue;
  }

  // Assemble the top-level sequence / union. `stack` stores the content from left to right.
  std::vector<RegexIR::State> segment;
  std::vector<std::vector<RegexIR::State>> union_segments;
  bool unioned = false;
  for (auto& entry : stack) {
    if (std::holds_alternative<char>(entry.item)) {
      char marker = std::get<char>(entry.item);
      if (marker == '|') {
        union_segments.push_back(std::move(segment));
        segment.clear();
        unioned = true;
        continue;
      }
      return ResultErr(error_at(entry.span_begin, "The parenthesis is not closed"));
    }
    segment.push_back(std::move(std::get<RegexIR::State>(entry.item)));
  }
  if (!unioned) {
    ir.states = std::move(segment);
  } else {
    union_segments.push_back(std::move(segment));
    RegexIR::Union union_state;
    for (auto& branch : union_segments) {
      RegexIR::Bracket bracket;
      bracket.states = std::move(branch);
      union_state.states.push_back(std::move(bracket));
    }
    ir.states.push_back(std::move(union_state));
  }
  return ResultOk(std::move(ir));
}

}  // namespace

Result<FSMWithStartEnd> RegexFSMBuilder::Build(
    const std::string& regex, GrammarBuilder* builder, const std::string& rule_hint
) {
  auto ir_result = ParseRegexToIR(regex, builder, rule_hint);
  if (ir_result.IsErr()) {
    return ResultErr(std::move(ir_result).UnwrapErr());
  }
  return std::move(ir_result).Unwrap().Build();
}

Result<bool> RegexFSMBuilder::MatchesEmpty(const std::string& regex) {
  auto ir_result = ParseRegexToIR(regex, nullptr, "");
  if (ir_result.IsErr()) {
    return ResultErr(std::move(ir_result).UnwrapErr());
  }
  auto ir = std::move(ir_result).Unwrap();
  return ResultOk(RegexIR::IsNullableSequence(ir.states));
}

Result<FSMWithStartEnd> RegexFSMBuilder::BuildWithForbiddenChars(
    const std::string& regex,
    const std::bitset<256>& forbidden_chars,
    GrammarBuilder* builder,
    const std::string& rule_hint
) {
  auto build_result = Build(regex, builder, rule_hint);
  if (build_result.IsErr() || forbidden_chars.none()) {
    return build_result;
  }
  auto fsm_wse = std::move(build_result).Unwrap();
  const auto& fsm = fsm_wse.GetFsm();
  FSM new_fsm(fsm_wse.NumStates());
  for (int state = 0; state < fsm_wse.NumStates(); ++state) {
    for (const auto& edge : fsm.GetEdges(state)) {
      if (!edge.IsCharRange()) {
        new_fsm.AddEdge(state, edge.target, edge.min, edge.max);
        continue;
      }
      // Split the character range into the maximal sub-ranges of allowed characters.
      int range_start = -1;
      for (int c = edge.min; c <= edge.max + 1; ++c) {
        if (c <= edge.max && !forbidden_chars[c]) {
          if (range_start == -1) {
            range_start = c;
          }
        } else if (range_start != -1) {
          new_fsm.AddEdge(state, edge.target, range_start, c - 1);
          range_start = -1;
        }
      }
    }
  }
  new_fsm.SetEdgeAuxData(std::vector<int32_t>(fsm.GetEdgeAuxData()));
  return ResultOk(FSMWithStartEnd(new_fsm, fsm_wse.GetStart(), fsm_wse.GetEnds()));
}

class TrieFSMBuilderImpl {
 public:
  TrieFSMBuilderImpl() = default;
  std::optional<FSMWithStartEnd> Build(
      const std::vector<std::string>& patterns,
      const std::vector<std::string>& excluded_patterns,
      std::vector<int32_t>* end_states,
      bool allow_overlap,
      bool add_back_edges
  );
  void AddBackEdges(
      FSM* fsm,
      int start,
      const std::unordered_set<int>& ends,
      std::unordered_set<int32_t>* dead_states
  );
};

std::optional<FSMWithStartEnd> TrieFSMBuilderImpl::Build(
    const std::vector<std::string>& patterns,
    const std::vector<std::string>& excluded_patterns,
    std::vector<int32_t>* end_states,
    bool allow_overlap,
    bool add_back_edges
) {
  FSM fsm(1);
  int start = 0;
  std::unordered_set<int> ends;

  if (end_states) {
    end_states->clear();
  }

  for (const auto& pattern : patterns) {
    // Check for empty patterns
    if (!allow_overlap && pattern.empty()) {
      return std::nullopt;
    }

    int current_state = 0;
    for (const auto& ch : pattern) {
      int32_t ch_int32 = static_cast<int32_t>(static_cast<uint8_t>(ch));
      int next_state = fsm.GetNextState(current_state, ch_int32);
      if (next_state == FSM::kNoNextState) {
        next_state = fsm.AddState();
        fsm.AddEdge(current_state, next_state, ch_int32, ch_int32);
      }
      current_state = next_state;
      if (!allow_overlap && ends.count(current_state) > 0) {
        return std::nullopt;
      }
    }
    if (!allow_overlap && fsm.GetEdges(current_state).size() > 0) {
      return std::nullopt;
    }
    ends.insert(current_state);
    if (end_states) {
      end_states->push_back(current_state);
    }
  }

  std::unordered_set<int32_t> dead_state_set;

  if (add_back_edges) {
    // Build trie for excluded patterns.
    for (const auto& excluded_pattern : excluded_patterns) {
      if (!allow_overlap && excluded_pattern.empty()) {
        return std::nullopt;
      }

      int current_state = 0;
      for (const auto& ch : excluded_pattern) {
        int32_t ch_int32 = static_cast<int32_t>(static_cast<uint8_t>(ch));
        int next_state = fsm.GetNextState(current_state, ch_int32);
        if (next_state == FSM::kNoNextState) {
          next_state = fsm.AddState();
          fsm.AddEdge(current_state, next_state, ch_int32, ch_int32);
        }
        current_state = next_state;
        if (!allow_overlap && ends.count(current_state) > 0) {
          return std::nullopt;
        }
      }
      if (!allow_overlap && fsm.GetEdges(current_state).size() > 0) {
        return std::nullopt;
      }

      ends.insert(current_state);
      dead_state_set.insert(current_state);
    }

    // Add back edges.
    AddBackEdges(&fsm, start, ends, &dead_state_set);

    // Remove the edges to excluded end states.
    if (dead_state_set.size() != 0) {
      for (int state = 0; state < fsm.NumStates(); state++) {
        std::vector<FSMEdge>& edges = fsm.GetEdges(state);
        std::vector<FSMEdge> new_edges;
        for (const auto& edge : edges) {
          if (dead_state_set.count(edge.target) == 0) {
            new_edges.push_back(edge);
          }
        }
        edges = std::move(new_edges);
      }
    }
  } else if (excluded_patterns.size() > 0) {
    XGRAMMAR_LOG(WARNING) << "Excluded patterns are ignored when back edges are not added.";
  }

  return FSMWithStartEnd(fsm, start, std::vector<int32_t>(ends.begin(), ends.end()));
}

void TrieFSMBuilderImpl::AddBackEdges(
    FSM* fsm,
    int start,
    const std::unordered_set<int>& ends,
    std::unordered_set<int32_t>* dead_states
) {
  // Build an Aho-Corasick automaton by adding back edges.
  // When matching on the trie fails at state u on byte b, the matcher must resume from
  // the longest proper suffix of u's prefix that is still a path in the trie (the
  // failure state), and retry b from there. Falling back only to the start state (or to
  // the start state's direct children) loses matches whose start lies inside an
  // already-followed branch of another pattern. Example: patterns {"bcd", "abce"} on
  // input "abcd" -- after following "abc" of the "abce" branch, 'd' must transition to
  // the "bcd" end state via the failure state "bc", not back to the start state.

  int num_states = fsm->NumStates();

  // Step 1. Record the BFS order of the trie (a tree at this point), so that shallower
  // states are always processed first.
  std::vector<int> bfs_order;
  bfs_order.reserve(num_states);
  bfs_order.push_back(start);
  for (size_t head = 0; head < bfs_order.size(); head++) {
    for (const auto& edge : fsm->GetEdges(bfs_order[head])) {
      XGRAMMAR_DCHECK(edge.min == edge.max && edge.min >= 0 && edge.min <= 255);
      bfs_order.push_back(edge.target);
    }
  }
  XGRAMMAR_DCHECK(static_cast<int>(bfs_order.size()) == num_states);

  // Step 2. Compute the failure link and the fully resolved transition table with the
  // textbook O(num_states * 256) dynamic program: delta[u][b] is the trie child when it
  // exists, and delta[fail[u]][b] otherwise -- fail[u] is strictly shallower than u, so
  // its row is already final when u is processed in BFS order.
  std::vector<int> fail(num_states, start);
  std::vector<std::array<int, 256>> delta(num_states);
  for (auto& row : delta) {
    row.fill(FSM::kNoNextState);
  }
  for (int u = 0; u < num_states; u++) {
    for (const auto& edge : fsm->GetEdges(u)) {
      delta[u][edge.min] = edge.target;
    }
  }
  for (int u : bfs_order) {
    // An explicit trie child can complete an excluded suffix without entering its
    // terminal state: with excludes {"abc", "xabcq"}, the state for "xabc" has
    // failure state "abc". It must be dead too, even if more trie edges remain.
    // BFS ensures the failure state's excluded status has already been propagated.
    if (dead_states->count(fail[u]) > 0) {
      dead_states->insert(u);
    }
    for (int byte = 0; byte < 256; byte++) {
      // Entries of deeper states are untouched so far, so a non-empty entry here is
      // exactly a trie child of u.
      int child = delta[u][byte];
      int fallback = (u == start) ? start : delta[fail[u]][byte];
      if (child == FSM::kNoNextState) {
        delta[u][byte] = fallback;
      } else {
        fail[child] = fallback;
      }
    }
  }

  // Step 3. Overwrite the edges of every non-end state with its resolved row,
  // compressing consecutive bytes with the same target into range edges.
  for (int u = 0; u < num_states; u++) {
    if (u != start && ends.count(u) > 0) {
      continue;
    }
    const auto& row = delta[u];
    std::vector<FSMEdge> new_edges;
    for (int byte = 0; byte < 256;) {
      int target = row[byte];
      int range_end = byte;
      while (range_end + 1 < 256 && row[range_end + 1] == target) {
        range_end++;
      }
      new_edges.push_back(FSMEdge(byte, range_end, target));
      byte = range_end + 1;
    }
    fsm->GetEdges(u) = std::move(new_edges);
  }
}

std::optional<FSMWithStartEnd> TrieFSMBuilder::Build(
    const std::vector<std::string>& patterns,
    const std::vector<std::string>& exclude_patterns,
    std::vector<int32_t>* end_states,
    bool allow_overlap,
    bool add_back_edges
) {
  return TrieFSMBuilderImpl().Build(
      patterns, exclude_patterns, end_states, allow_overlap, add_back_edges
  );
}

}  // namespace xgrammar
