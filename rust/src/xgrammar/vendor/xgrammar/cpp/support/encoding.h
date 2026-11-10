/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/support/encoding.h
 * \brief Encoding and decoding from/to UTF-8 and escape sequence to/from codepoints.
 */
#ifndef XGRAMMAR_SUPPORT_ENCODING_H_
#define XGRAMMAR_SUPPORT_ENCODING_H_
// TODO(yixin): enhance performance

#include <array>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "logging.h"

namespace xgrammar {

/*! \brief Represents a unicode codepoint. */
using TCodepoint = int32_t;

/*!
 * \brief Represents an error when handling characters. Will be returned as a special TCodepoint
 * value.
 */
enum CharHandlingError : TCodepoint {
  /*! \brief The UTF-8 string is invalid. */
  kInvalidUTF8 = -10,
  /*! \brief The escape sequence is invalid. */
  kInvalidEscape = -11,
  /*! \brief The Latin-1 string is invalid. */
  kInvalidLatin1 = -12,
};

/******************** UTF-8 Handling ********************/

/*!
 * \brief Print a codepoint to a UTF-8 string.
 * \param codepoint The codepoint.
 * \return The UTF-8 string.
 */
std::string CharToUTF8(TCodepoint codepoint);

/*!
 * \brief Handle the utf-8 first byte.
 * \returns (is_valid, total_number_of_bytes, initial_codepoint).
 */
std::tuple<bool, int, TCodepoint> HandleUTF8FirstByte(uint8_t byte);

/*!
 * \brief Parse all codepoints in a UTF-8 string.
 * \param utf8 The UTF-8 string.
 * \param perserve_invalid_bytes If the invalid UTF8 bytes will be preserved in the result.
 * \return All codepoints. If the UTF-8 string is invalid, when perserve_invalid_bytes is false,
 * the invalid bytes will be added to the result as a TCodepoint. Otherwise, the function will
 * return {CharHandlingError::kInvalidUTF8}.
 */
std::vector<TCodepoint> ParseUTF8(const char* utf8, bool perserve_invalid_bytes = false);

/*!
 * \brief Parse the first codepoint in a UTF-8 string.
 * \param utf8 The UTF-8 string.
 * \return The codepoint and the number of bytes consumed. If the UTF-8 string is invalid, return
 * {CharHandlingError::kInvalidUTF8, 0}.
 */
std::pair<TCodepoint, int32_t> ParseNextUTF8(const char* utf8);

/*!
 * \brief Convert a Latin-1 string to a byte sequence.
 * \param latin1 The Latin-1 string.
 * \return The byte sequence.
 */
std::optional<CharHandlingError> Latin1ToBytes(const std::string& latin1, std::string* result);

/******************** Escape Handling ********************/

/*!
 * \brief Convert a codepoint to a escaped string. If the codepoint is not printable, it will be
 * escaped. By default the function support escape sequences in C ("\n", "\t", "\u0123"). User
 * can specify more escape sequences using additional_escape_map.
 * \param codepoint The codepoint.
 * \param additional_escape_map A map from codepoint to escape sequence. If the codepoint is in
 * the map, it will be escaped using the corresponding escape sequence. e.g. {{'-', "\\-"}}.
 * \return The printable string.
 */
std::string EscapeString(
    TCodepoint codepoint,
    const std::unordered_map<TCodepoint, std::string>& additional_escape_map = {}
);

/*!
 * \brief Convert the given char to a escaped string that can be printed.
 * \return The escaped string.
 */
std::string EscapeString(uint8_t raw_char);

/*!
 * \brief Convert the given string to a escaped string that can be printed.
 * \return The escaped string.
 */
std::string EscapeString(std::string raw_str);

/*!
 * \brief Convert a hex character to an integer.
 * \param c The hex character: 0-9, a-f, A-F.
 * \return The integer value of the hex character. If the character is not a valid hex character,
 * return -1.
 */
int HexCharToInt(char c);

/*!
 * \brief Parse the first escaped codepoint from a escaped string. data must start with a '\'
 * character.
 * \param data The escaped string. Can be TCodepoint* (e.g. string decoded from UTF-8) or char*.
 * \param additional_escape_map A map from escape sequence to codepoint. If the escape sequence is
 * in the map, it will be converted to the corresponding codepoint. e.g. {{"\\-", '-'}}.
 * \return The codepoint and the number of bytes consumed.
 */
template <typename CharType>
std::pair<TCodepoint, int32_t> ParseNextEscaped(
    const CharType* data, const std::unordered_map<char, TCodepoint>& additional_escape_map = {}
);

/*!
 * \brief Parse the first codepoint from a UTF-8 string. Also checks escape sequences and converts
 * the escaped char to its original value.
 * \param utf8 The UTF-8 string or the escape sequence.
 * \param additional_escape_map A map from escape sequence to codepoint. If the escape sequence is
 * in the map, it will be converted to the corresponding codepoint. e.g. {{"\\-", '-'}}.
 * \return The codepoint and the number of bytes consumed. If the UTF-8 string is invalid, the
 * function returns (CharHandlingError::kInvalidUTF8, 0). If the escape sequence is invalid, the
 * function returns (CharHandlingError::kInvalidEscape, 0).
 */
std::pair<TCodepoint, int32_t> ParseNextUTF8OrEscaped(
    const char* utf8, const std::unordered_map<char, TCodepoint>& additional_escape_map = {}
);

/******************** Implementation ********************/

inline std::string CharToUTF8(TCodepoint codepoint) {
  XGRAMMAR_DCHECK(codepoint <= 0x10FFFF) << "Invalid codepoint: " << codepoint;
  std::string utf8;
  if (codepoint <= 0x7F) {
    // 1-byte sequence
    utf8 += static_cast<char>(codepoint);
  } else if (codepoint <= 0x7FF) {
    // 2-byte sequence
    utf8 += static_cast<char>(0xC0 | ((codepoint >> 6) & 0x1F));
    utf8 += static_cast<char>(0x80 | (codepoint & 0x3F));
  } else if (codepoint <= 0xFFFF) {
    // 3-byte sequence
    utf8 += static_cast<char>(0xE0 | ((codepoint >> 12) & 0x0F));
    utf8 += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
    utf8 += static_cast<char>(0x80 | (codepoint & 0x3F));
  } else {
    // 4-byte sequence
    utf8 += static_cast<char>(0xF0 | ((codepoint >> 18) & 0x07));
    utf8 += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F));
    utf8 += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
    utf8 += static_cast<char>(0x80 | (codepoint & 0x3F));
  }
  return utf8;
}

inline std::tuple<bool, int, TCodepoint> HandleUTF8FirstByte(uint8_t byte) {
  static const std::array<int8_t, 5> kFirstByteMask = {0x00, 0x7F, 0x1F, 0x0F, 0x07};
  // clang-format off
  static const std::array<int, 256> kUtf8Bytes = {
     1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
     1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
     1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
     1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
     1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
     1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
     1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
     1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
     2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
     2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
     3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
    4,  4,  4,  4,  4,  4,  4,  4, -1, -1, -1, -1, -1, -1, -1, -1,
  };
  // clang-format on
  auto num_bytes = kUtf8Bytes[static_cast<uint8_t>(byte)];
  if (num_bytes == -1) {
    return {false, 0, 0};
  }
  return {true, num_bytes, byte & kFirstByteMask[num_bytes]};
}

inline std::pair<TCodepoint, int32_t> ParseNextUTF8(const char* utf8) {
  auto [accepted, num_bytes, res] = HandleUTF8FirstByte(utf8[0]);
  if (accepted) {
    for (int i = 1; i < num_bytes; ++i) {
      if (utf8[i] == 0 || (static_cast<uint8_t>(utf8[i]) & 0xC0) != 0x80) {
        // invalid utf8
        accepted = false;
        break;
      }
      res = (res << 6) | (static_cast<uint8_t>(utf8[i]) & 0x3F);
    }
  }

  if (!accepted) {
    // invalid utf8
    return {CharHandlingError::kInvalidUTF8, 0};
  }

  return {res, num_bytes};
}

inline std::vector<TCodepoint> ParseUTF8(const char* utf8, bool perserve_invalid_bytes) {
  std::vector<TCodepoint> codepoints;
  while (*utf8 != 0) {
    auto [codepoint, num_bytes] = ParseNextUTF8(utf8);
    if (codepoint == CharHandlingError::kInvalidUTF8) {
      if (perserve_invalid_bytes) {
        codepoints.push_back(static_cast<TCodepoint>(static_cast<uint8_t>(utf8[0])));
        utf8 += 1;
        continue;
      } else {
        return {CharHandlingError::kInvalidUTF8};
      }
    }
    codepoints.push_back(codepoint);
    utf8 += num_bytes;
  }
  return codepoints;
}

/*!
  \brief Convert a Latin-1 string to a byte sequence.
  \param latin1 The Latin-1 string.
  \param result The output byte sequence.
  The function will convert each Latin-1 character to its corresponding byte(s).
  For characters in the range [0x00, 0x7F], the corresponding byte is the same as the character.
  Otherwise, the character should be encoded in two bytes in UTF-8:
    - First byte:  110xxxxx (0xC0 | (char >> 6))
    - Second byte: 10xxxxxx (0x80 | (char & 0x3F))
  Example:
    0xC3 0xBF -> 0xFF
    'A' -> 'A'
  \return std::nullopt if the conversion is successful. Otherwise, return
  CharHandlingError::kInvalidLatin1 if the Latin-1 string is invalid.
*/
inline std::optional<CharHandlingError> Latin1ToBytes(
    const std::string& latin1, std::string* result
) {
  result->clear();
  result->reserve(latin1.size());

  const size_t len = latin1.size();
  for (size_t i = 0; i < len; ++i) {
    unsigned char c1 = static_cast<unsigned char>(latin1[i]);
    if (c1 < 0x80) {
      result->push_back(static_cast<char>(c1));
    } else {
      if (i + 1 >= len) {
        return CharHandlingError::kInvalidLatin1;
      }

      unsigned char c2 = static_cast<unsigned char>(latin1[i + 1]);
      if ((c2 & 0xC0) != 0x80) {
        return CharHandlingError::kInvalidLatin1;
      }

      int code = ((c1 & 0x1F) << 6) | (c2 & 0x3F);
      if (code < 0x80 || code > 0xFF) {
        return CharHandlingError::kInvalidLatin1;
      }

      result->push_back(static_cast<char>(code));
      ++i;
    }
  }

  return std::nullopt;
}

/*!
 \brief Convert a byte sequence to a Latin-1 string.
 \param Bytes The input byte sequence.
 \param result The output Latin-1 string.
 The function will convert each byte in the input to a Latin-1 character.
 For bytes in the range [0x00, 0x7F], the corresponding Latin-1 character is the same as the byte.
 For bytes in the range [0x80, 0xFF], the corresponding Latin-1 character is represented by two
 bytes in UTF-8:
   - First byte:  110xxxxx (0xC0 | (byte >> 6))
   - Second byte: 10xxxxxx (0x80 | (byte & 0x3F))
  Example:
    0xFF -> 0xC3 0xBF
    'A' -> 'A'
*/
inline void ByteToLatin1(const std::string& bytes, std::string* result) {
  result->clear();
  result->reserve(bytes.size());

  for (unsigned char current_char : bytes) {
    // Ascii character, directly add to result.
    if (current_char <= 0x7F) {
      result->push_back(static_cast<char>(current_char));
      continue;
    }

    // not Ascii character, convert to Latin-1.
    unsigned char latin1_first_byte = 0;
    unsigned char latin1_second_byte = 0;

    latin1_first_byte = 0xC0 | (current_char >> 6);
    latin1_second_byte = 0x80 | (current_char & 0x3F);
    result->push_back(static_cast<char>(latin1_first_byte));
    result->push_back(static_cast<char>(latin1_second_byte));
  }
}

inline int HexCharToInt(char c) {
  if (c >= '0' && c <= '9') {
    return c - '0';
  } else if (c >= 'a' && c <= 'f') {
    return c - 'a' + 10;
  } else if (c >= 'A' && c <= 'F') {
    return c - 'A' + 10;
  } else {
    return -1;
  }
}

inline std::string EscapeString(
    TCodepoint codepoint, const std::unordered_map<TCodepoint, std::string>& additional_escape_map
) {
  static const std::unordered_map<TCodepoint, std::string> kCodepointToEscape = {
      {'\'', "\\\'"},
      {'\"', "\\\""},
      {'\?', "\\?"},
      {'\\', "\\\\"},
      {'\a', "\\a"},
      {'\b', "\\b"},
      {'\f', "\\f"},
      {'\n', "\\n"},
      {'\r', "\\r"},
      {'\t', "\\t"},
      {'\v', "\\v"},
      {'\0', "\\0"},
      {'\x1B', "\\e"}
  };

  if (auto it = additional_escape_map.find(codepoint); it != additional_escape_map.end()) {
    return it->second;
  }

  if (auto it = kCodepointToEscape.find(codepoint); it != kCodepointToEscape.end()) {
    return it->second;
  }

  if (codepoint >= 0x20 && codepoint <= 0x7E) {
    return std::string({static_cast<char>(codepoint)});
  }

  // convert codepoint to hex
  char prefix = codepoint <= 0xFF ? 'x' : codepoint <= 0xFFFF ? 'u' : 'U';
  int width = codepoint <= 0xFF ? 2 : codepoint <= 0xFFFF ? 4 : 8;
  std::stringstream ss;
  ss << std::setfill('0') << std::setw(width) << std::hex << codepoint;
  auto hex = ss.str();
  return std::string("\\") + prefix + hex;
}

inline std::string EscapeString(uint8_t raw_char) {
  return EscapeString(static_cast<TCodepoint>(raw_char));
}

inline std::string EscapeString(std::string raw_str) {
  std::string res;
  auto codepoints = ParseUTF8(raw_str.c_str(), true);
  for (auto c : codepoints) {
    res += EscapeString(c);
  }
  return res;
}

template <typename CharType>
std::pair<TCodepoint, int32_t> ParseNextEscaped(
    const CharType* data, const std::unordered_map<char, TCodepoint>& additional_escape_map
) {
  // C escape characters
  static const std::unordered_map<char, TCodepoint> kEscapeToCodepoint = {
      // clang-format off
      {'\'', '\''}, {'\"', '\"'}, {'?', '\?'}, {'\\', '\\'}, {'/', '/'}, {'a', '\a'},
      {'b', '\b'}, {'f', '\f'}, {'n', '\n'}, {'r', '\r'}, {'t', '\t'}, {'v', '\v'},
      {'0', '\0'}, {'e', '\x1B'}  // clang-format on
  };
  if (data[0] != '\\') {
    return {CharHandlingError::kInvalidEscape, 0};
  }

  bool escape_char_in_escape_range =
      static_cast<int32_t>(static_cast<unsigned char>(data[1])) <= 128;
  if (!escape_char_in_escape_range) {
    return {CharHandlingError::kInvalidEscape, 0};
  }

  if (auto it = additional_escape_map.find(static_cast<char>(data[1]));
      it != additional_escape_map.end()) {
    return {it->second, 2};
  }
  if (auto it = kEscapeToCodepoint.find(static_cast<char>(data[1]));
      it != kEscapeToCodepoint.end()) {
    return {it->second, 2};
  }

  if (data[1] == 'x') {
    // arbitrary length hex
    int len = 0;
    TCodepoint codepoint = 0;
    int32_t digit;
    while ((digit = HexCharToInt(data[2 + len])) != -1) {
      codepoint = codepoint * 16 + digit;
      ++len;
    }
    if (len == 0) {
      return {CharHandlingError::kInvalidEscape, 0};
    }
    return {codepoint, len + 2};
  } else if (data[1] == 'u' || data[1] == 'U') {
    // 4- or 8-digit hex
    int len = data[1] == 'u' ? 4 : 8;
    TCodepoint codepoint = 0;

    for (int i = 0; i < len; ++i) {
      auto digit = HexCharToInt(data[i + 2]);
      if (digit == -1) {
        return {CharHandlingError::kInvalidEscape, 0};
      }
      codepoint = codepoint * 16 + digit;
    }
    return {codepoint, len + 2};
  } else {
    return {CharHandlingError::kInvalidEscape, 0};
  }
}

inline std::pair<TCodepoint, int32_t> ParseNextUTF8OrEscaped(
    const char* utf8, const std::unordered_map<char, TCodepoint>& additional_escape_map
) {
  if (utf8[0] != '\\') {
    return ParseNextUTF8(utf8);
  }
  return ParseNextEscaped(utf8, additional_escape_map);
}

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_ENCODING_H_
