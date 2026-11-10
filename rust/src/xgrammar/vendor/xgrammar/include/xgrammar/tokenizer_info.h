/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/tokenizer_info.h
 * \brief The header for the tokenizer info.
 */

#ifndef XGRAMMAR_TOKENIZER_INFO_H_
#define XGRAMMAR_TOKENIZER_INFO_H_

#include <xgrammar/object.h>

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "xgrammar/exception.h"

namespace xgrammar {

enum class VocabType : int {
  RAW = 0,
  BYTE_FALLBACK = 1,
  BYTE_LEVEL = 2,
};

class TokenizerInfo {
 public:
  TokenizerInfo(
      const std::vector<std::string>& encoded_vocab,
      VocabType vocab_type = VocabType::RAW,
      std::optional<int> vocab_size = std::nullopt,
      std::optional<std::vector<int32_t>> stop_token_ids = std::nullopt,
      bool add_prefix_space = false
  );

  VocabType GetVocabType() const;
  bool GetAddPrefixSpace() const;
  int GetVocabSize() const;
  const std::vector<std::string>& GetDecodedVocab() const;
  const std::vector<int32_t>& GetStopTokenIds() const;
  const std::vector<int32_t>& GetSpecialTokenIds() const;
  const std::vector<std::pair<int32_t, std::string>>& GetSortedDecodedVocab() const;
  const std::vector<int32_t>& GetTrieSubtreeNodesRange() const;
  std::string DumpMetadata() const;

  /*!
   * \brief Create a tokenizer info from a vocabulary and metadata.
   * \param encoded_vocab The encoded vocabulary.
   * \param metadata The metadata.
   * \return The tokenizer info.
   */
  static TokenizerInfo FromVocabAndMetadata(
      const std::vector<std::string>& encoded_vocab, const std::string& metadata
  );

  /*!
   * \brief Detect the metadata from a Hugging Face backend string.
   * \param backend_str The Hugging Face backend string.
   * \return The metadata.
   */
  static std::string DetectMetadataFromHF(const std::string& backend_str);

  /*!
   * \brief Return the serialized JSON string of the tokenizer info.
   * \return The serialized JSON string.
   */
  std::string SerializeJSON() const;

  /*!
   * \brief Deserialize a tokenizer info from a JSON string.
   * \param json_string The JSON string to deserialize.
   * \return If the deserialization is successful, return the tokenizer info. Otherwise, return a
   * runtime error with the error message.
   */
  static std::variant<TokenizerInfo, SerializationError> DeserializeJSON(
      const std::string& json_string
  );

  XGRAMMAR_DEFINE_PIMPL_METHODS(TokenizerInfo);
};

}  // namespace xgrammar

#endif  // XGRAMMAR_TOKENIZER_INFO_H_
