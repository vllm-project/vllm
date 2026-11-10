#ifndef XGRAMMAR_EXCEPTION_H
#define XGRAMMAR_EXCEPTION_H

#include <stdexcept>
#include <string>
#include <variant>

namespace xgrammar {

/************** Exception Definitions **************/

/*!
 * \brief Exception thrown when the version in the serialized data does not follow the current
 * serialization version.
 */

struct XGrammarError : std::runtime_error {
  XGrammarError(const std::string& message) : std::runtime_error(message) {}
  virtual std::string GetType() const { return "XGrammarError"; }
};

struct DeserializeVersionError : XGrammarError {
  DeserializeVersionError(const std::string& message)
      : XGrammarError(std::string("Deserialize version error: ") + message) {}
  std::string GetType() const override { return "DeserializeVersionError"; }
};

/*!
 * \brief Exception thrown when the JSON is invalid.
 */
struct InvalidJSONError : XGrammarError {
  InvalidJSONError(const std::string& message)
      : XGrammarError(std::string("Invalid JSON error: ") + message) {}
  std::string GetType() const override { return "InvalidJSONError"; }
};

/*!
 * \brief Exception thrown when the serialized data does not follow the expected format.
 */
struct DeserializeFormatError : XGrammarError {
  DeserializeFormatError(const std::string& message)
      : XGrammarError(std::string("Deserialize format error: ") + message) {}
  std::string GetType() const override { return "DeserializeFormatError"; }
};

/*!
 * \brief Exception thrown when the JSON schema is invalid or not satisfiable.
 */
struct InvalidJSONSchemaError : XGrammarError {
  InvalidJSONSchemaError(const std::string& message)
      : XGrammarError(std::string("Invalid JSON schema error: ") + message) {}
  std::string GetType() const override { return "InvalidJSONSchemaError"; }
};

/*!
 * \brief Exception thrown when the structural tag is invalid.
 */
struct InvalidStructuralTagError : XGrammarError {
  InvalidStructuralTagError(const std::string& message)
      : XGrammarError(std::string("Invalid structural tag error: ") + message) {}
  std::string GetType() const override { return "InvalidStructuralTagError"; }
};

/************** Union Exceptions **************/

/*!
 * \brief Represents a serialization error.
 */
using SerializationError =
    std::variant<DeserializeVersionError, InvalidJSONError, DeserializeFormatError>;

/*!
 * \brief Represents an error from the structural tag conversion.
 */
using StructuralTagError =
    std::variant<InvalidJSONError, InvalidJSONSchemaError, InvalidStructuralTagError>;

}  // namespace xgrammar

#endif  // XGRAMMAR_EXCEPTION_H
