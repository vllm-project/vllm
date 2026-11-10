/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/converter_ext/kimi_k3.cc
 * \brief Kimi-K3 XML type attributes and key escaping.
 */
#include "../json_schema_converter_ext.h"

namespace xgrammar {

namespace converter_ext {

XMLWrapper GetKimiK3XMLWrapper() {
  // The key suffix (type attribute and <|sep|>) is generated in XMLKeySuffix.
  return {"<|open|>argument key=\"", "", "", "<|close|>argument<|sep|>"};
}

const XMLKeySuffix& GetKimiK3XMLKeySuffix() {
  static const XMLKeySuffix suffix = {
      "\" type=\"",
      {"string", "number", "integer", "boolean", "object", "array", "null"},
      "\"<|sep|>"
  };
  return suffix;
}

}  // namespace converter_ext

std::string XMLToolCallingConverter::EscapeAttrValue(const std::string& value) const {
  if (json_format_ != JSONFormat::kKimiK3XML) {
    return value;
  }
  // Kimi-K3's renderer escapes attribute values with & -> &amp; and " -> &quot;.
  std::string escaped;
  escaped.reserve(value.size());
  for (char c : value) {
    if (c == '&') {
      escaped += "&amp;";
    } else if (c == '"') {
      escaped += "&quot;";
    } else {
      escaped += c;
    }
  }
  return escaped;
}

}  // namespace xgrammar
