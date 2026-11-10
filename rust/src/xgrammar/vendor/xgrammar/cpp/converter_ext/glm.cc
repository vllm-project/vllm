/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/converter_ext/glm.cc
 * \brief GLM XML parameter format.
 */
#include "../json_schema_converter_ext.h"

namespace xgrammar {
namespace converter_ext {

XMLWrapper GetGLMXMLWrapper() { return {"<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"}; }

}  // namespace converter_ext
}  // namespace xgrammar
