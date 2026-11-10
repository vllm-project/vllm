/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/converter_ext/minimax.cc
 * \brief MiniMax XML parameter format.
 */
#include "../json_schema_converter_ext.h"

namespace xgrammar {
namespace converter_ext {

XMLWrapper GetMiniMaxXMLWrapper() { return {"<parameter name=\"", "\">", "", "</parameter>"}; }

}  // namespace converter_ext
}  // namespace xgrammar
