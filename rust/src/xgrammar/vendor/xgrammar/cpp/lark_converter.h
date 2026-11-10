/*!
 *  Copyright (c) 2026 by Contributors
 * \file xgrammar/lark_converter.h
 * \brief Convert Lark syntax to XGrammar Grammar IR.
 */

#ifndef XGRAMMAR_LARK_CONVERTER_H_
#define XGRAMMAR_LARK_CONVERTER_H_

#include <xgrammar/grammar.h>

#include <optional>
#include <string>
#include <vector>

namespace xgrammar {

Grammar LarkToGrammar(
    const std::string& lark_string,
    const std::optional<TokenizerInfo>& tokenizer_info = std::nullopt,
    const std::vector<NamedGrammar>& named_grammars = {}
);

}  // namespace xgrammar

#endif  // XGRAMMAR_LARK_CONVERTER_H_
