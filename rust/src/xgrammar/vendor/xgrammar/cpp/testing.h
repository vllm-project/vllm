/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/testing.h
 * \brief The header testing utilities.
 */
#ifndef XGRAMMAR_TESTING_H_
#define XGRAMMAR_TESTING_H_

#include <dlpack/dlpack.h>
#include <xgrammar/xgrammar.h>

#include <cstdint>
#include <string>
#include <vector>

namespace xgrammar {

std::string PrintTokenByIds(
    const std::vector<int32_t>& token_ids, const TokenizerInfo& tokenizer_info, int max_print_num
);

Grammar _EBNFToGrammarNoNormalization(
    const std::string& ebnf_string, const std::string& root_rule_name
);

std::string _PrintGrammarFSMs(const Grammar& grammar);

}  // namespace xgrammar

#endif  // XGRAMMAR_TESTING_H_
