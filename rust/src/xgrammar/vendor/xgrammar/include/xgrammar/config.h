/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/config.h
 * \brief Global configuration for XGrammar.
 */

#ifndef XGRAMMAR_CONFIG_H_
#define XGRAMMAR_CONFIG_H_

#include <string>

namespace xgrammar {

/*!
 * \brief Set the maximum recursion depth for the grammar.
 * \param max_recursion_depth The maximum recursion depth.
 */
void SetMaxRecursionDepth(int max_recursion_depth);

/*!
 * \brief Get the maximum recursion depth for the grammar.
 * \return The maximum recursion depth.
 */
int GetMaxRecursionDepth();

/*!
 * \brief Get the serialization version for the grammar.
 * \return The serialization version.
 * \note This is used to check the compatibility of the serialized grammar.
 */
std::string GetSerializationVersion();

}  // namespace xgrammar

#endif  // XGRAMMAR_CONFIG_H_
