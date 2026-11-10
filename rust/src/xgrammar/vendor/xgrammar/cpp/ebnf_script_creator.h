/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/ebnf_script_creator.h
 * \brief The header for the creating EBNF script.
 */

#ifndef XGRAMMAR_EBNF_SCRIPT_CREATOR_H_
#define XGRAMMAR_EBNF_SCRIPT_CREATOR_H_

#include <xgrammar/object.h>

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

#include "support/encoding.h"
#include "support/logging.h"
#include "support/utils.h"

namespace xgrammar {

/*!
 * \brief A class for creating EBNF grammar scripts.
 *
 * This class helps build EBNF (Extended Backus-Naur Form) grammar scripts
 * by managing rules and their content.
 */
class EBNFScriptCreator {
 public:
  /*! \brief Constructor */
  EBNFScriptCreator() = default;

  /*!
   * \brief Adds a new rule to the grammar with a suggested name
   * \param rule_name_hint Suggested name for the rule
   * \param rule_body The EBNF content/definition of the rule
   * \return The actual name assigned to the rule
   */
  std::string AddRule(const std::string& rule_name_hint, const std::string& rule_body) {
    return AddRuleWithAllocatedName(AllocateRuleName(rule_name_hint), rule_body);
  }

  /*!
   * \brief Generates a new rule name based on a suggested name
   * \param rule_name_hint Suggested name for the rule
   * \return The actual name assigned to the rule
   */
  std::string AllocateRuleName(const std::string& rule_name_hint) {
    if (rule_names_.find(rule_name_hint) == rule_names_.end()) {
      rule_names_.insert(rule_name_hint);
      return rule_name_hint;
    }
    for (int i = 0; i < NAME_SUFFIX_MAXIMUM; ++i) {
      std::string rule_name = rule_name_hint + "_" + std::to_string(i);
      if (rule_names_.find(rule_name) == rule_names_.end()) {
        rule_names_.insert(rule_name);
        return rule_name;
      }
    }
    XGRAMMAR_LOG(FATAL) << "Cannot find a unique rule name for " << rule_name_hint;
    XGRAMMAR_UNREACHABLE();
  }

  /*!
   * \brief Adds a new rule to the grammar with a allocated name. Used with AllocateRuleName()
   * \param rule_name The name of the rule to add
   * \param rule_body The EBNF content/definition of the rule
   * \return The actual name assigned to the rule
   */
  std::string AddRuleWithAllocatedName(const std::string& rule_name, const std::string& rule_body) {
    XGRAMMAR_CHECK(rule_names_.find(rule_name) != rule_names_.end())
        << "Rule name " << rule_name << " is not allocated";
    rules_.emplace_back(rule_name, rule_body);
    return rule_name;
  }

  /*!
   * \brief Concatenates a list of strings with a space separator
   * \param items The list of strings to concatenate
   * \return The concatenated string
   */
  static std::string Concat(const std::vector<std::string>& items) {
    std::stringstream ss;
    ss << "(";
    for (int i = 0; i < static_cast<int>(items.size()); ++i) {
      if (i > 0) {
        ss << " ";
      }
      ss << items[i];
    }
    ss << ")";
    return ss.str();
  }

  /*!
   * \brief Joins a list of strings with an OR operator
   * \param items The list of strings to join
   * \return The joined string
   */
  static std::string Or(const std::vector<std::string>& items) {
    std::stringstream ss;
    ss << "(";
    for (int i = 0; i < static_cast<int>(items.size()); ++i) {
      if (i > 0) {
        ss << " | ";
      }
      ss << items[i];
    }
    ss << ")";
    return ss.str();
  }

  /*!
   * \brief Escape and quote a string
   * \param str The string to escape and quote
   * \return The escaped and quoted string
   */
  static std::string Str(const std::string& str) {
    std::stringstream ss;
    ss << "\"" << EscapeString(str) << "\"";
    return ss.str();
  }

  /*!
   * \brief Repeats an item a given number of times
   * \param item The item to repeat
   * \param min The minimum number of times to repeat the item
   * \param max The maximum number of times to repeat the item
   * \return The repeated string
   */
  static std::string Repeat(const std::string& item, int min, int max) {
    std::stringstream ss;
    ss << item;
    if (min == 0 && max == 1) {
      ss << "?";
    } else if (min == 0 && max == -1) {
      ss << "*";
    } else if (min == 1 && max == -1) {
      ss << "+";
    } else if (min == 0 && max == 0) {
      return "";
    } else if (min == max) {
      ss << "{" << min << "}";
    } else if (max == -1) {
      ss << "{" << min << ",}";
    } else {
      ss << "{" << min << "," << max << "}";
    }
    return ss.str();
  }

  /*!
   * \brief Gets the complete EBNF grammar script
   * \return The full EBNF grammar script as a string
   */
  std::string GetScript() {
    std::string script = "";
    for (const auto& rule : rules_) {
      script += rule.first + " ::= " + rule.second + "\n";
    }
    return script;
  }

  /*!
   * \brief Retrieves the content/definition of a specific rule
   * \param rule_name The name of the rule to look up
   * \return The EBNF content/definition of the specified rule
   */
  std::string GetRuleContent(const std::string& rule_name) {
    auto it = std::find_if(rules_.begin(), rules_.end(), [rule_name](const auto& rule) {
      return rule.first == rule_name;
    });
    if (it != rules_.end()) {
      return it->second;
    }
    return "";
  }

 private:
  std::vector<std::pair<std::string, std::string>> rules_;
  std::unordered_set<std::string> rule_names_;
  const int NAME_SUFFIX_MAXIMUM = 10000;
};

}  // namespace xgrammar

#endif  // XGRAMMAR_EBNF_SCRIPT_CREATOR_H_
