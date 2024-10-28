/******************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

/**
 * \file
 * Utility for parsing command line arguments
 */

#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"

namespace cutlass {

/******************************************************************************
 * command_line
 ******************************************************************************/

/**
 * Utility for parsing command line arguments
 */
struct CommandLine {
  std::vector<std::string> keys;
  std::vector<std::string> values;
  std::vector<std::string> args;

  /**
   * Constructor
   */
  CommandLine(int argc, const char** argv) {
    using namespace std;

    for (int i = 1; i < argc; i++) {
      string arg = argv[i];

      if ((arg[0] != '-') || (arg[1] != '-')) {
        args.push_back(arg);
        continue;
      }

      string::size_type pos;
      string key, val;
      if ((pos = arg.find('=')) == string::npos) {
        key = string(arg, 2, arg.length() - 2);
        val = "";
      } else {
        key = string(arg, 2, pos - 2);
        val = string(arg, pos + 1, arg.length() - 1);
      }

      keys.push_back(key);
      values.push_back(val);
    }
  }

  /**
   * Checks whether a flag "--<flag>" is present in the commandline
   */
  bool check_cmd_line_flag(const char* arg_name) const {
    using namespace std;

    for (int i = 0; i < int(keys.size()); ++i) {
      if (keys[i] == string(arg_name)) return true;
    }
    return false;
  }

  /**
   * Returns number of naked (non-flag and non-key-value) commandline parameters
   */
  size_t num_naked_args() const {
    return args.size();
  }

  /**
   * Print naked (non-flag and non-key-value) commandline parameters
   */
  void print_naked_args(std::ostream &out) const {
    for (auto arg : args) {
      out << "   " << arg <<"\n";
    }
  }

  /**
   * Returns the commandline parameter for a given index (not including flags)
   */
  template <typename value_t>
  void get_cmd_line_argument(size_t index, value_t& val) const {
    using namespace std;
    if (index < args.size()) {
      istringstream str_stream(args[index]);
      str_stream >> val;
    }
  }

  /**
   * Obtains the boolean value specified for a given commandline parameter --<flag>=<bool>
   */
  void get_cmd_line_argument(const char* arg_name, bool& val, bool _default) const {
    val = _default;
    if (check_cmd_line_flag(arg_name)) {
      std::string value;
      get_cmd_line_argument(arg_name, value);

      val = !(value == "0" || value == "false");
    }
  }
  
  /**
   * Obtains the value specified for a given commandline parameter --<flag>=<value>
   */
  template <typename value_t>
  void get_cmd_line_argument(const char* arg_name,
                             value_t& val) const {

    get_cmd_line_argument(arg_name, val, val);
  }

  /**
   * Obtains the value specified for a given commandline parameter --<flag>=<value>
   */
  template <typename value_t>
  void get_cmd_line_argument(const char* arg_name,
                             value_t& val,
                             value_t const& _default) const {
    using namespace std;

    val = _default;

    for (int i = 0; i < int(keys.size()); ++i) {
      if (keys[i] == string(arg_name)) {
        istringstream str_stream(values[i]);
        str_stream >> val;
      }
    }
  }

  /**
   * Returns the values specified for a given commandline parameter --<flag>=<value>,<value>*
   */
  template <typename value_t>
  void get_cmd_line_arguments(const char* arg_name,
                              std::vector<value_t>& vals,
                              char sep = ',') const {
    using namespace std;

    if (check_cmd_line_flag(arg_name)) {
      // Clear any default values
      vals.clear();

      // Recover from multi-value string
      for (size_t i = 0; i < keys.size(); ++i) {
        if (keys[i] == string(arg_name)) {
          string val_string(values[i]);
          separate_string(val_string, vals, sep);
        }
      }
    }
  }

  /**
   * Returns the values specified for a given commandline parameter
   * --<flag>=<value>,<value_start:value_end>*
   */
  void get_cmd_line_argument_pairs(const char* arg_name,
                                   std::vector<std::pair<std::string, std::string> >& tokens,
                                   char delim = ',',
                                   char sep = ':') const {
    if (check_cmd_line_flag(arg_name)) {
      std::string value;
      get_cmd_line_argument(arg_name, value);

      tokenize(tokens, value, delim, sep);
    }
  }

  /**
   * Returns a list of ranges specified for a given commandline parameter
   * --<flag>=<key:value>,<key:value>*
   */
  void get_cmd_line_argument_ranges(const char* arg_name,
                                    std::vector<std::vector<std::string> >& vals,
                                    char delim = ',',
                                    char sep = ':') const {
    std::vector<std::string> ranges;
    get_cmd_line_arguments(arg_name, ranges, delim);

    for (std::vector<std::string>::const_iterator range = ranges.begin();
      range != ranges.end(); ++range) {

      std::vector<std::string> range_vals;
      separate_string(*range, range_vals, sep);
      vals.push_back(range_vals);
    }
  }

  /**
   * The number of pairs parsed
   */
  int parsed_argc() const { return (int)keys.size(); }

  //-------------------------------------------------------------------------
  // Utility functions
  //-------------------------------------------------------------------------

  /// Tokenizes a comma-delimited list of string pairs delimited by ':'
  static void tokenize(std::vector<std::pair<std::string, std::string> >& tokens,
                       std::string const& str,
                       char delim = ',',
                       char sep = ':') {
    // Home-built to avoid Boost dependency
    size_t s_idx = 0;
    size_t d_idx = std::string::npos;
    while (s_idx < str.size()) {
      d_idx = str.find_first_of(delim, s_idx);

      size_t end_idx = (d_idx != std::string::npos ? d_idx : str.size());
      size_t sep_idx = str.find_first_of(sep, s_idx);
      size_t offset = 1;
      if (sep_idx == std::string::npos || sep_idx >= end_idx) {
        sep_idx = end_idx;
        offset = 0;
      }

      std::pair<std::string, std::string> item(
          str.substr(s_idx, sep_idx - s_idx),
          str.substr(sep_idx + offset, end_idx - sep_idx - offset));

      tokens.push_back(item);
      s_idx = end_idx + 1;
    }
  }

  /// Tokenizes a comma-delimited list of string pairs delimited by ':'
  static void tokenize(std::vector<std::string>& tokens,
                       std::string const& str,
                       char delim = ',',
                       char sep = ':') {
    typedef std::vector<std::pair<std::string, std::string> > TokenVector;
    typedef TokenVector::const_iterator token_iterator;

    std::vector<std::pair<std::string, std::string> > token_pairs;
    tokenize(token_pairs, str, delim, sep);
    for (token_iterator tok = token_pairs.begin(); tok != token_pairs.end(); ++tok) {
      tokens.push_back(tok->first);
    }
  }

  template <typename value_t>
  static void separate_string(std::string const& str,
                              std::vector<value_t>& vals,
                              char sep = ',') {
    std::istringstream str_stream(str);
    std::string::size_type old_pos = 0;
    std::string::size_type new_pos = 0;

    // Iterate <sep>-delimited values
    value_t val;
    while ((new_pos = str.find(sep, old_pos)) != std::string::npos) {
      if (new_pos != old_pos) {
        str_stream.width(new_pos - old_pos);
        str_stream >> val;
        vals.push_back(val);
      }

      // skip over delimiter
      str_stream.ignore(1);
      old_pos = new_pos + 1;
    }

    // Read last value
    str_stream >> val;
    vals.push_back(val);
  }
};

}  // namespace cutlass
