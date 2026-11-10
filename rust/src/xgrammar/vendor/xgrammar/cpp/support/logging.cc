/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/support/logging.cc
 */
#include "logging.h"

namespace xgrammar {

#if XGRAMMAR_LOG_CUSTOMIZE == 0

LogFatal::Entry& LogFatal::GetEntry() {
  static thread_local LogFatal::Entry result;
  return result;
}

const char* LogMessage::level_strings_[] = {
    ": ",           // XGRAMMAR_LOG_LEVEL_INFO
    ": Debug: ",    // XGRAMMAR_LOG_LEVEL_DEBUG
    ": Warning: ",  // XGRAMMAR_LOG_LEVEL_WARNING
};

#endif  // XGRAMMAR_LOG_CUSTOMIZE

}  // namespace xgrammar
