/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/support/logging.h
 * \brief A logging library that supports logging at different levels.
 */
#ifndef XGRAMMAR_SUPPORT_LOGGING_H_
#define XGRAMMAR_SUPPORT_LOGGING_H_

#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "cpptrace.h"  // IWYU pragma: keep

/*!
 * \brief Whether or not customize the logging output.
 *  If log customize is enabled, the user must implement
 *  xgrammar::LogFatalImpl and xgrammar::LogMessageImpl.
 */
#ifndef XGRAMMAR_LOG_CUSTOMIZE
#define XGRAMMAR_LOG_CUSTOMIZE 0
#endif

namespace xgrammar {

/*!
 * \brief Error type for errors from XGRAMMAR_CHECK, XGRAMMAR_ICHECK, and XGRAMMAR_LOG(FATAL). This
 * error contains a backtrace of where it occurred.
 */
class LogFatalError : public std::runtime_error {
 public:
  /*! \brief Construct an error. Not recommended to use directly. Instead use XGRAMMAR_LOG(FATAL).
   *
   * \param file The file where the error occurred.
   * \param lineno The line number where the error occurred.
   * \param message The error message to display.
   * \param time The time at which the error occurred. This should be in local time.
   */
  LogFatalError(
      const std::string& file,
      int lineno,
      const std::string& message,
      std::time_t time = std::time(nullptr)
  )
      : std::runtime_error(message), file_(file), lineno_(lineno), time_(time) {
    std::ostringstream s;
    s << "[" << std::put_time(std::localtime(&time), "%H:%M:%S") << "] " << file << ":" << lineno
      << ": " << message << "\n";
    full_message_ = s.str();
  }

  /*! \return The file in which the error occurred. */
  const std::string& file() const { return file_; }
  /*! \return The time at which this error occurred. */
  const std::time_t& time() const { return time_; }
  /*! \return The line number at which this error occurred. */
  int lineno() const { return lineno_; }
  /*! \return The error message. */
  const char* what() const noexcept override { return full_message_.c_str(); }

 private:
  std::string file_;
  int lineno_;
  std::time_t time_;
  std::string full_message_;
};

// Provide support for customized logging.
#if XGRAMMAR_LOG_CUSTOMIZE
/*!
 * \brief Custom implementations of LogFatal.
 *
 * \sa XGRAMMAR_LOG_CUSTOMIZE
 */
[[noreturn]] void LogFatalImpl(const std::string& file, int lineno, const std::string& message);

/*!
 * \brief Custom implementations of LogMessage.
 *
 * \sa XGRAMMAR_LOG_CUSTOMIZE
 */
void LogMessageImpl(const std::string& file, int lineno, int level, const std::string& message);

/*!
 * \brief Class to accumulate an error message and throw it. Do not use
 * directly, instead use LOG(FATAL).
 */
class LogFatal {
 public:
  LogFatal(const std::string& file, int lineno) : file_(file), lineno_(lineno) {}
#ifdef _MSC_VER
#pragma disagnostic push
#pragma warning(disable : 4722)
#endif
  [[noreturn]] ~LogFatal() noexcept(false) { LogFatalImpl(file_, lineno_, stream_.str()); }
#ifdef _MSC_VER
#pragma disagnostic pop
#endif
  std::ostringstream& stream() { return stream_; }

 private:
  std::ostringstream stream_;
  std::string file_;
  int lineno_;
};

/*!
 * \brief Class to accumulate an log message. Do not use directly, instead use
 * LOG(INFO), LOG(WARNING), LOG(ERROR).
 */
class LogMessage {
 public:
  LogMessage(const std::string& file, int lineno, int level)
      : file_(file), lineno_(lineno), level_(level) {}
  ~LogMessage() { LogMessageImpl(file_, lineno_, level_, stream_.str()); }
  std::ostringstream& stream() { return stream_; }

 private:
  std::string file_;
  int lineno_;
  int level_;
  std::ostringstream stream_;
};

#else  // if XGRAMMAR_LOG_CUSTOMIZE

/*!
 * \brief Class to accumulate an error message and throw it. Do not use
 * directly, instead use XGRAMMAR_LOG(FATAL).
 * \note The `LogFatal` class is designed to be an empty class to reduce stack size usage.
 * To play this trick, we use the thread-local storage to store its internal data.
 */
class LogFatal {
 public:
  LogFatal(const std::string& file, int lineno) { GetEntry().Init(file, lineno); }
#ifdef _MSC_VER
#pragma disagnostic push
#pragma warning(disable : 4722)
#endif
  [[noreturn]] ~LogFatal() noexcept(false) {
    GetEntry().Finalize();
    throw;
  }
#ifdef _MSC_VER
#pragma disagnostic pop
#endif
  std::ostringstream& stream() { return GetEntry().stream_; }

 private:
  struct Entry {
    void Init(const std::string& file, int lineno) {
      this->stream_.str("");
      this->file_ = file;
      this->lineno_ = lineno;
    }
    [[noreturn]] LogFatalError Finalize() noexcept(false) {
      LogFatalError error(file_, lineno_, stream_.str());
      throw error;
    }
    std::ostringstream stream_;
    std::string file_;
    int lineno_;
  };

  static Entry& GetEntry();
};

/*!
 * \brief Class to accumulate an log message. Do not use directly, instead use
 * XGRAMMAR_LOG(INFO), XGRAMMAR_LOG(WARNING), XGRAMMAR_LOG(ERROR).
 */
class LogMessage {
 public:
  LogMessage(const std::string& file, int lineno, int level) {
    std::time_t t = std::time(nullptr);
    stream_ << "[" << std::put_time(std::localtime(&t), "%H:%M:%S") << "] " << file << ":" << lineno
            << level_strings_[level];
  }
  ~LogMessage() { std::cerr << (stream_.str() + "\n"); }
  std::ostringstream& stream() { return stream_; }

 private:
  std::ostringstream stream_;
  static const char* level_strings_[];
};

#endif  // XGRAMMAR_LOG_CUSTOMIZE

#define XGRAMMAR_LOG_LEVEL_INFO 0
#define XGRAMMAR_LOG_LEVEL_DEBUG 1
#define XGRAMMAR_LOG_LEVEL_WARNING 2
#define XGRAMMAR_LOG_LEVEL_FATAL 3

#define XGRAMMAR_LOG_INFO LogMessage(__FILE__, __LINE__, XGRAMMAR_LOG_LEVEL_INFO).stream()
#define XGRAMMAR_LOG_DEBUG LogMessage(__FILE__, __LINE__, XGRAMMAR_LOG_LEVEL_DEBUG).stream()
#define XGRAMMAR_LOG_WARNING LogMessage(__FILE__, __LINE__, XGRAMMAR_LOG_LEVEL_WARNING).stream()
#define XGRAMMAR_LOG_FATAL LogFatal(__FILE__, __LINE__).stream()

/*!
 * \brief Log a message at the given level.
 * \param level The level of the message. Can be INFO, DEBUG, WARNING, FATAL.
 */
#define XGRAMMAR_LOG(level) XGRAMMAR_LOG_##level

/*!
 * \brief Check if the condition is true. Used for checking the correctness of user inputs.
 * \param x The condition to check.
 */
#define XGRAMMAR_CHECK(x) \
  if (!(x)) LogFatal(__FILE__, __LINE__).stream() << "Check failed: (" #x << ") is false: "

/*!
 * \brief Check if the condition is true. Used to guarantee some internal conditions in the code.
 * \param x The condition to check.
 */
#define XGRAMMAR_ICHECK(x) \
  if (!(x)) LogFatal(__FILE__, __LINE__).stream() << "Internal check failed: (" #x << ") is false: "

/*!
 * \brief Check if the condition is true. Used to guarantee some internal conditions in the code.
 * \note This check is only enabled in debug mode. In release mode, it will be disabled for
 * efficiency. This should be used in preference to XGRAMMAR_ICHECK.
 * \param x The condition to check.
 */
#if XGRAMMAR_ENABLE_INTERNAL_CHECK
#define XGRAMMAR_DCHECK(x) XGRAMMAR_ICHECK(x)
#else
#define XGRAMMAR_DCHECK(x) \
  while (false) XGRAMMAR_ICHECK(x)
#endif  // XGRAMMAR_ENABLE_DCHECK

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_LOGGING_H_
