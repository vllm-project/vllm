/*!
 * Copyright (c) 2024 by Contributors
 * \file xgrammar/support/cpptrace.h
 * \details This file is an encapsulation of the cpptrace library. It helps debugging. This file
 * takes effect only when XGRAMMAR_ENABLE_CPPTRACE is set to 1, and only support Linux and
 * RelWithDebugInfo or Debug build.
 */
#ifndef XGRAMMAR_SUPPORT_CPPTRACE_H_
#define XGRAMMAR_SUPPORT_CPPTRACE_H_

#if XGRAMMAR_ENABLE_CPPTRACE == 1
#include <cpptrace/cpptrace.hpp>
#endif

#include <string>

namespace xgrammar {

#if XGRAMMAR_ENABLE_CPPTRACE == 1

// Flag to check if cpptrace feature is enabled
static constexpr bool CPPTRACE_ENABLED = true;

inline void PrintTrace() { cpptrace::generate_trace().print(); }
inline std::string GetTraceString() { return cpptrace::generate_trace().to_string(true); }

#else

static constexpr bool CPPTRACE_ENABLED = false;

// Provide empty implementation when cpptrace is disabled
inline void PrintTrace() {}
inline std::string GetTraceString() { return ""; }

#endif

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_CPPTRACE_H_
