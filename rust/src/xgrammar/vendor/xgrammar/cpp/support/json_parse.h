/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/support/json_parse.h
 * \brief picojson::parse wrappers whose nesting depth is bounded by RecursionGuard.
 * picojson recurses once per nesting level and has no depth limit of its own, so a deeply nested
 * input would overflow the stack before any of the recursion guards in xgrammar run.
 */
#ifndef XGRAMMAR_SUPPORT_JSON_PARSE_H_
#define XGRAMMAR_SUPPORT_JSON_PARSE_H_

#include <picojson.h>

#include <cstdint>
#include <string>

#include "logging.h"
#include "recursion_guard.h"

namespace xgrammar {

namespace detail {

/*!
 * \brief A picojson parse context that behaves like picojson::default_parse_context, but counts
 * every nested array or object in a RecursionGuard. The maximum recursion depth of xgrammar thus
 * applies while parsing, and exceeding it throws before the stack overflows.
 */
class DepthGuardedParseContext {
 public:
  DepthGuardedParseContext(picojson::value* out, int* depth) : out_(out), depth_(depth) {}
  DepthGuardedParseContext(const DepthGuardedParseContext&) = delete;
  DepthGuardedParseContext& operator=(const DepthGuardedParseContext&) = delete;

  bool set_null() {
    *out_ = picojson::value();
    return true;
  }
  bool set_bool(bool b) {
    *out_ = picojson::value(b);
    return true;
  }
  bool set_int64(int64_t i) {
    *out_ = picojson::value(i);
    return true;
  }
  bool set_number(double f) {
    *out_ = picojson::value(f);
    return true;
  }
  template <typename Iter>
  bool parse_string(picojson::input<Iter>& in) {
    *out_ = picojson::value(picojson::string_type, false);
    return picojson::_parse_string(out_->get<std::string>(), in);
  }
  bool parse_array_start() {
    *out_ = picojson::value(picojson::array_type, false);
    return true;
  }
  template <typename Iter>
  bool parse_array_item(picojson::input<Iter>& in, size_t) {
    picojson::array& a = out_->get<picojson::array>();
    a.push_back(picojson::value());
    RecursionGuard guard(depth_);
    DepthGuardedParseContext child(&a.back(), depth_);
    return picojson::_parse(child, in);
  }
  bool parse_array_stop(size_t) { return true; }
  bool parse_object_start() {
    *out_ = picojson::value(picojson::object_type, false);
    return true;
  }
  template <typename Iter>
  bool parse_object_item(picojson::input<Iter>& in, const std::string& key) {
    picojson::object& o = out_->get<picojson::object>();
    RecursionGuard guard(depth_);
    DepthGuardedParseContext child(&o[key], depth_);
    return picojson::_parse(child, in);
  }

 private:
  picojson::value* out_;
  int* depth_;
};

}  // namespace detail

/*!
 * \brief Parse the JSON value at the start of [begin, end). Same as
 * picojson::parse(out, begin, end, err), but the nesting depth is bounded by the maximum
 * recursion depth (see RecursionGuard).
 * \return The iterator past the parsed value; begin if the depth limit was exceeded.
 */
template <typename Iter>
inline Iter ParseJSON(picojson::value& out, Iter begin, Iter end, std::string* err) {
  int depth = 0;
  detail::DepthGuardedParseContext ctx(&out, &depth);
  try {
    return picojson::_parse(ctx, begin, end, err);
  } catch (const LogFatalError& error) {
    *err = error.what();
    return begin;
  }
}

/*!
 * \brief Parse a JSON string. Same as picojson::parse(out, json), but the nesting depth is bounded
 * by the maximum recursion depth (see RecursionGuard).
 * \return The error message, empty on success.
 */
inline std::string ParseJSON(picojson::value& out, const std::string& json) {
  std::string err;
  ParseJSON(out, json.begin(), json.end(), &err);
  return err;
}

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_JSON_PARSE_H_
