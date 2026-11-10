/*
 * Copyright 2009-2010 Cybozu Labs, Inc.
 * Copyright 2011-2014 Kazuho Oku
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once

#ifndef PICOJSON_USE_INT64
#define PICOJSON_USE_INT64
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS 1
#endif
#endif

// If PICOJSON_USE_ORDERED_OBJECT is set, picojson uses object_with_ordered_keys, which maintains
// the insertion order of keys, i.e. the order of keys in the json string.
// This macro is set by default.
#ifndef PICOJSON_USE_ORDERED_OBJECT
#define PICOJSON_USE_ORDERED_OBJECT 1
#endif

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// for isnan/isinf
#if __cplusplus >= 201103L
#include <cmath>
#else
extern "C" {
#ifdef _MSC_VER
#include <float.h>
#elif defined(__INTEL_COMPILER)
#include <mathimf.h>
#else
#include <math.h>
#endif
}
#endif

#ifndef PICOJSON_USE_RVALUE_REFERENCE
#if (defined(__cpp_rvalue_references) && __cpp_rvalue_references >= 200610) || \
    (defined(_MSC_VER) && _MSC_VER >= 1600)
#define PICOJSON_USE_RVALUE_REFERENCE 1
#else
#define PICOJSON_USE_RVALUE_REFERENCE 0
#endif
#endif  // PICOJSON_USE_RVALUE_REFERENCE

#ifndef PICOJSON_NOEXCEPT
#if PICOJSON_USE_RVALUE_REFERENCE
#define PICOJSON_NOEXCEPT noexcept
#else
#define PICOJSON_NOEXCEPT throw()
#endif
#endif

// experimental support for int64_t (see README.mkdn for detail)
#ifdef PICOJSON_USE_INT64
#include <errno.h>
#include <inttypes.h>
#endif

// to disable the use of localeconv(3), set PICOJSON_USE_LOCALE to 0
#ifndef PICOJSON_USE_LOCALE
#define PICOJSON_USE_LOCALE 1
#endif
#if PICOJSON_USE_LOCALE
extern "C" {
#include <locale.h>
}
#endif

#ifndef PICOJSON_ASSERT
#ifndef PICOJSON_DISABLE_EXCEPTION
#define PICOJSON_ASSERT(e)                  \
  do {                                      \
    if (!(e)) throw std::runtime_error(#e); \
  } while (0)
#else
#define PICOJSON_ASSERT(e)  \
  do {                      \
    if (!(e)) std::abort(); \
  } while (0)
#endif  // PICOJSON_DISABLE_EXCEPTION
#endif

#ifdef _MSC_VER
#define SNPRINTF _snprintf_s
#pragma warning(push)
#pragma warning(disable : 4244)  // conversion from int to char
#pragma warning(disable : 4127)  // conditional expression is constant
#pragma warning(disable : 4702)  // unreachable code
#else
#define SNPRINTF snprintf
#endif

namespace picojson {

enum {
  null_type,
  boolean_type,
  number_type,
  string_type,
  array_type,
  object_type
#ifdef PICOJSON_USE_INT64
  ,
  int64_type
#endif
};

enum { INDENT_WIDTH = 2 };

struct null {};

class object_with_ordered_keys;

class value {
 public:
  typedef std::vector<value> array;
#ifdef PICOJSON_USE_ORDERED_OBJECT
  typedef object_with_ordered_keys object;
#else
  typedef std::unordered_map<std::string, value> object;
#endif

  union _storage {
    bool boolean_;
    double number_;
#ifdef PICOJSON_USE_INT64
    int64_t int64_;
#endif
    std::string* string_;
    array* array_;
    object* object_;
  };

 protected:
  int type_;
  _storage u_;

 public:
  value();
  value(int type, bool);
  explicit value(bool b);
#ifdef PICOJSON_USE_INT64
  explicit value(int64_t i);
#endif
  explicit value(double n);
  explicit value(const std::string& s);
  explicit value(const array& a);
  explicit value(const object& o);
#if PICOJSON_USE_RVALUE_REFERENCE
  explicit value(std::string&& s);
  explicit value(array&& a);
  explicit value(object&& o);
#endif
  explicit value(const char* s);
  value(const char* s, size_t len);
  ~value();
  value(const value& x);
  value& operator=(const value& x);
#if PICOJSON_USE_RVALUE_REFERENCE
  value(value&& x) PICOJSON_NOEXCEPT;
  value& operator=(value&& x) PICOJSON_NOEXCEPT;
#endif
  void swap(value& x) PICOJSON_NOEXCEPT;
  template <typename T>
  bool is() const;
  template <typename T>
  const T& get() const;
  template <typename T>
  T& get();
  template <typename T>
  void set(const T&);
#if PICOJSON_USE_RVALUE_REFERENCE
  template <typename T>
  void set(T&&);
#endif
  bool evaluate_as_boolean() const;
  const value& get(const size_t idx) const;
  const value& get(const std::string& key) const;
  value& get(const size_t idx);
  value& get(const std::string& key);

  bool contains(const size_t idx) const;
  bool contains(const std::string& key) const;
  std::string to_str() const;
  template <typename Iter>
  void serialize(Iter os, bool prettify = false) const;
  std::string serialize(bool prettify = false) const;

 private:
  template <typename T>
  // NOLINTNEXTLINE(runtime/explicit)
  value(const T*);  // intentionally defined to block implicit conversion of
                    // pointer to bool
  template <typename Iter>
  static void _indent(Iter os, int indent);
  template <typename Iter>
  void _serialize(Iter os, int indent) const;
  std::string _serialize(int indent) const;
  void clear();
};

// The ordered version of hashmap. It has the same interface as std::unordered_map, but provides
// ordered_keys() to return the keys in the order they were inserted.
class object_with_ordered_keys : private std::unordered_map<std::string, value> {
 public:
  using typename std::unordered_map<std::string, value>::value_type;
  using typename std::unordered_map<std::string, value>::iterator;
  using typename std::unordered_map<std::string, value>::const_iterator;

  object_with_ordered_keys() = default;
  object_with_ordered_keys(const object_with_ordered_keys&) = default;
  object_with_ordered_keys(object_with_ordered_keys&&) = default;
  object_with_ordered_keys(std::initializer_list<value_type> init)
      : std::unordered_map<std::string, value>(init) {
    for (const auto& pair : init) {
      ordered_keys_.push_back(pair.first);
    }
  }
  object_with_ordered_keys& operator=(const object_with_ordered_keys&) = default;
  object_with_ordered_keys& operator=(object_with_ordered_keys&&) = default;

  using std::unordered_map<std::string, value>::begin;
  using std::unordered_map<std::string, value>::end;
  using std::unordered_map<std::string, value>::cbegin;
  using std::unordered_map<std::string, value>::cend;
  using std::unordered_map<std::string, value>::empty;
  using std::unordered_map<std::string, value>::size;
  using std::unordered_map<std::string, value>::at;
  using std::unordered_map<std::string, value>::count;
  using std::unordered_map<std::string, value>::find;
  using std::unordered_map<std::string, value>::reserve;

  value& operator[](const std::string& key) {
    if (count(key) == 0) {
      ordered_keys_.push_back(key);
    }
    return std::unordered_map<std::string, value>::operator[](key);
  }

  const value& operator[](const std::string& key) const {
    return std::unordered_map<std::string, value>::at(key);
  }

  void clear() {
    std::unordered_map<std::string, value>::clear();
    ordered_keys_.clear();
  }

  std::pair<iterator, bool> insert(const value_type& kv) {
    if (!count(kv.first)) {
      ordered_keys_.push_back(kv.first);
    }
    return std::unordered_map<std::string, value>::insert(kv);
  }

  template <class... Args>
  std::pair<iterator, bool> emplace(Args&&... args) {
    return insert(value_type(std::forward<Args>(args)...));
  }

  iterator erase(const_iterator it) {
    ordered_keys_.erase(std::find(ordered_keys_.begin(), ordered_keys_.end(), it->first));
    return std::unordered_map<std::string, value>::erase(it);
  }

  iterator erase(iterator it) {
    ordered_keys_.erase(std::find(ordered_keys_.begin(), ordered_keys_.end(), it->first));
    return std::unordered_map<std::string, value>::erase(it);
  }

  size_t erase(const std::string& key) {
    if (std::unordered_map<std::string, value>::erase(key)) {
      ordered_keys_.erase(std::find(ordered_keys_.begin(), ordered_keys_.end(), key));
      return 1;
    } else {
      return 0;
    }
  }

  const std::vector<std::string>& ordered_keys() const { return ordered_keys_; }

  friend bool operator==(const object_with_ordered_keys& lhs, const object_with_ordered_keys& rhs);

 private:
  std::vector<std::string> ordered_keys_;
};

inline bool operator==(const object_with_ordered_keys& lhs, const object_with_ordered_keys& rhs) {
  return static_cast<const std::unordered_map<std::string, value>&>(lhs) ==
         static_cast<const std::unordered_map<std::string, value>&>(rhs);
}

typedef value::array array;
typedef value::object object;

inline value::value() : type_(null_type), u_() {}

inline value::value(int type, bool) : type_(type), u_() {
  switch (type) {
#define INIT(p, v) \
  case p##type:    \
    u_.p = v;      \
    break
    INIT(boolean_, false);
    INIT(number_, 0.0);
#ifdef PICOJSON_USE_INT64
    INIT(int64_, 0);
#endif
    INIT(string_, new std::string());
    INIT(array_, new array());
    INIT(object_, new object());
#undef INIT
    default:
      break;
  }
}

inline value::value(bool b) : type_(boolean_type), u_() { u_.boolean_ = b; }

#ifdef PICOJSON_USE_INT64
inline value::value(int64_t i) : type_(int64_type), u_() { u_.int64_ = i; }
#endif

inline value::value(double n) : type_(number_type), u_() {
  if (
#ifdef _MSC_VER
      !_finite(n)
#elif __cplusplus >= 201103L
      std::isnan(n) || std::isinf(n)
#else
      isnan(n) || isinf(n)
#endif
  ) {
#ifndef PICOJSON_DISABLE_EXCEPTION
    throw std::overflow_error("");
#else
    std::abort();
#endif
  }
  u_.number_ = n;
}

inline value::value(const std::string& s) : type_(string_type), u_() {
  u_.string_ = new std::string(s);
}

inline value::value(const array& a) : type_(array_type), u_() { u_.array_ = new array(a); }

inline value::value(const object& o) : type_(object_type), u_() { u_.object_ = new object(o); }

#if PICOJSON_USE_RVALUE_REFERENCE
inline value::value(std::string&& s) : type_(string_type), u_() {
  u_.string_ = new std::string(std::move(s));
}

inline value::value(array&& a) : type_(array_type), u_() { u_.array_ = new array(std::move(a)); }

inline value::value(object&& o) : type_(object_type), u_() {
  u_.object_ = new object(std::move(o));
}
#endif

inline value::value(const char* s) : type_(string_type), u_() { u_.string_ = new std::string(s); }

inline value::value(const char* s, size_t len) : type_(string_type), u_() {
  u_.string_ = new std::string(s, len);
}

inline void value::clear() {
  switch (type_) {
#define DEINIT(p) \
  case p##type:   \
    delete u_.p;  \
    break
    DEINIT(string_);
    DEINIT(array_);
    DEINIT(object_);
#undef DEINIT
    default:
      break;
  }
}

inline value::~value() { clear(); }

inline value::value(const value& x) : type_(x.type_), u_() {
  switch (type_) {
#define INIT(p, v) \
  case p##type:    \
    u_.p = v;      \
    break
    INIT(string_, new std::string(*x.u_.string_));
    INIT(array_, new array(*x.u_.array_));
    INIT(object_, new object(*x.u_.object_));
#undef INIT
    default:
      u_ = x.u_;
      break;
  }
}

inline value& value::operator=(const value& x) {
  if (this != &x) {
    value t(x);
    swap(t);
  }
  return *this;
}

#if PICOJSON_USE_RVALUE_REFERENCE
inline value::value(value&& x) PICOJSON_NOEXCEPT : type_(null_type), u_() { swap(x); }
inline value& value::operator=(value&& x) PICOJSON_NOEXCEPT {
  swap(x);
  return *this;
}
#endif
inline void value::swap(value& x) PICOJSON_NOEXCEPT {
  std::swap(type_, x.type_);
  std::swap(u_, x.u_);
}

#define IS(ctype, jtype)                 \
  template <>                            \
  inline bool value::is<ctype>() const { \
    return type_ == jtype##_type;        \
  }
IS(null, null)
IS(bool, boolean)
#ifdef PICOJSON_USE_INT64
IS(int64_t, int64)
#endif
IS(std::string, string)
IS(array, array)
IS(object, object)
#undef IS
template <>
inline bool value::is<double>() const {
  return type_ == number_type
#ifdef PICOJSON_USE_INT64
         || type_ == int64_type
#endif
      // NOLINTNEXTLINE(whitespace/semicolon)
      ;
}

#define GET(ctype, var)                                                                  \
  template <>                                                                            \
  inline const ctype& value::get<ctype>() const {                                        \
    PICOJSON_ASSERT("type mismatch! call is<type>() before get<type>()" && is<ctype>()); \
    return var;                                                                          \
  }                                                                                      \
  template <>                                                                            \
  inline ctype& value::get<ctype>() {                                                    \
    PICOJSON_ASSERT("type mismatch! call is<type>() before get<type>()" && is<ctype>()); \
    return var;                                                                          \
  }
GET(bool, u_.boolean_)
GET(std::string, *u_.string_)
GET(array, *u_.array_)
GET(object, *u_.object_)
#ifdef PICOJSON_USE_INT64
GET(double,
    (type_ == int64_type && (const_cast<value*>(this)->type_ = number_type,
                             (const_cast<value*>(this)->u_.number_ = u_.int64_)),
     u_.number_))
GET(int64_t, u_.int64_)
#else
GET(double, u_.number_)
#endif
#undef GET

#define SET(ctype, jtype, setter)                    \
  template <>                                        \
  inline void value::set<ctype>(const ctype& _val) { \
    clear();                                         \
    type_ = jtype##_type;                            \
    setter                                           \
  }
SET(bool, boolean, u_.boolean_ = _val;)
SET(std::string, string, u_.string_ = new std::string(_val);)
SET(array, array, u_.array_ = new array(_val);)
SET(object, object, u_.object_ = new object(_val);)
SET(double, number, u_.number_ = _val;)
#ifdef PICOJSON_USE_INT64
SET(int64_t, int64, u_.int64_ = _val;)
#endif
#undef SET

#if PICOJSON_USE_RVALUE_REFERENCE
#define MOVESET(ctype, jtype, setter)            \
  template <>                                    \
  inline void value::set<ctype>(ctype && _val) { \
    clear();                                     \
    type_ = jtype##_type;                        \
    setter                                       \
  }
MOVESET(std::string, string, u_.string_ = new std::string(std::move(_val));)
MOVESET(array, array, u_.array_ = new array(std::move(_val));)
MOVESET(object, object, u_.object_ = new object(std::move(_val));)
#undef MOVESET
#endif

inline bool value::evaluate_as_boolean() const {
  switch (type_) {
    case null_type:
      return false;
    case boolean_type:
      return u_.boolean_;
    case number_type:
      return u_.number_ != 0;
#ifdef PICOJSON_USE_INT64
    case int64_type:
      return u_.int64_ != 0;
#endif
    case string_type:
      return !u_.string_->empty();
    default:
      return true;
  }
}

inline const value& value::get(const size_t idx) const {
  static value s_null;
  PICOJSON_ASSERT(is<array>());
  return idx < u_.array_->size() ? (*u_.array_)[idx] : s_null;
}

inline value& value::get(const size_t idx) {
  static value s_null;
  PICOJSON_ASSERT(is<array>());
  return idx < u_.array_->size() ? (*u_.array_)[idx] : s_null;
}

inline const value& value::get(const std::string& key) const {
  static value s_null;
  PICOJSON_ASSERT(is<object>());
  object::const_iterator i = u_.object_->find(key);
  return i != u_.object_->end() ? i->second : s_null;
}

inline value& value::get(const std::string& key) {
  static value s_null;
  PICOJSON_ASSERT(is<object>());
  object::iterator i = u_.object_->find(key);
  return i != u_.object_->end() ? i->second : s_null;
}

inline bool value::contains(const size_t idx) const {
  PICOJSON_ASSERT(is<array>());
  return idx < u_.array_->size();
}

inline bool value::contains(const std::string& key) const {
  PICOJSON_ASSERT(is<object>());
  object::const_iterator i = u_.object_->find(key);
  return i != u_.object_->end();
}

inline std::string value::to_str() const {
  switch (type_) {
    case null_type:
      return "null";
    case boolean_type:
      return u_.boolean_ ? "true" : "false";
#ifdef PICOJSON_USE_INT64
    case int64_type: {
      char buf[sizeof("-9223372036854775808")];
      SNPRINTF(buf, sizeof(buf), "%" PRId64, u_.int64_);
      return buf;
    }
#endif
    case number_type: {
      char buf[256];
      double tmp;
      SNPRINTF(
          buf,
          sizeof(buf),
          fabs(u_.number_) < (1ULL << 53) && modf(u_.number_, &tmp) == 0 ? "%.f" : "%.17g",
          u_.number_
      );
#if PICOJSON_USE_LOCALE
      char* decimal_point = localeconv()->decimal_point;
      if (strcmp(decimal_point, ".") != 0) {
        size_t decimal_point_len = strlen(decimal_point);
        for (char* p = buf; *p != '\0'; ++p) {
          if (strncmp(p, decimal_point, decimal_point_len) == 0) {
            return std::string(buf, p) + "." + (p + decimal_point_len);
          }
        }
      }
#endif
      return buf;
    }
    case string_type:
      return *u_.string_;
    case array_type:
      return "array";
    case object_type:
      return "object";
    default:
      PICOJSON_ASSERT(0);
#ifdef _MSC_VER
      __assume(0);
#endif
  }
  return std::string();
}

template <typename Iter>
void copy(const std::string& s, Iter oi) {
  std::copy(s.begin(), s.end(), oi);
}

template <typename Iter>
struct serialize_str_char {
  Iter oi;
  void operator()(char c) {
    switch (c) {
#define MAP(val, sym) \
  case val:           \
    copy(sym, oi);    \
    break
      MAP('"', "\\\"");
      MAP('\\', "\\\\");
      MAP('\b', "\\b");
      MAP('\f', "\\f");
      MAP('\n', "\\n");
      MAP('\r', "\\r");
      MAP('\t', "\\t");
#undef MAP
      default:
        if (static_cast<unsigned char>(c) < 0x20 || c == 0x7f) {
          char buf[7];
          SNPRINTF(buf, sizeof(buf), "\\u%04x", c & 0xff);
          copy(buf, buf + 6, oi);
        } else {
          *oi++ = c;
        }
        break;
    }
  }
};

template <typename Iter>
void serialize_str(const std::string& s, Iter oi) {
  *oi++ = '"';
  serialize_str_char<Iter> process_char = {oi};
  std::for_each(s.begin(), s.end(), process_char);
  *oi++ = '"';
}

template <typename Iter>
void value::serialize(Iter oi, bool prettify) const {
  return _serialize(oi, prettify ? 0 : -1);
}

inline std::string value::serialize(bool prettify) const { return _serialize(prettify ? 0 : -1); }

template <typename Iter>
void value::_indent(Iter oi, int indent) {
  *oi++ = '\n';
  for (int i = 0; i < indent * INDENT_WIDTH; ++i) {
    *oi++ = ' ';
  }
}

template <typename Iter>
void value::_serialize(Iter oi, int indent) const {
  switch (type_) {
    case string_type:
      serialize_str(*u_.string_, oi);
      break;
    case array_type: {
      *oi++ = '[';
      if (indent != -1) {
        ++indent;
      }
      for (array::const_iterator i = u_.array_->begin(); i != u_.array_->end(); ++i) {
        if (i != u_.array_->begin()) {
          *oi++ = ',';
        }
        if (indent != -1) {
          _indent(oi, indent);
        }
        i->_serialize(oi, indent);
      }
      if (indent != -1) {
        --indent;
        if (!u_.array_->empty()) {
          _indent(oi, indent);
        }
      }
      *oi++ = ']';
      break;
    }
    case object_type: {
      *oi++ = '{';
      if (indent != -1) {
        ++indent;
      }

#if PICOJSON_USE_ORDERED_OBJECT
      for (auto i = u_.object_->ordered_keys().begin(); i != u_.object_->ordered_keys().end();
           ++i) {
        if (i != u_.object_->ordered_keys().begin()) {
          *oi++ = ',';
        }
        if (indent != -1) {
          _indent(oi, indent);
        }
        serialize_str(*i, oi);
        *oi++ = ':';
        if (indent != -1) {
          *oi++ = ' ';
        }
        u_.object_->at(*i)._serialize(oi, indent);
      }
#else
      for (object::const_iterator i = u_.object_->begin(); i != u_.object_->end(); ++i) {
        if (i != u_.object_->begin()) {
          *oi++ = ',';
        }
        if (indent != -1) {
          _indent(oi, indent);
        }
        serialize_str(i->first, oi);
        *oi++ = ':';
        if (indent != -1) {
          *oi++ = ' ';
        }
        i->second._serialize(oi, indent);
      }
#endif
      if (indent != -1) {
        --indent;
        if (!u_.object_->empty()) {
          _indent(oi, indent);
        }
      }
      *oi++ = '}';
      break;
    }
    default:
      copy(to_str(), oi);
      break;
  }
  if (indent == 0) {
    *oi++ = '\n';
  }
}

inline std::string value::_serialize(int indent) const {
  std::string s;
  _serialize(std::back_inserter(s), indent);
  return s;
}

template <typename Iter>
class input {
 protected:
  Iter cur_, end_;
  bool consumed_;
  int line_;

 public:
  input(const Iter& first, const Iter& last)
      : cur_(first), end_(last), consumed_(false), line_(1) {}
  int getc() {
    if (consumed_) {
      if (*cur_ == '\n') {
        ++line_;
      }
      ++cur_;
    }
    if (cur_ == end_) {
      consumed_ = false;
      return -1;
    }
    consumed_ = true;
    return *cur_ & 0xff;
  }
  void ungetc() { consumed_ = false; }
  Iter cur() const {
    if (consumed_) {
      input<Iter>* self = const_cast<input<Iter>*>(this);
      self->consumed_ = false;
      ++self->cur_;
    }
    return cur_;
  }
  int line() const { return line_; }
  void skip_ws() {
    while (1) {
      int ch = getc();
      if (!(ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r')) {
        ungetc();
        break;
      }
    }
  }
  bool expect(const int expected) {
    skip_ws();
    if (getc() != expected) {
      ungetc();
      return false;
    }
    return true;
  }
  bool match(const std::string& pattern) {
    for (std::string::const_iterator pi(pattern.begin()); pi != pattern.end(); ++pi) {
      if (getc() != *pi) {
        ungetc();
        return false;
      }
    }
    return true;
  }
};

template <typename Iter>
// NOLINTNEXTLINE(runtime/references)
inline int _parse_quadhex(input<Iter>& in) {
  int uni_ch = 0, hex;
  for (int i = 0; i < 4; i++) {
    if ((hex = in.getc()) == -1) {
      return -1;
    }
    if ('0' <= hex && hex <= '9') {
      hex -= '0';
    } else if ('A' <= hex && hex <= 'F') {
      hex -= 'A' - 0xa;
    } else if ('a' <= hex && hex <= 'f') {
      hex -= 'a' - 0xa;
    } else {
      in.ungetc();
      return -1;
    }
    uni_ch = uni_ch * 16 + hex;
  }
  return uni_ch;
}

template <typename String, typename Iter>
// NOLINTNEXTLINE(runtime/references)
inline bool _parse_codepoint(String& out, input<Iter>& in) {
  int uni_ch;
  if ((uni_ch = _parse_quadhex(in)) == -1) {
    return false;
  }
  if (0xd800 <= uni_ch && uni_ch <= 0xdfff) {
    if (0xdc00 <= uni_ch) {
      // a second 16-bit of a surrogate pair appeared
      return false;
    }
    // first 16-bit of surrogate pair, get the next one
    if (in.getc() != '\\' || in.getc() != 'u') {
      in.ungetc();
      return false;
    }
    int second = _parse_quadhex(in);
    if (!(0xdc00 <= second && second <= 0xdfff)) {
      return false;
    }
    uni_ch = ((uni_ch - 0xd800) << 10) | ((second - 0xdc00) & 0x3ff);
    uni_ch += 0x10000;
  }
  if (uni_ch < 0x80) {
    out.push_back(static_cast<char>(uni_ch));
  } else {
    if (uni_ch < 0x800) {
      out.push_back(static_cast<char>(0xc0 | (uni_ch >> 6)));
    } else {
      if (uni_ch < 0x10000) {
        out.push_back(static_cast<char>(0xe0 | (uni_ch >> 12)));
      } else {
        out.push_back(static_cast<char>(0xf0 | (uni_ch >> 18)));
        out.push_back(static_cast<char>(0x80 | ((uni_ch >> 12) & 0x3f)));
      }
      out.push_back(static_cast<char>(0x80 | ((uni_ch >> 6) & 0x3f)));
    }
    out.push_back(static_cast<char>(0x80 | (uni_ch & 0x3f)));
  }
  return true;
}

template <typename String, typename Iter>
// NOLINTNEXTLINE(runtime/references)
inline bool _parse_string(String& out, input<Iter>& in) {
  while (1) {
    int ch = in.getc();
    if (ch < ' ') {
      in.ungetc();
      return false;
    } else if (ch == '"') {
      return true;
    } else if (ch == '\\') {
      if ((ch = in.getc()) == -1) {
        return false;
      }
      switch (ch) {
#define MAP(sym, val)   \
  case sym:             \
    out.push_back(val); \
    break
        MAP('"', '\"');
        MAP('\\', '\\');
        MAP('/', '/');
        MAP('b', '\b');
        MAP('f', '\f');
        MAP('n', '\n');
        MAP('r', '\r');
        MAP('t', '\t');
#undef MAP
        case 'u':
          if (!_parse_codepoint(out, in)) {
            return false;
          }
          break;
        default:
          return false;
      }
    } else {
      out.push_back(static_cast<char>(ch));
    }
  }
  return false;
}

template <typename Context, typename Iter>
// NOLINTNEXTLINE(runtime/references)
inline bool _parse_array(Context& ctx, input<Iter>& in) {
  if (!ctx.parse_array_start()) {
    return false;
  }
  size_t idx = 0;
  if (in.expect(']')) {
    return ctx.parse_array_stop(idx);
  }
  do {
    if (!ctx.parse_array_item(in, idx)) {
      return false;
    }
    idx++;
  } while (in.expect(','));
  return in.expect(']') && ctx.parse_array_stop(idx);
}

template <typename Context, typename Iter>
// NOLINTNEXTLINE(runtime/references)
inline bool _parse_object(Context& ctx, input<Iter>& in) {
  if (!ctx.parse_object_start()) {
    return false;
  }
  if (in.expect('}')) {
    return true;
  }
  do {
    std::string key;
    if (!in.expect('"') || !_parse_string(key, in) || !in.expect(':')) {
      return false;
    }
    if (!ctx.parse_object_item(in, key)) {
      return false;
    }
  } while (in.expect(','));
  return in.expect('}');
}

template <typename Iter>
// NOLINTNEXTLINE(runtime/references)
inline std::string _parse_number(input<Iter>& in) {
  std::string num_str;
  while (1) {
    int ch = in.getc();
    if (('0' <= ch && ch <= '9') || ch == '+' || ch == '-' || ch == 'e' || ch == 'E') {
      num_str.push_back(static_cast<char>(ch));
    } else if (ch == '.') {
#if PICOJSON_USE_LOCALE
      num_str += localeconv()->decimal_point;
#else
      num_str.push_back('.');
#endif
    } else {
      in.ungetc();
      break;
    }
  }
  return num_str;
}

template <typename Context, typename Iter>
// NOLINTNEXTLINE(runtime/references)
inline bool _parse(Context& ctx, input<Iter>& in) {
  in.skip_ws();
  int ch = in.getc();
  switch (ch) {
#define IS(ch, text, op)        \
  case ch:                      \
    if (in.match(text) && op) { \
      return true;              \
    } else {                    \
      return false;             \
    }
    IS('n', "ull", ctx.set_null());
    IS('f', "alse", ctx.set_bool(false));
    IS('t', "rue", ctx.set_bool(true));
#undef IS
    case '"':
      return ctx.parse_string(in);
    case '[':
      return _parse_array(ctx, in);
    case '{':
      return _parse_object(ctx, in);
    default:
      if (('0' <= ch && ch <= '9') || ch == '-') {
        double f;
        char* endp;
        in.ungetc();
        std::string num_str(_parse_number(in));
        if (num_str.empty()) {
          return false;
        }
#ifdef PICOJSON_USE_INT64
        {
          errno = 0;
          intmax_t ival = strtoimax(num_str.c_str(), &endp, 10);
          if (errno == 0 && std::numeric_limits<int64_t>::min() <= ival &&
              ival <= std::numeric_limits<int64_t>::max() &&
              endp == num_str.c_str() + num_str.size()) {
            ctx.set_int64(ival);
            return true;
          }
        }
#endif
        f = strtod(num_str.c_str(), &endp);
        if (endp == num_str.c_str() + num_str.size()) {
          ctx.set_number(f);
          return true;
        }
        return false;
      }
      break;
  }
  in.ungetc();
  return false;
}

class deny_parse_context {
 public:
  bool set_null() { return false; }
  bool set_bool(bool) { return false; }
#ifdef PICOJSON_USE_INT64
  bool set_int64(int64_t) { return false; }
#endif
  bool set_number(double) { return false; }
  template <typename Iter>
  bool parse_string(input<Iter>&) {
    return false;
  }
  bool parse_array_start() { return false; }
  template <typename Iter>
  bool parse_array_item(input<Iter>&, size_t) {
    return false;
  }
  bool parse_array_stop(size_t) { return false; }
  bool parse_object_start() { return false; }
  template <typename Iter>
  bool parse_object_item(input<Iter>&, const std::string&) {
    return false;
  }
};

class default_parse_context {
 protected:
  value* out_;

 public:
  // NOLINTNEXTLINE(runtime/explicit)
  default_parse_context(value* out) : out_(out) {}
  bool set_null() {
    *out_ = value();
    return true;
  }
  bool set_bool(bool b) {
    *out_ = value(b);
    return true;
  }
#ifdef PICOJSON_USE_INT64
  bool set_int64(int64_t i) {
    *out_ = value(i);
    return true;
  }
#endif
  bool set_number(double f) {
    *out_ = value(f);
    return true;
  }
  template <typename Iter>
  // NOLINTNEXTLINE(runtime/references)
  bool parse_string(input<Iter>& in) {
    *out_ = value(string_type, false);
    return _parse_string(out_->get<std::string>(), in);
  }
  bool parse_array_start() {
    *out_ = value(array_type, false);
    return true;
  }
  template <typename Iter>
  // NOLINTNEXTLINE(runtime/references)
  bool parse_array_item(input<Iter>& in, size_t) {
    array& a = out_->get<array>();
    a.push_back(value());
    default_parse_context ctx(&a.back());
    return _parse(ctx, in);
  }
  bool parse_array_stop(size_t) { return true; }
  bool parse_object_start() {
    *out_ = value(object_type, false);
    return true;
  }
  template <typename Iter>
  // NOLINTNEXTLINE(runtime/references)
  bool parse_object_item(input<Iter>& in, const std::string& key) {
    object& o = out_->get<object>();
    default_parse_context ctx(&o[key]);
    return _parse(ctx, in);
  }

 private:
  default_parse_context(const default_parse_context&);
  default_parse_context& operator=(const default_parse_context&);
};

class null_parse_context {
 public:
  struct dummy_str {
    void push_back(int) {}
  };

 public:
  null_parse_context() {}
  bool set_null() { return true; }
  bool set_bool(bool) { return true; }
#ifdef PICOJSON_USE_INT64
  bool set_int64(int64_t) { return true; }
#endif
  bool set_number(double) { return true; }
  template <typename Iter>
  // NOLINTNEXTLINE(runtime/references)
  bool parse_string(input<Iter>& in) {
    dummy_str s;
    return _parse_string(s, in);
  }
  bool parse_array_start() { return true; }
  template <typename Iter>
  // NOLINTNEXTLINE(runtime/references)
  bool parse_array_item(input<Iter>& in, size_t) {
    return _parse(*this, in);
  }
  bool parse_array_stop(size_t) { return true; }
  bool parse_object_start() { return true; }
  template <typename Iter>
  // NOLINTNEXTLINE(runtime/references)
  bool parse_object_item(input<Iter>& in, const std::string&) {
    return _parse(*this, in);
  }

 private:
  null_parse_context(const null_parse_context&);
  null_parse_context& operator=(const null_parse_context&);
};

// obsolete, use the version below
template <typename Iter>
// NOLINTNEXTLINE(runtime/references)
inline std::string parse(value& out, Iter& pos, const Iter& last) {
  std::string err;
  pos = parse(out, pos, last, &err);
  return err;
}

template <typename Context, typename Iter>
// NOLINTNEXTLINE(runtime/references)
inline Iter _parse(Context& ctx, const Iter& first, const Iter& last, std::string* err) {
  input<Iter> in(first, last);
  if (!_parse(ctx, in) && err != NULL) {
    char buf[64];
    SNPRINTF(buf, sizeof(buf), "syntax error at line %d near: ", in.line());
    *err = buf;
    while (1) {
      int ch = in.getc();
      if (ch == -1 || ch == '\n') {
        break;
      } else if (ch >= ' ') {
        err->push_back(static_cast<char>(ch));
      }
    }
  }
  return in.cur();
}

template <typename Iter>
// NOLINTNEXTLINE(runtime/references)
inline Iter parse(value& out, const Iter& first, const Iter& last, std::string* err) {
  default_parse_context ctx(&out);
  return _parse(ctx, first, last, err);
}

// NOLINTNEXTLINE(runtime/references)
inline std::string parse(value& out, const std::string& s) {
  std::string err;
  parse(out, s.begin(), s.end(), &err);
  return err;
}

// NOLINTNEXTLINE(runtime/references)
inline std::string parse(value& out, std::istream& is) {
  std::string err;
  parse(out, std::istreambuf_iterator<char>(is.rdbuf()), std::istreambuf_iterator<char>(), &err);
  return err;
}

template <typename T>
struct last_error_t {
  static std::string s;
};
template <typename T>
// NOLINTNEXTLINE(runtime/string)
std::string last_error_t<T>::s;

inline void set_last_error(const std::string& s) { last_error_t<bool>::s = s; }

inline const std::string& get_last_error() { return last_error_t<bool>::s; }

inline bool operator==(const value& x, const value& y) {
  if (x.is<null>()) return y.is<null>();
#define PICOJSON_CMP(type) \
  if (x.is<type>()) return y.is<type>() && x.get<type>() == y.get<type>()
  PICOJSON_CMP(bool);
  PICOJSON_CMP(double);
  PICOJSON_CMP(std::string);
  PICOJSON_CMP(array);
  PICOJSON_CMP(object);
#undef PICOJSON_CMP
  PICOJSON_ASSERT(0);
#ifdef _MSC_VER
  __assume(0);
#endif
  return false;
}

inline bool operator!=(const value& x, const value& y) { return !(x == y); }
}  // namespace picojson

#if !PICOJSON_USE_RVALUE_REFERENCE
namespace std {
template <>
inline void swap(picojson::value& x, picojson::value& y) {
  x.swap(y);
}
}  // namespace std
#endif

inline std::istream& operator>>(std::istream& is, picojson::value& x) {
  picojson::set_last_error(std::string());
  const std::string err(picojson::parse(x, is));
  if (!err.empty()) {
    picojson::set_last_error(err);
    is.setstate(std::ios::failbit);
  }
  return is;
}

inline std::ostream& operator<<(std::ostream& os, const picojson::value& x) {
  x.serialize(std::ostream_iterator<char>(os));
  return os;
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif
