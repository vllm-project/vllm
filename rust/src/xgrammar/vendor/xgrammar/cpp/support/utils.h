/*!
 * Copyright (c) 2024 by Contributors
 * \file xgrammar/support/utils.h
 * \brief Utility functions.
 */
#ifndef XGRAMMAR_SUPPORT_UTILS_H_
#define XGRAMMAR_SUPPORT_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

#include "logging.h"

/****************** Hash Library ******************/

namespace xgrammar {

/*!
 * \brief Hash and combine value into seed.
 * \ref https://www.boost.org/doc/libs/1_84_0/boost/intrusive/detail/hash_combine.hpp
 */
inline void HashCombineBinary(uint64_t& seed, uint64_t value) {
  seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
}

/*!
 * \brief Find the hash sum of several size_t args.
 */
template <typename... Args>
inline uint64_t HashCombine(Args... args) {
  uint64_t seed = 0;
  (..., HashCombineBinary(seed, args));
  return seed;
}

/*!
 * \brief Helper class to define the hash function for a struct by its members.
 */
template <class T, auto... Members>
struct HashByMembers {
  std::size_t operator()(T const& x) const noexcept {
    return HashCombine(std::hash<std::decay_t<decltype(x.*Members)>>{}(x.*Members)...);
  }
};

}  // namespace xgrammar

/*!
 * \brief Define a hash function for a struct by its members in namespace std. Should be used
 * outside of namespace xgrammar.
 * \param Type The type of the struct.
 * \param ... The member pointers of the struct.
 * \example
 * \code
 * // In the global namespace
 * XGRAMMAR_HASH_BY_MEMBERS(Type, &Type::member1, &Type::member2, &Type::member3);
 * \endcode
 */
#define XGRAMMAR_HASH_BY_MEMBERS(Type, ...)                                 \
  namespace std {                                                           \
  template <>                                                               \
  struct hash<Type> : public xgrammar::HashByMembers<Type, __VA_ARGS__> {}; \
  }

/*!
 * \brief Empty specialization of XGRAMMAR_HASH_BY_MEMBERS.
 */
#define XGRAMMAR_HASH_BY_MEMBERS_EMPTY(Type)                   \
  namespace std {                                              \
  template <>                                                  \
  struct hash<Type> : public xgrammar::HashByMembers<Type> {}; \
  }

namespace std {

/*!
 * \brief Define the hash function for std::pair.
 */
template <typename T, typename U>
struct hash<std::pair<T, U>> {
  size_t operator()(const std::pair<T, U>& pair) const noexcept {
    return xgrammar::HashCombine(std::hash<T>{}(pair.first), std::hash<U>{}(pair.second));
  }
};

/*!
 * \brief Define the hash function for std::tuple.
 */
template <typename... Args>
struct hash<std::tuple<Args...>> {
  size_t operator()(const std::tuple<Args...>& tuple) const noexcept {
    return std::apply(
        [](const Args&... args) { return xgrammar::HashCombine(std::hash<Args>{}(args)...); }, tuple
    );
  }
};

/*!
 * \brief Define the hash function for std::vector.
 */
template <typename T>
struct hash<std::vector<T>> {
  size_t operator()(const std::vector<T>& vec) const {
    uint32_t seed = 0;
    for (const auto& item : vec) {
      xgrammar::HashCombineBinary(seed, std::hash<T>{}(item));
    }
    return seed;
  }
};

}  // namespace std

namespace xgrammar {

/****************** Result Library ******************/

/*!
 * \brief A partial result type that can be used to construct a Result. Holds a result value or an
 * error value.
 * \tparam T The type of the value
 * \tparam IsOk Whether the result is ok
 */
template <typename T, bool IsOk>
struct PartialResult {
  template <typename... Args>
  PartialResult(Args&&... args) : value(std::forward<Args>(args)...) {}
  T value;
};

/*!
 * \brief Construct a success result with the arguments to construct a T.
 * \tparam T The type of the success value
 * \tparam Args The types of the arguments to construct a T
 * \param args The arguments to construct a T
 * \return A PartialResult with the arguments to construct a T
 * \example
 * \code
 * // Call the constructor of T with the arguments
 * return ResultOk<T>(1, 2, 3);
 * \endcode
 */
template <typename T, typename... Args>
inline PartialResult<T, true> ResultOk(Args&&... args) {
  return PartialResult<T, true>{std::forward<Args>(args)...};
}

/*!
 * \brief Construct a success result with a universal reference (both lvalue and rvalue)
 * \tparam T The type of the success value
 * \param value The universal reference to the success value
 * \return A PartialResult with the universal reference to the success value
 * \example
 * \code
 * T value = T(1, 2, 3);
 * // Move the value to the PartialResult
 * return ResultOk(std::move(value));
 * \endcode
 */
template <typename T>
inline PartialResult<T&&, true> ResultOk(T&& value) {
  return PartialResult<T&&, true>{std::forward<T>(value)};
}

/*!
 * \brief Construct a error result with the arguments to construct a E.
 * \tparam E The type of the error value. Default to std::runtime_error.
 * \tparam Args The types of the arguments to construct a E
 * \param args The arguments to construct a E
 * \return A PartialResult with the arguments to construct a E
 * \example
 * \code
 * // Construct a std::runtime_error with a error
 * std::runtime_error error("Message");
 * return ResultErr(std::move(error));
 * \endcode
 * \code
 * // Construct a std::runtime_error with its argument
 * return ResultErr("Error");
 * \endcode
 * \code
 * // Construct an E error with its argument
 * return ResultErr<E>("Error");
 * \endcode
 */
template <typename E = std::runtime_error, typename... Args>
inline PartialResult<E, false> ResultErr(Args&&... args) {
  return PartialResult<E, false>{std::forward<Args>(args)...};
}

/*!
 * \brief Construct a error result with a universal reference (both lvalue and rvalue)
 * \tparam E The type of the error value
 * \param err The universal reference to the error value
 * \return A PartialResult with the universal reference to the error value
 * \example
 * \code
 * E err = E("Error");
 * // Move the err to the PartialResult
 * return ResultErr(std::move(err));
 * \endcode
 */
template <typename E>
inline PartialResult<E&&, false> ResultErr(E&& err) {
  return PartialResult<E&&, false>{std::forward<E>(err)};
}

/*!
 * \brief An always-move Result type similar to Rust's Result, representing either success (Ok) or
 * failure (Err). It always uses move semantics for the success and error values.
 * \tparam T The type of the success value
 * \tparam E The type of the error value
 *
 * \note The Ok and Err constructor, and all methods of this class (except for ValueRef and ErrRef)
 * accept only rvalue references as parameters for performance reasons. You should use std::move to
 * convert a Result to an rvalue reference before invoking these methods. Examples for move
 * semantics are shown below.
 *
 * \example Construct a success result with a rvalue reference
 * \code
 * T value;
 * return Result<T, std::string>::Ok(std::move(value));
 * \endcode
 * \example Construct a error result with a rvalue reference of std::runtime_error
 * \code
 * std::runtime_error error_msg = std::runtime_error("Error");
 * return Result<T>::Err(std::move(error_msg));
 * \endcode
 * \example Construct a error result with a std::runtime_error object constructed with a string
 * \code
 * std::string error_msg = "Error";
 * return Result<T>::Err(std::move(error_msg));
 * \endcode
 * \example Unwrap the rvalue reference of the result
 * \code
 * Result<T> result = func();
 * if (result.IsOk()) {
 *   T result_val = std::move(result).Unwrap();
 * } else {
 *   std::runtime_error error_msg = std::move(result).UnwrapErr();
 * }
 * \endcode
 */
template <typename T, typename E = std::runtime_error>
class Result {
 private:
  static_assert(!std::is_same_v<T, E>, "T and E cannot be the same type");

 public:
  /*! \brief Default constructor is deleted to avoid accidental use */
  Result() = delete;

  /*! \brief Construct from Result::Ok */
  template <typename U, typename = std::enable_if_t<std::is_constructible_v<T, std::decay_t<U>>>>
  Result(PartialResult<U, true>&& partial_result)
      : data_(std::in_place_type<T>, std::forward<U>(partial_result.value)) {}

  /*! \brief Construct from Result::Err */
  template <typename V, typename = std::enable_if_t<std::is_constructible_v<E, std::decay_t<V>>>>
  Result(PartialResult<V, false>&& partial_result)
      : data_(std::in_place_type<E>, std::forward<V>(partial_result.value)) {}

  /*! \brief Check if Result contains success value */
  bool IsOk() const { return std::holds_alternative<T>(data_); }

  /*! \brief Check if Result contains error */
  bool IsErr() const { return std::holds_alternative<E>(data_); }

  /*! \brief Get the success value. It assumes (or checks if in debug mode) the result is ok. */
  T Unwrap() && {
    XGRAMMAR_DCHECK(IsOk()) << "Called Unwrap() on an Err value";
    return std::get<T>(std::move(data_));
  }

  /*! \brief Get the error value. It assumes (or checks if in debug mode) the result is an error. */
  E UnwrapErr() && {
    XGRAMMAR_DCHECK(IsErr()) << "Called UnwrapErr() on an Ok value";
    return std::get<E>(std::move(data_));
  }

  /*! \brief Get the success value if present, otherwise return the provided default */
  T UnwrapOr(T default_value) && {
    return IsOk() ? std::get<T>(std::move(data_)) : std::move(default_value);
  }

  /*! \brief Map success value to new type using provided function */
  template <typename F, typename U = std::decay_t<std::invoke_result_t<F, T>>>
  Result<U, E> Map(F&& f) && {
    if (IsOk()) {
      return ResultOk(f(std::get<T>(std::move(data_))));
    }
    return ResultErr(std::get<E>(std::move(data_)));
  }

  /*! \brief Map error value to new type using provided function */
  template <typename F, typename V = std::decay_t<std::invoke_result_t<F, E>>>
  Result<T, V> MapErr(F&& f) && {
    if (IsErr()) {
      return ResultErr(f(std::get<E>(std::move(data_))));
    }
    return ResultOk(std::get<T>(std::move(data_)));
  }

  /*!
   * \brief Convert a Result<U, V> to a Result<T, E>. U should be convertible to T, and V should be
   * convertible to E.
   */
  template <typename U, typename V>
  static Result<T, E> Convert(Result<U, V>&& result) {
    if (result.IsOk()) {
      return ResultOk<T>(std::move(result).Unwrap());
    }
    return ResultErr<E>(std::move(result).UnwrapErr());
  }

  /*! \brief Get a std::variant<T, E> from the result. */
  std::variant<T, E> ToVariant() && { return std::move(data_); }

  /*!
   * \brief Get a reference to the success value. It assumes (or checks if in debug mode) the
   * result is ok.
   */
  T& ValueRef() & {
    XGRAMMAR_DCHECK(IsOk()) << "Called ValueRef() on an Err value";
    return std::get<T>(data_);
  }

  /*!
   * \brief Get a reference to the error value. It assumes (or checks if in debug mode) the
   * result is an error.
   */
  E& ErrRef() & {
    XGRAMMAR_DCHECK(IsErr()) << "Called ErrRef() on an Ok value";
    return std::get<E>(data_);
  }

 private:
  // in-place construct T in variant
  template <typename... Args>
  explicit Result(std::in_place_type_t<T>, Args&&... args)
      : data_(std::in_place_type<T>, std::forward<Args>(args)...) {}

  // in-place construct E in variant
  template <typename... Args>
  explicit Result(std::in_place_type_t<E>, Args&&... args)
      : data_(std::in_place_type<E>, std::forward<Args>(args)...) {}

  std::variant<T, E> data_;
};

/****************** Misc ******************/

// Sometimes GCC fails to detect some branches will not return, such as when we use LOG(FATAL)
// to raise an error. This macro manually mark them as unreachable to avoid warnings.
#ifdef __GNUC__
#define XGRAMMAR_UNREACHABLE() __builtin_unreachable()
#else
#define XGRAMMAR_UNREACHABLE()
#endif

/*!
 * \brief An error class that contains a type. The type can be an enum.
 */
template <typename T>
class TypedError : public std::runtime_error {
 public:
  explicit TypedError(T type, const std::string& msg) : std::runtime_error(msg), type_(type) {}
  const T& Type() const noexcept { return type_; }

 private:
  T type_;
};

/**
 * \brief Helper function to compare two objects by their members.
 */
template <class T, auto... Ms>
constexpr bool EqualByMembers(const T& lhs, const T& rhs) noexcept {
  return std::tie(lhs.*Ms...) == std::tie(rhs.*Ms...);
}

/**
 * \brief Define == and != operator for a struct by its members.
 * \param Type The type of the struct. Must be under namespace xgrammar.
 * \param ... The member pointers of the struct.
 * \example
 * \code
 * struct Type {
 *   int member1;
 *   std::string member2;
 *   double member3;
 *
 *   XGRAMMAR_EQUAL_BY_MEMBERS(Type, &Type::member1, &Type::member2, &Type::member3);
 * };
 * \endcode
 */
#define XGRAMMAR_EQUAL_BY_MEMBERS(Type, ...)                          \
  friend bool operator==(const Type& lhs, const Type& rhs) noexcept { \
    return EqualByMembers<Type, __VA_ARGS__>(lhs, rhs);               \
  }                                                                   \
  friend bool operator!=(const Type& lhs, const Type& rhs) noexcept { return !(lhs == rhs); }

/*!
 * \brief Empty specialization of XGRAMMAR_EQUAL_BY_MEMBERS.
 */
#define XGRAMMAR_EQUAL_BY_MEMBERS_EMPTY(Type)                                        \
  friend bool operator==(const Type& lhs, const Type& rhs) noexcept { return true; } \
  friend bool operator!=(const Type& lhs, const Type& rhs) noexcept { return false; }

/*!
 * \brief Throw an error from a variant of multiple error types.
 * \param error_variant The variant of multiple error types.
 * \tparam Args The types of the error types. Each type should inherit from std::runtime_error.
 */
template <typename... Args>
[[noreturn]] void ThrowVariantError(const std::variant<Args...>& error_variant) {
  std::visit([](const auto& e) { throw e; }, error_variant);
  XGRAMMAR_UNREACHABLE();
}

/*!
 * \brief Get the message from a variant of multiple error types.
 * \param error_variant The variant of multiple error types.
 * \return The message from the error variant.
 * \tparam Args The types of the error types. Each type should inherit from std::runtime_error.
 */
template <typename... Args>
std::string GetMessageFromVariantError(const std::variant<Args...>& error_variant) {
  return std::visit([](const auto& e) { return e.what(); }, error_variant);
}

/*!
 * \brief Get the type name from a variant of XGrammarError types (each has GetType()).
 * \param error_variant The variant of multiple error types.
 * \return The type string from the error variant (e.g. "DeserializeVersionError").
 * \tparam Args The types of the error types. Each type should have GetType() const.
 */
template <typename... Args>
std::string GetTypeFromVariantError(const std::variant<Args...>& error_variant) {
  return std::visit([](const auto& e) { return e.GetType(); }, error_variant);
}

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_UTILS_H_
