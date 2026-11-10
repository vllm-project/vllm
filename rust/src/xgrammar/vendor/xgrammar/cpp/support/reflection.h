/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/support/reflection.h
 * \brief The header for compile-time reflection.
 */

#ifndef XGRAMMAR_SUPPORT_REFLECTION_H_
#define XGRAMMAR_SUPPORT_REFLECTION_H_

#include <array>
#include <cstddef>
#include <optional>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace xgrammar {

/******************** Core Reflection Types ********************/

/*!
 * \brief The type of the member trait.
 */
enum class member_type {
  kNone = 0,    // this is default, which has no member trait
  kConfig = 1,  // this is a config with member pointers
};

/**
 * \brief Base trait for member traits.
 *
 * \tparam T the type whose members are being reflected
 * \details Provides a default trait indicating no members.
 */
template <typename T>
struct member_trait {
  static constexpr auto value = member_type::kNone;
};

/******************** STL and Custom Type Traits ********************/

template <typename>
struct is_std_array : std::false_type {};

template <typename T, std::size_t N>
struct is_std_array<std::array<T, N>> : std::true_type {};

template <typename>
struct is_std_pair : std::false_type {};

template <typename... R>
struct is_std_pair<std::pair<R...>> : std::true_type {};

template <typename T>
struct is_std_tuple : std::false_type {};

template <typename... R>
struct is_std_tuple<std::tuple<R...>> : std::true_type {};

template <typename>
struct is_std_optional : std::false_type {};

template <typename T>
struct is_std_optional<std::optional<T>> : std::true_type {};

template <typename>
struct is_std_vector : std::false_type {};

template <typename... R>
struct is_std_vector<std::vector<R...>> : std::true_type {};

template <typename>
struct is_std_unordered_map : std::false_type {};

template <typename... R>
struct is_std_unordered_map<std::unordered_map<R...>> : std::true_type {};

template <typename T>
struct is_std_unordered_set : std::false_type {};

template <typename... R>
struct is_std_unordered_set<std::unordered_set<R...>> : std::true_type {};

/*!
 * \brief XGrammar specific: Check if a class is a PImpl class.
 */
template <typename T, typename = void>
struct is_pimpl_class : std::false_type {};

/*!
 * \brief XGrammar specific: Check if a class is a PImpl class. It's true iff the class has a
 * member `Impl` and the class is not the same as the `Impl` type.
 */
template <typename T>
struct is_pimpl_class<
    T,
    std::void_t<typename T::Impl, std::enable_if_t<!std::is_same_v<T, typename T::Impl>, void>>>
    : std::true_type {};

/*!
 * \brief A helper class to print the value when the condition is false.
 */
template <auto condition, auto value>
struct DebugAssert {
  static_assert(condition);
};

/******************** Implementation Details ********************/

namespace detail::reflection {

// We cannot use `static_assert(false)` even in unreachable code in `if constexpr`.
// See https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/p2593r1.html
// for more details.
// TL;DR: We use the following `false_v` as a workaround.
template <typename T>
inline constexpr bool false_v = false;

// Note that we don't allow empty tables now (that's uncommon).
template <typename X, typename Y, typename... Args>
inline constexpr auto make_member_table(X, Y second, Args... args) {
  static_assert(sizeof...(args) % 2 == 0, "member table must be even");
  static_assert(std::is_same_v<X, const char*>, "first member must be a c-string");
  static_assert(std::is_member_pointer_v<Y>, "second member must be a member pointer");
  if constexpr (sizeof...(args) == 0) {
    return std::make_tuple(second);
  } else {
    return std::tuple_cat(std::make_tuple(second), make_member_table(args...));
  }
}

template <std::size_t... Idx, typename Tuple>
inline constexpr auto make_name_table_aux(std::index_sequence<Idx...>, Tuple tuple) {
  return std::array{std::get<Idx * 2>(tuple)...};
}

template <typename... Args>
inline constexpr auto make_name_table(Args... args) {
  constexpr auto N = sizeof...(args);
  static_assert(N % 2 == 0, "name table must be even");
  return make_name_table_aux(std::make_index_sequence<N / 2>{}, std::make_tuple(args...));
}

template <typename Ftor, typename Fn, std::size_t... Idx>
inline void visit_config_impl(Fn&& fn, std::index_sequence<Idx...>) {
  // This is a helper function to visit each member of the config.
  // It uses fold expression to apply the function to each member.
  static_assert(Ftor::value == member_type::kConfig, "T must be a config type");
  static constexpr auto get_name = [](std::size_t idx) {
    if constexpr (Ftor::has_names) {
      return Ftor::names[idx];
    } else {
      return "";
    }
  };
  return (fn(std::get<Idx>(Ftor::members), get_name(Idx), Idx), ...);
}

}  // namespace detail::reflection

/******************** Member Functors and Visitors ********************/

/*!
 * \brief A functor that provides access to the members of a config type.
 * It extracts the members from the `member_trait` specialization for the type `T`.
 * A valid `member_trait` specialization must meet the following requirements:
 * - It must have a static member `value` of type `member_type`,
 *   which must be either `kNone` or `kConfig`.
 */
template <typename T, member_type = member_trait<T>::value>
struct member_functor {
  static_assert(detail::reflection::false_v<T>, "This specialization should never be used");
};

/*!
 * \brief A specialization of `member_functor` for config types.
 * A valid `member_trait` specialization for a config type must meet the following:
 * - It must have a static member `value` of type `member_type::kConfig`.
 * - It must have a static tuple `members` that contains the member pointers.
 * - It must have a static array `names` that contains the names of the members.
 * - The size of `names` must be either 0 or equal to the number of members in `members`.
 *   - In the first case, `names` will be empty.
 *   - In the second case, `names` represent the printed name of each member.
 */
template <typename T>
struct member_functor<T, member_type::kConfig> {
 private:
  using _trait_t = member_trait<T>;
  using _members_t = std::decay_t<decltype(_trait_t::members)>;
  using _names_t = std::decay_t<decltype(_trait_t::names)>;

 public:
  static constexpr auto value = member_type::kConfig;
  static constexpr auto members = _trait_t::members;
  static constexpr auto names = _trait_t::names;
  static constexpr auto member_count = std::tuple_size_v<_members_t>;
  static constexpr auto has_names = names.size() == member_count;

  // some static_asserts to check the member list and name list
  static_assert(is_std_tuple<_members_t>::value, "Member list must be a tuple");
  static_assert(is_std_array<_names_t>::value, "Name list must be an array");
  static_assert(member_count > 0, "Member list must not be empty");
  static_assert(
      names.size() == member_count || names.size() == 0,
      "Name list must be empty or have the same size as member list"
  );
};

/*!
 * \brief Visit the members of a config type.
 * \tparam T The type of the config.
 * \tparam Fn The type of the function to visit the members.
 * \param fn The function to visit the members. fn's signature should be:
 * \code{.cpp}
 * (auto ptr, const char* name, size_t idx) -> void
 * \endcode
 * where `ptr` is the pointer to the member, `name` is the name of the member, and `idx` is the
 * index of the member.
 */
template <typename T, typename Fn>
inline void visit_config(Fn&& fn) {
  using Ftor = member_functor<T>;
  return detail::reflection::visit_config_impl<Ftor>(
      fn, std::make_index_sequence<Ftor::member_count>{}
  );
}

/******************** Registration Macros ********************/

/**
 * \brief Macros to define member traits for types.
 * \details These macros are used to define the structural information of types
 * for serialization and reflection purposes.
 *
 * Macros:
 *   - \c XGRAMMAR_MEMBER_TABLE: Defines a type with a table of (name, member pointer) pairs.
 *   - \c XGRAMMAR_MEMBER_ARRAY: Defines a type with an array of member pointers.
 *
 * Use the `_TEMPLATE` variants for template types.
 *
 * \example
 * \code{.cpp}
 * // Example of using XGRAMMAR_MEMBER_TABLE to register (name, member pointer) pairs
 * struct SimpleClass {
 *   int a;
 *   double b;
 * };
 * XGRAMMAR_MEMBER_TABLE(SimpleClass, "name_a", &SimpleClass::a, "name_b", &SimpleClass::b);
 *
 * // Or register members as an array with XGRAMMAR_MEMBER_ARRAY
 * XGRAMMAR_MEMBER_ARRAY(SimpleClass, &SimpleClass::a, &SimpleClass::b);
 *
 * // Example of using XGRAMMAR_MEMBER_ARRAY to register members from a derived class
 * struct Derived : SimpleClass {
 *   std::string c;
 * };
 * XGRAMMAR_MEMBER_TABLE(Derived, "name_a", &Derived::a, "name_b", &Derived::b, "name_c",
 * &Derived::c);
 *
 * // Example of using XGRAMMAR_MEMBER_ARRAY_TEMPLATE for a template type
 * // If the default constructor/member is private, you need to declare a friend for member_trait.
 * template <typename T>
 * struct TemplateClass {
 * private:
 *   T value;
 *   TemplateClass() = default;
 *   friend struct member_trait<TemplateClass>;
 * };
 * template <typename T>
 * XGRAMMAR_MEMBER_ARRAY_TEMPLATE(TemplateClass<T>, &TemplateClass<T>::value);
 * \endcode
 */
#define XGRAMMAR_MEMBER_TABLE_TEMPLATE(Type, ...)                                       \
  struct member_trait<Type> {                                                           \
    static constexpr auto value = member_type::kConfig;                                 \
    static constexpr auto members = detail::reflection::make_member_table(__VA_ARGS__); \
    static constexpr auto names = detail::reflection::make_name_table(__VA_ARGS__);     \
  }

#define XGRAMMAR_MEMBER_ARRAY_TEMPLATE(Type, ...)                 \
  struct member_trait<Type> {                                     \
    static constexpr auto value = member_type::kConfig;           \
    static constexpr auto members = std::make_tuple(__VA_ARGS__); \
    static constexpr auto names = std::array<const char*, 0>{};   \
  }

#define XGRAMMAR_MEMBER_TABLE(Type, ...) \
  template <>                            \
  XGRAMMAR_MEMBER_TABLE_TEMPLATE(Type, __VA_ARGS__)

#define XGRAMMAR_MEMBER_ARRAY(Type, ...) \
  template <>                            \
  XGRAMMAR_MEMBER_ARRAY_TEMPLATE(Type, __VA_ARGS__)

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_REFLECTION_H_
