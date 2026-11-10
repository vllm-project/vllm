/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/support/memory_size.h
 * \brief Compute the memory consumption of a container in heap memory.
 */

#ifndef XGRAMMAR_SUPPORT_MEMORY_SIZE_H_
#define XGRAMMAR_SUPPORT_MEMORY_SIZE_H_

#include <cstddef>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "reflection.h"

namespace xgrammar {

/******************* MemorySize Procotol *******************/

template <typename T>
inline constexpr std::size_t MemorySize(const T& value);

template <typename T1, typename T2>
inline constexpr std::size_t MemorySize(const std::pair<T1, T2>& pair);

template <typename... Ts>
inline constexpr std::size_t MemorySize(const std::tuple<Ts...>& tpl);

template <typename T>
inline constexpr std::size_t MemorySize(const std::optional<T>& optional_value);

inline std::size_t MemorySize(const std::vector<bool>& value);

/******************* MemorySize Implementations *******************/

namespace detail::memory_size {

/*!
 * \brief Get the element type of a container.
 */
template <typename Container>
using ElementType = std::decay_t<decltype(*std::begin(Container()))>;

/*!
 * \brief A false value for static_assert.
 */
template <typename>
inline constexpr bool false_v = false;

}  // namespace detail::memory_size

/*!
 * \brief Compute the memory consumption of a value.
 * \tparam T The type of the value.
 * \param value The value.
 * \return The memory consumption in heap memory of the value in bytes.
 */
template <typename T>
inline constexpr std::size_t MemorySize(const T& value) {
  if constexpr (is_pimpl_class<T>::value) {
    // Customized MemorySize
    return MemorySize(*value.ImplPtr());
  } else if constexpr (std::is_trivially_copyable_v<T>) {
    // Primitive type
    return 0;
  } else if constexpr (std::is_trivially_copyable_v<detail::memory_size::ElementType<T>>) {
    // Container of primitive type
    return sizeof(detail::memory_size::ElementType<T>) * std::size(value);
  } else if constexpr (!std::is_trivially_copyable_v<detail::memory_size::ElementType<T>>) {
    // Container of non-primitive type: sum up the memory size of all elements
    std::size_t size = sizeof(detail::memory_size::ElementType<T>) * std::size(value);
    for (const auto& element : value) {
      size += MemorySize(element);
    }
    return size;
  } else {
    static_assert(detail::memory_size::false_v<T>, "MemorySize is not implemented for this type");
  }
}

/*!
 * \brief Compute the memory consumption of a pair.
 * \tparam T1 The type of the first element.
 * \tparam T2 The type of the second element.
 * \param pair The pair.
 * \return The memory consumption in heap memory of the pair.
 */
template <typename T1, typename T2>
inline constexpr std::size_t MemorySize(const std::pair<T1, T2>& pair) {
  return MemorySize(pair.first) + MemorySize(pair.second);
}

/*!
 * \brief Compute the memory consumption of a tuple.
 * \tparam Ts The types of the tuple.
 * \param tpl The tuple.
 * \return The memory consumption in heap memory of the tuple.
 */
template <typename... Ts>
inline constexpr std::size_t MemorySize(const std::tuple<Ts...>& tpl) {
  return std::apply([](auto&&... elems) { return (MemorySize(elems) + ... + 0); }, tpl);
}

/*!
 * \brief Compute the memory consumption in heap memory. This function is specialized for
 * std::optional.
 * \tparam Tp The type of the optional.
 * \param range The optional.
 * \return The memory consumption in heap memory of the optional.
 */
template <typename T>
inline constexpr std::size_t MemorySize(const std::optional<T>& optional_value) {
  return optional_value.has_value() ? MemorySize(*optional_value) : 0;
}

/*!
 * \brief Compute the memory consumption of a std::vector<bool>, which is bit-packed: it stores
 * one bit (not one byte) per element. The generic container overload would over-count it by 8x.
 * \param value The vector<bool>.
 * \return The memory consumption in heap memory of the vector<bool> in bytes.
 */
inline std::size_t MemorySize(const std::vector<bool>& value) { return (value.size() + 7) / 8; }

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_MEMORY_SIZE_H_
