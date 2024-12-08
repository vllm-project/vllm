#pragma once

template <typename A, typename B>
static inline constexpr auto div_ceil(A a, B b) {
  return (a + b - 1) / b;
}

// Compute the next multiple of a that is greater than or equal to b
template <typename A, typename B>
static inline constexpr auto next_multiple_of(A a, B b) {
  return div_ceil(b, a) * a;
}

// Compute the largest multiple of a that is less than or equal to b
template <typename A, typename B>
static inline constexpr auto prev_multiple_of(A a, B b) {
  return (b / a) * a;
}