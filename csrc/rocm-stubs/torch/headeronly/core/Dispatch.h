#pragma once

#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/Exception.h>

#define THO_PRIVATE_CASE_TYPE_USING_HINT_TMPL(PRELUDE, enum_type, HINT, ...) \
  case enum_type: {                                                          \
    PRELUDE(enum_type);                                                      \
    using HINT =                                                             \
        torch::headeronly::impl::ScalarTypeToCPPTypeT<enum_type>;            \
    [&]() -> decltype(auto) { return __VA_ARGS__(); }();                     \
  } break;

#define THO_DISPATCH_CASE_TMPL(CASE_TYPE_USING_HINT, enum_type, ...) \
  CASE_TYPE_USING_HINT(enum_type, scalar_t, __VA_ARGS__)

inline torch::headeronly::ScalarType scalar_type(torch::headeronly::ScalarType s) { return s; }

#define THO_DISPATCH_SWITCH_TMPL(                                           \
    PRELUDE, CHECK_NOT_IMPLEMENTED, TYPE, NAME, ...)                        \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    constexpr const char* at_dispatch_name = NAME;                          \
    torch::headeronly::ScalarType _st = ::scalar_type(the_type);    \
    PRELUDE(at_dispatch_name, _st);                                         \
    switch (_st) {                                                          \
      __VA_ARGS__                                                           \
      default:                                                              \
        CHECK_NOT_IMPLEMENTED(                                              \
            false,                                                          \
            '"',                                                            \
            at_dispatch_name,                                               \
            "\" not implemented for '",                                     \
            torch::headeronly::toString(_st),                               \
            "'");                                                           \
    }                                                                       \
  }()

#define THO_EMPTY(...)

#define THO_PRIVATE_CASE_TYPE_USING_HINT(enum_type, HINT, ...) \
  THO_PRIVATE_CASE_TYPE_USING_HINT_TMPL(THO_EMPTY, enum_type, HINT, __VA_ARGS__)

#define THO_DISPATCH_SWITCH(TYPE, NAME, ...) \
  THO_DISPATCH_SWITCH_TMPL(THO_EMPTY, STD_TORCH_CHECK, TYPE, NAME, __VA_ARGS__)

#define THO_DISPATCH_CASE(enum_type, ...) \
  THO_DISPATCH_CASE_TMPL(THO_PRIVATE_CASE_TYPE_USING_HINT, enum_type, __VA_ARGS__)
