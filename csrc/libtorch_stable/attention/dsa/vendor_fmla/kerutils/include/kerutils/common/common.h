#pragma once

namespace kerutils {}

#define KU_PRINTLN(fmt, ...)         \
  {                                  \
    cute::print(fmt, ##__VA_ARGS__); \
    print("\n");                     \
  }

namespace ku = kerutils;
