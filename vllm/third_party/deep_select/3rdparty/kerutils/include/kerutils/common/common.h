#pragma once

namespace kerutils {}

#define KU_PRINTLN(fmt, ...) { cute::print(fmt, ##__VA_ARGS__); print("\n"); }

namespace ku = kerutils;

#ifdef __CUDACC__
#define KERUTILS_IS_BUILD_ON_CUDA
#endif

#ifndef KERUTILS_IS_BUILD_ON_CUDA
#error "KERUTILS_IS_BUILD_ON_CUDA must be defined. It is defined automatically when compiling with NVCC; pass `-DKERUTILS_IS_BUILD_ON_CUDA` when compiling with a host compiler."
#endif
