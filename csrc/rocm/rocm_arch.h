#pragma once

#if defined(__HIPCC__) && (defined(__gfx908__) || defined(__gfx90a__) || \
                           defined(__gfx942__) || defined(__gfx950__))
  #define __HIP__GFX9__
  #if defined(__gfx908__)
    #define __HIP__GFX9__CNDA__ 1
  #elif defined(__gfx90a__)
    #define __HIP__GFX9__CNDA__ 2
  #elif defined(__gfx942__)
    #define __HIP__GFX9__CNDA__ 3
    #define __HIP__GFX9__CDNA_FP8_EN__
  #elif defined(__gfx950__)
    #define __HIP__GFX9__CNDA__ 3
    #define __HIP__GFX9__CDNA_FP8_EN__
    #define __HIP__GFX9__CDNA_FP4_EN__
  #endif
#endif

#if defined(__HIPCC__) && (defined(__gfx942__) || defined(__gfx950__))
  #define __HIP__MI3XX__
#endif

#if defined(__HIPCC__) && (defined(__gfx1100__) || defined(__gfx1101__))
  #define __HIP__GFX11__
#endif

#if defined(__HIPCC__) && (defined(__gfx1200__) || defined(__gfx1201__))
  #define __HIP__GFX12__
#endif

#if defined(__gfx950__)
  #define LDS_SIZE 160 * 1024
#else
  #define LDS_SIZE 64 * 1024
#endif
