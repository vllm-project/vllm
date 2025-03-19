// clang-format will break include orders
// clang-format off
#include <cudaTypedefs.h>

#if defined CUDA_VERSION && CUDA_VERSION >= 12020
#include "sparse_scaled_mm_c3x.cuh"
// clang-format on

using namespace cute;
using namespace vllm;

struct GemmCallerTraits {
  using return_type = void;

  template <typename GemmConfig, typename... Args>
  static return_type invoke(Args&&... args) {
    return cutlass_sparse_gemm_caller<GemmConfig>(std::forward<Args>(args)...);
  }
};

struct GemmCompressorTraits {
  using return_type = CompressorResult;

  template <typename GemmConfig, typename... Args>
  static return_type invoke(Args&&... args) {
    return cutlass_sparse_compress<GemmConfig>(std::forward<Args>(args)...);
  }
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue,
          typename DispatchFunc, typename... Args>
typename DispatchFunc::return_type cutlass_gemm_sm90_fp8_dispatch(
    uint32_t m, uint32_t n, Args&&... args) {
  static_assert(std::is_same_v<InType, cutlass::float_e4m3_t>);

  using Cutlass3xGemmDefault =
      typename sm90_config_default<InType, OutType, Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemmM64 =
      typename sm90_fp8_config_M64<InType, OutType, Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemmM128 =
      typename sm90_fp8_config_M128<InType, OutType, Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemmM256 =
      typename sm90_fp8_config_M256<InType, OutType, Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemmM512 =
      typename sm90_fp8_config_M512<InType, OutType, Epilogue>::Cutlass3xGemm;

  using Cutlass3xGemm1 =
      typename sm90_fp8_config_1<InType, OutType, Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemm2 =
      typename sm90_fp8_config_2<InType, OutType, Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemm3 =
      typename sm90_fp8_config_3<InType, OutType, Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemm4 =
      typename sm90_fp8_config_4<InType, OutType, Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemm5 =
      typename sm90_fp8_config_5<InType, OutType, Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemm6 =
      typename sm90_fp8_config_6<InType, OutType, Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemm7 =
      typename sm90_fp8_config_7<InType, OutType, Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemm8 =
      typename sm90_fp8_config_8<InType, OutType, Epilogue>::Cutlass3xGemm;

  uint32_t const mp2 =
      std::max(static_cast<uint32_t>(64), next_pow_2(m));  // next power of 2

  if (mp2 <= 64) {
    if (n == 28672) {
      return DispatchFunc::template invoke<Cutlass3xGemm2>(
          std::forward<Args>(args)...);
    } else if (n == 4096 || n == 6144) {
      return DispatchFunc::template invoke<Cutlass3xGemm1>(
          std::forward<Args>(args)...);
    }
  } else if (mp2 <= 128) {
    if (n == 4096) {
      return DispatchFunc::template invoke<Cutlass3xGemm3>(
          std::forward<Args>(args)...);
    } else if (n == 28672) {
      return DispatchFunc::template invoke<Cutlass3xGemm5>(
          std::forward<Args>(args)...);
    } else if (n == 6144) {
      return DispatchFunc::template invoke<Cutlass3xGemm4>(
          std::forward<Args>(args)...);
    }
  } else if (mp2 <= 256) {
    if (n == 4096) {
      return DispatchFunc::template invoke<Cutlass3xGemm6>(
          std::forward<Args>(args)...);
    } else if (n == 28672) {
      return DispatchFunc::template invoke<Cutlass3xGemm8>(
          std::forward<Args>(args)...);
    } else if (n == 6144) {
      return DispatchFunc::template invoke<Cutlass3xGemm7>(
          std::forward<Args>(args)...);
    }
  } else {
    if (n == 6144 || n == 28672) {
      return DispatchFunc::template invoke<Cutlass3xGemm8>(
          std::forward<Args>(args)...);
    } else if (n == 4096) {
      return DispatchFunc::template invoke<Cutlass3xGemm7>(
          std::forward<Args>(args)...);
    }
  }

  // Otherwise the default heuristic
  if (mp2 <= 64) {
    // n in [1, 64]
    return DispatchFunc::template invoke<Cutlass3xGemmM64>(
        std::forward<Args>(args)...);
  } else if (mp2 <= 128) {
    // n in (64, 128]
    return DispatchFunc::template invoke<Cutlass3xGemmM128>(
        std::forward<Args>(args)...);
  } else if (mp2 <= 256) {
    // n in (128, 256]
    return DispatchFunc::template invoke<Cutlass3xGemmM256>(
        std::forward<Args>(args)...);
  } else {
    // n in (256, inf)
    return DispatchFunc::template invoke<Cutlass3xGemmM512>(
        std::forward<Args>(args)...);
  }
}

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue,
          typename DispatchFunc, typename... Args>
typename DispatchFunc::return_type cutlass_gemm_sm90_16bit_dispatch(
    uint32_t m, uint32_t n, Args&&... args) {
  using Cutlass3xGemmDefault =
      typename sm90_config_default<InType, OutType, Epilogue>::Cutlass3xGemm;

  return DispatchFunc::template invoke<Cutlass3xGemmDefault>(
      std::forward<Args>(args)...);
}

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue,
          typename DispatchFunc, typename... Args>
typename DispatchFunc::return_type cutlass_gemm_sm90_int8_dispatch(
    uint32_t m, uint32_t n, Args&&... args) {
  static_assert(std::is_same_v<InType, int8_t>);

  using Cutlass3xGemmDefault =
      typename sm90_config_default<InType, OutType, Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemmM128 =
      typename sm90_int8_config_M128<InType, OutType, Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemmM64 =
      typename sm90_int8_config_M64<InType, OutType, Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemmM32NBig =
      typename sm90_int8_config_M32_NBig<InType, OutType,
                                         Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemmM32NSmall =
      typename sm90_int8_config_M32_NSmall<InType, OutType,
                                           Epilogue>::Cutlass3xGemm;

  bool const is_small_n = n < 8192;
  uint32_t const mp2 =
      std::max(static_cast<uint32_t>(32), next_pow_2(m));  // next power of 2

  if (mp2 <= 32) {
    // m in [1, 32]
    if (is_small_n) {
      return DispatchFunc::template invoke<Cutlass3xGemmM32NSmall>(
          std::forward<Args>(args)...);
    } else {
      return DispatchFunc::template invoke<Cutlass3xGemmM32NBig>(
          std::forward<Args>(args)...);
    }
  } else if (mp2 <= 64) {
    // m in (32, 64]
    return DispatchFunc::template invoke<Cutlass3xGemmM64>(
        std::forward<Args>(args)...);
  } else if (mp2 <= 128) {
    // m in (64, 128]
    return DispatchFunc::template invoke<Cutlass3xGemmM128>(
        std::forward<Args>(args)...);
  } else {
    // m in (128, inf)
    return DispatchFunc::template invoke<Cutlass3xGemmDefault>(
        std::forward<Args>(args)...);
  }
}

// Dispatch to GEMM implementations based on element types
template <template <typename, typename, typename> typename Epilogue,
          typename... EpilogueArgs>
void cutlass_scaled_sparse_mm_sm90_epilogue(torch::Tensor& out,
                                            torch::Tensor const& a,
                                            torch::Tensor const& bt_nzs,
                                            torch::Tensor const& bt_meta,
                                            EpilogueArgs&&... epilogue_args) {
  uint32_t const m = out.size(0);
  uint32_t const n = out.size(1);

  // TODO: add dispatch functions to all of these
  TORCH_CHECK(bt_meta.dtype() == torch::kUInt8);
  if (a.dtype() == torch::kInt8) {
    TORCH_CHECK(bt_nzs.dtype() == torch::kInt8);

    if (out.dtype() == torch::kBFloat16) {
      return cutlass_gemm_sm90_int8_dispatch<int8_t, cutlass::bfloat16_t,
                                             Epilogue, GemmCallerTraits>(
          m, n, out, a, bt_nzs, bt_meta,
          std::forward<EpilogueArgs>(epilogue_args)...);
    } else {
      TORCH_CHECK(out.dtype() == torch::kFloat16);
      return cutlass_gemm_sm90_int8_dispatch<int8_t, cutlass::half_t, Epilogue,
                                             GemmCallerTraits>(
          m, n, out, a, bt_nzs, bt_meta,
          std::forward<EpilogueArgs>(epilogue_args)...);
    }
  } else if (a.dtype() == torch::kFloat8_e4m3fn) {
    TORCH_CHECK(bt_nzs.dtype() == torch::kFloat8_e4m3fn);

    if (out.dtype() == torch::kBFloat16) {
      return cutlass_gemm_sm90_fp8_dispatch<cutlass::float_e4m3_t,
                                            cutlass::bfloat16_t, Epilogue,
                                            GemmCallerTraits>(
          m, n, out, a, bt_nzs, bt_meta,
          std::forward<EpilogueArgs>(epilogue_args)...);
    } else {
      TORCH_CHECK(out.dtype() == torch::kFloat16);
      return cutlass_gemm_sm90_fp8_dispatch<
          cutlass::float_e4m3_t, cutlass::half_t, Epilogue, GemmCallerTraits>(
          m, n, out, a, bt_nzs, bt_meta,
          std::forward<EpilogueArgs>(epilogue_args)...);
    }
  } else if (a.dtype() == torch::kFloat16) {
    TORCH_CHECK(bt_nzs.dtype() == torch::kFloat16);
    TORCH_CHECK(out.dtype() == torch::kFloat16);

    return cutlass_gemm_sm90_16bit_dispatch<cutlass::half_t, cutlass::half_t,
                                            Epilogue, GemmCallerTraits>(
        m, n, out, a, bt_nzs, bt_meta,
        std::forward<EpilogueArgs>(epilogue_args)...);
  } else {  // a.dtype() == torch::kBFloat16
    TORCH_CHECK(a.dtype() == torch::kBFloat16);
    TORCH_CHECK(bt_nzs.dtype() == torch::kBFloat16);
    TORCH_CHECK(out.dtype() == torch::kBFloat16);

    return cutlass_gemm_sm90_16bit_dispatch<
        cutlass::bfloat16_t, cutlass::bfloat16_t, Epilogue, GemmCallerTraits>(
        m, n, out, a, bt_nzs, bt_meta,
        std::forward<EpilogueArgs>(epilogue_args)...);
  }
}

void cutlass_scaled_sparse_mm_sm90(torch::Tensor& out, torch::Tensor const& a,
                                   torch::Tensor const& bt_nzs,
                                   torch::Tensor const& bt_meta,
                                   torch::Tensor const& a_scales,
                                   torch::Tensor const& b_scales,
                                   std::optional<torch::Tensor> const& bias) {
  TORCH_CHECK(bt_meta.dtype() == torch::kUInt8);
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);

  if (bias) {
    TORCH_CHECK(bias->dtype() == out.dtype(),
                "CUTLASS scaled_mm bias dtype must match output dtype ",
                out.dtype());
    return cutlass_scaled_sparse_mm_sm90_epilogue<
        c3x::ScaledEpilogueColumnBias>(out, a, bt_nzs, bt_meta, b_scales,
                                       a_scales, *bias);
  } else {
    return cutlass_scaled_sparse_mm_sm90_epilogue<c3x::ScaledEpilogue>(
        out, a, bt_nzs, bt_meta, b_scales, a_scales);
  }
}

CompressorResult cutlass_sparse_compress_sm90(torch::Tensor const& a) {
  // These m and n variables are fordispatching to different GEMM algorithms.
  uint32_t const m = 1;  // Set M to 1 for compression
  uint32_t const n = a.size(1);

  // Note: For correctess, the compressed format must be invariant in:
  //  - M, the flattened number of tokens
  //  - Whether output dtype is fp16 or bf16
  //  - CUTLASS epilogues

  if (a.dtype() == torch::kInt8) {
    return cutlass_gemm_sm90_int8_dispatch<int8_t, cutlass::bfloat16_t,
                                           c3x::TrivialEpilogue,
                                           GemmCompressorTraits>(m, n, a);
  } else if (a.dtype() == torch::kFloat8_e4m3fn) {
    return cutlass_gemm_sm90_fp8_dispatch<
        cutlass::float_e4m3_t, cutlass::bfloat16_t, c3x::TrivialEpilogue,
        GemmCompressorTraits>(m, n, a);
  } else if (a.dtype() == torch::kFloat16) {
    return cutlass_gemm_sm90_16bit_dispatch<
        cutlass::bfloat16_t, cutlass::bfloat16_t, c3x::TrivialEpilogue,
        GemmCompressorTraits>(m, n, a);
  } else {
    TORCH_CHECK(a.dtype() == torch::kBFloat16,
                "cutlass_sparse_compress only supports int8, fp8_e4m3, fp16, "
                "and bf16 datatypes");
    return cutlass_gemm_sm90_16bit_dispatch<cutlass::half_t, cutlass::half_t,
                                            c3x::TrivialEpilogue,
                                            GemmCompressorTraits>(m, n, a);
  }
}

#endif
